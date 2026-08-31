"""An async OpenAI-compatible chat client shared by all three providers.

All three providers speak the OpenAI chat-completions protocol, so this is one adapter
with different base URLs rather than three SDKs -- which is the reason the project can
afford three model families at all.

Call order is deliberate: **cache, then quota, then network.** A cached call costs no
quota, and quota is checked *before* the request rather than discovered via a 429, so
budget is spent on purpose.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from codesage.config.settings import ModelSpec, Registry, Settings
from codesage.llm.cache import ResponseCache, make_key
from codesage.llm.errors import (
    PermanentProviderError,
    QuotaExhaustedError,
    RateLimitedError,
    TransientProviderError,
)
from codesage.llm.quota import QuotaTracker, QuotaVerdict

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    model_key: str
    family: str
    usage: dict[str, int]
    cached: bool = False
    attempts: int = 1
    # The assistant message exactly as the provider returned it. The agent loop needs
    # this whole object -- tool calls and all -- to append back into the conversation;
    # reconstructing it from `content` would drop the tool_calls array.
    raw_message: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(self.usage.get("total_tokens", 0))

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self.raw_message.get("tool_calls") or []


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough pre-flight token estimate, used only for quota reservation.

    Deliberately crude -- ~4 characters per token. It never needs to be accurate, only
    conservative enough that we do not overshoot a daily cap by a wide margin.
    """
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return chars // 4 + 256


class LLMClient:
    def __init__(
        self,
        registry: Registry,
        settings: Settings,
        *,
        cache: ResponseCache | None = None,
        quota: QuotaTracker | None = None,
        http: httpx.AsyncClient | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.cache = cache or ResponseCache(settings.cache_dir)
        self.quota = quota or QuotaTracker(
            settings.state_db, {n: p.limits for n, p in registry.providers.items()}
        )
        self._http = http
        self._owns_http = http is None
        # Injectable so the tests can exercise retry and backoff logic without
        # actually waiting for it.
        self._sleep = sleeper

    async def __aenter__(self) -> LLMClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.settings.request_timeout)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.settings.request_timeout)
        return self._http

    async def complete(
        self,
        model: ModelSpec,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str = "review",
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """One chat completion, with cache, quota, and retry handled.

        When `tools` is given, JSON mode is switched off: providers reject the two
        together, since a tool call is not the JSON object `response_format` promises.
        """
        provider = self.registry.provider_for(model)
        params: dict[str, Any] = {
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_output_tokens,
            "json_mode": json_mode and not tools,
            # Tool names go in the cache key: the same prompt with and without tools is
            # a different request and will get a different answer, so they must not
            # collide on one cache entry.
            "tools": sorted(t["function"]["name"] for t in tools) if tools else None,
        }

        key = make_key(model_key=model.key, messages=messages, params=params)
        if (hit := self.cache.get(key)) is not None:
            log.debug("cache hit for %s", model.key)
            return LLMResponse(
                content=hit.content,
                model_key=model.key,
                family=model.family,
                usage=hit.usage,
                cached=True,
                raw_message=hit.raw_message or {"role": "assistant", "content": hit.content},
            )

        if not provider.configured:
            raise QuotaExhaustedError(
                f"no API key set ({provider.api_key_env})",
                provider=provider.name,
                model=model.id,
            )

        estimate = _estimate_tokens(messages)
        last_error: Exception | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            decision = self.quota.check(provider.name, estimated_tokens=estimate)

            if decision.verdict is QuotaVerdict.EXHAUSTED:
                self.quota.note_degradation(provider.name, stage, decision.reason)
                raise QuotaExhaustedError(decision.reason, provider=provider.name, model=model.id)

            if decision.verdict is QuotaVerdict.RATE_LIMITED:
                log.info(
                    "%s rate limited, sleeping %.1fs (%s)",
                    provider.name,
                    decision.wait_seconds,
                    decision.reason,
                )
                await self._sleep(decision.wait_seconds)
                continue

            try:
                content, usage, raw_message = await self._post(model, messages, params, tools)
            except RateLimitedError as exc:
                last_error = exc
                # The provider knows better than our local counter; respect its delay.
                self.quota.record(provider.name, tokens=0)
                log.info("%s returned 429, backing off %.1fs", provider.name, exc.retry_after)
                await self._sleep(exc.retry_after)
                continue
            except TransientProviderError as exc:
                last_error = exc
                backoff = min(2.0**attempt, 30.0)
                log.info(
                    "%s transient failure (%s), retrying in %.1fs", provider.name, exc, backoff
                )
                await self._sleep(backoff)
                continue

            self.quota.record(provider.name, tokens=int(usage.get("total_tokens", estimate)))
            self.cache.put(key, content=content, usage=usage, raw_message=raw_message)
            return LLMResponse(
                content=content,
                model_key=model.key,
                family=model.family,
                usage=usage,
                attempts=attempt,
                raw_message=raw_message,
            )

        raise TransientProviderError(
            f"exhausted {self.settings.max_retries} attempts: {last_error}",
            provider=provider.name,
            model=model.id,
        )

    async def _post(
        self,
        model: ModelSpec,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, int], dict[str, Any]]:
        provider = self.registry.provider_for(model)
        body: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
        }
        if params.get("json_mode"):
            # Not every free-tier model honours this, which is exactly why the parser
            # downstream also strips fences and hunts for the first JSON object.
            body["response_format"] = {"type": "json_object"}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        headers = {"Authorization": f"Bearer {provider.api_key}"}
        if provider.name == "openrouter":
            # OpenRouter asks callers to identify themselves; it also affects free-tier
            # routing priority.
            headers["HTTP-Referer"] = "https://github.com/codesage-review/codesage"
            headers["X-Title"] = "CodeSage"

        try:
            resp = await self.http.post(
                f"{provider.base_url}/chat/completions", json=body, headers=headers
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientProviderError(
                f"transport failure: {exc}", provider=provider.name, model=model.id
            ) from exc

        if resp.status_code == 429:
            raise RateLimitedError(
                "provider rate limit",
                retry_after=_retry_after(resp),
                provider=provider.name,
                model=model.id,
            )
        if resp.status_code >= 500:
            raise TransientProviderError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                provider=provider.name,
                model=model.id,
            )
        if resp.status_code >= 400:
            raise PermanentProviderError(
                f"HTTP {resp.status_code}: {resp.text[:300]}",
                provider=provider.name,
                model=model.id,
            )

        data = resp.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise PermanentProviderError(
                f"unexpected response shape: {str(data)[:200]}",
                provider=provider.name,
                model=model.id,
            ) from exc

        # A message carrying tool calls has `content: null`, which is not an error --
        # the tool calls are the payload. Normalising to "" keeps every caller that
        # only wants text from having to guard against None.
        content = message.get("content") or ""
        usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int)}
        return content, usage, message


def _retry_after(resp: httpx.Response) -> float:
    raw = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-requests")
    if raw:
        try:
            return min(float(str(raw).rstrip("s")), 60.0)
        except ValueError:
            pass
    return 5.0
