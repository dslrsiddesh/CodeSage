"""The agent loop: let a model call tools until it is ready to answer.

This is the harness. Everything an agent framework would give you, written out where it
can be read, because the interesting parts are the failure cases rather than the happy
path.

**The loop.** Send messages plus tool schemas. Tool calls get executed and appended;
content gets parsed as the final answer. Repeat up to `max_steps`.

**Four things that make it a harness rather than a while loop:**

1. *Bounded exploration.* Without `max_steps`, a model that keeps grepping never
   terminates -- on a free tier that is the day's quota spent on one file.
2. *Graceful loss of tool calling.* Some free-tier models reject the `tools` parameter;
   some accept it and never emit a call. A rejection retries once without tools; a model
   that ignores them still reviews from the context it was given.
3. *Forced landing.* On the final step tools are withdrawn and the model must answer. A
   loop ending mid-exploration wastes everything it spent getting there.
4. *Failure containment.* Any exception costs that lens one opinion, never the review.

A framework would do this in ten lines and hide all four -- which are the parts that
matter under a budget, and the parts worth being able to explain.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from codesage.agents.tools import TOOL_SCHEMAS, RepoTools, ToolCall
from codesage.config.settings import ModelSpec
from codesage.llm.client import LLMClient
from codesage.llm.errors import PermanentProviderError, ProviderError
from codesage.llm.structured import parse as parse_structured

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FINAL_STEP_NUDGE = (
    "You have no more tool calls available. Give your final answer now, as a single "
    "JSON object, using what you already know. If you did not find anything worth "
    "reporting, return an empty findings list -- that is a normal outcome."
)


@dataclass
class AgentTrace:
    """What the agent actually did. Surfaced in the dashboard and the run manifest.

    An agent that reached its answer after checking three callers is doing something
    different from one that answered immediately, and the difference should be visible
    rather than buried in a log file.
    """

    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    tokens: int = 0
    cached: bool = False
    tools_supported: bool = True
    hit_step_limit: bool = False
    error: str | None = None

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)

    def describe(self) -> str:
        if self.error:
            return f"failed after {self.steps} step(s): {self.error}"
        detail = f"{self.steps} step(s)"
        if self.tool_calls:
            detail += f", {len(self.tool_calls)} tool call(s): {', '.join(self.tool_calls[:4])}"
        if self.hit_step_limit:
            detail += " [hit step limit]"
        if not self.tools_supported:
            detail += " [model does not support tools]"
        return detail


def _extract_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Read tool calls out of an OpenAI-shaped assistant message.

    Arguments arrive as a JSON *string*, and models get that string wrong often enough
    that it has to be handled rather than assumed: a malformed one becomes an empty
    argument dict, the tool reports what it needed, and the model can correct itself on
    the next turn.
    """
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


async def run_agent(
    client: LLMClient,
    model: ModelSpec,
    messages: list[dict[str, Any]],
    schema: type[T],
    *,
    tools: RepoTools | None = None,
    max_steps: int = 4,
    stage: str = "review",
) -> tuple[T | None, AgentTrace]:
    """Drive one agent to a structured answer, allowing tool calls along the way.

    Returns `(parsed, trace)`. `parsed` is None when the model could not be coaxed into
    valid output -- the caller treats that as one lost opinion, not a failed review.
    """
    trace = AgentTrace()
    conversation = list(messages)
    use_tools = tools is not None

    for step in range(1, max_steps + 1):
        trace.steps = step
        last_step = step == max_steps

        if last_step and use_tools and trace.used_tools:
            conversation.append({"role": "user", "content": FINAL_STEP_NUDGE})

        try:
            response = await client.complete(
                model,
                conversation,
                stage=stage,
                tools=TOOL_SCHEMAS if (use_tools and not last_step) else None,
            )
        except PermanentProviderError as exc:
            # The most likely cause is that this model rejects the `tools` parameter.
            # Retry once without it before giving up -- the model can still review from
            # the context it was handed.
            if use_tools and "tool" in str(exc).lower():
                log.info("%s rejected tool calling; retrying without tools", model.key)
                trace.tools_supported = False
                use_tools = False
                continue
            trace.error = str(exc)[:160]
            return None, trace
        except ProviderError as exc:
            trace.error = str(exc)[:160]
            return None, trace
        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"[:160]
            log.warning("agent %s raised unexpectedly", model.key, exc_info=True)
            return None, trace

        trace.tokens += response.total_tokens
        trace.cached = trace.cached or response.cached

        calls = _extract_tool_calls(response.raw_message)
        if calls and use_tools and tools is not None:
            conversation.append(response.raw_message)
            for call in calls:
                result = tools.execute(call)
                trace.tool_calls.append(call.name)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": _call_id(response.raw_message, call.name),
                        "name": call.name,
                        "content": result.content,
                    }
                )
            continue

        parsed, error = parse_structured(response.content, schema)
        if parsed is not None:
            return parsed, trace

        # No tool calls and no parseable answer. One nudge, then give up: a model that
        # has failed to produce valid JSON twice will not manage it on the third try,
        # and those attempts come out of the same budget as a real review.
        if last_step:
            trace.error = error
            return None, trace
        conversation.append({"role": "assistant", "content": response.content[:1500]})
        conversation.append(
            {
                "role": "user",
                "content": (
                    f"That could not be parsed ({error}). Reply again as a single valid "
                    f"JSON object and nothing else -- no markdown fences, no commentary."
                ),
            }
        )

    trace.hit_step_limit = True
    trace.error = trace.error or "exhausted step budget without a parseable answer"
    return None, trace


def _call_id(message: dict[str, Any], name: str) -> str:
    """Recover the id the provider assigned to a tool call, so results can be matched."""
    for raw in message.get("tool_calls") or []:
        if (raw.get("function") or {}).get("name") == name:
            return str(raw.get("id") or name)
    return name
