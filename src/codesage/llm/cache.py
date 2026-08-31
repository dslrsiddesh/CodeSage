"""Content-addressed cache for model responses.

Not an optimisation -- load-bearing. On free tiers a full evaluation sweep would blow the
daily quota many times over; with the cache only the first run of a given (model, prompt)
pair costs anything and every re-run is free and instant. It is also why the test suite
runs offline with no API key.

It is what actually protects a killed run, too: re-running the same command replays every
completed call for free and only spends quota on what had not finished.

Keys hash everything that could change the response, including a schema version. Bumping
`VERSION` invalidates the whole cache on purpose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERSION = "2"


def make_key(*, model_key: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
    """Deterministic key. Dict ordering is normalised so it cannot drift."""
    blob = json.dumps(
        {"v": VERSION, "model": model_key, "messages": messages, "params": params},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class CacheEntry:
    content: str
    usage: dict[str, int] = field(default_factory=dict)
    # The full assistant message. A cached reply carrying tool calls has to replay them
    # too, or a resumed agent loop silently loses the model's exploration.
    raw_message: dict[str, Any] = field(default_factory=dict)


class ResponseCache:
    """A directory of JSON files, sharded by key prefix to keep any one small."""

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        if not self.enabled or not (path := self._path(key)).exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A truncated write from an interrupted run. Treat as a miss and let it be
            # overwritten rather than crashing a long review.
            self.misses += 1
            return None
        self.hits += 1
        return CacheEntry(
            content=data.get("content", ""),
            usage=data.get("usage", {}),
            raw_message=data.get("raw_message", {}),
        )

    def put(
        self, key: str, *, content: str, usage: dict[str, int], raw_message: dict[str, Any]
    ) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename, so a crash mid-write cannot leave a half-written entry that
        # still looks valid.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"content": content, "usage": usage, "raw_message": raw_message}),
            encoding="utf-8",
        )
        tmp.replace(path)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
