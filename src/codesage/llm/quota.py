"""Per-provider quota accounting, persisted in SQLite.

Free tiers impose two kinds of limit that fail in opposite ways, and the correct
response to each is the opposite of the other: a per-minute rate limit is worth *waiting
out*, a per-day cap is not. Conflating them either wastes a review or wastes an hour.

The rule that matters: when a provider is exhausted we never silently proceed with fewer
opinions. It is recorded and surfaced in the report, because a review that ran one model
instead of three is a different experiment.

The clock is injectable so tests can drive it without sleeping.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from codesage.config.settings import ProviderLimits


class QuotaVerdict(StrEnum):
    OK = "ok"
    RATE_LIMITED = "rate_limited"  # transient: wait and retry
    EXHAUSTED = "exhausted"  # daily cap: do not retry today


@dataclass(frozen=True)
class QuotaDecision:
    verdict: QuotaVerdict
    wait_seconds: float = 0.0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is QuotaVerdict.OK


@dataclass
class Degradation:
    """Recorded whenever quota forced us to do less than intended."""

    provider: str
    stage: str
    reason: str

    def describe(self) -> str:
        return f"{self.provider} unavailable during {self.stage}: {self.reason}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    provider TEXT NOT NULL, day TEXT NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, day)
);
CREATE TABLE IF NOT EXISTS recent (provider TEXT NOT NULL, ts REAL NOT NULL);
"""


class QuotaTracker:
    def __init__(
        self,
        db_path: Path,
        limits: dict[str, ProviderLimits],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self.clock = clock
        self.degradations: list[Degradation] = []
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def _today(self) -> str:
        return datetime.fromtimestamp(self.clock(), tz=UTC).strftime("%Y-%m-%d")

    def usage(self, provider: str) -> tuple[int, int]:
        """(requests, tokens) used today."""
        row = self._db.execute(
            "SELECT requests, tokens FROM usage WHERE provider=? AND day=?",
            (provider, self._today()),
        ).fetchone()
        return (row[0], row[1]) if row else (0, 0)

    def check(self, provider: str, *, estimated_tokens: int = 0) -> QuotaDecision:
        """Decide whether a call to `provider` may proceed right now."""
        limits = self.limits.get(provider)
        if limits is None:
            return QuotaDecision(QuotaVerdict.OK)

        requests, tokens = self.usage(provider)
        if requests >= limits.requests_per_day:
            return QuotaDecision(
                QuotaVerdict.EXHAUSTED,
                reason=f"daily request cap reached ({requests}/{limits.requests_per_day})",
            )
        if tokens + estimated_tokens > limits.tokens_per_day:
            return QuotaDecision(
                QuotaVerdict.EXHAUSTED,
                reason=f"daily token cap reached ({tokens}/{limits.tokens_per_day})",
            )

        cutoff = self.clock() - 60.0
        recent = self._db.execute(
            "SELECT COUNT(*), MIN(ts) FROM recent WHERE provider=? AND ts>?", (provider, cutoff)
        ).fetchone()
        if recent and recent[0] >= limits.requests_per_minute:
            wait = max(0.0, 60.0 - (self.clock() - recent[1])) + 0.5
            return QuotaDecision(
                QuotaVerdict.RATE_LIMITED,
                wait_seconds=wait,
                reason=f"{recent[0]}/{limits.requests_per_minute} requests in the last minute",
            )
        return QuotaDecision(QuotaVerdict.OK)

    def record(self, provider: str, *, tokens: int = 0) -> None:
        """Book one request against the provider's budget."""
        now = self.clock()
        self._db.execute(
            """INSERT INTO usage (provider, day, requests, tokens) VALUES (?, ?, 1, ?)
               ON CONFLICT(provider, day) DO UPDATE SET
                   requests = requests + 1, tokens = tokens + excluded.tokens""",
            (provider, self._today(), tokens),
        )
        self._db.execute("INSERT INTO recent (provider, ts) VALUES (?, ?)", (provider, now))
        # Anything older than an hour cannot affect a per-minute window.
        self._db.execute("DELETE FROM recent WHERE ts < ?", (now - 3600,))
        self._db.commit()

    def note_degradation(self, provider: str, stage: str, reason: str) -> None:
        self.degradations.append(Degradation(provider, stage, reason))

    def report(self) -> dict[str, dict[str, int]]:
        """Today's consumption, for the run manifest."""
        return {
            provider: {
                "requests": self.usage(provider)[0],
                "requests_limit": limits.requests_per_day,
                "tokens": self.usage(provider)[1],
                "tokens_limit": limits.tokens_per_day,
            }
            for provider, limits in self.limits.items()
        }

    def close(self) -> None:
        self._db.close()
