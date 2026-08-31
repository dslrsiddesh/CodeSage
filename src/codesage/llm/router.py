"""Assigns models to review lenses under a family-diversity constraint.

This is where the project's central claim is either honoured or quietly broken. The
theme is combining *diverse model perspectives*; two checkpoints of the same base model
are not two perspectives, they are one perspective sampled twice. So the unit of
diversity here is `family`, never `id`, and the router will return a smaller ensemble
rather than pad it with a second model from a family it already used.

The other rule: when quota forces a smaller ensemble, that is recorded as a degradation
and ends up in the report. A review that ran one model instead of three is a different
experiment, and silently substituting one for the other would make every number the
evaluation produces meaningless.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from codesage.config.settings import ModelSpec, Registry, Settings
from codesage.domain import Lens
from codesage.llm.quota import QuotaTracker, QuotaVerdict

log = logging.getLogger(__name__)


@dataclass
class Assignment:
    """The models chosen for one lens, plus what we could not get.

    Two different things can go wrong under quota pressure and they are not equally
    serious, so they are tracked separately:

      * *substituted* -- we got the full ensemble, but from lower-preference families
        because the preferred ones were spent. Mild: the lens still has the intended
        number of independent opinions.
      * *degraded* -- we got fewer models than asked for. Serious: this lens now has
        less cross-family agreement behind it than every other lens in the report, and
        the scorer must not treat its findings as equally corroborated.
    """

    lens: Lens
    models: list[ModelSpec]
    requested: int
    preferred: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)

    @property
    def families(self) -> list[str]:
        return [m.family for m in self.models]

    @property
    def degraded(self) -> bool:
        return len(self.models) < self.requested

    @property
    def substituted(self) -> bool:
        return not self.degraded and bool(self.unavailable)

    def describe(self) -> str:
        got = ", ".join(self.families) or "nothing"
        if self.degraded:
            return (
                f"{self.lens}: {got} -- DEGRADED, wanted {self.requested} families, "
                f"unavailable: {', '.join(self.unavailable)}"
            )
        if self.substituted:
            return f"{self.lens}: {got} (substituted for preferred {', '.join(self.unavailable)})"
        return f"{self.lens}: {got}"


class ModelRouter:
    def __init__(self, registry: Registry, quota: QuotaTracker, settings: Settings) -> None:
        self.registry = registry
        self.quota = quota
        self.settings = settings
        self._usage = Counter[str]()

    def _family_is_usable(self, family: str) -> ModelSpec | None:
        """First model in `family` whose provider is configured and has quota left.

        Trying every model in the family matters: the same family is often served by two
        providers, so a Groq daily cap does not have to cost us the family entirely.
        """
        for model in self.registry.models_in_family(family):
            decision = self.quota.check(model.provider)
            if decision.verdict is not QuotaVerdict.EXHAUSTED:
                return model
        return None

    def assign_for_lens(
        self,
        lens: Lens,
        *,
        size: int | None = None,
        exclude_families: set[str] | None = None,
    ) -> Assignment:
        """Pick `size` models from distinct families for this lens."""
        wanted = size or self.settings.ensemble_size
        excluded = exclude_families or set()
        preferences = self.registry.lens_preferences.get(str(lens), [])

        # Preference order first, then any remaining configured family, so an unusual
        # key configuration still produces an ensemble rather than nothing.
        order = preferences + [
            f for f in self.registry.available_families() if f not in preferences
        ]

        chosen: list[ModelSpec] = []
        used: set[str] = set()
        unavailable: list[str] = []

        for family in order:
            if len(chosen) >= wanted:
                break
            if family in used or family in excluded:
                continue
            model = self._family_is_usable(family)
            if model is None:
                if family in preferences[:wanted]:
                    unavailable.append(family)
                continue
            chosen.append(model)
            used.add(family)
            self._usage[family] += 1

        assignment = Assignment(
            lens=lens,
            models=chosen,
            requested=wanted,
            preferred=preferences[:wanted],
            unavailable=unavailable,
        )
        if assignment.degraded:
            self.quota.note_degradation(
                provider=",".join(unavailable) or "all",
                stage=f"review:{lens}",
                reason=f"only {len(chosen)}/{wanted} model families available",
            )
            log.warning("degraded ensemble -- %s", assignment.describe())
        elif assignment.substituted:
            log.info("substituted families -- %s", assignment.describe())
        return assignment

    def assign_planner(self) -> ModelSpec | None:
        """A model for repository triage.

        Planning reads only an outline -- filenames, sizes, counts -- so it is a cheap
        call, and any family can do it. Takes the first with quota rather than a
        preferred one, so planning never consumes budget a specialist lens needs.
        """
        for family in self.registry.available_families():
            if (model := self._family_is_usable(family)) is not None:
                self._usage[family] += 1
                return model
        log.info("no family had quota for planning; falling back to deterministic triage")
        return None

    def assign_critic(self, exclude_families: set[str]) -> ModelSpec | None:
        """A model to argue *against* a finding, from a family that did not raise it.

        Structural separation of proposer and critic is the whole point. If no
        independent family has quota left we return None, and the scorer treats the
        finding as un-challenged rather than pretending it survived a challenge.
        """
        for family in self.registry.critic_preferences:
            if family in exclude_families:
                continue
            if (model := self._family_is_usable(family)) is not None:
                self._usage[family] += 1
                return model
        log.info("no independent family available to critique (excluded: %s)", exclude_families)
        return None

    def usage_report(self) -> dict[str, int]:
        """How many calls each family was assigned. Goes into the run manifest."""
        return dict(self._usage)

    def diversity(self) -> int:
        """Number of distinct families that actually ran."""
        return len(self._usage)
