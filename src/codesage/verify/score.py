"""Turning a cluster of findings into a single confidence number.

A transparent weighted sum over four signals, three of which are things we measured
rather than things a model told us about itself:

    support      how many distinct model *families* raised it
    corroborated whether the linter independently flagged the same lines
    critic       whether an independent family failed to refute it
    confidence   the mean self-reported confidence of the proposers

Self-reported confidence is included but weighted lowest on purpose. It is the only
input a model controls directly, and models are poorly calibrated about their own
correctness -- it is a tiebreaker, not evidence.

`unchallenged` scores below `upheld` rather than equal to it. "No independent family had
quota left to challenge this" is weaker support than "someone tried and could not refute
it", and collapsing them would inflate confidence in exactly the runs where the budget
ran out.

This deliberately replaced a Dawid-Skene EM consensus. That estimated per-family
reliability from the agreement structure with no labels, and beat majority voting when
annotators are unequally reliable -- a real result. It was cut because the whole scoring
step is now something a reader can verify by arithmetic, and a number you can check
beats a better number you have to trust.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from codesage.domain import CriticVerdict, Severity, StaticFinding, Verdict
from codesage.verify.cluster import FindingCluster

log = logging.getLogger(__name__)

WEIGHTS = {"support": 0.40, "corroborated": 0.25, "critic": 0.25, "confidence": 0.10}

# How far a lint finding can sit from a cluster and still count as corroboration.
WINDOW = 3


@dataclass
class ScoredCluster:
    cluster: FindingCluster
    score: float
    families_agreeing: int = 0
    families_available: int = 0
    corroborating_rules: list[str] = field(default_factory=list)
    critic_verdict: Verdict = Verdict.UNCHALLENGED
    critic_reasoning: str = ""

    @property
    def severity(self) -> Severity:
        return self.cluster.severity

    @property
    def rank_key(self) -> float:
        """What the report sorts by: confidence weighted by how much it would matter."""
        return self.score * self.severity.weight

    @property
    def refuted(self) -> bool:
        return self.critic_verdict is Verdict.REJECTED

    def explain(self) -> str:
        parts = [f"{self.families_agreeing}/{self.families_available} families agree"]
        if self.corroborating_rules:
            parts.append(f"linter agrees ({', '.join(self.corroborating_rules)})")
        parts.append(
            {
                Verdict.UPHELD: "survived challenge",
                Verdict.REJECTED: "refuted by an independent model",
                Verdict.UNCHALLENGED: "not challenged",
            }[self.critic_verdict]
        )
        return "; ".join(parts)


def _support(cluster: FindingCluster, available: int) -> float:
    """Fraction of the families that could have raised this, which did.

    Measured against the families that actually ran this cluster's lenses, not the
    whole ensemble -- otherwise a finding from a two-family lens is capped at 2/5
    however complete its agreement was.
    """
    if available <= 0:
        return 0.0
    if available == 1:
        # A single family cannot corroborate itself. Neutral rather than full marks, or
        # a degraded run would emit maximum-confidence findings.
        return 0.5
    return min(1.0, cluster.support / available)


def _corroboration(cluster: FindingCluster, lint: list[StaticFinding]) -> tuple[float, list[str]]:
    """Whether the linter flagged the same region.

    Independent confirmation from a non-LLM source is the strongest single signal
    available, because a linter cannot hallucinate a line number.
    """
    hits = [
        f
        for f in lint
        if f.file == cluster.file
        and f.line_start <= cluster.line_end + WINDOW
        and f.line_end >= cluster.line_start - WINDOW
    ]
    if not hits:
        return 0.0, []
    return 1.0, sorted({f.rule_id for f in hits})[:4]


def _critic(verdict: CriticVerdict | None) -> float:
    if verdict is None or verdict.verdict is Verdict.UNCHALLENGED:
        return 0.35
    if verdict.verdict is Verdict.REJECTED:
        # Scaled by the critic's own confidence, so a hesitant refutation does not erase
        # a finding three families agreed on.
        return max(0.0, 0.25 * (1.0 - verdict.confidence))
    return min(1.0, 0.6 + 0.4 * verdict.confidence)


def families_for(cluster: FindingCluster, lens_families: dict[str, set[str]] | None) -> int:
    """How many families could plausibly have raised this finding."""
    if not lens_families:
        return 0
    eligible: set[str] = set()
    for lens in cluster.lenses:
        eligible |= lens_families.get(lens, set())
    return len(eligible)


def score_all(
    clusters: list[FindingCluster],
    *,
    families_available: int,
    lint: list[StaticFinding],
    critics: dict[int, CriticVerdict] | None = None,
    lens_families: dict[str, set[str]] | None = None,
) -> list[ScoredCluster]:
    """Score every cluster and sort by how much it should command attention."""
    critics = critics or {}
    scored: list[ScoredCluster] = []

    for cluster in clusters:
        available = families_for(cluster, lens_families) or families_available
        critic = critics.get(cluster.id)
        support = _support(cluster, available)
        corroborated, rules = _corroboration(cluster, lint)
        critic_score = _critic(critic)

        total = min(
            1.0,
            support * WEIGHTS["support"]
            + corroborated * WEIGHTS["corroborated"]
            + critic_score * WEIGHTS["critic"]
            + cluster.mean_confidence * WEIGHTS["confidence"],
        )
        scored.append(
            ScoredCluster(
                cluster=cluster,
                score=total,
                families_agreeing=cluster.support,
                families_available=available,
                corroborating_rules=rules,
                critic_verdict=critic.verdict if critic else Verdict.UNCHALLENGED,
                critic_reasoning=critic.reasoning if critic else "",
            )
        )

    scored.sort(key=lambda s: -s.rank_key)
    log.info(
        "scored %d clusters; %d refuted, %d with multi-family support",
        len(scored),
        sum(1 for s in scored if s.refuted),
        sum(1 for s in scored if s.cluster.support > 1),
    )
    return scored
