"""Grouping findings that describe the same defect.

Without this, three models spotting one bug produce three bullet points and the report
looks three times as alarming as the code deserves. With it, "three families
independently found this" becomes a statement that means something -- and cross-family
agreement is what the scoring rests on.

Two findings merge when they are in the same file, their line ranges overlap, and their
claims share enough distinctive words. The location test does most of the work: once two
findings are known to be about the same few lines, the words only have to separate
"off-by-one in the bound" from "missing null check". That is why plain word overlap is
enough and there is no embedding model here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from codesage.domain import Finding, Severity

log = logging.getLogger(__name__)

OVERLAP_THRESHOLD = 0.30
WORD_THRESHOLD = 0.25

# Words too common in review prose to distinguish one finding from another. Without
# this, every pair scores near the threshold on "the function value may cause".
COMMON = frozenset(
    """
    a an the this that these those is are was were be been being do does did it its of
    to in on at for with by from as and or not no if then than when where which who what
    how why can could should would may might must will has have had here there very more
    most some any all each both code function method class value values variable line
    lines file issue problem potential possible cause caused using used use
    """.split()  # noqa: SIM905 -- a readable wordlist beats a 400-character list literal
)


def words(text: str) -> set[str]:
    """Distinctive words, split on camelCase and snake_case.

    Splitting matters: one model writes `total_cents`, another `totalCents`, a third
    "the total cents". All three should share words.
    """
    out: set[str] = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        for part in re.split(r"[_]+|(?<=[a-z])(?=[A-Z])", token):
            lowered = part.lower()
            if len(lowered) > 2 and lowered not in COMMON:
                out.add(lowered)
    return out


def similar(a: str, b: str) -> float:
    """Fraction of distinctive words the two claims share."""
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def overlap(a: Finding, b: Finding) -> float:
    """Intersection over union of two line ranges."""
    lo = max(a.raw.line_start, b.raw.line_start)
    hi = min(a.raw.line_end, b.raw.line_end)
    if hi < lo:
        return 0.0
    union = max(a.raw.line_end, b.raw.line_end) - min(a.raw.line_start, b.raw.line_start) + 1
    return (hi - lo + 1) / union


def same_defect(a: Finding, b: Finding) -> bool:
    return (
        a.raw.file == b.raw.file
        and overlap(a, b) >= OVERLAP_THRESHOLD
        and similar(a.raw.claim, b.raw.claim) >= WORD_THRESHOLD
    )


@dataclass
class FindingCluster:
    """One defect, as described by one or more models."""

    id: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def families(self) -> set[str]:
        """Distinct model *families*. Not model ids -- see the router."""
        return {f.provenance.family for f in self.findings}

    @property
    def lenses(self) -> set[str]:
        return {str(f.provenance.lens) for f in self.findings}

    @property
    def support(self) -> int:
        return len(self.families)

    @property
    def representative(self) -> Finding:
        """Whose wording is shown. The others still count toward support."""
        return max(self.findings, key=lambda f: (f.raw.confidence, len(f.raw.evidence)))

    @property
    def severity(self) -> Severity:
        """The most severe assessment offered, not the average.

        Averaging would let two models calling something `low` bury one that spotted a
        genuine `critical`, and those two errors do not cost the same.
        """
        return max((f.raw.severity for f in self.findings), key=lambda s: s.weight)

    @property
    def file(self) -> str:
        return self.representative.raw.file

    @property
    def line_start(self) -> int:
        return min(f.raw.line_start for f in self.findings)

    @property
    def line_end(self) -> int:
        return max(f.raw.line_end for f in self.findings)

    @property
    def symbol(self) -> str | None:
        return self.representative.raw.symbol

    @property
    def location(self) -> str:
        base = f"{self.file}:{self.line_start}-{self.line_end}"
        return f"{base} ({self.symbol})" if self.symbol else base

    @property
    def mean_confidence(self) -> float:
        return sum(f.raw.confidence for f in self.findings) / len(self.findings)


def cluster(findings: list[Finding]) -> list[FindingCluster]:
    """Greedily merge findings that describe the same defect.

    Greedy rather than union-find: a review produces tens of findings, not thousands,
    so the quadratic scan is instant and this version can be read top to bottom.
    """
    clusters: list[FindingCluster] = []
    for finding in findings:
        for existing in clusters:
            if any(same_defect(finding, member) for member in existing.findings):
                existing.findings.append(finding)
                break
        else:
            clusters.append(FindingCluster(id=len(clusters), findings=[finding]))

    log.info(
        "clustered %d findings into %d (%d with cross-family support)",
        len(findings),
        len(clusters),
        sum(1 for c in clusters if c.support > 1),
    )
    return clusters
