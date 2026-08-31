"""Deterministic verification that a finding points at code that exists.

This is the highest value-per-line code in the project, and it contains no model call.
Every finding names a file, a line range and usually a symbol. Those are checked against
the repository index: does the file exist, is the range inside it, is there really a
function by that name, and does that function actually span the lines cited. Anything
that fails is dropped.

Note that this only ever asks *per-file* questions, answered from a real `ast` parse.
It never consults the caller index, which matches on bare names and is therefore
approximate -- good enough to point an agent at a caller, nowhere near good enough to
confirm that a claim is true.

The drop rate is not an internal detail -- it is the system's own measured hallucination
rate, and it is reported. That is the point: an LLM reviewer that invents a finding
about `validate_token` on line 340 of a 120-line file is caught here by arithmetic, not
by asking a second model whether it believes the first one.

Two deliberate choices:

*Symbol mismatch is a warning, not a rejection.* A model that names the right defect but
attributes it to the enclosing class rather than the method has made a labelling error,
not a hallucination. Those are repaired -- we substitute the symbol the graph says owns
that line -- and counted separately, because conflating a sloppy label with an invented
claim would overstate the hallucination rate.

*A finding with no symbol is still checkable.* Module-level code has no enclosing
function, so the symbol is optional; the file and line checks still apply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from codesage.domain import Finding, GroundStatus
from codesage.index.code import CodeIndex, ParsedFile
from codesage.ingest.inventory import Inventory

log = logging.getLogger(__name__)


@dataclass
class GroundReport:
    """Outcome of grounding a batch of findings."""

    grounded: list[Finding] = field(default_factory=list)
    rejected: list[Finding] = field(default_factory=list)
    repaired_symbols: int = 0

    @property
    def total(self) -> int:
        return len(self.grounded) + len(self.rejected)

    @property
    def hallucination_rate(self) -> float:
        """Fraction of findings that pointed at code which does not exist."""
        return len(self.rejected) / self.total if self.total else 0.0

    def rejections_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.rejected:
            key = str(finding.ground_status)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def rejections_by_family(self) -> dict[str, int]:
        """Which model families hallucinate most. Feeds the evaluation, not the report."""
        counts: dict[str, int] = {}
        for finding in self.rejected:
            family = finding.provenance.family
            counts[family] = counts.get(family, 0) + 1
        return counts

    def describe(self) -> str:
        if not self.total:
            return "no findings to ground"
        return (
            f"{len(self.grounded)}/{self.total} findings grounded, "
            f"{len(self.rejected)} rejected ({self.hallucination_rate:.0%}), "
            f"{self.repaired_symbols} symbol labels repaired"
        )


def check_one(
    finding: Finding,
    inventory: Inventory,
    index: CodeIndex,
) -> Finding:
    """Set `ground_status` on one finding. Returns the same object, mutated."""
    raw = finding.raw

    source = inventory.by_path(raw.file)
    if source is None:
        finding.ground_status = GroundStatus.NO_SUCH_FILE
        finding.ground_detail = f"{raw.file} is not a file that was reviewed"
        return finding

    if raw.line_start > source.n_lines:
        finding.ground_status = GroundStatus.LINE_OUT_OF_RANGE
        finding.ground_detail = (
            f"cites line {raw.line_start} but {raw.file} has {source.n_lines} lines"
        )
        return finding

    module = index.file(raw.file)
    if module is None or not raw.symbol:
        # No parse available, or a module-level finding with nothing to name. The file
        # and line checks already passed, which is all we can verify.
        finding.ground_status = GroundStatus.GROUNDED
        return finding

    named = _find_symbol(module, raw.symbol)
    # The graph knows which symbol actually owns the cited line -- `symbol_at` returns
    # the innermost one, so a line inside a method resolves to the method rather than
    # its class.
    owner = module.symbol_at(raw.line_start)

    if named is None:
        if owner is not None:
            # The model found the right place and invented a name for it. That is a
            # labelling error, not a fabricated claim: repair it and count the repair.
            _repair(finding, owner.qualname, f"no symbol named {raw.symbol!r}")
            return finding
        finding.ground_status = GroundStatus.NO_SUCH_SYMBOL
        finding.ground_detail = f"no symbol named {raw.symbol!r} in {raw.file}"
        return finding

    if not _overlaps(named.line_start, named.line_end, raw.line_start, raw.line_end):
        if owner is not None:
            _repair(
                finding,
                owner.qualname,
                f"{raw.symbol!r} spans {named.line_start}-{named.line_end}, "
                f"not the cited {raw.line_start}-{raw.line_end}",
            )
            return finding
        finding.ground_status = GroundStatus.SYMBOL_LINE_MISMATCH
        finding.ground_detail = (
            f"{raw.symbol!r} spans lines {named.line_start}-{named.line_end}, "
            f"but the finding cites {raw.line_start}-{raw.line_end}"
        )
        return finding

    # The symbol is real and covers the cited lines, but it may be coarser than
    # necessary -- a model reporting a bug inside a method often names the enclosing
    # class. Normalise to the innermost owner so that two models describing the same
    # defect at different granularity end up with the same symbol, and therefore
    # cluster together instead of being counted as two independent findings.
    if owner is not None and owner.qualname != named.qualname:
        _repair(finding, owner.qualname, "narrowed to the innermost enclosing symbol")
        return finding

    finding.ground_status = GroundStatus.GROUNDED
    return finding


def _repair(finding: Finding, correct_symbol: str, why: str) -> None:
    finding.ground_status = GroundStatus.GROUNDED
    finding.ground_detail = (
        f"symbol repaired: {finding.raw.symbol!r} -> {correct_symbol!r} ({why})"
    )
    finding.raw.symbol = correct_symbol


def check_all(
    findings: list[Finding],
    inventory: Inventory,
    index: CodeIndex,
) -> GroundReport:
    """Ground a batch and split it into kept and rejected."""
    report = GroundReport()
    for finding in findings:
        checked = check_one(finding, inventory, index)
        if checked.ground_detail and "repaired" in checked.ground_detail:
            report.repaired_symbols += 1
        (report.grounded if checked.is_grounded else report.rejected).append(checked)

    log.info("ground check -- %s", report.describe())
    if report.rejected:
        log.info("  rejections by reason: %s", report.rejections_by_reason())
    return report


def _find_symbol(module: ParsedFile, name: str) -> object | None:
    """Match a symbol by qualified name, then by bare name.

    Models cite `withdraw` as often as `Account.withdraw`, and both are legitimate ways
    to refer to the same method. A bare name is only accepted when it is unambiguous
    within the file.
    """
    return module.find(name)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end
