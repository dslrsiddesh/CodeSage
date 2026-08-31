"""Assembling the review report and its manifest.

The report deliberately leads with what it does *not* know. A reader deciding how much
weight to give these findings needs to see, before the findings themselves, how many
model families actually ran, what fraction of raw findings were thrown out for pointing
at nothing, and whether quota forced the ensemble to shrink. A report that buries those
numbers is asking to be trusted more than it deserves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from codesage.domain import Severity
from codesage.index.pipeline import IndexResult
from codesage.orchestration.state import RunManifest
from codesage.verify.score import ScoredCluster


@dataclass
class Report:
    manifest: RunManifest
    findings: list[ScoredCluster] = field(default_factory=list)
    index: IndexResult | None = None

    # Findings below this score are counted but not shown individually. The threshold is
    # a display decision, not a truth claim -- everything is in the JSON output.
    display_threshold: float = 0.35

    @property
    def shown(self) -> list[ScoredCluster]:
        return [f for f in self.findings if f.score >= self.display_threshold and not f.refuted]

    @property
    def suppressed(self) -> list[ScoredCluster]:
        return [f for f in self.findings if f.score < self.display_threshold or f.refuted]

    def by_severity(self) -> dict[str, int]:
        counts = dict.fromkeys((str(s) for s in Severity), 0)
        for finding in self.shown:
            counts[str(finding.severity)] += 1
        return {k: v for k, v in counts.items() if v}

    @property
    def headline(self) -> str:
        counts = self.by_severity()
        if not counts:
            return "No findings met the confidence threshold."
        parts = [f"{n} {sev}" for sev, n in counts.items()]
        return ", ".join(parts)


def build(
    *,
    run_id: str,
    target: str,
    index: IndexResult,
    scored: list[ScoredCluster],
    state: dict,
    router_usage: dict[str, int],
    cache_hit_rate: float,
    quota_report: dict,
    started_at: float,
    ensemble_size: int,
    tool_calls: list[str] | None = None,
    plan: object | None = None,
) -> Report:
    findings = state.get("findings", [])
    rejected = state.get("rejected", [])
    clusters = state.get("clusters", [])

    manifest = RunManifest(
        run_id=run_id,
        target=target,
        commit=index.checkout.commit,
        started_at=datetime.fromtimestamp(started_at, tz=UTC).isoformat(timespec="seconds"),
        duration_s=round(time.time() - started_at, 1),
        files_reviewed=sorted(state.get("packs", {})),
        files_skipped=max(0, len(index.inventory.files) - len(state.get("packs", {}))),
        lenses=sorted({str(f.provenance.lens) for f in findings}),
        families_used=router_usage,
        ensemble_size=ensemble_size,
        findings_raw=len(findings) + len(state.get("out_of_scope", [])),
        findings_grounded=len(state.get("grounded", [])),
        findings_rejected=len(rejected),
        hallucination_rate=round(state.get("hallucination_rate", 0.0), 3),
        symbol_repairs=state.get("symbol_repairs", 0),
        clusters=len(clusters),
        clusters_multi_family=sum(1 for c in clusters if c.support > 1),
        lint_available=index.lint.available,
        lint_findings=len(index.lint.findings),
        plan_source=state.get("plan_source", "fallback"),
        plan_rationale=state.get("plan_rationale", ""),
        tool_calls=len(tool_calls or []),
        agent_traces=list(state.get("traces", []))[:40],
        degradations=state.get("degradations", []),
        cache_hit_rate=cache_hit_rate,
        quota=quota_report,
    )
    return Report(manifest=manifest, findings=scored, index=index)
