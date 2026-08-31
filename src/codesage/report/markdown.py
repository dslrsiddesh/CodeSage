"""Rendering the report.

The report leads with what it does *not* know. A reader deciding how much weight to give
these findings needs to see, before the findings themselves, how many model families
actually ran, what fraction of raw findings were thrown out for pointing at nothing, and
whether quota forced the ensemble to shrink. Burying those is asking to be trusted more
than the run deserves.

Each finding carries its evidence trail -- which families agreed, what the critic said,
what the linter independently confirmed. That trail is the product. A bare list of LLM
claims is worth little; the same list annotated with how each was corroborated is
something a reviewer can triage.
"""

from __future__ import annotations

from codesage.domain import Severity, Verdict
from codesage.report.build import Report
from codesage.verify.score import ScoredCluster

MARK = {
    Severity.CRITICAL: "!!!", Severity.HIGH: "!!", Severity.MEDIUM: "!",
    Severity.LOW: "~", Severity.INFO: "-",
}


def render(report: Report) -> str:
    m = report.manifest
    out = [
        f"# Code review: {m.target}",
        "",
        f"`{m.commit[:10]}` · {m.started_at} · {m.duration_s}s",
        "",
        f"**{report.headline}**",
        "",
        "## How much to trust this report",
        "",
    ]

    families = len(m.families_used)
    out.append(
        f"{families} independent model families reviewed this code."
        if families >= 3
        else "2 model families reviewed this code — enough to cross-check, but thin."
        if families == 2
        else f"Only {families} model family was available. Cross-family agreement is the "
        f"main signal CodeSage uses, so these findings rest on much weaker evidence than usual."
    )
    out += ["", "| | |", "|---|---|"]
    out += [
        f"| Model families | {', '.join(f'{f} ({n})' for f, n in sorted(m.families_used.items()))} |",
        f"| Findings discarded as ungrounded | {m.findings_rejected} of {m.findings_raw} "
        f"({m.hallucination_rate:.0%}) — cited a file, line, or symbol that does not exist |",
        f"| Findings with cross-family support | {m.clusters_multi_family} of {m.clusters} |",
        f"| Static analysis | {'ruff, ' + str(m.lint_findings) + ' findings' if m.lint_available else 'ruff not installed'} |",
        f"| File selection | {'planner agent' if m.plan_source == 'agent' else 'deterministic fallback'} |",
        f"| Agent tool calls | {m.tool_calls} |",
        "",
    ]

    if m.degradations:
        out += [
            "> **This run was degraded.** Quota limits meant part of the intended ensemble",
            "> did not run, so some findings have less support than their scores suggest:",
            ">",
            *(f"> - {d}" for d in m.degradations),
            "",
        ]

    out += ["## Findings", ""]
    if not report.shown:
        out += ["Nothing met the confidence threshold.", ""]
    for i, finding in enumerate(report.shown, 1):
        out += _finding(i, finding)

    if report.suppressed:
        refuted = sum(1 for f in report.suppressed if f.refuted)
        out += [
            "---",
            "",
            f"*{len(report.suppressed)} further findings scored below the display threshold"
            + (f", including {refuted} refuted by an independent model" if refuted else "")
            + ". All are in the JSON output.*",
            "",
        ]

    out += _coverage(report) + _method(report)
    return "\n".join(out)


def _finding(index: int, scored: ScoredCluster) -> list[str]:
    cluster = scored.cluster
    raw = cluster.representative.raw

    out = [
        f"### {index}. {MARK[cluster.severity]} {raw.claim}",
        "",
        f"**{cluster.location}** · {cluster.severity} · confidence {scored.score:.2f}",
        "",
        raw.evidence,
        "",
    ]
    if raw.suggested_fix:
        out += [f"**Suggested fix.** {raw.suggested_fix}", ""]

    out.append(
        f"- Raised by **{scored.families_agreeing} of {scored.families_available}** families: "
        f"{', '.join(sorted(cluster.families))}"
    )
    out.append(f"- Lenses: {', '.join(sorted(cluster.lenses))}")
    if scored.corroborating_rules:
        out.append(f"- Independently flagged by ruff: {', '.join(scored.corroborating_rules)}")
    out.append(
        {
            Verdict.UPHELD: f"- Survived challenge by an independent model: *{scored.critic_reasoning}*",
            Verdict.REJECTED: f"- **Disputed** by an independent model: *{scored.critic_reasoning}*",
            Verdict.UNCHALLENGED: "- Not challenged — no independent family had quota remaining",
        }[scored.critic_verdict]
    )

    # Only alternative wordings that actually differ; repeating identical text is padding.
    others = [
        f.raw.claim for f in cluster.findings if f.raw.claim != raw.claim
    ]
    if others:
        out.append("- Also described as: " + "; ".join(f'"{o}"' for o in others[:2]))
    out.append("")
    return out


def _coverage(report: Report) -> list[str]:
    if report.index is None:
        return []
    m = report.manifest
    out = [
        "## What was reviewed",
        "",
        f"{len(m.files_reviewed)} of {len(report.index.inventory.files)} reviewable files. "
        f"The remaining {m.files_skipped} were not reviewed — a free-tier budget does not "
        f"stretch to a whole repository, and pretending otherwise would be misleading.",
        "",
    ]
    if m.plan_rationale:
        out += [f"> **Triage strategy.** {m.plan_rationale}", ""]
    out += [f"- `{path}`" for path in m.files_reviewed]
    out.append("")
    return out


def _method(report: Report) -> list[str]:
    m = report.manifest
    return [
        "## Method",
        "",
        "Every finding was produced by an LLM and then filtered three ways:",
        "",
        "1. **Grounded** — the cited file, line range and symbol are checked against a parse",
        "   of the repository. Anything pointing at code that does not exist is discarded",
        f"   ({m.hallucination_rate:.0%} of raw findings were, in this run).",
        "2. **Corroborated** — findings raised independently by more than one model *family*",
        "   score higher. Two checkpoints of the same base model count once.",
        "3. **Challenged** — each surviving finding goes to a model from a family that did",
        "   not propose it, which is asked to refute it.",
        "",
        f"Agents had read-only tools for exploring the repository and used them "
        f"{m.tool_calls} time{'s' if m.tool_calls != 1 else ''}.",
        "",
        f"*Cache hit rate {m.cache_hit_rate:.0%} · run id `{m.run_id}`*",
        "",
    ]
