"""One call that runs a complete review: index, orchestrate, report."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from codesage.agents.tools import RepoTools
from codesage.config.settings import Registry, Settings
from codesage.index.pipeline import IndexResult
from codesage.index.pipeline import run as run_index
from codesage.llm.cache import ResponseCache
from codesage.llm.client import LLMClient
from codesage.llm.quota import QuotaTracker
from codesage.llm.router import ModelRouter
from codesage.orchestration.graph import ReviewDeps, build_graph
from codesage.report import markdown
from codesage.report.build import Report
from codesage.report.build import build as build_report

log = logging.getLogger(__name__)


async def review(
    target: str,
    settings: Settings,
    registry: Registry,
    *,
    max_files: int | None = None,
    refresh: bool = False,
    use_tools: bool = True,
    no_cache: bool = False,
    on_event: Callable[[str, dict], None] | None = None,
    index: IndexResult | None = None,
) -> Report:
    """Review `target` end to end and return the report.

    `index` can be supplied to reuse a deterministic stage that has already run --
    the evaluation harness does this to avoid re-parsing the same repository across
    every ablation.
    """
    started = time.time()
    run_id = uuid.uuid4().hex[:12]
    settings.ensure_dirs()

    if index is None:
        index = run_index(target, settings, refresh=refresh)
    if max_files is not None:
        settings = settings.model_copy(update={"max_files": max_files})

    cache = ResponseCache(settings.cache_dir, enabled=not no_cache)
    quota = QuotaTracker(
        settings.state_db, {n: p.limits for n, p in registry.providers.items()}
    )

    async with LLMClient(registry, settings, cache=cache, quota=quota) as client:
        router = ModelRouter(registry, quota, settings)
        deps = ReviewDeps(
            client=client,
            router=router,
            settings=settings,
            index=index,
            tools=RepoTools(index.code),
            use_tools=use_tools,
            on_event=on_event,
        )
        graph = build_graph(deps)
        state = await graph.ainvoke(
            {"target": target, "run_id": run_id},
            # Enough headroom for a fan-out of files x lenses x families; the default
            # of 25 is far too low and fails with a confusing recursion error.
            {"recursion_limit": 200},
        )

    return build_report(
        run_id=run_id,
        target=target,
        index=index,
        scored=state.get("scored", []),
        state=state,
        router_usage=router.usage_report(),
        cache_hit_rate=cache.hit_rate,
        quota_report=quota.report(),
        started_at=started,
        ensemble_size=settings.ensemble_size,
        tool_calls=deps.tools.call_log,
        plan=deps.plan,
    )


def write_outputs(report: Report, out_dir: Path) -> tuple[Path, Path]:
    """Write the Markdown report and the machine-readable JSON beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.manifest.run_id}"

    md_path = out_dir / f"review-{stem}.md"
    md_path.write_text(markdown.render(report), encoding="utf-8")

    json_path = out_dir / f"review-{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "manifest": report.manifest.model_dump(),
                # Everything, including findings below the display threshold and those
                # the critic refuted. The Markdown is a view; this is the record.
                "findings": [
                    {
                        "score": round(f.score, 4),
                        "severity": str(f.severity),
                        "refuted": f.refuted,
                        "shown": f in report.shown,
                        "file": f.cluster.file,
                        "line_start": f.cluster.line_start,
                        "line_end": f.cluster.line_end,
                        "symbol": f.cluster.symbol,
                        "claim": f.cluster.representative.raw.claim,
                        "evidence": f.cluster.representative.raw.evidence,
                        "suggested_fix": f.cluster.representative.raw.suggested_fix,
                        "families": sorted(f.cluster.families),
                        "lenses": sorted(f.cluster.lenses),
                        "support": f.families_agreeing,
                        "available": f.families_available,
                        "corroborating_rules": f.corroborating_rules,
                        "critic_verdict": str(f.critic_verdict),
                        "critic_reasoning": f.critic_reasoning,
                    }
                    for f in report.findings
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return md_path, json_path
