"""FastAPI service and streaming dashboard.

A review takes minutes on free tiers, so what the user watches while waiting matters.
LangGraph emits an event per node; `ReviewDeps.on_event` forwards them to a per-run queue
the browser consumes over Server-Sent Events.

SSE rather than WebSockets: the traffic is one-directional, SSE reconnects on its own,
and it is `EventSource` in the browser with no library and no build step.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from codesage.config.settings import get_registry, get_settings
from codesage.orchestration.runner import review as run_review
from codesage.orchestration.runner import write_outputs
from codesage.report import markdown
from codesage.report.build import Report

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


class ReviewRequest(BaseModel):
    target: str = Field(description="GitHub URL, owner/repo, PR link, or a local path")
    max_files: int | None = Field(default=None, ge=1, le=200)


@dataclass
class Run:
    """One review in flight, plus the events it has produced so far."""

    run_id: str
    target: str
    status: str = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    report: Report | None = None
    error: str | None = None
    task: asyncio.Task | None = None

    def publish(self, kind: str, payload: dict) -> None:
        event = {"kind": kind, **payload}
        # Kept as well as queued: a browser that connects late, or reconnects after a
        # dropped connection, gets the backlog replayed rather than an empty screen.
        self.events.append(event)
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(event)


RUNS: dict[str, Run] = {}

app = FastAPI(title="CodeSage", description="Multi-LLM code review with grounding and consensus")


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    index = WEB_DIR / "index.html"
    if not index.exists():
        return "<h1>CodeSage</h1><p>Dashboard not found. Expected web/index.html.</p>"
    return index.read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict:
    registry = get_registry()
    configured = [n for n, p in registry.providers.items() if p.configured]
    return {
        "status": "ok",
        "providers_configured": configured,
        "families": sorted({m.family for m in registry.available_models()}),
        "ready": len(configured) >= 1,
    }


@app.post("/api/reviews")
async def start_review(request: ReviewRequest) -> dict:
    registry = get_registry()
    if not any(p.configured for p in registry.providers.values()):
        raise HTTPException(
            status_code=400,
            detail="No provider API keys are configured. See `codesage doctor`.",
        )

    run = Run(run_id=uuid.uuid4().hex[:12], target=request.target)
    RUNS[run.run_id] = run
    run.task = asyncio.create_task(_execute(run, request))
    return {"run_id": run.run_id, "status": run.status}


async def _execute(run: Run, request: ReviewRequest) -> None:
    settings = get_settings()
    registry = get_registry()
    run.status = "running"
    run.publish("started", {"target": run.target})

    loop = asyncio.get_running_loop()

    def on_event(kind: str, payload: dict) -> None:
        # The graph runs on this loop, but the callback is synchronous; scheduling the
        # publish keeps the queue write off the graph's critical path.
        loop.call_soon(run.publish, kind, payload)

    try:
        report = await run_review(
            run.target,
            settings,
            registry,
            max_files=request.max_files,
            on_event=on_event,
        )
    except Exception as exc:
        log.exception("review %s failed", run.run_id)
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        run.publish("failed", {"error": run.error})
        return

    write_outputs(report, settings.report_dir)
    run.report = report
    run.status = "complete"
    run.publish("complete", _summary(report))


def _summary(report: Report) -> dict:
    m = report.manifest
    return {
        "headline": report.headline,
        "findings": len(report.shown),
        "suppressed": len(report.suppressed),
        "families": m.families_used,
        "hallucination_rate": m.hallucination_rate,
        "findings_raw": m.findings_raw,
        "findings_rejected": m.findings_rejected,
        "clusters": m.clusters,
        "clusters_multi_family": m.clusters_multi_family,
        "plan_source": m.plan_source,
        "plan_rationale": m.plan_rationale,
        "tool_calls": m.tool_calls,
        "degradations": m.degradations,
        "duration_s": m.duration_s,
        "cache_hit_rate": m.cache_hit_rate,
        "files_reviewed": m.files_reviewed,
    }


@app.get("/api/reviews/{run_id}")
async def get_review(run_id: str) -> dict:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")

    payload: dict[str, Any] = {
        "run_id": run.run_id, "target": run.target,
        "status": run.status, "error": run.error, "events": run.events,
    }
    if run.report is not None:
        payload["summary"] = _summary(run.report)
        payload["markdown"] = markdown.render(run.report)
        payload["findings"] = [
            {
                "score": round(f.score, 3),
                "severity": str(f.severity),
                "location": f.cluster.location,
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
                "refuted": f.refuted,
            }
            for f in run.report.findings
        ]
        payload["agreement"] = _agreement(run.report)
    return payload


def _agreement(report: Report) -> dict:
    """How often each pair of families raised the same finding.

    The one view that makes the multi-model thesis visible at a glance: if the families
    are interchangeable, the extra calls are buying nothing.
    """
    families = sorted({f for c in report.findings for f in c.cluster.families})
    counts = {a: dict.fromkeys(families, 0) for a in families}
    for scored in report.findings:
        present = sorted(scored.cluster.families)
        for a in present:
            for b in present:
                counts[a][b] += 1
    return {"families": families, "counts": counts}


@app.get("/api/reviews/{run_id}/stream")
async def stream(run_id: str) -> EventSourceResponse:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")

    async def generator():
        # The events *list* is the single source of truth and a cursor walks it; the
        # queue is only a wake-up signal. Emitting from both -- replaying the list and
        # then draining the queue -- delivers every event twice, since `publish` writes
        # to both. Using one as the record and the other as the doorbell also means a
        # client that reconnects mid-run simply replays from cursor zero.
        cursor = 0
        while True:
            while cursor < len(run.events):
                yield {"data": json.dumps(run.events[cursor])}
                cursor += 1

            if run.status not in ("queued", "running"):
                break

            try:
                await asyncio.wait_for(run.queue.get(), timeout=15.0)
            except TimeoutError:
                # Keep-alive: proxies drop idle SSE connections, and a review can sit
                # quiet for a long time waiting out a rate limit.
                yield {"event": "ping", "data": "{}"}

    return EventSourceResponse(generator())


@app.get("/api/reviews")
async def list_runs() -> dict:
    return {
        "runs": [
            {"run_id": r.run_id, "target": r.target, "status": r.status}
            for r in RUNS.values()
        ]
    }
