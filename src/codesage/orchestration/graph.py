"""The LangGraph review pipeline.

Shape:

    plan ──► review (fan-out: files x lenses x families, in parallel)
             │
             ▼
          ground ──► cluster ──► critique ──► consensus ──► finish

`Send` fans out one task per (file, lens, model), so the review calls run concurrently,
bounded by the quota tracker rather than by Python.

*Known gap.* `build_graph` accepts a LangGraph `checkpointer`, but the runner does not
pass one -- a killed process does not resume mid-graph. What actually protects a crashed
run is the response cache: re-running replays every completed call for free and re-walks
the graph from the start. Weaker than true mid-node resume, and worth saying so.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from codesage.agents import context as context_builder
from codesage.agents import lenses as lens_runner
from codesage.agents import planner as planner_agent
from codesage.agents.context import ContextPack
from codesage.agents.tools import RepoTools
from codesage.config.settings import ModelSpec, Settings
from codesage.domain import Finding, GroundStatus, Lens
from codesage.index.pipeline import IndexResult
from codesage.llm.client import LLMClient
from codesage.llm.router import ModelRouter
from codesage.orchestration.state import ReviewState
from codesage.verify import cluster as clustering
from codesage.verify.ground_check import check_all
from codesage.verify.score import score_all

log = logging.getLogger(__name__)


@dataclass
class ReviewDeps:
    """Everything the nodes need. Passed via config rather than closed over, so the
    graph can be built once and reused across runs."""

    client: LLMClient
    router: ModelRouter
    settings: Settings
    index: IndexResult
    tools: RepoTools
    packs: dict[str, ContextPack] = field(default_factory=dict)
    plan: planner_agent.PlanResult | None = None
    use_tools: bool = True
    on_event: Callable[[str, dict], None] | None = None

    @property
    def agent_tools(self) -> RepoTools | None:
        """Tools, unless this run has them disabled for an ablation."""
        return self.tools if self.use_tools else None

    def emit(self, kind: str, payload: dict) -> None:
        """Progress events for the dashboard. Never allowed to break a review."""
        if self.on_event is None:
            return
        try:
            self.on_event(kind, payload)
        except Exception:
            log.debug("event handler raised; continuing", exc_info=True)


# --------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------


async def plan_node(state: ReviewState, deps: ReviewDeps) -> dict:
    """Triage the repository with the planner agent, then assign models to lenses."""
    planner_model = deps.router.assign_planner()
    if planner_model is not None:
        plan = await planner_agent.plan(
            deps.client, planner_model, deps.index.code, deps.index.lint,
            budget=deps.settings.max_files,
        )
    else:
        plan = planner_agent.fallback_plan(
            deps.index.code, deps.index.lint, deps.settings.max_files
        )
        plan.rationale = "No model had quota for planning; used deterministic triage."

    deps.plan = plan
    deps.emit(
        "plan",
        {
            "source": plan.source,
            "files": [f.path for f in plan.files],
            "rationale": plan.rationale,
            "dropped": plan.dropped,
        },
    )

    packs = {
        planned.path: context_builder.build(
            source, deps.index.code, deps.index.lint.for_file(planned.path),
            plan_reason=planned.reason,
        )
        for planned in plan.files
        if (source := deps.index.inventory.by_path(planned.path)) is not None
    }
    deps.packs = packs

    assignments = {}
    for lens in Lens:
        assignments[str(lens)] = assignment = deps.router.assign_for_lens(lens)
        deps.emit(
            "assignment",
            {"lens": str(lens), "families": assignment.families, "degraded": assignment.degraded},
        )

    families = sorted({f for a in assignments.values() for f in a.families})
    log.info("plan: %d files, families=%s", len(packs), families)
    return {
        "assignments": assignments,
        "families_available": families,
        "degradations": [d.describe() for d in deps.client.quota.degradations],
        "packs": packs,
        "plan_source": plan.source,
        "plan_rationale": plan.rationale,
    }


def make_fan_out(deps: ReviewDeps):
    """One `Send` per (file, lens, model). This is the parallelism.

    Only the lenses the planner assigned to each file are dispatched, which is where
    the planner's judgement turns into saved budget: a pure-arithmetic utility does not
    get a security review just because every other file did.
    """

    def fan_out(state: ReviewState) -> list[Send]:
        sends: list[Send] = []
        for path in state["packs"]:
            wanted = deps.plan.lenses_for(path) if deps.plan else list(Lens)
            for lens_name, assignment in state["assignments"].items():
                if Lens(lens_name) not in wanted:
                    continue
                for model in assignment.models:
                    sends.append(
                        Send("review", {"path": path, "lens": lens_name, "model": model})
                    )
        log.info("fanning out %d review tasks", len(sends))
        return sends or [Send("review", {"path": None, "lens": None, "model": None})]

    return fan_out


async def review_node(task: dict, deps: ReviewDeps) -> dict:
    """Run one lens on one file with one model."""
    if task.get("path") is None:
        return {"findings": [], "out_of_scope": [], "lens_errors": [], "traces": []}

    path: str = task["path"]
    lens = Lens(task["lens"])
    model: ModelSpec = task["model"]
    pack = deps.packs[path]

    deps.emit("review_start", {"path": path, "lens": str(lens), "family": model.family})
    run = await lens_runner.run_lens(
        deps.client, model, lens, pack, deps.agent_tools, max_steps=deps.settings.max_agent_steps
    )
    deps.emit(
        "review_done",
        {
            "path": path,
            "lens": str(lens),
            "family": model.family,
            "findings": len(run.findings),
            "tool_calls": run.trace.tool_calls,
            "steps": run.trace.steps,
            "error": run.trace.error,
        },
    )

    return {
        "findings": run.findings,
        "out_of_scope": run.out_of_scope,
        "lens_errors": [run.describe()] if not run.ok else [],
        "traces": [run.describe()],
    }


async def ground_node(state: ReviewState, deps: ReviewDeps) -> dict:
    """Reject findings that point at code which does not exist."""
    findings: list[Finding] = state.get("findings", [])
    report = check_all(findings, deps.index.inventory, deps.index.code)

    # Claims about files this model was never shown are ungrounded too. They were
    # separated earlier so they never pollute the clustering, but they belong in the
    # hallucination count -- excluding them would understate the rate.
    out_of_scope: list[Finding] = state.get("out_of_scope", [])
    for finding in out_of_scope:
        finding.ground_status = GroundStatus.NO_SUCH_FILE
        finding.ground_detail = "reported on a file this model was not shown"
    rejected = report.rejected + out_of_scope

    total = len(report.grounded) + len(rejected)
    rate = len(rejected) / total if total else 0.0

    deps.emit(
        "grounded",
        {"kept": len(report.grounded), "rejected": len(rejected), "rate": round(rate, 3)},
    )
    return {
        "grounded": report.grounded,
        "rejected": rejected,
        "hallucination_rate": rate,
        "symbol_repairs": report.repaired_symbols,
    }


async def cluster_node(state: ReviewState, deps: ReviewDeps) -> dict:
    """Merge findings that describe the same defect."""
    clusters = clustering.cluster(state.get("grounded", []))
    deps.emit(
        "clustered",
        {
            "clusters": len(clusters),
            "multi_family": sum(1 for c in clusters if c.support > 1),
        },
    )
    return {"clusters": clusters}


async def critique_node(state: ReviewState, deps: ReviewDeps) -> dict:
    """Ask an independent family to refute each cluster.

    Only clusters that could plausibly be shown are challenged. Spending a critic call
    on a finding that will rank near the bottom anyway wastes budget that a borderline
    finding needs, and on a free tier that trade is real.
    """
    clusters = state.get("clusters", [])

    async def challenge(cluster: clustering.FindingCluster):
        model = deps.router.assign_critic(cluster.families)
        if model is None:
            return cluster.id, None
        pack = deps.packs.get(cluster.file)
        if pack is None:
            return cluster.id, None
        verdict = await lens_runner.run_critic(
            deps.client,
            model,
            claim=cluster.representative.raw.claim,
            evidence=cluster.representative.raw.evidence,
            location=cluster.location,
            pack=pack,
            tools=deps.agent_tools,
        )
        deps.emit(
            "critique",
            {"cluster": cluster.id, "family": model.family, "verdict": str(verdict.verdict)},
        )
        return cluster.id, verdict

    results = await asyncio.gather(*(challenge(c) for c in clusters))
    critics = {cid: verdict for cid, verdict in results if verdict is not None}
    log.info("critiqued %d/%d clusters", len(critics), len(clusters))
    return {"critics": critics}


async def consensus_node(state: ReviewState, deps: ReviewDeps) -> dict:
    """Score every cluster by weighted vote."""
    clusters = state.get("clusters", [])

    # Which families reviewed through which lens. Support has to be measured against
    # these, not against every family that ran somewhere -- otherwise a finding from a
    # two-family lens is capped at 2/5 no matter how complete its agreement was.
    lens_families = {
        lens: set(assignment.families) for lens, assignment in state["assignments"].items()
    }

    scored = score_all(
        clusters,
        families_available=len(state.get("families_available", [])),
        lint=deps.index.lint.findings,
        critics=state.get("critics", {}),
        lens_families=lens_families,
    )
    deps.emit("consensus", {"clusters": len(scored)})
    return {"scored": scored}


async def finish_node(state: ReviewState, deps: ReviewDeps) -> dict:
    scored = state.get("scored", [])
    stats = {
        "findings_raw": len(state.get("findings", [])),
        "findings_grounded": len(state.get("grounded", [])),
        "hallucination_rate": round(state.get("hallucination_rate", 0.0), 3),
        "clusters": len(state.get("clusters", [])),
        "clusters_multi_family": sum(1 for s in scored if s.cluster.support > 1),
        "tool_calls": len(deps.tools.call_log),
    }
    log.info("review complete: %s", stats)
    deps.emit("finished", stats)
    return {"stats": stats}


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def build_graph(deps: ReviewDeps, checkpointer=None):
    """Compile the review graph.

    Nodes are bound to `deps` here rather than reading them from LangGraph's config,
    which keeps the node signatures honest -- each one declares exactly what it needs.
    """
    builder = StateGraph(ReviewState)

    # `partial`, not a lambda. A lambda wrapping an async function is itself sync, so
    # LangGraph calls it, gets an un-awaited coroutine back, and raises. `partial`
    # preserves coroutine-function detection, so the node is awaited properly.
    builder.add_node("plan", partial(plan_node, deps=deps))
    builder.add_node("review", partial(review_node, deps=deps))
    builder.add_node("ground", partial(ground_node, deps=deps))
    builder.add_node("cluster", partial(cluster_node, deps=deps))
    builder.add_node("critique", partial(critique_node, deps=deps))
    builder.add_node("consensus", partial(consensus_node, deps=deps))
    builder.add_node("finish", partial(finish_node, deps=deps))

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", make_fan_out(deps), ["review"])
    builder.add_edge("review", "ground")
    builder.add_edge("ground", "cluster")
    builder.add_edge("cluster", "critique")
    builder.add_edge("critique", "consensus")
    builder.add_edge("consensus", "finish")
    builder.add_edge("finish", END)

    return builder.compile(checkpointer=checkpointer)
