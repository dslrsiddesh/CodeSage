"""The state that flows through the review graph.

LangGraph merges state updates from parallel nodes, so anything written concurrently
needs a reducer that says *how* to merge. Findings from four lenses running at once are
appended, not overwritten -- getting this wrong is the classic LangGraph bug, and it
fails silently: the graph runs, and three quarters of the findings vanish.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from codesage.agents.context import ContextPack
from codesage.domain import CriticVerdict, Finding
from codesage.llm.router import Assignment
from codesage.verify.cluster import FindingCluster
from codesage.verify.score import ScoredCluster


class ReviewState(TypedDict, total=False):
    """State for one review run.

    `findings` and `lens_errors` use `operator.add` because the lens nodes fan out and
    write to them simultaneously. Everything else is written by exactly one node.
    """

    # -- inputs, set once ------------------------------------------------------------
    target: str
    run_id: str

    # -- deterministic stage ---------------------------------------------------------
    packs: dict[str, ContextPack]
    families_available: list[str]
    assignments: dict[str, Assignment]
    plan_source: str
    plan_rationale: str

    # -- fan-out: concurrent writes, hence the reducers -------------------------------
    findings: Annotated[list[Finding], operator.add]
    out_of_scope: Annotated[list[Finding], operator.add]
    lens_errors: Annotated[list[str], operator.add]
    traces: Annotated[list[str], operator.add]

    # -- consolidation ---------------------------------------------------------------
    grounded: list[Finding]
    rejected: list[Finding]
    hallucination_rate: float
    symbol_repairs: int
    clusters: list[FindingCluster]
    critics: dict[int, CriticVerdict]
    scored: list[ScoredCluster]

    # -- bookkeeping -----------------------------------------------------------------
    degradations: list[str]
    stats: dict[str, Any]


class ReviewTask(BaseModel):
    """One unit of fan-out work: review this file, through this lens, with this model."""

    path: str
    lens: str
    model_key: str


class RunManifest(BaseModel):
    """Everything needed to reproduce or interpret a run.

    Written alongside the report. Without it, a report is an assertion; with it, the
    reader can see which families actually ran, how much of the ensemble was available,
    and whether the numbers rest on three opinions or one.
    """

    run_id: str
    target: str
    commit: str
    started_at: str
    duration_s: float = 0.0

    files_reviewed: list[str] = Field(default_factory=list)
    files_skipped: int = 0
    lenses: list[str] = Field(default_factory=list)
    families_used: dict[str, int] = Field(default_factory=dict)
    ensemble_size: int = 0

    findings_raw: int = 0
    findings_grounded: int = 0
    findings_rejected: int = 0
    hallucination_rate: float = 0.0
    symbol_repairs: int = 0
    clusters: int = 0
    clusters_multi_family: int = 0

    plan_source: str = "fallback"
    plan_rationale: str = ""
    tool_calls: int = 0
    agent_traces: list[str] = Field(default_factory=list)

    lint_available: bool = True
    lint_findings: int = 0

    degradations: list[str] = Field(default_factory=list)
    cache_hit_rate: float = 0.0
    quota: dict[str, dict[str, int | str]] = Field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return bool(self.degradations)
