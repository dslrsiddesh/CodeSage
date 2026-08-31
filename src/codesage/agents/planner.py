"""The planner agent: deciding what to review, and through which lenses.

This replaced a statistical risk model -- z-scored fan-in, churn, complexity, and lint
density, clipped and summed. That model worked, and swapping it for an LLM is a real
trade with costs on both sides, so it is worth stating honestly rather than pretending
the agent is strictly better.

**What the formula did well.** It was deterministic, free, instant, and every ranking
came with an arithmetic explanation. It never hallucinated a filename.

**Why an agent instead.** The formula could only rank on what it could count. It had no
way to know that `auth.py` matters more than `colours.py` at equal complexity, that a
file named `migrations/` is usually not worth review budget, or that a payments module
deserves the security lens while a formatter deserves correctness. Those are judgements
about *what the code is for*, and a model that has read the outline can make them where
arithmetic cannot.

**How the cost is contained.** The planner is one cheap call over a compact repo
outline -- filenames, sizes, function counts, lint hits -- never file contents. Its
output is validated against reality: a plan naming files that do not exist has those
entries dropped, and if nothing survives, a deterministic fallback ranks by lint density
and size. The agent gets to be smart; it does not get to be trusted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from codesage.config.settings import ModelSpec
from codesage.domain import Lens
from codesage.harness.loop import run_agent
from codesage.index.code import CodeIndex
from codesage.index.lint import LintReport
from codesage.llm.client import LLMClient
from codesage.llm.errors import ProviderError

log = logging.getLogger(__name__)


class PlannedFile(BaseModel):
    model_config = {"extra": "ignore"}

    path: str = Field(description="Repo-relative path from the outline")
    reason: str = Field(default="", description="Why this file is worth review budget")
    lenses: list[str] = Field(
        default_factory=list,
        description="Which lenses to apply: correctness, security, design, testing",
    )


class ReviewPlan(BaseModel):
    model_config = {"extra": "ignore"}

    files: list[PlannedFile] = Field(default_factory=list)
    rationale: str = Field(default="", description="One sentence on the overall strategy")


@dataclass
class PlanResult:
    """A validated plan, plus how it was produced."""

    files: list[PlannedFile] = field(default_factory=list)
    rationale: str = ""
    source: str = "agent"  # "agent" | "fallback"
    dropped: list[str] = field(default_factory=list)
    trace: str = ""

    def lenses_for(self, path: str) -> list[Lens]:
        """Lenses for one file, defaulting to all four when the plan did not say."""
        for planned in self.files:
            if planned.path == path:
                chosen = [
                    Lens(name) for name in planned.lenses if name in {str(x) for x in Lens}
                ]
                return chosen or list(Lens)
        return list(Lens)

    def describe(self) -> str:
        detail = f"{len(self.files)} files via {self.source}"
        if self.dropped:
            detail += f", dropped {len(self.dropped)} hallucinated path(s)"
        return detail


PLANNER_SYSTEM = """You are triaging a Python repository for code review.

A review budget is limited, so you are choosing which files are most worth spending it \
on, and which review perspectives each one needs.

Prefer files where a defect would actually matter: authentication, payments, permissions, \
data validation, parsing of untrusted input, concurrency, money or time arithmetic, and \
anything with many callers. Deprioritise generated code, configuration, migrations, \
constants, thin wrappers, and files that mostly re-export other modules.

Available lenses:
- correctness: logic errors, edge cases, unhandled None, resource leaks
- security: injection, authorization, secrets, unsafe deserialization, weak crypto
- design: coupling, duplication, unclear contracts, dead code
- testing: risky behaviour with no test covering it

Assign only the lenses each file actually warrants. A pure-logic utility rarely needs \
the security lens; an input parser almost always does. Assigning every lens to every \
file wastes budget that a second file needed.

Choose paths only from the outline given. Do not invent filenames.

Reply with a single JSON object and nothing else:

{"rationale": "<one sentence on your strategy>",
 "files": [{"path": "<exact path from the outline>",
            "reason": "<why this file>",
            "lenses": ["correctness", "security"]}]}"""


def fallback_plan(
    index: CodeIndex, lint: LintReport, budget: int
) -> PlanResult:
    """Deterministic triage when the agent is unavailable or produced nothing usable.

    Ranks by lint findings first, then by how much code there is to be wrong. Crude
    compared to the old z-scored model, but this is a safety net rather than the main
    path, and a simple net is easier to trust than a clever one.
    """
    hits: dict[str, int] = {}
    for finding in lint.findings:
        hits[finding.file] = hits.get(finding.file, 0) + 1

    ranked = sorted(
        index.files.values(),
        key=lambda f: (-hits.get(f.path, 0), -len(f.symbols), f.path),
    )
    return PlanResult(
        files=[
            PlannedFile(
                path=f.path,
                reason=(
                    f"{hits.get(f.path, 0)} lint findings, {len(f.symbols)} symbols"
                ),
                lenses=[str(lens) for lens in Lens],
            )
            for f in ranked[:budget]
        ],
        rationale="Deterministic fallback: ranked by static-analysis density, then size.",
        source="fallback",
    )


def _build_outline(index: CodeIndex, lint: LintReport, budget: int) -> str:
    """The planner's whole view of the repository: names and counts, never contents."""
    hits: dict[str, int] = {}
    for finding in lint.findings:
        hits[finding.file] = hits.get(finding.file, 0) + 1

    rows = []
    for path, file in sorted(index.files.items()):
        flags = []
        if hits.get(path):
            flags.append(f"{hits[path]} static findings")
        if any("test" in c.file for s in file.symbols for c in index.callers_of(s.name)):
            flags.append("has tests")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        rows.append(f"{path}  ({file.n_lines} lines, {len(file.symbols)} symbols){suffix}")

    return (
        f"Repository: {len(index.files)} Python files.\n"
        f"Budget: choose at most {budget} files.\n\n"
        + "\n".join(rows)
    )


async def plan(
    client: LLMClient,
    model: ModelSpec,
    index: CodeIndex,
    lint: LintReport,
    *,
    budget: int,
) -> PlanResult:
    """Ask a model which files deserve the review budget, then validate its answer."""
    if not index.files:
        return PlanResult(rationale="Nothing to review.", source="fallback")

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": _build_outline(index, lint, budget)},
    ]

    try:
        parsed, trace = await run_agent(
            client, model, messages, ReviewPlan, tools=None, max_steps=2, stage="plan"
        )
    except ProviderError as exc:
        log.info("planner unavailable (%s); falling back", exc)
        return fallback_plan(index, lint, budget)

    if parsed is None or not parsed.files:
        log.info("planner produced no usable plan (%s); falling back", trace.describe())
        result = fallback_plan(index, lint, budget)
        result.trace = trace.describe()
        return result

    # Validate every path against the index. A planner that invents a plausible filename
    # would otherwise send the review stage chasing a file that does not exist.
    kept, dropped = [], []
    for entry in parsed.files:
        normalised = entry.path.strip().lstrip("./")
        if normalised in index.files:
            kept.append(PlannedFile(path=normalised, reason=entry.reason, lenses=entry.lenses))
        else:
            dropped.append(entry.path)

    if not kept:
        log.warning("every planned path was hallucinated (%s); falling back", dropped[:3])
        result = fallback_plan(index, lint, budget)
        result.dropped = dropped
        return result

    result = PlanResult(
        files=kept[:budget],
        rationale=parsed.rationale,
        source="agent",
        dropped=dropped,
        trace=trace.describe(),
    )
    log.info("plan -- %s", result.describe())
    return result
