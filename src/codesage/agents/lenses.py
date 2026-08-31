"""Running one lens with one model, and running the critic.

Both are thin wrappers over the agent loop. The only thing worth noting is what this
module refuses to do: a model's output is parsed into `RawFinding` objects and nothing
else. `Provenance` -- which model, which family, which lens -- is attached here, by us,
from the arguments we passed in. If a model could report its own family, every agreement
statistic in the report would rest on self-reported data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from codesage.agents.context import ContextPack
from codesage.agents.prompts import build_critic_messages, build_review_messages
from codesage.agents.tools import RepoTools
from codesage.config.settings import ModelSpec
from codesage.domain import CriticVerdict, Finding, Lens, Provenance, RawFindingList, Verdict
from codesage.harness.loop import AgentTrace, run_agent
from codesage.llm.client import LLMClient

log = logging.getLogger(__name__)


@dataclass
class LensRun:
    lens: Lens
    model: ModelSpec
    path: str
    findings: list[Finding] = field(default_factory=list)
    out_of_scope: list[Finding] = field(default_factory=list)
    trace: AgentTrace = field(default_factory=AgentTrace)

    @property
    def ok(self) -> bool:
        return self.trace.error is None

    def describe(self) -> str:
        head = f"{self.lens}/{self.model.family} on {self.path}"
        if self.trace.error:
            return f"{head}: {self.trace.describe()}"
        scope = f", {len(self.out_of_scope)} out of scope" if self.out_of_scope else ""
        return f"{head}: {len(self.findings)} findings{scope} ({self.trace.describe()})"


async def run_lens(
    client: LLMClient,
    model: ModelSpec,
    lens: Lens,
    pack: ContextPack,
    tools: RepoTools | None = None,
    *,
    max_steps: int = 4,
) -> LensRun:
    """Review one file through one lens with one model, with tools available."""
    run = LensRun(lens=lens, model=model, path=pack.path)
    parsed, trace = await run_agent(
        client,
        model,
        build_review_messages(lens, pack),
        RawFindingList,
        tools=tools,
        max_steps=max_steps,
        stage=f"review:{lens}",
    )
    run.trace = trace

    if parsed is None:
        log.info("%s", run.describe())
        return run

    provenance = Provenance(
        model_id=model.id, family=model.family, provider=model.provider, lens=lens
    )
    for raw in parsed.findings:
        finding = Finding(raw=raw, provenance=provenance)
        # A claim about a file the agent only saw named -- rather than the one it was
        # asked to review -- is separated out. It still counts toward the hallucination
        # rate; dropping it silently would flatter every model in the report.
        if raw.file != pack.path:
            run.out_of_scope.append(finding)
            continue
        run.findings.append(finding)

    log.info("%s", run.describe())
    return run


async def run_critic(
    client: LLMClient,
    model: ModelSpec,
    *,
    claim: str,
    evidence: str,
    location: str,
    pack: ContextPack,
    tools: RepoTools | None = None,
) -> CriticVerdict:
    """Ask an independent family to refute a finding.

    The critic gets tools too, and that is the point: the strongest refutation is
    usually "no caller can reach this state", which requires actually looking at the
    callers rather than reasoning about the file in isolation.

    A failed critic call returns UNCHALLENGED rather than UPHELD. "Nobody could
    challenge this" is weaker support than "someone tried and failed", and collapsing
    the two would inflate confidence in exactly the runs where quota ran out.
    """
    parsed, trace = await run_agent(
        client,
        model,
        build_critic_messages(claim, evidence, location, pack),
        CriticVerdict,
        tools=tools,
        max_steps=3,
        stage="critic",
    )
    if parsed is None:
        log.info("critic %s unavailable: %s", model.key, trace.describe())
        return CriticVerdict(
            verdict=Verdict.UNCHALLENGED, reasoning=f"critic unavailable ({trace.describe()})"
        )
    return parsed
