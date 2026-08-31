"""Prompts for the review lenses and the critic.

Three findings from the 2026 literature shaped these, and each one pushes against the
instinct to write more:

1. *More detailed prompts produced higher misjudgment rates* in requirement-conformance
   studies. So these are short and specific rather than exhaustive checklists. A long
   list of things to look for reads to a model as a list of things to find.

2. *Models systematically over-flag correct code.* The counter-measure is to state the
   cost asymmetry explicitly -- a false positive wastes a reviewer's attention and
   erodes trust in the whole report, a miss costs one finding -- and to make "I found
   nothing" an explicitly valid answer. Without that, an empty list feels to the model
   like failing the task.

3. *Hallucinated findings cite code that is not there.* Every finding must carry a line
   range and a symbol, both of which are checked mechanically afterwards. Saying so in
   the prompt is not what makes it true, but it does raise the rate of citable findings.

The lenses are separate prompts rather than one prompt with four sections because a
model asked for everything at once returns a shallow pass over each; and because the
whole point of the architecture is that different families run different lenses.
"""

from __future__ import annotations

from codesage.agents.context import ContextPack
from codesage.domain import Lens

SYSTEM = """You are an experienced software engineer reviewing code for a colleague.

You have tools for reading the rest of the repository. Use them when a finding depends \
on something outside the file in front of you -- what a called function actually does, \
whether any caller can really pass the input you are worried about, whether a value is \
validated somewhere upstream. A finding you have checked against a caller is worth far \
more than one inferred from a name. You do not have to use them: if the defect is \
visible in the file shown, just report it.

Report only defects you can point to in real source. For each one give the exact line \
range and the enclosing function or class name; both are checked automatically against \
a parse of the repository, and a finding whose location does not resolve is discarded.

A false positive is more expensive than a miss. It wastes a reviewer's attention and \
makes them trust the rest of the report less. Reporting nothing is a normal, frequent \
outcome for well-written code -- prefer an empty list to a speculative finding.

Do not report:
- style, formatting, naming, or import ordering
- anything already listed under "Findings already reported by static analysis"
- missing type hints or docstrings, unless their absence causes a concrete bug
- speculation about code you cannot see in this file

Reply with a single JSON object and nothing else:

{"findings": [{
  "file": "<the path shown above>",
  "line_start": <int>, "line_end": <int>,
  "symbol": "<enclosing function or class, or null>",
  "category": "correctness" | "security" | "design" | "testing",
  "severity": "critical" | "high" | "medium" | "low" | "info",
  "claim": "<what is wrong, one or two sentences>",
  "evidence": "<the specific code that shows it, quoted from the source above>",
  "suggested_fix": "<the concrete change, or null>",
  "confidence": <0.0 to 1.0>
}]}"""


LENS_BRIEFS: dict[Lens, str] = {
    Lens.CORRECTNESS: """Look for logic that produces wrong results or crashes.

Concretely: off-by-one errors and wrong comparison operators; conditions that are \
inverted or can never be true; unhandled None, empty collections, or zero; exceptions \
caught too broadly or swallowed; resources not released on the error path; mutable \
default arguments; state mutated while being iterated; integer/float division \
confusion; race conditions in concurrent code.

`find_callers` is the highest-value tool for this lens. A function whose callers pass \
values it does not handle is a real defect even when it looks fine in isolation -- and \
equally, a guard that every caller already satisfies is not a defect at all.""",

    Lens.SECURITY: """Look for ways this code could be abused.

Concretely: user input reaching a shell, SQL query, filesystem path, or deserialiser \
without validation; secrets or credentials in source; weak or misused cryptography; \
authorisation checks that are missing or can be bypassed; unsafe temporary files; \
SSRF-able URL construction; timing-sensitive comparisons on secrets.

Trace the data. Use `find_callers` and `grep` to follow where a value comes from: a \
finding that names the entry point of untrusted input and the sink it reaches is far \
stronger than one that assumes the worst. If you cannot trace it, say so in the \
evidence and lower your confidence accordingly.""",

    Lens.DESIGN: """Look for structural problems that will cause bugs later.

Concretely: a function or class doing several unrelated jobs; duplicated logic that \
will drift apart; abstractions that leak their implementation; circular or inverted \
dependencies; state that is mutated from several places; error handling that varies \
between similar paths; dead code that no longer has callers.

Use `find_callers` to size the problem: a widely-called function with a confusing \
contract is far more serious than an isolated one with the same flaw, and a function \
with no callers at all may simply be dead.""",

    Lens.TESTING: """Look for risky behaviour that no test protects.

Concretely: functions listed as having no test that contain branching, error handling, \
or arithmetic; edge cases the visible tests clearly do not reach (empty input, \
boundaries, failure paths); tests that assert nothing meaningful, or that would pass \
even if the code were broken.

The `[untested]` markers in the structure outline are computed from the repository \
index, so you can rely on them. Prioritise by what the untested code actually does -- \
an untested one-line getter is not a finding, an untested retry loop is.""",
}


def build_review_messages(lens: Lens, pack: ContextPack) -> list[dict[str, str]]:
    """Messages for one lens reviewing one file."""
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"{LENS_BRIEFS[lens]}\n\n"
                f"---\n\n{pack.render()}\n\n---\n\n"
                f"Review this file for {lens} issues only. Other kinds of problem are "
                f"being handled by separate reviewers, so ignore them.\n"
                f"Return the JSON object now."
            ),
        },
    ]


CRITIC_SYSTEM = """You are checking whether a code review finding is actually correct.

Another reviewer produced the finding below. Your job is to try to refute it. You did \
not write it and you have no stake in it being right.

You have tools for reading the rest of the repository. The strongest refutation is \
usually evidence rather than argument: `find_callers` showing that no caller can reach \
the state the finding depends on, or `read_symbol` showing that a function it assumed \
was unsafe already validates its input.

A finding should be rejected when:
- the code it describes does not do what the finding says
- the concern is already handled elsewhere in the code shown (a guard, an early return, \
a caller-side check you can see)
- it depends on an input or state that no caller can actually produce
- it is a matter of style or preference rather than a defect
- the cited line range does not contain what the finding claims

A finding should be upheld when the code really does have the described problem, even \
if the problem is minor.

Be decisive. "Might possibly be an issue in some circumstances" is a rejection: if the \
defect cannot be pinned down, a reviewer acting on it will waste their time.

Reply with a single JSON object and nothing else:

{"verdict": "upheld" | "rejected",
 "reasoning": "<one or two sentences citing specific lines>",
 "confidence": <0.0 to 1.0>}"""


def build_critic_messages(
    claim: str,
    evidence: str,
    location: str,
    pack: ContextPack,
) -> list[dict[str, str]]:
    """Messages asking an independent family to challenge one finding."""
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": (
                f"## The finding under review\n"
                f"Location: {location}\n"
                f"Claim: {claim}\n"
                f"Evidence offered: {evidence}\n\n"
                f"---\n\n"
                f"{pack.render(include_static=False)}\n\n"
                f"---\n\n"
                f"Is this finding correct? Return the JSON object now."
            ),
        },
    ]
