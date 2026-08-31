# CodeSage

Multi-agent code review with mechanical grounding and cross-family consensus.

Point it at a GitHub repository and it produces a structured review report. The
interesting part is not that it calls several models — it is how it decides which of
their claims to believe, and how the agents go looking for the evidence.

## The thesis

Studies of LLM code review are consistent: models invent findings that aren't in the
code, and flag correct code as broken. So CodeSage only shows a finding if it:

1. **points at code that actually exists** — checked deterministically against a parse
   of the repository, not by asking another model;
2. **is raised by more than one model family** — two checkpoints of the same base model
   are one opinion sampled twice, not two opinions;
3. **survives a challenge** from a model that didn't propose it.

Findings that fail the first check are dropped *and counted*. That drop rate is the
system's own hallucination rate, and it is reported at the top of every run.

## The agentic core

Four things make this a multi-agent system rather than a prompt chain:

**A planner agent triages the repository.** It sees an outline — filenames, sizes,
function counts, lint hits — and decides which files deserve the budget and which lenses
each one needs. A payments module gets the security lens; a formatter doesn't. Its
output is validated against the index, so a plan naming files that don't exist has those
entries dropped, with a deterministic fallback if nothing survives.

**Review agents pull their own context.** Rather than being handed everything up front,
each agent starts with the file and a structural outline, then calls tools —
`read_symbol`, `find_callers`, `outline_file`, `grep` — to fetch what the specific bug
actually requires. An agent reviewing `total_cents` calls `find_callers` and discovers a
caller passing an empty list. It pulled two functions instead of being handed forty.

**A critic agent gets the same tools.** Prosecution and defence come from different
model families, and the strongest refutation is usually evidence rather than argument:
`find_callers` showing that no caller can reach the state the finding depends on.

Everything downstream is plain arithmetic. Findings are clustered by line overlap plus
word overlap, and scored by a weighted sum of four signals you can check by hand. There
is no statistical machinery to take on trust.

**The harness is written out, not imported.** The agent loop bounds exploration, forces
a landing on the final step, degrades when a model doesn't support tool calling, and
contains failures to one agent. Those four behaviours are the whole reason it isn't a
`while` loop, and they're the parts that matter under a free-tier budget.

## Status

229 tests, fully offline. ~3,100 lines of executable code. Not yet run against live
models — that needs API keys.

## Quick start

```bash
make setup          # install deps, create .env
make test           # 229 tests, fully offline, no API key needed
make index REPO=.   # deterministic stage only — no LLM calls, no keys needed
```

Then add at least two provider keys to `.env` (free tiers: [Groq](https://console.groq.com/keys),
[Cerebras](https://cloud.cerebras.ai), [OpenRouter](https://openrouter.ai/keys)) and:

```bash
make doctor                                 # which models are actually reachable
make review REPO=https://github.com/owner/x # full review -> reports/
make serve                                  # dashboard on localhost:8000
make eval                                   # mutation benchmark -> RESULTS.md
```

## How it works

```
GitHub URL
   │
1. INGEST     clone → file inventory                    ┐
   ▼                                                    │ deterministic,
2. INDEX      ast parse → repo map (symbols, callers)   │ no model calls,
   ▼          ruff → static findings                    ┘ ~19% of the code
3. PLAN       planner agent picks files and lenses      ┐
   ▼                                                    │
4. REVIEW     lens agents × model families, in parallel │ agentic,
   ▼          each calling tools to pull context        │ ~81% of the code
5. VERIFY     ground check → cluster → critic → score   │
   ▼                                                    ┘
6. REPORT     Markdown + JSON, streamed to a dashboard
```

## Evaluation

`make eval` injects known defects into copies of a real repository using seven AST-level
mutation operators, then measures how many the reviewer locates. Ground truth is exact
and needs no labelling.

| ablation | question |
|---|---|
| ± agent tools | does letting the agent explore the repo find more bugs? |
| 1 family vs 2 | does cross-family diversity beat a single model? |
| ± lint seeding | does seeding the lenses with linter findings help? |

The tools ablation is the one this project most needs to answer honestly: tool calling
is its central bet, it costs a round trip per hop, and "the agent explored the codebase"
is exactly the kind of claim that sounds good and might buy nothing.

Recall is the headline metric and precision deliberately is not: whether a finding that
misses the injected line is a false positive depends on whether the original code had a
real defect there, which is the one thing this benchmark cannot know.

## Design notes

**Why tools instead of a fat prompt.** The obvious design pre-computes everything an
agent might want and sends one enormous prompt. Most of that context is irrelevant to
the specific bug, it costs tokens on every call, and it caps out on any file worth
reviewing. The trade is real in both directions — a naive agent can spend more tokens
exploring than the fat prompt would have used — which is why `max_agent_steps` bounds it
and why the ablation measures whether it actually helps.

**Why the caller index is allowed to be approximate.** There are two consumers with
opposite requirements. The ground check asks only per-file questions, answered from a
real parse with no guessing. Tools and context only need to be *useful* — `find_callers`
matches on bare names, so a spurious caller costs one wasted read. Keeping those apart
is the whole design; the moment the ground check trusts the caller index, approximate
becomes dangerous.

**Why one static analyzer, not three.** Ruff's `S` rules are a reimplementation of
Bandit. Running both meant every security finding was reported twice, and the "two
independent tools agree" corroboration signal in the scorer was double-counting one
tool's opinion.

**Why there is no EM consensus.** An earlier version estimated per-family reliability
with Dawid–Skene, unsupervised, and it beat majority voting when annotators are unequally
reliable. It was cut anyway: the whole scoring step is now something a reader can verify
by arithmetic, and a number you can check beats a better number you have to trust.

**Why the quota tracker is load-bearing.** Free tiers impose two kinds of limit that
fail in opposite ways: a per-minute rate limit is worth waiting out, a per-day cap is
not. When quota forces a smaller ensemble, that is recorded and surfaced — a review that
ran one model instead of three is a different experiment.

## Licence

MIT
