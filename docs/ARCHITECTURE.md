# Architecture

Written to be read before an interview. Every section says *what* a component does,
*why* it exists, and *what would break* without it — because "why is it built that way"
is the question that actually gets asked.

---

## The one-sentence version

> A finding is only shown if it points at code that actually exists, is raised by more
> than one model family, and survives a challenge from a model that did not propose it.

The agents go and find the evidence for those checks themselves.

## Shape of the system

```
1. INGEST     clone → file inventory                    ┐  deterministic
2. INDEX      ast parse → repo map; ruff → findings     ┘  ~19% of the code
3. PLAN       planner agent picks files and lenses      ┐
4. REVIEW     lens agents × families, calling tools     │  agentic
5. VERIFY     ground → cluster → critic → score         │  ~81% of the code
6. REPORT     markdown + json + streaming dashboard     ┘
```

Stages 1–2 never call a model. By the time anything expensive runs, the facts it will be
checked against are fixed and fully unit-tested.

---

## The agentic layer

### Tool surface — `agents/tools.py`

Four read-only tools: `read_symbol`, `find_callers`, `outline_file`, `grep`.

**The bet.** The obvious design pre-computes everything an agent might want and sends
one enormous prompt. That fails in a specific way: most of the context is irrelevant to
the specific bug, it costs tokens on every call, and it caps out at the context window
on any file worth reviewing. Instead an agent starts small and *asks* for the rest.

**The counter-argument, which you should raise yourself.** Tool calling costs a round
trip per hop, and each hop resends the whole conversation — so exploration is quadratic
in tokens, not linear. A naive agent can spend more than the fat prompt would have.
`max_agent_steps` bounds it, results are truncated at 6,000 characters, and the
`± tools` ablation exists precisely because this might not pay off.

**Security.** Every tool is read-only and repo-scoped. Nothing writes, executes, or
reaches the network. These run whatever an untrusted model asks for, and path traversal
out of the checkout is the obvious way that goes wrong.

**Robustness.** No tool ever raises. An invented tool name, a malformed regex, a missing
argument — each returns an error string as a normal tool result so the model can recover
next turn. An exception would abort the review over one bad guess.

### Agent loop — `harness/loop.py`

The harness, written out rather than imported. Four behaviours make it more than a
`while` loop:

1. **Bounded exploration.** `max_steps` caps round trips. Without it, a model that keeps
   grepping never terminates — and on a free tier that's the day's quota on one file.
2. **Graceful loss of tool calling.** Not every free-tier model supports `tools`. Some
   400, some accept it and never emit a call. A rejection retries once without tools; a
   model that ignores them still reviews from what it was given. Tools accelerate, they
   aren't a dependency.
3. **Forced landing.** On the final step the tools are withdrawn and the model is told
   to answer now. A loop that ends mid-exploration wastes everything it spent.
4. **Failure containment.** Any exception returns an empty result for that one agent.
   One model failing costs a lens one opinion; it must not abort the review.

*Why not LangChain's agent executor?* It would do this in ten lines and hide all four.
Those are the parts that matter under a budget, and the parts worth being able to explain.

### Planner agent — `agents/planner.py`

**What it replaced.** A statistical risk model — z-scored fan-in, churn, complexity and
lint density, clipped and summed. That model was deterministic, free, instant, and
explainable. Be honest that it had real virtues.

**Why an agent.** The formula could only rank on what it could count. It had no way to
know `auth.py` matters more than `colours.py` at equal complexity, or that a payments
module deserves the security lens while a formatter deserves correctness. Those are
judgements about *what code is for*.

**How the cost is contained.** One cheap call over an outline — names and counts, never
file contents. Output validated against the index: hallucinated paths are dropped, and
if nothing survives there's a deterministic fallback. The agent gets to be smart; it
does not get to be trusted.

### Lenses and critic — `agents/lenses.py`, `agents/prompts.py`

Four lenses (correctness, security, design, testing), each run by two different model
*families*. The critic comes from a family that did not propose the finding — and gets
the same tools, because the strongest refutation is usually evidence: `find_callers`
showing no caller can reach the state the finding depends on.

**Prompt design, informed by the literature.** Studies found that *more detailed* prompts
raised misjudgment rates, and that models systematically over-flag correct code. So the
prompts are short, state the cost asymmetry explicitly (a false positive is worse than a
miss), and make "I found nothing" an explicitly valid answer.

---

## The verification layer

### Ground check — `verify/ground_check.py`

**What.** Every finding names a file, a line range, usually a symbol. Those are checked
against the parse. Anything that doesn't resolve is dropped.

**Why it's the best code in the project.** ~150 lines, no model call, and it catches the
single failure mode that makes LLM review untrustworthy. The drop rate *is* the measured
hallucination rate, printed at the top of every report.

**The subtlety.** A model naming the enclosing class instead of the method has
*mislabelled*, not hallucinated. Those are repaired — and normalised to the innermost
symbol owning the line, which has a useful side effect: two models describing the same
defect at different granularity get the same symbol and therefore cluster together.

**The accuracy split.** The ground check only asks *per-file* questions, answered from a
real `ast` parse. It never consults the caller index, which matches bare names and is
approximate. Keeping those apart is the design — the moment the ground check trusts
approximate data, a hallucinated finding gets confirmed as real.

### Clustering and scoring — `verify/cluster.py`, `verify/score.py`

Two findings merge when their line ranges overlap **and** their claims are textually
similar. The location gate does most of the discriminating, which is why token overlap
suffices for the text half.

Score is a transparent weighted sum: family support, static corroboration, critic
verdict, self-reported confidence (weighted lowest — it's the only input a model
controls directly). `unchallenged` scores below `upheld`: "nobody could challenge this"
is weaker than "someone tried and failed."

### Why there is no EM consensus

An earlier version estimated per-family reliability with Dawid–Skene: treat each model
as a noisy annotator, run EM over the agreement structure, recover a confusion matrix
per family and a posterior per finding — all without labels. It worked, and it beat
majority voting (0.945 vs 0.895) on synthetic annotators with known reliability.

It was removed deliberately. Three reasons, and this is a good "what did you cut and
why" answer:

1. **Every number in the report is now checkable by hand.** The weighted sum is four
   terms with fixed weights. A reader can verify a score with a calculator.
2. **EM has three failure modes** — local optima, label symmetry, and missing votes —
   each needing a guard, and each one more surface to defend.
3. **It bought little here.** With two families per lens, there is rarely enough
   disagreement structure for EM to learn from; it fell back to the weighted vote on
   most real runs anyway.

The lesson worth stating: a sophisticated estimator you cannot fully justify is worse
than a simple one you can.

---

## Free-tier engineering — `llm/`

**Two limits, opposite responses.** A per-minute rate limit is worth waiting out; a
per-day cap is not. Conflating them wastes either a review or an hour.

**The cache isn't an optimisation.** A full ablation sweep would blow the daily quota
many times over. With it, only the first run of a given (model, prompt) pair costs
anything — and the test suite runs offline with no API key as a direct consequence. The
cache stores the full assistant message including tool calls, or a replayed agent loop
would silently lose the model's exploration.

**Call order: cache → quota → network.** A cached call costs no quota, and quota is
checked *before* the request rather than discovered via a 429.

**Family, not model id, is the unit of diversity.** Two checkpoints of one base model are
one opinion sampled twice. The router returns a *smaller* ensemble rather than pad it,
and distinguishes *substitution* (full ensemble, lower-preference families — mild) from
*degradation* (fewer models — serious, and reported).

---

## Bugs worth telling the story of

**1. `DiGraph` → `MultiDiGraph`** *(in the version this replaced)*. A test function
calling its target had both a `CALLS` and a `TESTS` edge; on a `DiGraph` the second
overwrote the first, losing call edges for exactly the files with tests.

**2. Support measured against the wrong denominator.** A correctness finding was scored
against all 5 families that ran anywhere, but only 2 ran the correctness lens — so it
could never exceed 2/5. Fixed per-lens, confidence went **0.46 → 0.70**.

**4. Duplicate SSE delivery.** The stream replayed the events list *and* drained the
queue — `publish` writes to both, so every event arrived twice. Now the list is the
single source of truth and the queue is a wake-up signal.

All four *silently produced wrong numbers* rather than crashing, which is the dangerous
kind.

---

## Known gaps — say these before they're found

**Checkpointing isn't wired.** `build_graph()` takes an optional LangGraph
`checkpointer`, but the runner never passes one, so a killed process doesn't resume
mid-graph. The property people care about — not re-paying quota after a crash — comes
from the response cache: re-running replays completed calls for free and re-walks the
graph from the start. Weaker than true resume. Fixing it means
`langgraph-checkpoint-sqlite` keyed by a stable `(target, commit)` thread id.

**Tool calling support varies by model.** Handled (retry without tools), but it means the
agentic behaviour is uneven across free-tier providers, and the `± tools` ablation may
be measuring provider capability as much as the idea.

**The caller index is name-based.** `save()` matches every `save` in the repo. Fine for
context; explicitly *not* trusted by the ground check.

---

## Deliberately not built

| Skipped | Why |
|---|---|
| tree-sitter | Python-only project; `ast` ships in the standard library |
| Dawid–Skene EM consensus | Worked, but the weighted vote is checkable by hand — see above |
| NetworkX / any graph library | Three dicts do what this asks |
| Bandit, Semgrep | Ruff's `S` rules *are* Bandit. Running both double-counted corroboration |
| Git churn signal | Pure statistics, no agentic value, and the planner reasons about purpose instead |
| Trained validity classifier | Needs a labelled dataset. The weighted vote is fitted on mutation ground truth instead |
| JS/TS support | Python-only keeps the parser honest; the interface is additive |
| GitHub App / webhooks | Auth and deployment plumbing, zero intellectual content |
| React/Vite frontend | One HTML file with `EventSource` demos identically, no build step |

---

## Testing philosophy

229 tests, fully offline, no API key. ~3,100 lines of executable code.

- **Exact-structure tests** for the parser and index — the precise symbol set, not "at
  least N". A parser regression silently discards true findings.
- **Adversarial tests** for the ground check: fabricated findings of the shapes models
  actually produce (invented symbol, line 9999, a file never reviewed).
- **Robustness tests** for tools: every tool, every malformed input, asserting nothing
  raises.
- **Harness tests** mapping one-to-one onto the four loop behaviours.
- **A full graph run against a scripted model**, which is what makes "a hallucinated
  finding never reaches the report" a test rather than an aspiration.
