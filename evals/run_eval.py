"""Build a mutation benchmark, run the ablations, write RESULTS.md.

The grid is the point. Each row asks whether something the architecture claims is
actually true, and any of them can come back negative:

    full          the pipeline as shipped
    no-tools      does letting an agent explore the repository find more bugs?
    one-family    does cross-family diversity beat a single model?
    no-lint-seed  does seeding the lenses with linter findings help?

`no-tools` is the one this project most needs to answer honestly. Tool calling is its
central bet, it costs a round trip per hop, and "the agent explored the codebase" is
exactly the kind of claim that sounds good and might buy nothing.

Recall is the headline and precision deliberately is not. Whether a finding that misses
the injected line is a *false positive* depends on whether the original code had a real
defect there -- the one thing this benchmark cannot know. Reporting precision would mean
quietly assuming the source repository is defect-free.

A negative result is reported as a negative result. Quietly dropping an ablation that
came back flat would make every remaining number untrustworthy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from evals.mutate import Mutation, mutate
from codesage.config.settings import Settings, get_registry
from codesage.index.pipeline import run as run_index
from codesage.ingest.inventory import discover
from codesage.orchestration.runner import review

# A finding may reasonably cite the enclosing statement rather than the exact line -- a
# loop-bound bug is often reported at the loop header.
WINDOW = 2


@dataclass
class Case:
    """One repository copy containing exactly one injected defect."""

    case_id: str
    root: Path
    mutation: Mutation


@dataclass
class Result:
    case_id: str
    kind: str
    detected: bool
    score: float = 0.0
    other_findings: int = 0
    ungrounded: int = 0
    error: str | None = None


@dataclass
class Config:
    name: str
    question: str
    use_tools: bool = True
    use_lint_seed: bool = True
    ensemble_size: int = 2


GRID = [
    Config("full", "the pipeline as shipped"),
    Config("no-tools", "does exploring the repository find more bugs?", use_tools=False),
    Config("one-family", "does cross-family diversity beat one model?", ensemble_size=1),
    Config("no-lint-seed", "does seeding the lenses with linter findings help?", use_lint_seed=False),
]


def build_cases(repo: Path, work: Path, *, count: int, seed: int) -> list[Case]:
    """Make `count` independent copies of `repo`, each with one injected defect.

    A full copy per case rather than one repo mutated in place: each case is reviewed
    independently, and a shared tree would let one case's mutation leak into another's
    context.
    """
    rng = random.Random(seed)
    work.mkdir(parents=True, exist_ok=True)

    clean = work / "clean"
    if clean.exists():
        shutil.rmtree(clean)
    shutil.copytree(repo, clean, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"))

    # Test files are not mutation targets: a broken assertion is a broken test, not the
    # kind of production defect the reviewer is meant to catch.
    files = [f for f in discover(clean).source_files if f.n_lines > 15]
    rng.shuffle(files)

    cases: list[Case] = []
    for source in files:
        if len(cases) >= count:
            break
        for mutated, mutation in mutate(source.read(), source.path, rng=rng, limit=2):
            if len(cases) >= count:
                break
            case_id = f"case{len(cases):03d}"
            root = work / case_id
            if root.exists():
                shutil.rmtree(root)
            shutil.copytree(clean, root)
            (root / source.path).write_text(mutated, encoding="utf-8")
            cases.append(Case(case_id=case_id, root=root, mutation=mutation))

    if not cases:
        raise SystemExit(f"no mutable source files found in {repo}")
    return cases


async def run_case(case: Case, config: Config, settings: Settings, registry) -> Result:
    """Review one mutated repository and check whether the defect was found.

    The planner chooses files on its own rather than being forced onto the mutated one.
    That makes triage part of what is measured: a review that never looks at the
    defective file has genuinely missed the bug.
    """
    tuned = settings.model_copy(update={"ensemble_size": config.ensemble_size})
    try:
        index = run_index(str(case.root), tuned)
        if not config.use_lint_seed:
            index.lint.findings.clear()
        report = await review(
            str(case.root), tuned, registry, use_tools=config.use_tools, index=index
        )
    except Exception as exc:
        return Result(case.case_id, case.mutation.kind, False, error=f"{type(exc).__name__}: {exc}"[:120])

    m = case.mutation
    hits = [
        f
        for f in report.shown
        if f.cluster.file == m.file
        and f.cluster.line_start - WINDOW <= m.line <= f.cluster.line_end + WINDOW
    ]
    return Result(
        case_id=case.case_id,
        kind=m.kind,
        detected=bool(hits),
        score=max((f.score for f in hits), default=0.0),
        other_findings=len(report.shown) - len(hits),
        ungrounded=report.manifest.findings_rejected,
    )


def rate(results: list[Result]) -> float:
    return sum(1 for r in results if r.detected) / len(results) if results else 0.0


def bootstrap_ci(results: list[Result], *, seed: int = 0, iterations: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap for the detection rate.

    The interval is wide at this sample size, which is the point: a configuration that
    looks several points better very often is not.
    """
    if not results:
        return (0.0, 0.0)
    rng = random.Random(seed)
    flags = [1.0 if r.detected else 0.0 for r in results]
    means = sorted(
        sum(flags[rng.randrange(len(flags))] for _ in flags) / len(flags)
        for _ in range(iterations)
    )
    return means[int(0.025 * iterations)], means[int(0.975 * iterations)]


def paired_diff(a: list[Result], b: list[Result], *, iterations: int = 2000) -> dict:
    """Paired bootstrap over the same injected defects.

    Paired because both configurations reviewed the identical set of defects. Some
    defects are simply easier than others, and pairing removes that shared variance
    instead of letting it swamp the comparison.
    """
    by_a = {r.case_id: r.detected for r in a}
    by_b = {r.case_id: r.detected for r in b}
    shared = sorted(set(by_a) & set(by_b))
    if not shared:
        return {"difference": 0.0, "ci": (0.0, 0.0), "significant": False}

    rng = random.Random(0)
    diffs = sorted(
        (
            lambda sample: sum(by_a[c] for c in sample) / len(sample)
            - sum(by_b[c] for c in sample) / len(sample)
        )([shared[rng.randrange(len(shared))] for _ in shared])
        for _ in range(iterations)
    )
    lo, hi = diffs[int(0.025 * iterations)], diffs[int(0.975 * iterations)]
    observed = (
        sum(by_a[c] for c in shared) / len(shared) - sum(by_b[c] for c in shared) / len(shared)
    )
    # The interval excluding zero is the whole reason for reporting it.
    return {"difference": observed, "ci": (lo, hi), "significant": lo > 0.0 or hi < 0.0}


def render(results: dict[str, list[Result]], cases: list[Case], seconds: float) -> str:
    kinds: dict[str, int] = {}
    for case in cases:
        kinds[case.mutation.kind] = kinds.get(case.mutation.kind, 0) + 1

    out = [
        "# Evaluation results",
        "",
        f"{len(cases)} cases, each an independent copy of the repository with exactly one "
        f"injected defect at a known line "
        f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))}). "
        f"Detection is unambiguous and needs no labelling.",
        "",
        f"*Generated in {seconds:.0f}s.*",
        "",
        "## Why recall and not precision",
        "",
        "Whether a finding that misses the injected line is a *false positive* depends on "
        "whether the original code had a real defect there -- the one thing this benchmark "
        "cannot know. Reporting precision would mean quietly assuming the source repository "
        "is defect-free. So the headline is detection rate; `other` is the count of findings "
        "unrelated to the injected defect, useful only for comparing configurations.",
        "",
        "## Ablations",
        "",
        "| configuration | detection | 95% CI | other | question |",
        "|---|---|---|---|---|",
    ]
    by_name = {c.name: c for c in GRID}
    for name, rows in results.items():
        lo, hi = bootstrap_ci(rows)
        other = sum(r.other_findings for r in rows) / len(rows) if rows else 0.0
        detected = sum(1 for r in rows if r.detected)
        out.append(
            f"| `{name}` | {rate(rows):.0%} ({detected}/{len(rows)}) "
            f"| {lo:.0%} to {hi:.0%} | {other:.1f} "
            f"| {by_name[name].question if name in by_name else ''} |"
        )
    out.append("")

    if "full" in results and len(results) > 1:
        out += [
            "## Paired comparisons against the full pipeline",
            "",
            "Paired bootstrap over the identical set of defects. An interval including zero "
            "means the difference is not distinguishable from noise at this sample size.",
            "",
            "| vs full | difference | 95% CI | distinguishable? |",
            "|---|---|---|---|",
        ]
        for name, rows in results.items():
            if name == "full":
                continue
            stats = paired_diff(results["full"], rows)
            lo, hi = stats["ci"]
            out.append(
                f"| `{name}` | {stats['difference']:+.0%} | {lo:+.0%} to {hi:+.0%} "
                f"| {'yes' if stats['significant'] else 'no'} |"
            )
        out.append("")

    if "full" in results:
        per_kind: dict[str, list[int]] = {}
        for r in results["full"]:
            found, total = per_kind.get(r.kind, [0, 0])
            per_kind[r.kind] = [found + int(r.detected), total + 1]
        out += ["## Detection by defect type", "", "| defect | detected |", "|---|---|"]
        out += [
            f"| {kind} | {found}/{total} ({found / total:.0%}) |"
            for kind, (found, total) in sorted(per_kind.items())
        ]
        out.append("")

    out += [
        "## Reading these numbers",
        "",
        "The intervals are wide because the benchmark is small. That is a property of the "
        "experiment, not a presentation choice. Any row whose interval spans zero should be "
        "read as *no measured effect* -- including rows where the architecture predicted one.",
        "",
    ]
    return "\n".join(out)


async def main_async(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.ensure_dirs()
    registry = get_registry()

    if not any(p.configured for p in registry.providers.values()):
        raise SystemExit("No API keys set. The evaluation needs at least two families.")

    cases = build_cases(Path(args.repo), Path(args.work_dir), count=args.cases, seed=args.seed)
    print(f"{len(cases)} cases built\n")

    grid = [c for c in GRID if not args.only or c.name in args.only]
    started = time.time()
    results: dict[str, list[Result]] = {}

    for config in grid:
        print(f"Running: {config.name} ...")
        rows = [await run_case(case, config, settings, registry) for case in cases]
        results[config.name] = rows
        print(f"  detection {rate(rows):.0%}\n")

    out = Path(args.out)
    out.write_text(render(results, cases, time.time() - started), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                name: [
                    {
                        "case": r.case_id, "kind": r.kind, "detected": r.detected,
                        "score": r.score, "other": r.other_findings, "error": r.error,
                    }
                    for r in rows
                ]
                for name, rows in results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out} and {out.with_suffix('.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CodeSage mutation benchmark.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--cases", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default=".codesage/eval")
    parser.add_argument("--out", default="RESULTS.md")
    parser.add_argument("--only", nargs="*")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
