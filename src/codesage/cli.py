"""CodeSage command line."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from codesage.config.settings import get_registry, get_settings

app = typer.Typer(
    name="codesage",
    help="Multi-agent code review with mechanical grounding and cross-family consensus.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _logging(verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else getattr(logging, get_settings().log_level, logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@app.command()
def doctor(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Check which providers and models are reachable with your keys."""
    _logging(verbose)
    settings = get_settings()
    settings.ensure_dirs()
    registry = get_registry()

    print("Provider status")
    for name, p in registry.providers.items():
        configured = "yes" if p.configured else "no"
        print(
            f"  - {name:<10} {p.api_key_env:<20} {configured:<5} "
            f"{p.limits.requests_per_day} req / {p.limits.tokens_per_day:,} tok"
        )

    if not any(p.configured for p in registry.providers.values()):
        print("\nNo API keys set. Copy .env.example to .env and set at least two provider keys.")
        raise typer.Exit(code=1)

    print("\nConfigured models:")
    configured_models = [m for m in registry.models if registry.provider_for(m).configured]
    if configured_models:
        for model in configured_models:
            print(f"  - {model.id:<35} family={model.family:<10} provider={model.provider:<10}")
    else:
        print("  - none")

    live = sorted({m.family for m in configured_models})
    print(f"\nAvailable families: {', '.join(live) or 'none'}")
    if len(live) < 2:
        print("Warning: fewer than two families are configured; cross-family agreement will be weak.")
        raise typer.Exit(code=1)

    print(f"\nReady. {len(live)} families configured.")


@app.command()
def index(
    target: str = typer.Argument(..., help="GitHub URL, owner/repo, or a local path"),
    outline: str = typer.Option(None, "--outline", help="Show the structure of one file"),
    refresh: bool = typer.Option(False, "--refresh", help="Re-clone instead of reusing the cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the deterministic stage only: acquire, parse, index, lint. No LLM calls."""
    _logging(verbose)
    from codesage.index.pipeline import run

    with console.status(f"Indexing {target}..."):
        result = run(target, get_settings(), refresh=refresh)

    s = result.summary()
    console.print(f"\n[bold]{s['repo']}[/bold] @ {s['commit']}")
    console.print(
        f"  {s['files']} files, {s['lines']:,} lines, {s['symbols']} symbols, "
        f"{s['lint_findings']} lint findings"
        + ("" if s["lint_available"] else "  [yellow](ruff not installed)[/yellow]")
    )

    if outline:
        console.print(f"\n{result.code.outline(outline)}")
        return

    console.print("\n[bold]Repository outline[/bold] [dim](what the planner agent sees)[/dim]")
    console.print(result.code.repo_outline())

    if result.lint.findings:
        table = Table("rule", "sev", "location", "message", title="\nLint seeds", header_style="bold")
        for f in result.lint.findings[:12]:
            table.add_row(f.rule_id, str(f.severity), f"{f.file}:{f.line_start}", f.message[:56])
        console.print(table)

    console.print(
        "\n[dim]Which files get reviewed is decided by the planner agent at review "
        "time, not here.[/dim]"
    )


@app.command()
def review(
    target: str = typer.Argument(..., help="GitHub URL, owner/repo, PR link, or a local path"),
    max_files: int = typer.Option(None, "--max-files", "-n", help="Override the review budget"),
    out: str = typer.Option(None, "--out", "-o", help="Directory for the report"),
    no_tools: bool = typer.Option(False, "--no-tools", help="Disable agent tool calling (ablation)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached model responses"),
    refresh: bool = typer.Option(False, "--refresh", help="Re-clone instead of reusing the cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Review a repository end to end and write a report."""
    _logging(verbose)
    from codesage.orchestration.runner import review as run_review
    from codesage.orchestration.runner import write_outputs

    settings, registry = get_settings(), get_registry()
    if not any(p.configured for p in registry.providers.values()):
        console.print(
            "[red]No API keys set.[/red] Run [cyan]codesage doctor[/cyan] for setup, or "
            "[cyan]codesage index[/cyan] to run the deterministic stage without keys."
        )
        raise typer.Exit(code=1)

    done = {"n": 0}

    def on_event(kind: str, payload: dict) -> None:
        if kind == "plan":
            status.update(f"Planner chose {len(payload['files'])} file(s)...")
        elif kind == "review_done":
            done["n"] += 1
            status.update(f"Reviewing... {done['n']} calls ({payload['lens']} on {payload['path']})")
        elif kind == "grounded":
            status.update(f"Grounding... kept {payload['kept']}, discarded {payload['rejected']}")

    with console.status(f"Indexing {target}...") as status:
        report = asyncio.run(
            run_review(
                target, settings, registry, max_files=max_files, refresh=refresh,
                use_tools=not no_tools, no_cache=no_cache, on_event=on_event,
            )
        )

    md_path, json_path = write_outputs(report, Path(out) if out else settings.report_dir)
    m = report.manifest

    console.print(f"\n[bold]{m.target}[/bold] @ {m.commit[:10]}  ({m.duration_s}s)\n")
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column(style="cyan")
    summary.add_column()
    for label, value in [
        ("findings", report.headline),
        ("families", ", ".join(sorted(m.families_used)) or "none"),
        ("discarded as ungrounded", f"{m.findings_rejected}/{m.findings_raw} ({m.hallucination_rate:.0%})"),
        ("cross-family support", f"{m.clusters_multi_family}/{m.clusters}"),
        ("file selection", m.plan_source),
        ("agent tool calls", str(m.tool_calls)),
        ("cache hit rate", f"{m.cache_hit_rate:.0%}"),
    ]:
        summary.add_row(label, str(value))
    console.print(summary)

    if m.degradations:
        console.print("\n[yellow]This run was degraded:[/yellow]")
        for d in m.degradations:
            console.print(f"  - {d}")

    if report.shown:
        table = Table("conf", "sev", "location", "finding", title="\nTop findings", header_style="bold")
        for f in report.shown[:10]:
            table.add_row(
                f"{f.score:.2f}", str(f.severity), f.cluster.location,
                f.cluster.representative.raw.claim[:60],
            )
        console.print(table)

    console.print(f"\n[green]Report:[/green] {md_path}\n[green]Data:  [/green] {json_path}\n")


@app.command()
def config() -> None:
    """Show effective settings."""
    settings, registry = get_settings(), get_registry()
    table = Table(show_header=False)
    table.add_column(style="cyan")
    table.add_column()
    for label, value in [
        ("max_files (review budget)", settings.max_files),
        ("ensemble_size (families/lens)", settings.ensemble_size),
        ("max_agent_steps (tool hops)", settings.max_agent_steps),
        ("cache_dir", settings.cache_dir),
        ("registry families", ", ".join(sorted({m.family for m in registry.models}))),
    ]:
        table.add_row(str(label), str(value))
    console.print(table)


if __name__ == "__main__":
    app()
