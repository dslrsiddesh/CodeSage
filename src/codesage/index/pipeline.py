"""The deterministic stage: acquire, index, lint. No model calls.

Everything here is cheap, fast, and fully testable. It exists to give the agentic stage
two things: a repo map exact enough for the ground check to trust, and a set of static
findings to seed the security lens with. Choosing *what to review* is no longer part of
this stage -- that moved to the planner agent.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from codesage.config.settings import Settings
from codesage.index import code as code_parser
from codesage.index.code import CodeIndex
from codesage.index.lint import LintReport
from codesage.index.lint import run as run_lint
from codesage.ingest.inventory import Inventory, discover
from codesage.ingest.repo import Checkout, RepoRef, acquire

log = logging.getLogger(__name__)


@dataclass
class IndexResult:
    checkout: Checkout
    inventory: Inventory
    code: CodeIndex
    lint: LintReport
    duration_s: float = 0.0
    unparseable: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "repo": self.checkout.ref.describe(),
            "commit": self.checkout.short_commit,
            "files": len(self.inventory.files),
            "lines": self.inventory.total_lines,
            "symbols": self.code.n_symbols,
            "unparseable": len(self.unparseable),
            "lint_findings": len(self.lint.findings),
            "lint_available": self.lint.available,
            "duration_s": round(self.duration_s, 1),
        }


def run(target: str, settings: Settings, *, refresh: bool = False) -> IndexResult:
    start = time.perf_counter()
    settings.ensure_dirs()

    ref = RepoRef.parse(target)
    log.info("acquiring %s", ref.describe())
    checkout = acquire(ref, settings.work_dir, refresh=refresh)

    include = _changed_files(checkout) if ref.is_pr else None
    inventory = discover(
        checkout.root, max_file_bytes=settings.max_file_bytes, include=include
    )
    if not inventory.files:
        raise RuntimeError(
            f"no reviewable Python files found in {ref.describe()}. "
            f"CodeSage reviews Python only."
        )

    parsed = []
    unparseable = []
    for source in inventory.files:
        file = code_parser.parse(source.read(), source.path)
        parsed.append(file)
        if not file.ok:
            unparseable.append(source.path)

    index = CodeIndex.build(parsed)
    lint = run_lint(checkout.root, targets=[f.path for f in inventory.files])

    result = IndexResult(
        checkout=checkout,
        inventory=inventory,
        code=index,
        lint=lint,
        duration_s=time.perf_counter() - start,
        unparseable=unparseable,
    )
    log.info("index complete: %s", result.summary())
    return result


def _changed_files(checkout: Checkout) -> list[str]:
    """Files touched by a PR, for diff-scoped review."""
    for candidate in ("origin/main", "origin/master", "origin/HEAD"):
        proc = subprocess.run(
            ["git", "diff", "--name-only", f"{candidate}...HEAD"],
            cwd=checkout.root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            log.info("PR mode: %d changed files", len(files))
            return files
    log.warning("could not determine changed files; reviewing the whole repository")
    return []


def graph_cache_path(settings: Settings, checkout: Checkout) -> Path:
    return settings.cache_dir / "index" / f"{checkout.commit}.pkl"
