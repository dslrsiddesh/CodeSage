"""Ruff as the single static-analysis seed.

This replaced three adapters -- Ruff, Bandit, and Semgrep -- with one, for a reason
worth being able to state: **Ruff's `S` rules are a reimplementation of Bandit.** Running
both meant every `shell=True` and every weak-hash call was reported twice, and the
"two independent tools agree" corroboration signal in the scorer was double-counting one
tool's opinion. Semgrep added a large install and a network fetch for rule packs on top.

One tool, three rule families, all bug-shaped rather than style-shaped:

    F   pyflakes      undefined names, unused imports -- real errors
    B   bugbear       mutable default arguments, and friends
    S   flake8-bandit the security rules Bandit would have run
    E9  syntax        errors that stop the file working at all

Findings serve two purposes downstream: they *seed* the security lens with regions a
deterministic tool already flagged, and they *corroborate* an agent's finding that lands
on the same lines. A missing tool is recorded and reported, never silently treated as
"found nothing" -- those look identical in the output and mean opposite things.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from codesage.domain import Severity, StaticFinding

log = logging.getLogger(__name__)

RULES = "F,B,S,E9"

# Rules that fire on every line of every test file. Measured on this project's own
# repository, `assert` alone was 258 of 279 findings -- 92% of the signal was noise, and
# it was enough to push test files above real source in any ranking.
TEST_NOISE = frozenset({"S101", "S105", "S106", "S107", "S311"})

_HIGH_PREFIXES = ("F82", "E9", "S1", "S3", "S5", "S6", "S7")
_MEDIUM_PREFIXES = ("F4", "F5", "F6", "F7", "F8", "B0")


def severity_of(code: str) -> Severity:
    if code.startswith(_HIGH_PREFIXES):
        return Severity.HIGH
    if code.startswith(_MEDIUM_PREFIXES):
        return Severity.MEDIUM
    return Severity.LOW


def _is_test(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        "test" in path.split("/")[:-1]
        or "tests" in path.split("/")[:-1]
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


@dataclass
class LintReport:
    findings: list[StaticFinding] = field(default_factory=list)
    available: bool = True
    error: str | None = None

    def for_file(self, path: str) -> list[StaticFinding]:
        return sorted(
            (f for f in self.findings if f.file == path), key=lambda f: f.line_start
        )

    def describe(self) -> str:
        if not self.available:
            return "ruff: not installed (the security lens runs unseeded)"
        if self.error:
            return f"ruff: failed ({self.error})"
        return f"ruff: {len(self.findings)} findings"


def run(root: Path, targets: list[str] | None = None) -> LintReport:
    if not shutil.which("ruff"):
        return LintReport(available=False)

    # --isolated ignores any pyproject.toml in the repository under review. Inheriting
    # the target's own lint config would mean a project that disables a rule also hides
    # it from the agents, so seeding would vary silently from repo to repo.
    cmd = [
        "ruff", "check", "--output-format=json",
        f"--select={RULES}", "--no-cache", "--isolated",
        *(targets or ["."]),
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True, timeout=300, check=False
        )
        raw = json.loads(proc.stdout or "[]")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return LintReport(error=str(exc)[:120])
    except json.JSONDecodeError:
        return LintReport(error=f"unparseable output: {proc.stderr[:100]}")

    if not isinstance(raw, list):
        return LintReport(error="unexpected JSON shape")

    findings = []
    for item in raw:
        location = item.get("location") or {}
        filename = item.get("filename")
        code = item.get("code") or "?"
        if not (filename and location):
            continue
        try:
            rel = Path(filename).resolve().relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            rel = str(filename).lstrip("./")
        if _is_test(rel) and code in TEST_NOISE:
            continue
        findings.append(
            StaticFinding(
                tool="ruff",
                rule_id=code,
                file=rel,
                line_start=location.get("row", 1),
                line_end=(item.get("end_location") or location).get("row", 1),
                severity=severity_of(code),
                message=item.get("message", ""),
            )
        )

    report = LintReport(findings=findings)
    log.info("static analysis -- %s", report.describe())
    return report
