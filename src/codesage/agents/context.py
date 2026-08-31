"""The starting context a review agent gets, before it goes looking for more.

The design rule is *seed, don't dump*. An agent starts with the file under review, its
structural outline, and any static findings already on it -- then uses tools to pull
whatever else the specific bug requires. Pre-loading callers, callees, and neighbouring
files would spend tokens on every call for context most reviews never touch.

Line numbers are rendered into the source because every finding has to carry a location
the ground check can verify, and a model cannot cite a line it was never shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from codesage.agents.tools import number_lines
from codesage.domain import StaticFinding
from codesage.index.code import CodeIndex
from codesage.ingest.inventory import SourceFile

# Files longer than this are truncated. The agent is told, so it does not report on code
# it cannot see -- and it can always call read_symbol for anything past the cut.
MAX_SOURCE_LINES = 700


@dataclass
class ContextPack:
    path: str
    numbered_source: str
    outline: str
    n_lines: int
    static_findings: list[StaticFinding] = field(default_factory=list)
    plan_reason: str = ""
    truncated: bool = False

    def render(self, *, include_static: bool = True) -> str:
        parts = [f"# File under review: {self.path}"]
        if self.plan_reason:
            parts.append(f"Selected because: {self.plan_reason}")

        parts.append(
            "\n## Source — line numbers are authoritative, cite them exactly\n"
            f"```python\n{self.numbered_source}\n```"
        )
        if self.truncated:
            parts.append(
                f"*Truncated at {MAX_SOURCE_LINES} lines. Do not report on code you "
                f"cannot see; use `read_symbol` to fetch anything below the cut.*"
            )

        parts.append(f"\n## Structure\n```\n{self.outline}\n```")

        if include_static and self.static_findings:
            parts.append(
                "\n## Already reported by static analysis\n"
                "Confirmed by a deterministic tool. Do not repeat these; use them as a "
                "hint about where to look, and report only what they miss or understate.\n"
                + "\n".join(
                    f"- L{f.line_start}: [{f.rule_id}] {f.message}"
                    for f in self.static_findings
                )
            )
        return "\n".join(parts)


def build(
    source: SourceFile,
    index: CodeIndex,
    static_findings: list[StaticFinding],
    *,
    plan_reason: str = "",
) -> ContextPack:
    text = source.read()
    lines = text.splitlines()
    truncated = len(lines) > MAX_SOURCE_LINES
    if truncated:
        text = "\n".join(lines[:MAX_SOURCE_LINES])

    return ContextPack(
        path=source.path,
        numbered_source=number_lines(text),
        outline=index.outline(source.path),
        n_lines=len(lines),
        static_findings=sorted(static_findings, key=lambda f: f.line_start),
        plan_reason=plan_reason,
        truncated=truncated,
    )
