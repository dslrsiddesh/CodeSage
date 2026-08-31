"""The tools a review agent can call to pull context on demand.

This is the project's bet about context engineering. The obvious alternative
pre-computes everything an agent might want -- source, callers, callees, metrics --
concatenates it and sends one enormous prompt. That is simpler, and it fails specifically:
most of the context is irrelevant to the bug in front of the model, it costs tokens on
every call, and it caps out on any file worth reviewing.

Instead an agent starts with the file and its outline, then *asks*. Reviewing
`total_cents` it calls `find_callers`, finds a caller passing an empty list, and reads
that one function -- rather than being handed forty.

The trade is real in both directions: each hop resends the whole conversation, so
exploration is quadratic in tokens. `max_agent_steps` bounds it, results are truncated,
and the evaluation measures whether it actually pays.

Every tool is read-only and repo-scoped -- nothing writes, executes, or reaches the
network. That is a security property, not an accident: these run whatever an untrusted
model asks for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from codesage.index.code import CodeIndex

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 5_000


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    name: str
    content: str
    ok: bool = True


def number_lines(text: str, start: int = 1) -> str:
    """Prefix each line with its number.

    A model cannot cite a line it was never shown, and every finding needs a location
    precise enough for the ground check. Handing over a bare snippet and trusting the
    model to count is the easiest way to manufacture ungrounded findings.
    """
    lines = text.splitlines()
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{start + i:>{width}}| {line}" for i, line in enumerate(lines))


def _schema(name: str, description: str, arg: str, arg_description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {arg: {"type": "string", "description": arg_description}},
                "required": [arg],
            },
        },
    }


# Descriptions are prompt text -- the only guidance the model gets about when a tool is
# worth reaching for, so they say *why*, not just what comes back.
TOOL_SCHEMAS = [
    _schema(
        "read_symbol",
        "Read the full source of a function or class anywhere in the repository. Use "
        "this when a finding depends on what another function actually does, rather "
        "than on what its name suggests.",
        "name",
        "Function or class name, e.g. 'total_cents' or 'Cart.total_cents'",
    ),
    _schema(
        "find_callers",
        "List the functions that call a given function, with file and line. Use this to "
        "check whether callers can actually pass the input your finding depends on -- a "
        "bug no caller can trigger is not a bug.",
        "name",
        "Function name",
    ),
    _schema(
        "grep",
        "Search the repository with a regular expression. Use this to find where a value "
        "is validated before it reaches the code you are reviewing.",
        "pattern",
        "Regular expression",
    ),
]

TOOL_NAMES = frozenset(s["function"]["name"] for s in TOOL_SCHEMAS)


class RepoTools:
    """Executes tool calls against the code index. Read-only and never raises."""

    def __init__(self, index: CodeIndex) -> None:
        self.index = index
        self.call_log: list[str] = []

    def execute(self, call: ToolCall) -> ToolResult:
        """Dispatch one tool call.

        Never raises. A model that invents a tool name or passes a malformed regex gets
        an error string back as a normal result and can recover next turn; an exception
        would abort the review over one bad guess.
        """
        self.call_log.append(call.name)
        handler = {
            "read_symbol": self._read_symbol,
            "find_callers": self._find_callers,
            "grep": self._grep,
        }.get(call.name)

        if handler is None:
            return ToolResult(
                call.name,
                f"No tool named {call.name!r}. Available: {', '.join(sorted(TOOL_NAMES))}.",
                ok=False,
            )
        try:
            result = handler(call.arguments)
        except Exception as exc:
            log.info("tool %s failed: %s", call.name, exc)
            return ToolResult(call.name, f"Tool failed: {type(exc).__name__}: {exc}", ok=False)

        if len(result.content) > MAX_RESULT_CHARS:
            result.content = result.content[:MAX_RESULT_CHARS] + "\n... [truncated]"
        return result

    def _read_symbol(self, args: dict[str, Any]) -> ToolResult:
        name = str(args.get("name", "")).strip()
        symbol = self.index.find_symbol(name)
        if symbol is None:
            return ToolResult("read_symbol", f"No symbol named {name!r}.", ok=False)
        return ToolResult(
            "read_symbol",
            f"{symbol.file}  lines {symbol.line_start}-{symbol.line_end}\n"
            + number_lines(symbol.source, symbol.line_start),
        )

    def _find_callers(self, args: dict[str, Any]) -> ToolResult:
        name = str(args.get("name", "")).strip()
        callers = self.index.callers_of(name)
        if not callers:
            return ToolResult(
                "find_callers",
                f"Nothing calls {name!r}. It may be dead code, an entry point, or "
                f"called dynamically.",
            )
        rows = [f"{c.file}:{c.line_start}  {c.qualname}" for c in callers[:20]]
        return ToolResult("find_callers", f"{len(callers)} caller(s):\n" + "\n".join(rows))

    def _grep(self, args: dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return ToolResult("grep", "No pattern given.", ok=False)
        try:
            hits = self.index.grep(pattern)
        except ValueError as exc:
            return ToolResult("grep", str(exc), ok=False)
        if not hits:
            return ToolResult("grep", f"No matches for {pattern!r}.")
        rows = [f"{path}:{line}  {text.strip()}" for path, line, text in hits]
        return ToolResult("grep", f"{len(hits)} match(es):\n" + "\n".join(rows))
