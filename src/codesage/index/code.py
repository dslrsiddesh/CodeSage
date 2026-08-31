"""The code index: what exists in the repository, where, and who calls whom.

One module, built with the standard library `ast`. It replaced a tree-sitter parser and
a NetworkX graph, and the reason is worth stating plainly: this project reviews Python,
and everything downstream needs only four facts.

  * `symbol_at(line)`  the ground check asks "what owns this line?"
  * `find(name)`       resolve a name to a definition
  * `source`           the `read_symbol` tool hands an agent a function on demand
  * `callers_of`       the `find_callers` tool, and the untested markers in an outline

**The accuracy split that matters.** Two consumers, opposite requirements:

  * The **ground check** must be exact. It only asks per-file questions -- does this file
    exist, this line, this symbol -- answered from a real parse with no guessing.
  * **Tools and context** only need to be useful. `callers_of` matches on the bare name a
    call was written with, so `save()` matches every `save` in the repo. A spurious
    caller costs an agent one wasted read; a spurious *ground* fact confirms a
    hallucinated finding as real.

Names are never resolved to definitions during parsing. That needs type inference, and a
wrong edge is worse than a missing one.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class Symbol:
    name: str
    qualname: str  # "Cart.total" for a method, "helper" for a module-level function
    kind: str  # "function" | "method" | "class"
    file: str
    line_start: int
    line_end: int
    source: str
    calls: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.file}::{self.qualname}"

    def contains(self, line: int) -> bool:
        return self.line_start <= line <= self.line_end

    def signature(self) -> str:
        return self.source.splitlines()[0].strip() if self.source else self.qualname


@dataclass
class ParsedFile:
    path: str
    n_lines: int
    lines: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def symbol_at(self, line: int) -> Symbol | None:
        """Innermost symbol containing `line` -- a method, not its class."""
        holders = [s for s in self.symbols if s.contains(line)]
        return min(holders, key=lambda s: s.line_end - s.line_start) if holders else None

    def find(self, name: str) -> Symbol | None:
        """Qualified name first, then an unambiguous bare name."""
        for symbol in self.symbols:
            if symbol.qualname == name:
                return symbol
        bare = name.rsplit(".", 1)[-1]
        matches = [s for s in self.symbols if s.name == bare]
        return matches[0] if len(matches) == 1 else None


def _call_name(node: ast.AST) -> str | None:
    """Flatten a call target into a dotted string, as written."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _calls_in(node: ast.AST) -> list[str]:
    """Calls made directly inside `node`, not descending into nested definitions."""
    out: list[str] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if isinstance(child, ast.Call) and (name := _call_name(child.func)):
            out.append(name)
        out.extend(_calls_in(child))
    return out


def parse(source: str, path: str) -> ParsedFile:
    lines = source.splitlines()
    result = ParsedFile(path=path, n_lines=len(lines), lines=lines)

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        # One unparseable file is skipped and recorded, not fatal to the run.
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    def visit(node: ast.AST, scope: list[tuple[str, bool]]) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                visit(child, scope)
                continue

            is_class = isinstance(child, ast.ClassDef)
            # A def is a method only when its immediate enclosing scope is a class; a
            # closure inside a function is still a function.
            kind = "class" if is_class else ("method" if scope and scope[-1][1] else "function")
            # Decorators sit above the `def` line, so include them in the range.
            start = min([child.lineno, *(d.lineno for d in child.decorator_list)])
            end = child.end_lineno or child.lineno

            result.symbols.append(
                Symbol(
                    name=child.name,
                    qualname=".".join([n for n, _ in scope] + [child.name]),
                    kind=kind,
                    file=path,
                    line_start=start,
                    line_end=end,
                    source="\n".join(lines[start - 1 : end]),
                    calls=[] if is_class else _calls_in(child),
                )
            )
            visit(child, [*scope, (child.name, is_class)])

    visit(tree, [])
    return result


@dataclass
class CodeIndex:
    """Every parsed file, plus a name-keyed caller index."""

    files: dict[str, ParsedFile] = field(default_factory=dict)
    _by_name: dict[str, list[Symbol]] = field(default_factory=dict)
    _callers: dict[str, list[Symbol]] = field(default_factory=dict)

    @classmethod
    def build(cls, parsed: list[ParsedFile]) -> CodeIndex:
        index = cls(files={f.path: f for f in parsed})
        for file in parsed:
            for symbol in file.symbols:
                index._by_name.setdefault(symbol.name, []).append(symbol)
                # Key on the last dotted component so `self.total()` and `cart.total()`
                # both find `total`.
                for call in symbol.calls:
                    index._callers.setdefault(call.rsplit(".", 1)[-1], []).append(symbol)
        log.info(
            "index: %d files, %d symbols",
            len(index.files),
            sum(len(f.symbols) for f in parsed),
        )
        return index

    # -- exact: safe for the ground check ----------------------------------------------

    def file(self, path: str) -> ParsedFile | None:
        return self.files.get(path)

    # -- approximate: tools and context only -------------------------------------------

    def find_symbol(self, name: str) -> Symbol | None:
        for file in self.files.values():
            if (hit := file.find(name)) is not None and hit.qualname == name:
                return hit
        candidates = self._by_name.get(name.rsplit(".", 1)[-1], [])
        return candidates[0] if candidates else None

    def callers_of(self, name: str) -> list[Symbol]:
        return list(self._callers.get(name.rsplit(".", 1)[-1], []))

    def is_tested(self, symbol: Symbol) -> bool:
        return any("test" in c.file for c in self.callers_of(symbol.name))

    def grep(self, pattern: str, *, limit: int = 30) -> list[tuple[str, int, str]]:
        import re

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc

        hits = []
        for path, file in sorted(self.files.items()):
            for n, line in enumerate(file.lines, start=1):
                if regex.search(line):
                    hits.append((path, n, line.rstrip()[:160]))
                    if len(hits) >= limit:
                        return hits
        return hits

    # -- summaries ---------------------------------------------------------------------

    def outline(self, path: str) -> str:
        """Signatures and line ranges for one file. Bodies come from `read_symbol`."""
        file = self.files.get(path)
        if file is None:
            return f"{path}: not indexed"
        if not file.symbols:
            return f"{path}: no functions or classes ({file.n_lines} lines)"

        rows = []
        for symbol in file.symbols:
            notes = []
            if symbol.kind != "class":
                if n := len(self.callers_of(symbol.name)):
                    notes.append(f"{n} caller{'s' if n != 1 else ''}")
                if not self.is_tested(symbol):
                    notes.append("untested")
            suffix = f"  [{', '.join(notes)}]" if notes else ""
            rows.append(f"L{symbol.line_start}-{symbol.line_end}  {symbol.signature()}{suffix}")
        return f"{path} ({file.n_lines} lines)\n" + "\n".join(rows)

    def repo_outline(self, limit: int = 60) -> str:
        """One line per file -- what the planner agent sees."""
        rows = [
            f"{path}  ({f.n_lines} lines, {len(f.symbols)} symbols)"
            for path, f in sorted(self.files.items())[:limit]
        ]
        if len(self.files) > limit:
            rows.append(f"... and {len(self.files) - limit} more")
        return "\n".join(rows)

    @property
    def n_symbols(self) -> int:
        return sum(len(f.symbols) for f in self.files.values())
