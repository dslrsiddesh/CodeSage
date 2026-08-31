"""Injecting known bugs, so the reviewer can be measured against ground truth.

The evaluation problem for automated code review is that there is no ground truth.
Reference datasets are noisy, review is one-to-many, and asking a model to judge another
model's review just moves the problem.

Mutation sidesteps it. Take working code, inject a defect whose location you recorded,
and detection is unambiguous at any scale with no labelling: a finding on the mutated
line is a hit, one anywhere else is not.

*Surgical splices, not `ast.unparse`.* Round-tripping through `unparse` would reformat
the file, strip comments, and shift every line number -- the reviewer would see obviously
machine-generated code and the recorded lines would be meaningless. Each operator
computes the exact source span and splices a replacement, leaving every other byte alone.

Four operators, chosen to be *plausible* rather than merely detectable. A swapped
comparison or a dropped `None` guard is the kind of defect that survives real code
review; replacing a function body with `pass` would be trivially findable and would
flatter every model equally.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass

SWAPS = {
    "comparison": {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="},
    "arithmetic": {"+": "-", "-": "+", "+=": "-=", "-=": "+="},
}


@dataclass(frozen=True)
class Mutation:
    """One injected defect, with everything needed to score a finding against it."""

    kind: str
    file: str
    line: int
    before: str
    after: str

    def describe(self) -> str:
        return f"{self.file}:{self.line} [{self.kind}] {self.before!r} -> {self.after!r}"


def _spans(source: str, tree: ast.AST) -> list[tuple[str, int, int, str, str]]:
    """Every (kind, start, end, original, replacement) a mutation could be applied at."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def index(node, attr_line="lineno", attr_col="col_offset") -> int:
        return offsets[getattr(node, attr_line) - 1] + getattr(node, attr_col)

    def end(node) -> int:
        return offsets[node.end_lineno - 1] + node.end_col_offset

    out = []
    for node in ast.walk(tree):
        # Both mutations below apply to Compare nodes, and the None-guard case has to be
        # tested first: `is not None` is a single-op comparison too, so an ordering that
        # checks the operator swap first swallows it and the guard mutation never fires.
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and node.comparators:
            is_none_guard = isinstance(node.ops[0], ast.IsNot) and isinstance(
                node.comparators[0], ast.Constant
            ) and node.comparators[0].value is None

            if is_none_guard:
                # `x is not None` -> `True`: the guard stops guarding.
                lo, hi = index(node), end(node)
                out.append(("drop_none_check", lo, hi, source[lo:hi], "True"))
            else:
                # `a < b` -> `a <= b`
                op = {
                    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
                    ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
                }.get(type(node.ops[0]))
                lo, hi = end(node.left), index(node.comparators[0])
                if op and op in source[lo:hi]:
                    at = lo + source[lo:hi].index(op)
                    out.append(("comparison", at, at + len(op), op, SWAPS["comparison"][op]))

        # `total += x` -> `total -= x`, and the plain binary forms.
        elif isinstance(node, ast.AugAssign):
            op = {ast.Add: "+=", ast.Sub: "-="}.get(type(node.op))
            lo, hi = end(node.target), index(node.value)
            if op and op in source[lo:hi]:
                at = lo + source[lo:hi].index(op)
                out.append(("arithmetic", at, at + len(op), op, SWAPS["arithmetic"][op]))

        elif isinstance(node, ast.BinOp):
            op = {ast.Add: "+", ast.Sub: "-"}.get(type(node.op))
            lo, hi = end(node.left), index(node.right)
            if op and op in source[lo:hi]:
                at = lo + source[lo:hi].index(op)
                out.append(("arithmetic", at, at + len(op), op, SWAPS["arithmetic"][op]))

        # `raise ValueError(...)` -> `pass`: the error is swallowed.
        elif isinstance(node, ast.Raise):
            lo, hi = index(node), end(node)
            if "\n" not in source[lo:hi]:
                out.append(("swallow_error", lo, hi, source[lo:hi], "pass"))

    return out


def mutate(source: str, path: str, *, rng: random.Random, limit: int = 1) -> list[tuple[str, Mutation]]:
    """Produce up to `limit` single-defect variants of `source`.

    Each variant contains exactly one injected bug. Several at once would make recall
    ambiguous -- a review that found one of three defects is not 33% correct in any
    useful sense -- and the interactions would muddy what is being measured.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    candidates = _spans(source, tree)
    rng.shuffle(candidates)

    variants = []
    for kind, start, stop, before, after in candidates:
        if len(variants) >= limit:
            break
        mutated = source[:start] + after + source[stop:]
        if mutated == source:
            continue
        try:
            # A mutation that breaks the parse is a typo, not a plausible bug.
            ast.parse(mutated)
        except SyntaxError:
            continue
        variants.append(
            (
                mutated,
                Mutation(
                    kind=kind,
                    file=path,
                    line=source.count("\n", 0, start) + 1,
                    before=before,
                    after=after,
                ),
            )
        )
    return variants
