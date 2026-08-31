"""Turning whatever a model actually said into a validated Pydantic object.

Free-tier open-weight models are noticeably worse than frontier models at honouring a
schema, and they fail in a small number of predictable ways. Each is handled
deterministically before spending a second call on a repair:

  1. clean JSON                                    -> parse directly
  2. wrapped in a ```json fence                    -> strip the fence
  3. preceded by "Sure, here is the analysis"      -> brace-match the first object
  4. a bare `[...]` array instead of `{...}`       -> wrap it in the expected field

Anything still unparseable gets one reprompt from the agent loop, then is dropped. A
model that has failed twice will not manage it on the third try, and those attempts come
out of the same budget as a real review.

Nothing here regexes fields out of prose. A half-parsed finding is worse than no
finding, because it reaches the report with the same confidence as a real one.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if there is one."""
    s = text.strip()
    for fence in ("```json", "```JSON", "```"):
        if s.startswith(fence):
            s = s[len(fence) :].lstrip("\n")
            if (end := s.rfind("```")) != -1:
                s = s[:end]
            return s.strip()
    return s


def find_json(text: str) -> str | None:
    """The first balanced JSON object or array in `text`.

    Brace counting has to respect string literals and escapes, otherwise a `}` inside a
    quoted code snippet truncates the object -- a very common failure precisely because
    we ask models to quote code back at us in the `evidence` field.
    """
    start = next((i for i, c in enumerate(text) if c in "{["), None)
    if start is None:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = escaped = False

    for i in range(start, len(text)):
        char = text[i]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse(text: str, schema: type[T]) -> tuple[T | None, str | None]:
    """Deterministic parse. Returns `(parsed, error)` -- never raises."""
    if not text or not text.strip():
        return None, "empty response"

    stripped = strip_fences(text)
    for candidate in (stripped, find_json(stripped), find_json(text)):
        if candidate is None:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        # Models routinely return `[{...}]` when asked for `{"findings": [...]}`. That
        # is a formatting slip, not a content failure, so repair rather than discard
        # findings that may be perfectly good.
        if isinstance(value, list):
            list_fields = [
                name
                for name, f in schema.model_fields.items()
                if getattr(f.annotation, "__origin__", None) is list
            ]
            if len(list_fields) == 1:
                value = {list_fields[0]: value}

        try:
            return schema.model_validate(value), None
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3]
            )
            return None, f"schema validation failed -- {detail}"

    return None, "no JSON object or array found in the response"
