"""Core domain types shared across every stage of the pipeline.

The split between `RawFinding` and `Finding` is deliberate and load-bearing: a model
is only ever allowed to fill in `RawFinding`. Provenance (which model said it, which
lens produced it, whether it survived grounding) is attached by us afterwards. If the
model could write its own provenance it could also lie about it, and the whole
consensus argument would rest on self-reported data.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def weight(self) -> float:
        """Numeric weight used when ranking findings for the report."""
        return {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25, "info": 0.1}[self.value]


class Category(StrEnum):
    CORRECTNESS = "correctness"
    SECURITY = "security"
    DESIGN = "design"
    TESTING = "testing"


class Lens(StrEnum):
    """A review perspective. Each lens is run by two different model families."""

    CORRECTNESS = "correctness"
    SECURITY = "security"
    DESIGN = "design"
    TESTING = "testing"


# --------------------------------------------------------------------------------------
# What a model is allowed to produce
# --------------------------------------------------------------------------------------


class RawFinding(BaseModel):
    """A single finding exactly as emitted by a model. Nothing here is trusted yet.

    Every field that the ground check needs (`file`, `line_start`, `line_end`, `symbol`)
    is mandatory. A model that will not commit to a location cannot have its claim
    verified, so we would have to drop it anyway -- better to reject at parse time.
    """

    model_config = {"extra": "ignore"}

    file: str = Field(description="Repo-relative path the finding refers to")
    line_start: int = Field(ge=1, description="First line of the offending range")
    line_end: int = Field(ge=1, description="Last line of the offending range")
    symbol: str | None = Field(default=None, description="Enclosing function or class name, if any")
    category: Category
    severity: Severity
    claim: str = Field(min_length=10, description="What is wrong, in one or two sentences")
    evidence: str = Field(
        min_length=1, description="Why the model believes it, referencing the code shown"
    )
    suggested_fix: str | None = Field(default=None, description="Concrete change to make")
    confidence: float = Field(ge=0.0, le=1.0, description="Model's own confidence")

    @field_validator("file")
    @classmethod
    def _normalise_path(cls, v: str) -> str:
        return v.strip().lstrip("./").replace("\\", "/")

    @model_validator(mode="after")
    def _check_range(self) -> RawFinding:
        if self.line_end < self.line_start:
            self.line_end = self.line_start
        return self


class RawFindingList(BaseModel):
    """Wrapper so models emit an object rather than a bare array.

    Many open-weight models handle `{"findings": [...]}` far more reliably than a
    top-level JSON array, and it leaves room for the model to say nothing at all.
    """

    findings: list[RawFinding] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# What we attach afterwards
# --------------------------------------------------------------------------------------


class GroundStatus(StrEnum):
    """Outcome of the deterministic ground check."""

    GROUNDED = "grounded"
    NO_SUCH_FILE = "no_such_file"
    LINE_OUT_OF_RANGE = "line_out_of_range"
    NO_SUCH_SYMBOL = "no_such_symbol"
    SYMBOL_LINE_MISMATCH = "symbol_line_mismatch"

    @property
    def ok(self) -> bool:
        return self is GroundStatus.GROUNDED


class Provenance(BaseModel):
    """Who produced a finding. Filled by the orchestrator, never by a model."""

    model_id: str
    family: str
    provider: str
    lens: Lens


class Finding(BaseModel):
    """A raw finding plus everything we learned about it."""

    raw: RawFinding
    provenance: Provenance
    ground_status: GroundStatus | None = None
    ground_detail: str | None = None

    @property
    def location_key(self) -> str:
        return f"{self.raw.file}:{self.raw.line_start}-{self.raw.line_end}"

    @property
    def is_grounded(self) -> bool:
        return self.ground_status is not None and self.ground_status.ok


# --------------------------------------------------------------------------------------
# Static analysis
# --------------------------------------------------------------------------------------


class StaticFinding(BaseModel):
    """One static-analysis finding, from Ruff.

    These serve two roles: they seed the security and correctness lenses with
    high-signal regions, and they act as independent corroboration when scoring a
    finding that lands on the same lines.
    """

    tool: str
    rule_id: str
    file: str
    line_start: int
    line_end: int
    severity: Severity
    message: str

    @field_validator("file")
    @classmethod
    def _normalise_path(cls, v: str) -> str:
        return v.strip().lstrip("./").replace("\\", "/")


# --------------------------------------------------------------------------------------
# Adversarial verification
# --------------------------------------------------------------------------------------


class Verdict(StrEnum):
    UPHELD = "upheld"
    REJECTED = "rejected"
    UNCHALLENGED = "unchallenged"  # no independent family had quota left


class CriticVerdict(BaseModel):
    """A challenge to one finding, from a family that did not propose it."""

    model_config = {"extra": "ignore"}

    verdict: Verdict
    reasoning: str = Field(default="", description="Why, citing specific lines")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def refuted(self) -> bool:
        return self.verdict is Verdict.REJECTED
