"""Runtime settings and the model registry loader."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REGISTRY_PATH = Path(__file__).parent / "models.yaml"


def _load_local_env() -> None:
    """Load repo-root .env into os.environ, without overriding already-exported values."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()


class ProviderLimits(BaseModel):
    requests_per_minute: int
    requests_per_day: int
    tokens_per_day: int


class ProviderSpec(BaseModel):
    name: str
    base_url: str
    api_key_env: str
    limits: ProviderLimits

    @property
    def api_key(self) -> str | None:
        key = os.environ.get(self.api_key_env, "").strip()
        return key or None

    @property
    def configured(self) -> bool:
        return self.api_key is not None


class ModelSpec(BaseModel):
    id: str
    provider: str
    family: str
    context: int
    strengths: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identity for caching and quota accounting."""
        return f"{self.provider}/{self.id}"


class Registry(BaseModel):
    """The parsed contents of models.yaml, with a few convenience lookups."""

    providers: dict[str, ProviderSpec]
    models: list[ModelSpec]
    lens_preferences: dict[str, list[str]]
    critic_preferences: list[str]

    @classmethod
    def load(cls, path: Path | None = None) -> Registry:
        data = yaml.safe_load((path or REGISTRY_PATH).read_text())
        providers = {
            name: ProviderSpec(name=name, **spec) for name, spec in data["providers"].items()
        }
        return cls(
            providers=providers,
            models=[ModelSpec(**m) for m in data["models"]],
            lens_preferences=data["lens_preferences"],
            critic_preferences=data["critic_preferences"],
        )

    def provider_for(self, model: ModelSpec) -> ProviderSpec:
        return self.providers[model.provider]

    def available_models(self) -> list[ModelSpec]:
        """Models whose provider has an API key set. Does not check the network."""
        return [m for m in self.models if self.providers[m.provider].configured]

    def models_in_family(self, family: str, *, configured_only: bool = True) -> list[ModelSpec]:
        pool = self.available_models() if configured_only else self.models
        return [m for m in pool if m.family == family]

    def available_families(self) -> list[str]:
        seen: list[str] = []
        for m in self.available_models():
            if m.family not in seen:
                seen.append(m.family)
        return seen


class Settings(BaseSettings):
    """Process-wide configuration. Everything is overridable by environment variable."""

    model_config = SettingsConfigDict(
        env_prefix="CODESAGE_", env_file=".env", extra="ignore", case_sensitive=False
    )

    cache_dir: Path = Path(".codesage/cache")
    state_db: Path = Path(".codesage/state.sqlite")
    work_dir: Path = Path(".codesage/repos")
    report_dir: Path = Path("reports")
    log_level: str = "INFO"

    # Review budget. This is what makes a whole-repo review fit in a free tier.
    max_files: int = Field(default=25, description="Top-K files by risk score to review")
    ensemble_size: int = Field(default=2, description="Distinct model families per lens")
    max_file_bytes: int = Field(
        default=120_000, description="Skip files larger than this; they are usually generated"
    )
    max_agent_steps: int = Field(
        default=4,
        description=(
            "Tool-call round trips one agent may take before it must answer. This is the "
            "main lever on cost: each step is a full request carrying the whole "
            "conversation so far, so exploration is quadratic in tokens, not linear."
        ),
    )

    # LLM call behaviour
    request_timeout: float = 180.0
    max_retries: int = 3
    temperature: float = 0.1
    max_output_tokens: int = 4096

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.state_db.parent, self.work_dir, self.report_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    return Registry.load()
