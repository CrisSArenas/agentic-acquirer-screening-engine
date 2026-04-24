"""
Centralized configuration using Pydantic Settings.

Loads environment variables from .env file. All secrets (API keys) come from
environment, never from code. This module is the single source of truth for
runtime configuration — no magic strings scattered across the codebase.
"""
from __future__ import annotations

from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Secrets ---
    anthropic_api_key: SecretStr = Field(
        ..., description="Anthropic API key. Load from .env, never commit."
    )

    # --- Model configuration ---
    model: str = Field(default="claude-sonnet-4-6", description="Claude model ID")
    max_tokens: int = Field(
        default=2000, ge=256, le=8192,
        description="Hard cap on output tokens per call. 2000 leaves comfortable "
                    "headroom for a ~1000-token rationale JSON + any accidental "
                    "prose. Prevents truncation that breaks JSON parsing.",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Concurrency ---
    max_concurrent_requests: int = Field(
        default=2, ge=1, le=20,
        description="Cap on parallel LLM calls. Sized for Anthropic tier-1: "
                    "8,000 output tokens per minute. 2 concurrent x 2000 max = "
                    "4000 tokens per wave. At ~15s/wave, 5 waves/min = 20k/min "
                    "theoretical max, but actual output is ~1000 tokens/call, "
                    "so real throughput stays under the 8k/min cap."
    )
    retry_max_attempts: int = Field(default=3, ge=1, le=10)

    # --- Observability ---
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False, description="Emit JSON logs for prod")

    # --- Data ---
    csv_path: Path = Field(default=Path("data/ma_transactions_500.csv"))

    @property
    def api_key(self) -> str:
        """Return the raw API key string. Kept as SecretStr in the class to
        prevent accidental logging."""
        return self.anthropic_api_key.get_secret_value()


# Singleton — instantiated once at import time.
settings = Settings()  # type: ignore[call-arg]
