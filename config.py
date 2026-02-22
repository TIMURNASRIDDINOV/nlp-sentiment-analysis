"""Centralised configuration for the NLP Sentiment application.

Settings are resolved in the following order (last wins):
  1. Defaults defined in this module
  2. Values from a ``.env`` file (loaded automatically)
  3. Real environment variables
  4. CLI flags (override everything)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Try to load .env — completely optional at runtime.
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent / ".env"

if _ENV_PATH.is_file():
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_PATH)
    except ImportError:
        # python-dotenv is not installed — skip silently.
        pass


def _env(name: str, default: str) -> str:
    """Read an environment variable with a fallback default."""
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Application settings dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Immutable application-wide settings."""

    # -- Model ---------------------------------------------------------------
    model: str = field(default_factory=lambda: _env("NLP_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"))
    device: int = field(default_factory=lambda: _env_int("NLP_DEVICE", -1))
    top_k: int = field(default_factory=lambda: _env_int("NLP_TOP_K", 1))

    # -- Web server ----------------------------------------------------------
    host: str = field(default_factory=lambda: _env("NLP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("NLP_PORT", 5000))
    cors_origins: str = field(default_factory=lambda: _env("NLP_CORS_ORIGINS", "*"))

    # -- Limits --------------------------------------------------------------
    max_batch_size: int = field(default_factory=lambda: _env_int("NLP_MAX_BATCH_SIZE", 64))
    max_text_length: int = field(default_factory=lambda: _env_int("NLP_MAX_TEXT_LENGTH", 5000))
    request_timeout: int = field(default_factory=lambda: _env_int("NLP_REQUEST_TIMEOUT", 30))

    # -- Misc ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("NLP_LOG_LEVEL", "INFO"))
    debug: bool = field(default_factory=lambda: _env_bool("NLP_DEBUG", False))


# Singleton — importable from anywhere as ``from config import settings``.
settings = Settings()
