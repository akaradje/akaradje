"""Centralized configuration for DeepSeek V4 Pro chatbot.

All tuning knobs live here. Reading code should never call os.getenv directly.

Key design choice: we use a SINGLE model (deepseek-v4-pro) with variable
reasoning_effort rather than routing between multiple models. This mirrors
how Claude Opus 4.7 uses effort levels (low/medium/high/xhigh/max) on a
single model rather than switching between Haiku/Sonnet/Opus.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class ReasoningEffort(str, Enum):
    """Maps to DeepSeek V4's `reasoning_effort` parameter."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class ThinkingMode(str, Enum):
    """Maps to DeepSeek V4's `thinking.type` parameter."""
    DISABLED = "disabled"
    ENABLED = "enabled"


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # ── API connection ──────────────────────────────────────────────────
    api_key: str
    base_url: str

    # ── Model ───────────────────────────────────────────────────────────
    # Single model for all paths. DeepSeek V4 Pro handles everything via
    # reasoning_effort levels — no need for separate fast/standard/deep.
    model: str

    # ── Effort mapping (Router → API parameter) ─────────────────────────
    # Which reasoning_effort to use for each complexity tier.
    effort_trivial: ThinkingMode       # thinking disabled for trivial
    effort_standard: ReasoningEffort   # medium for standard queries
    effort_complex: ReasoningEffort    # high/max for complex queries

    # ── Pipeline switches ───────────────────────────────────────────────
    router_enabled: bool
    verifier_enabled: bool
    best_of_n: int
    max_react_iterations: int

    # ── Task Budget (simulated — DeepSeek has no native equivalent) ─────
    # Total token budget for a full agent loop. Set to 0 to disable.
    task_budget_tokens: int
    # At what % of budget consumed do we force wrap-up.
    task_budget_wrapup_threshold: float

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            effort_trivial=ThinkingMode.DISABLED,
            effort_standard=ReasoningEffort(
                os.getenv("EFFORT_STANDARD", "medium")
            ),
            effort_complex=ReasoningEffort(
                os.getenv("EFFORT_COMPLEX", "high")
            ),
            router_enabled=_bool("ROUTER_ENABLED", True),
            verifier_enabled=_bool("VERIFIER_ENABLED", True),
            best_of_n=_int("BEST_OF_N", 3),
            max_react_iterations=_int("MAX_REACT_ITERATIONS", 8),
            task_budget_tokens=_int("TASK_BUDGET_TOKENS", 128000),
            task_budget_wrapup_threshold=_float("TASK_BUDGET_WRAPUP_PCT", 0.80),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        if self.best_of_n < 1:
            raise ValueError("BEST_OF_N must be >= 1")
        if not (0.5 <= self.task_budget_wrapup_threshold <= 0.99):
            raise ValueError("TASK_BUDGET_WRAPUP_PCT must be between 0.5 and 0.99")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )
