"""Centralized configuration loaded from environment variables.

All tuning knobs live here. Reading code should never call os.getenv directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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
    # API
    api_key: str
    base_url: str

    # Models for different paths in the pipeline
    model_fast: str
    model_standard: str
    model_deep: str
    model_verifier: str

    # Pipeline switches
    router_enabled: bool
    verifier_enabled: bool
    best_of_n: int
    max_react_iterations: int

    # Sampling
    temperature_generate: float
    temperature_verifier: float

    # Logging
    log_level: str

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model_fast=os.getenv("MODEL_FAST", "deepseek-chat"),
            model_standard=os.getenv("MODEL_STANDARD", "deepseek-chat"),
            model_deep=os.getenv("MODEL_DEEP", "deepseek-reasoner"),
            model_verifier=os.getenv("MODEL_VERIFIER", "deepseek-reasoner"),
            router_enabled=_bool("ROUTER_ENABLED", True),
            verifier_enabled=_bool("VERIFIER_ENABLED", True),
            best_of_n=_int("BEST_OF_N", 3),
            max_react_iterations=_int("MAX_REACT_ITERATIONS", 8),
            temperature_generate=_float("TEMPERATURE_GENERATE", 0.7),
            temperature_verifier=_float("TEMPERATURE_VERIFIER", 0.1),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        if self.best_of_n < 1:
            raise ValueError("BEST_OF_N must be >= 1")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )
