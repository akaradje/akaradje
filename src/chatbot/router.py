"""Complexity router.

Decides which path the orchestrator should take. The whole point is to avoid
spending deep-reasoning tokens on questions that don't deserve them.

Tiers:
    TRIVIAL  — greetings, lookups, single-line code, simple math
    STANDARD — explanations, refactors, summaries, common debugging
    COMPLEX  — multi-step reasoning, system design, hard debugging
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum

from .client import LLMClient
from .config import Config

log = logging.getLogger(__name__)


class Complexity(str, Enum):
    TRIVIAL = "TRIVIAL"
    STANDARD = "STANDARD"
    COMPLEX = "COMPLEX"


_ROUTER_SYSTEM = """You are a query complexity classifier.

Given a user query, output exactly one of: TRIVIAL, STANDARD, COMPLEX.

Definitions:
- TRIVIAL: greetings, simple factual lookups, one-line answers, basic arithmetic.
- STANDARD: explanations, single-file code changes, summaries, common Q&A, typical debugging.
- COMPLEX: multi-step reasoning, system design, ambiguous tasks, hard debugging,
  anything requiring planning or tool orchestration.

Output JSON only, no prose:
{"label": "TRIVIAL"|"STANDARD"|"COMPLEX", "reason": "<short reason>"}"""


# Cheap heuristics applied before paying for an API call. They only catch
# obvious cases — ambiguous ones still go to the LLM classifier.
_TRIVIAL_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey|yo|sup|สวัสดี|หวัดดี)\b", re.IGNORECASE),
    re.compile(r"^\s*(thanks|thank you|ขอบคุณ)\b", re.IGNORECASE),
    re.compile(r"^\s*(bye|goodbye|ลาก่อน)\b", re.IGNORECASE),
]

_COMPLEX_HINTS = [
    "design", "architect", "refactor entire",
    "debug", "why is", "trace through", "step by step",
    "compare and contrast", "trade-off", "tradeoff",
    "ออกแบบ", "วิเคราะห์", "เปรียบเทียบ",
]


class Router:
    def __init__(self, client: LLMClient, config: Config):
        self._client = client
        self._cfg = config

    def _heuristic(self, query: str) -> Complexity | None:
        q = query.strip()
        if not q:
            return Complexity.TRIVIAL
        for pat in _TRIVIAL_PATTERNS:
            if pat.search(q):
                return Complexity.TRIVIAL
        # Very short queries are usually trivial
        if len(q) <= 20 and "?" not in q and not any(h in q.lower() for h in _COMPLEX_HINTS):
            return Complexity.TRIVIAL
        # Strong complex hints — but defer to LLM if also short
        if len(q) > 60 and any(h in q.lower() for h in _COMPLEX_HINTS):
            return Complexity.COMPLEX
        return None

    async def classify(self, query: str) -> Complexity:
        if not self._cfg.router_enabled:
            return Complexity.STANDARD

        heuristic = self._heuristic(query)
        if heuristic is not None:
            log.info("router: heuristic → %s", heuristic.value)
            return heuristic

        try:
            text = await self._client.chat_text(
                model=self._cfg.model_fast,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=80,
            )
            label = self._parse_label(text)
            log.info("router: llm → %s", label.value)
            return label
        except Exception as exc:  # pragma: no cover — network path
            log.warning("router: classifier failed (%s) — defaulting to STANDARD", exc)
            return Complexity.STANDARD

    @staticmethod
    def _parse_label(text: str) -> Complexity:
        # Try strict JSON first
        try:
            obj = json.loads(text)
            label = str(obj.get("label", "")).upper()
            return Complexity(label)
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: scan for the label keyword
        upper = text.upper()
        for c in (Complexity.COMPLEX, Complexity.STANDARD, Complexity.TRIVIAL):
            if c.value in upper:
                return c
        return Complexity.STANDARD
