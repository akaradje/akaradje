"""Observability — per-query metrics, cost tracking, and structured logging.

Tracks:
- Token usage (prompt, completion, reasoning, total)
- Cost in USD (using DeepSeek V4 Pro pricing)
- Latency (wall-clock time per operation)
- Cache hit rate
- Effort level used
- Correction/escalation events
- Query complexity classification

Exports as structured JSON for ingestion by any logging/monitoring tool.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)


# DeepSeek V4 Pro pricing (as of May 2026, per 1M tokens)
# With 75% discount active until May 31, 2026
PRICING = {
    "deepseek-v4-pro": {
        "input_cache_hit": 0.028,    # 1/10 of miss
        "input_cache_miss": 1.74,     # full price (pre-discount: 1.74)
        "output": 3.48,
        "reasoning": 3.48,            # reasoning tokens billed as output
    },
    "deepseek-v4-flash": {
        "input_cache_hit": 0.014,
        "input_cache_miss": 0.14,
        "output": 0.28,
        "reasoning": 0.28,
    },
}


@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""
    query_id: str = ""
    timestamp: str = ""
    query_preview: str = ""

    # Classification
    complexity: str = ""
    thinking_mode: str = ""
    reasoning_effort: str = ""
    path: str = ""

    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0

    # Cost
    cost_usd: float = 0.0

    # Performance
    latency_ms: int = 0
    iterations: int = 0
    tool_calls: int = 0

    # Verification
    verifier_score: int | None = None
    verifier_passed: bool | None = None
    correction_attempts: int = 0
    effort_escalated: bool = False

    # Voting
    candidates_sampled: int = 0


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int = 0,
    model: str = "deepseek-v4-pro",
) -> float:
    """Calculate cost in USD for a single API call."""
    prices = PRICING.get(model, PRICING["deepseek-v4-pro"])

    # Input cost (split by cache hit/miss)
    if cache_hit_tokens or cache_miss_tokens:
        input_cost = (
            cache_hit_tokens * prices["input_cache_hit"] / 1_000_000
            + cache_miss_tokens * prices["input_cache_miss"] / 1_000_000
        )
    else:
        # No cache info — assume all miss
        input_cost = prompt_tokens * prices["input_cache_miss"] / 1_000_000

    # Output cost
    output_cost = completion_tokens * prices["output"] / 1_000_000

    # Reasoning cost (billed as output)
    reasoning_cost = reasoning_tokens * prices["reasoning"] / 1_000_000

    return input_cost + output_cost + reasoning_cost


class MetricsCollector:
    """Collects and persists query metrics."""

    def __init__(self, log_path: str | Path | None = None):
        self._log_path = Path(log_path) if log_path else None
        self._metrics: list[QueryMetrics] = []
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._query_count: int = 0

        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def track_query(self, query: str, query_id: str = "") -> AsyncIterator[QueryMetrics]:
        """Context manager that times a query and records metrics."""
        metrics = QueryMetrics(
            query_id=query_id or f"q-{self._query_count + 1}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_preview=query[:100],
        )
        start = time.perf_counter()

        yield metrics

        # After the query completes
        metrics.latency_ms = int((time.perf_counter() - start) * 1000)
        metrics.cost_usd = calculate_cost(
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            reasoning_tokens=metrics.reasoning_tokens,
            cache_hit_tokens=metrics.cache_hit_tokens,
            cache_miss_tokens=metrics.cache_miss_tokens,
        )

        self._metrics.append(metrics)
        self._total_cost += metrics.cost_usd
        self._total_tokens += metrics.total_tokens
        self._query_count += 1

        log.info(
            "metrics: id=%s cost=$%.4f tokens=%d latency=%dms complexity=%s effort=%s",
            metrics.query_id, metrics.cost_usd, metrics.total_tokens,
            metrics.latency_ms, metrics.complexity, metrics.reasoning_effort,
        )

        if self._log_path:
            self._persist(metrics)

    def _persist(self, metrics: QueryMetrics) -> None:
        """Append metrics as JSON line."""
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(metrics)) + "\n")
        except Exception as exc:
            log.warning("metrics: persist failed: %s", exc)

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def avg_cost_per_query(self) -> float:
        if self._query_count == 0:
            return 0.0
        return self._total_cost / self._query_count

    def summary(self) -> dict[str, Any]:
        """Return a summary of all collected metrics."""
        return {
            "total_queries": self._query_count,
            "total_cost_usd": round(self._total_cost, 4),
            "total_tokens": self._total_tokens,
            "avg_cost_per_query": round(self.avg_cost_per_query, 4),
            "avg_latency_ms": (
                sum(m.latency_ms for m in self._metrics) // max(1, len(self._metrics))
            ),
            "complexity_distribution": self._count_by("complexity"),
            "effort_distribution": self._count_by("reasoning_effort"),
        }

    def _count_by(self, field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self._metrics:
            val = getattr(m, field, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts
