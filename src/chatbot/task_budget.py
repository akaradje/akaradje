"""Task Budget — simulated Opus 4.7-style token countdown.

Claude Opus 4.7 has a native `task_budget` parameter: you tell the model
how many tokens it has for the entire agentic loop, and the model sees a
running countdown to self-moderate its work.

DeepSeek V4 Pro has NO native equivalent. We simulate it by:
1. Tracking cumulative token usage across all iterations of the loop.
2. Injecting a "budget status" system message each iteration so the model
   is AWARE of its remaining capacity.
3. Forcing a wrap-up when usage crosses the threshold.

This gives the model the same behavioral nudge: finish gracefully within
budget, prioritize important work first, don't waste tokens on tangents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .client import TokenUsage
from .config import Config

log = logging.getLogger(__name__)


@dataclass
class BudgetStatus:
    """Current state of the token budget."""
    total_budget: int          # total tokens allowed (0 = unlimited)
    tokens_used: int = 0      # cumulative tokens consumed so far
    iterations: int = 0       # how many API calls made
    wrapup_threshold: float = 0.80

    @property
    def unlimited(self) -> bool:
        return self.total_budget <= 0

    @property
    def tokens_remaining(self) -> int:
        if self.unlimited:
            return 999_999
        return max(0, self.total_budget - self.tokens_used)

    @property
    def pct_used(self) -> float:
        if self.unlimited:
            return 0.0
        return self.tokens_used / self.total_budget

    @property
    def should_wrapup(self) -> bool:
        """True when we've crossed the threshold and should ask model to finish."""
        if self.unlimited:
            return False
        return self.pct_used >= self.wrapup_threshold

    @property
    def exhausted(self) -> bool:
        """True when budget is fully consumed."""
        if self.unlimited:
            return False
        return self.tokens_used >= self.total_budget

    def record(self, usage: TokenUsage) -> None:
        """Record tokens from one API call."""
        self.tokens_used += usage.total_tokens
        self.iterations += 1
        log.debug(
            "budget: +%d tokens (total=%d/%d, %.0f%% used, iter=%d)",
            usage.total_tokens,
            self.tokens_used,
            self.total_budget,
            self.pct_used * 100,
            self.iterations,
        )

    def to_system_message(self) -> str | None:
        """Generate a system-level budget awareness message.

        Returns None if budget is unlimited (no need to inject).
        """
        if self.unlimited:
            return None

        remaining = self.tokens_remaining
        pct = self.pct_used * 100

        if self.should_wrapup:
            return (
                f"[TASK BUDGET] {remaining:,} tokens remaining ({pct:.0f}% used). "
                f"You are approaching the budget limit. Prioritize completing your "
                f"current task and provide a final answer. Avoid unnecessary tool calls."
            )
        else:
            return (
                f"[TASK BUDGET] {remaining:,} tokens remaining ({pct:.0f}% used). "
                f"Budget is healthy. Continue working."
            )


class TaskBudgetManager:
    """Creates and manages a BudgetStatus for an agent loop."""

    def __init__(self, config: Config):
        self._cfg = config

    def create(self) -> BudgetStatus:
        """Create a fresh budget for a new agent loop."""
        return BudgetStatus(
            total_budget=self._cfg.task_budget_tokens,
            wrapup_threshold=self._cfg.task_budget_wrapup_threshold,
        )
