"""Best-of-N voting with diversity through prompt variation.

Since DeepSeek V4 IGNORES temperature in thinking mode, we can't just
sample with different temps. Instead, we achieve diversity by giving each
parallel sample a slightly different "approach angle" in the system prompt.

This is MORE principled than temperature diversity anyway — it forces the
model to explore genuinely different reasoning paths rather than just
random token sampling.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .config import Config, ReasoningEffort, ThinkingMode
from .executor import Executor, ExecutorResult
from .task_budget import BudgetStatus, TaskBudgetManager
from .verifier import VerificationResult, Verifier

log = logging.getLogger(__name__)


# Different "approach angle" suffixes to inject diversity.
# Each candidate gets a different thinking style directive.
_DIVERSITY_SUFFIXES = [
    "",  # default — no modification
    "\nApproach: Start by identifying potential edge cases and failure modes.",
    "\nApproach: Begin with the simplest possible solution, then refine.",
    "\nApproach: Think about this from first principles. Question assumptions.",
    "\nApproach: Consider multiple strategies before committing to one.",
    "\nApproach: Focus on correctness first, clarity second.",
]


@dataclass
class Candidate:
    answer: str
    executor: ExecutorResult
    verdict: VerificationResult

    @property
    def score(self) -> float:
        base = self.verdict.score
        return base if self.verdict.passed else base - 3.0


@dataclass
class VotingResult:
    winner: Candidate
    candidates: list[Candidate]

    @property
    def n(self) -> int:
        return len(self.candidates)


class Voter:
    def __init__(
        self,
        executor: Executor,
        verifier: Verifier,
        config: Config,
        budget_manager: TaskBudgetManager,
    ):
        self._executor = executor
        self._verifier = verifier
        self._cfg = config
        self._budget_mgr = budget_manager

    async def vote(
        self,
        query: str,
        *,
        thinking: ThinkingMode = ThinkingMode.ENABLED,
        reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH,
        n: int | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        file_context: str | None = None,
    ) -> VotingResult:
        n = n if n is not None else self._cfg.best_of_n
        n = max(1, min(n, len(_DIVERSITY_SUFFIXES)))
        log.info("voter: sampling %d candidates with effort=%s", n, reasoning_effort.value)

        # Each candidate gets its own budget allocation
        # (total budget split equally among candidates)
        exec_tasks = []
        for i in range(n):
            # Create per-candidate budget (fraction of total)
            budget = self._budget_mgr.create()
            if not budget.unlimited:
                budget.total_budget = budget.total_budget // n

            # Inject diversity suffix into prior_messages
            varied_prior = self._inject_diversity(prior_messages, i)

            exec_tasks.append(
                self._executor.run(
                    query,
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                    budget=budget,
                    prior_messages=varied_prior,
                    file_context=file_context,
                )
            )

        raw_results = await asyncio.gather(*exec_tasks, return_exceptions=True)
        executor_results: list[ExecutorResult] = [
            r for r in raw_results if not isinstance(r, BaseException)
        ]

        # Verify each (verifier sees only Q, A — never the reasoning trace)
        verdict_tasks = [
            self._verifier.verify(query, er.answer) for er in executor_results
        ]
        verdicts: list[VerificationResult] = list(await asyncio.gather(*verdict_tasks))

        candidates = [
            Candidate(answer=er.answer, executor=er, verdict=v)
            for er, v in zip(executor_results, verdicts)
        ]

        # Pick highest score
        ranked = sorted(
            enumerate(candidates),
            key=lambda iv: (-iv[1].score, iv[0]),
        )
        winner_idx, winner = ranked[0]
        log.info(
            "voter: winner=#%d score=%.1f passed=%s (scores=%s)",
            winner_idx, winner.score, winner.verdict.passed,
            [round(c.score, 1) for c in candidates],
        )
        return VotingResult(winner=winner, candidates=candidates)

    @staticmethod
    def _inject_diversity(
        prior_messages: list[dict[str, Any]] | None,
        index: int,
    ) -> list[dict[str, Any]]:
        """Add a diversity suffix to the system message for this candidate."""
        suffix = _DIVERSITY_SUFFIXES[index % len(_DIVERSITY_SUFFIXES)]
        if not suffix:
            return prior_messages or []

        prior = list(prior_messages or [])
        # Append as a system-level nudge
        prior.append({"role": "system", "content": suffix.strip()})
        return prior
