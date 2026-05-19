"""Best-of-N voting.

For COMPLEX queries we sample N independent candidate answers from the
generator (each running its own ReAct loop), then have the verifier score
all of them, and return the highest-scoring one.

Why parallel sampling instead of iterative self-refinement?
    - Self-refine tends to anchor on the first attempt and only patch
      surface issues.
    - Independent samples explore different reasoning paths, which is what
      "self-consistency" research showed empirically improves accuracy.
    - It parallelizes trivially.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .config import Config
from .executor import Executor, ExecutorResult
from .verifier import VerificationResult, Verifier

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    answer: str
    executor: ExecutorResult
    verdict: VerificationResult

    @property
    def score(self) -> float:
        # Penalize failed verdicts so a high-overall but failed answer
        # doesn't beat a slightly-lower passing one.
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
    def __init__(self, executor: Executor, verifier: Verifier, config: Config):
        self._executor = executor
        self._verifier = verifier
        self._cfg = config

    async def vote(
        self,
        query: str,
        *,
        model: str,
        n: int | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> VotingResult:
        n = n if n is not None else self._cfg.best_of_n
        n = max(1, n)
        log.info("voter: sampling %d candidates on %s", n, model)

        # Slight temperature jitter so the N samples are actually diverse.
        # All samples >= 1 use a step above the base temperature.
        temps = [self._cfg.temperature_generate] + [
            min(1.2, self._cfg.temperature_generate + 0.1 * i) for i in range(1, n)
        ]

        exec_tasks = [
            self._executor.run(
                query,
                model=model,
                temperature=temps[i],
                prior_messages=prior_messages,
            )
            for i in range(n)
        ]
        executor_results: list[ExecutorResult] = list(await asyncio.gather(*exec_tasks))

        # Verify each in parallel. Verifier sees only (query, answer) — the
        # executor transcript is intentionally NOT passed in.
        verdict_tasks = [
            self._verifier.verify(query, er.answer) for er in executor_results
        ]
        verdicts: list[VerificationResult] = list(await asyncio.gather(*verdict_tasks))

        candidates = [
            Candidate(answer=er.answer, executor=er, verdict=v)
            for er, v in zip(executor_results, verdicts)
        ]

        # Pick highest score, with passed-status as the tiebreaker. Stable
        # sort so on exact ties the first-sampled candidate wins.
        ranked = sorted(
            enumerate(candidates),
            key=lambda iv: (-iv[1].score, iv[0]),
        )
        winner_idx, winner = ranked[0]
        log.info(
            "voter: winner=#%d score=%.1f passed=%s (scores=%s)",
            winner_idx,
            winner.score,
            winner.verdict.passed,
            [round(c.score, 1) for c in candidates],
        )
        return VotingResult(winner=winner, candidates=candidates)
