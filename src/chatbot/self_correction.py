"""Iterative Self-Correction with Backtracking.

Instead of regenerating the entire answer when the verifier fails,
we use a targeted correction strategy:

1. Verifier identifies SPECIFIC issues (e.g., "factual error in point 3")
2. Corrector asks the model to fix ONLY those issues
3. Re-verify the corrected answer
4. If still failing, escalate effort level and retry

This is more efficient than Best-of-N for cases where the answer is
mostly correct but has localized errors.

Research basis:
- "Thought-ICS" (2026): 20-40% correction lift with localized backtracking
- PAG (2024): separating generator and verifier roles
- Key insight: correction works ONLY when you tell the model exactly what's wrong
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .client import LLMClient
from .config import Config, ReasoningEffort, ThinkingMode
from .verifier import VerificationResult, Verifier

log = logging.getLogger(__name__)


_CORRECTION_PROMPT = """You previously answered a question, but the answer has issues that need fixing.

ORIGINAL QUESTION:
{question}

YOUR PREVIOUS ANSWER:
{answer}

IDENTIFIED ISSUES:
{issues}

INSTRUCTIONS:
- Fix ONLY the issues listed above
- Keep everything else that was correct
- Do NOT apologize or explain what you changed
- Just provide the corrected answer directly"""


@dataclass
class CorrectionResult:
    """Result of a self-correction attempt."""
    final_answer: str
    correction_attempts: int
    final_verdict: VerificationResult
    improved: bool  # did correction actually improve the score?
    effort_escalated: bool


class SelfCorrector:
    """Manages iterative self-correction with effort escalation."""

    def __init__(
        self,
        client: LLMClient,
        verifier: Verifier,
        config: Config,
        *,
        max_corrections: int = 2,
        min_score_for_pass: int = 7,
    ):
        self._client = client
        self._verifier = verifier
        self._cfg = config
        self._max_corrections = max_corrections
        self._min_score = min_score_for_pass

    async def correct_if_needed(
        self,
        question: str,
        answer: str,
        verdict: VerificationResult,
        *,
        current_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> CorrectionResult:
        """Attempt to correct the answer if the verifier found issues.

        Returns the (possibly improved) answer and metadata.
        """
        # Already passing — no correction needed
        if verdict.passed and verdict.overall >= self._min_score:
            return CorrectionResult(
                final_answer=answer,
                correction_attempts=0,
                final_verdict=verdict,
                improved=False,
                effort_escalated=False,
            )

        log.info(
            "self-correction: triggered (score=%d, issues=%d)",
            verdict.overall, len(verdict.issues),
        )

        current_answer = answer
        current_verdict = verdict
        best_score = verdict.overall
        best_answer = answer
        effort_escalated = False

        for attempt in range(1, self._max_corrections + 1):
            # Format issues for the correction prompt
            issues_text = "\n".join(f"- {issue}" for issue in current_verdict.issues)
            if not issues_text:
                issues_text = f"- Overall score too low ({current_verdict.overall}/10)"

            # Ask model to correct
            corrected = await self._client.chat_text(
                messages=[{
                    "role": "user",
                    "content": _CORRECTION_PROMPT.format(
                        question=question,
                        answer=current_answer,
                        issues=issues_text,
                    ),
                }],
                thinking=ThinkingMode.ENABLED,
                reasoning_effort=current_effort,
                max_tokens=8192,
            )

            # Re-verify
            new_verdict = await self._verifier.verify(question, corrected)

            log.info(
                "self-correction: attempt %d → score %d→%d (pass=%s)",
                attempt, current_verdict.overall, new_verdict.overall, new_verdict.passed,
            )

            # Track best result
            if new_verdict.overall > best_score:
                best_score = new_verdict.overall
                best_answer = corrected

            # Success
            if new_verdict.passed and new_verdict.overall >= self._min_score:
                return CorrectionResult(
                    final_answer=corrected,
                    correction_attempts=attempt,
                    final_verdict=new_verdict,
                    improved=True,
                    effort_escalated=effort_escalated,
                )

            # Not fixed yet — escalate effort if possible
            current_answer = corrected
            current_verdict = new_verdict
            escalated_effort = self._escalate(current_effort)
            if escalated_effort != current_effort:
                current_effort = escalated_effort
                effort_escalated = True
                log.info("self-correction: escalating effort to %s", current_effort.value)

        # Exhausted correction attempts — return best we found
        return CorrectionResult(
            final_answer=best_answer,
            correction_attempts=self._max_corrections,
            final_verdict=current_verdict,
            improved=best_score > verdict.overall,
            effort_escalated=effort_escalated,
        )

    @staticmethod
    def _escalate(current: ReasoningEffort) -> ReasoningEffort:
        """Escalate to the next effort level."""
        order = [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH, ReasoningEffort.MAX]
        idx = order.index(current)
        if idx < len(order) - 1:
            return order[idx + 1]
        return current  # already at max
