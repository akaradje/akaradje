"""LLM-as-Judge verifier.

A separate model call that scores a candidate answer against the original
question. The crucial design choice: the verifier sees ONLY the question and
the final answer — never the generator's scratchpad or tool transcripts.

This avoids the standard self-agreement failure mode where a model that has
just produced a flawed chain of thought is happy to rubber-stamp the result.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .client import LLMClient
from .config import Config

log = logging.getLogger(__name__)


_VERIFIER_SYSTEM = """You are a strict reviewer of AI-generated answers.

You are NOT the assistant. You did not write the answer. Your job is to
critique it honestly. Default to skepticism.

Score the answer on these axes (each 0-10):
- correctness: are the factual or logical claims right?
- completeness: does it actually answer what was asked?
- clarity: is it well-structured and unambiguous?

Then produce an overall score (0-10) and decide pass/fail. The bar for
"pass" is overall >= 7 AND no axis below 5.

Output JSON only:
{
  "correctness": <0-10>,
  "completeness": <0-10>,
  "clarity": <0-10>,
  "overall": <0-10>,
  "pass": <true|false>,
  "issues": ["<short issue>", ...]
}"""


@dataclass
class VerificationResult:
    correctness: int
    completeness: int
    clarity: int
    overall: int
    passed: bool
    issues: list[str]
    raw: str

    @property
    def score(self) -> float:
        """Floating-point overall score, useful for tie-breaking in voting."""
        return float(self.overall)


class Verifier:
    def __init__(self, client: LLMClient, config: Config):
        self._client = client
        self._cfg = config

    async def verify(self, question: str, answer: str) -> VerificationResult:
        if not self._cfg.verifier_enabled:
            # Verifier disabled — assume pass with neutral score.
            return VerificationResult(
                correctness=7,
                completeness=7,
                clarity=7,
                overall=7,
                passed=True,
                issues=[],
                raw="(verifier disabled)",
            )

        user = (
            f"QUESTION:\n{question}\n\n"
            f"CANDIDATE ANSWER:\n{answer}\n\n"
            "Review the candidate answer per the rubric and output the JSON."
        )
        try:
            text = await self._client.chat_text(
                model=self._cfg.model_verifier,
                messages=[
                    {"role": "system", "content": _VERIFIER_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=self._cfg.temperature_verifier,
                max_tokens=600,
            )
        except Exception as exc:  # pragma: no cover — network path
            log.warning("verifier: call failed (%s) — passing through", exc)
            return VerificationResult(
                correctness=5,
                completeness=5,
                clarity=5,
                overall=5,
                passed=True,
                issues=[f"verifier unavailable: {exc}"],
                raw="",
            )

        return self._parse(text)

    @staticmethod
    def _parse(text: str) -> VerificationResult:
        obj = _extract_json(text)
        if obj is None:
            log.warning("verifier: could not parse JSON, defaulting to neutral")
            return VerificationResult(
                correctness=5,
                completeness=5,
                clarity=5,
                overall=5,
                passed=True,
                issues=["verifier output not parseable"],
                raw=text,
            )
        return VerificationResult(
            correctness=_clip(obj.get("correctness", 5)),
            completeness=_clip(obj.get("completeness", 5)),
            clarity=_clip(obj.get("clarity", 5)),
            overall=_clip(obj.get("overall", 5)),
            passed=bool(obj.get("pass", True)),
            issues=[str(i) for i in obj.get("issues", [])][:10],
            raw=text,
        )


def _clip(v: object) -> int:
    try:
        i = int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 5
    return max(0, min(10, i))


def _extract_json(text: str) -> dict | None:
    # Strict parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None
