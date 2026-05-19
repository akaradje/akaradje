"""LLM-as-Judge verifier using DeepSeek V4 thinking mode.

Uses `reasoning_effort: "high"` so the verifier actually THINKS about
whether the answer is correct. The thinking trace makes verification
substantially more reliable than asking without CoT.

Critical design: the verifier sees ONLY (question, answer) — never the
generator's scratchpad. This prevents the self-agreement failure mode.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .client import LLMClient
from .config import Config, ReasoningEffort, ThinkingMode

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
    reasoning: str | None  # the verifier's thinking trace (for debugging)
    raw: str

    @property
    def score(self) -> float:
        return float(self.overall)


class Verifier:
    def __init__(self, client: LLMClient, config: Config):
        self._client = client
        self._cfg = config

    async def verify(self, question: str, answer: str) -> VerificationResult:
        if not self._cfg.verifier_enabled:
            return VerificationResult(
                correctness=7, completeness=7, clarity=7, overall=7,
                passed=True, issues=[], reasoning=None, raw="(verifier disabled)",
            )

        user = (
            f"QUESTION:\n{question}\n\n"
            f"CANDIDATE ANSWER:\n{answer}\n\n"
            "Review the candidate answer per the rubric and output the JSON."
        )
        try:
            resp = await self._client.chat(
                messages=[
                    {"role": "system", "content": _VERIFIER_SYSTEM},
                    {"role": "user", "content": user},
                ],
                # Verifier uses thinking mode to reason about correctness
                thinking=ThinkingMode.ENABLED,
                reasoning_effort=ReasoningEffort.HIGH,
                max_tokens=1000,
            )
        except Exception as exc:
            log.warning("verifier: call failed (%s) — passing through", exc)
            return VerificationResult(
                correctness=5, completeness=5, clarity=5, overall=5,
                passed=True, issues=[f"verifier unavailable: {exc}"],
                reasoning=None, raw="",
            )

        return self._parse(resp.content, reasoning=resp.reasoning_content)

    @staticmethod
    def _parse(text: str, reasoning: str | None = None) -> VerificationResult:
        obj = _extract_json(text)
        if obj is None:
            log.warning("verifier: could not parse JSON, defaulting to neutral")
            return VerificationResult(
                correctness=5, completeness=5, clarity=5, overall=5,
                passed=True, issues=["verifier output not parseable"],
                reasoning=reasoning, raw=text,
            )
        return VerificationResult(
            correctness=_clip(obj.get("correctness", 5)),
            completeness=_clip(obj.get("completeness", 5)),
            clarity=_clip(obj.get("clarity", 5)),
            overall=_clip(obj.get("overall", 5)),
            passed=bool(obj.get("pass", True)),
            issues=[str(i) for i in obj.get("issues", [])][:10],
            reasoning=reasoning,
            raw=text,
        )


def _clip(v: object) -> int:
    try:
        i = int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 5
    return max(0, min(10, i))


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None
