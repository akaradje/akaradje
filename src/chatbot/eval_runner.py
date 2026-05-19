"""Eval Framework — automated testing harness for the chatbot.

Runs predefined test cases against the pipeline and scores results.
This is how you measure whether a change actually improves quality.

Test cases are YAML/JSON files with:
- question: the input query
- expected: what a correct answer should contain
- rubric: scoring criteria
- min_score: minimum acceptable verifier score
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """A single evaluation test case."""
    id: str
    question: str
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    min_score: int = 7
    max_tokens: int = 50000
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of running a single eval case."""
    case_id: str
    question: str
    answer: str
    score: int
    passed: bool
    latency_ms: int
    tokens_used: int
    contains_check: bool
    not_contains_check: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Summary report of an eval run."""
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    avg_latency_ms: int
    total_tokens: int
    results: list[EvalResult]
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 3),
            "avg_score": round(self.avg_score, 2),
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens,
            "results": [
                {
                    "case_id": r.case_id,
                    "passed": r.passed,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "errors": r.errors,
                }
                for r in self.results
            ],
        }


# Built-in eval cases for smoke testing
DEFAULT_EVAL_CASES = [
    EvalCase(
        id="math-basic",
        question="What is 2^10 + 3^5?",
        expected_contains=["1267"],
        tags=["math", "trivial"],
    ),
    EvalCase(
        id="code-factorial",
        question="Write a Python function to compute factorial recursively. Show the code.",
        expected_contains=["def", "factorial", "return"],
        tags=["code", "standard"],
    ),
    EvalCase(
        id="reasoning-logic",
        question="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        expected_contains=["cannot", "no"],
        expected_not_contains=["yes, some roses fade"],
        tags=["reasoning", "standard"],
    ),
    EvalCase(
        id="design-ratelimiter",
        question="Design a token bucket rate limiter for a REST API. Explain the algorithm.",
        expected_contains=["token", "bucket", "refill"],
        min_score=6,
        tags=["design", "complex"],
    ),
    EvalCase(
        id="greeting",
        question="Hello!",
        expected_contains=[],
        min_score=5,
        tags=["trivial"],
    ),
]


class EvalRunner:
    """Runs evaluation cases against the orchestrator."""

    def __init__(self, orchestrator: Any):
        self._orch = orchestrator

    async def run_case(self, case: EvalCase) -> EvalResult:
        """Run a single eval case."""
        start = time.perf_counter()
        errors: list[str] = []

        try:
            answer = await self._orch.ask(case.question)
            answer_text = answer.text
            score = answer.verdict.overall if answer.verdict else 7
            tokens = answer.tokens_used
        except Exception as exc:
            return EvalResult(
                case_id=case.id, question=case.question,
                answer="", score=0, passed=False,
                latency_ms=int((time.perf_counter() - start) * 1000),
                tokens_used=0, contains_check=False,
                not_contains_check=True, errors=[str(exc)],
            )

        latency_ms = int((time.perf_counter() - start) * 1000)

        # Check expected_contains
        contains_ok = True
        for expected in case.expected_contains:
            if expected.lower() not in answer_text.lower():
                contains_ok = False
                errors.append(f"missing expected: '{expected}'")

        # Check expected_not_contains
        not_contains_ok = True
        for not_expected in case.expected_not_contains:
            if not_expected.lower() in answer_text.lower():
                not_contains_ok = False
                errors.append(f"contains unexpected: '{not_expected}'")

        passed = (
            score >= case.min_score
            and contains_ok
            and not_contains_ok
            and tokens <= case.max_tokens
        )

        if tokens > case.max_tokens:
            errors.append(f"exceeded max_tokens: {tokens} > {case.max_tokens}")

        return EvalResult(
            case_id=case.id,
            question=case.question,
            answer=answer_text[:500],
            score=score,
            passed=passed,
            latency_ms=latency_ms,
            tokens_used=tokens,
            contains_check=contains_ok,
            not_contains_check=not_contains_ok,
            errors=errors,
        )

    async def run_all(
        self,
        cases: list[EvalCase] | None = None,
        *,
        parallel: bool = False,
    ) -> EvalReport:
        """Run all eval cases and produce a report."""
        cases = cases or DEFAULT_EVAL_CASES

        if parallel:
            results = list(await asyncio.gather(*[
                self.run_case(c) for c in cases
            ]))
        else:
            results = []
            for case in cases:
                results.append(await self.run_case(case))

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        scores = [r.score for r in results]

        return EvalReport(
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / max(1, total),
            avg_score=sum(scores) / max(1, len(scores)),
            avg_latency_ms=sum(r.latency_ms for r in results) // max(1, total),
            total_tokens=sum(r.tokens_used for r in results),
            results=results,
        )

    @staticmethod
    def load_cases(path: Path) -> list[EvalCase]:
        """Load eval cases from a JSON file."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            EvalCase(
                id=c["id"],
                question=c["question"],
                expected_contains=c.get("expected_contains", []),
                expected_not_contains=c.get("expected_not_contains", []),
                min_score=c.get("min_score", 7),
                tags=c.get("tags", []),
            )
            for c in data
        ]
