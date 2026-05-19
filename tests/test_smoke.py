"""Network-free smoke tests.

These exercise only the pure-Python logic (parsing, scoring, retrieval,
local tools). LLM-calling code paths are covered by separate integration
tests that need a live API key.
"""

from __future__ import annotations

import pytest

from chatbot.executor import _truncate
from chatbot.memory import ConversationMemory
from chatbot.router import Complexity, Router
from chatbot.tools import (
    CalculatorTool,
    PythonExecTool,
    ToolRegistry,
    WebSearchTool,
)
from chatbot.verifier import Verifier, VerificationResult
from chatbot.voter import Candidate
from chatbot.executor import ExecutorResult


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("1 + 2", "3"),
        ("(2 ** 10) - 24", "1000"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
        ("-5 + 3", "-2"),
    ],
)
async def test_calculator_basic(expr, expected):
    out = await CalculatorTool().run({"expression": expr})
    assert out == expected


async def test_calculator_rejects_unsafe():
    out = await CalculatorTool().run({"expression": "__import__('os').system('echo pwned')"})
    assert out.startswith("ERROR")


async def test_calculator_missing_arg():
    assert (await CalculatorTool().run({})).startswith("ERROR")


# ---------------------------------------------------------------------------
# Python exec
# ---------------------------------------------------------------------------


async def test_python_exec_basic():
    out = await PythonExecTool().run({"code": "print(2 + 2)"})
    assert out.strip() == "4"


async def test_python_exec_timeout():
    tool = PythonExecTool()
    tool.timeout_seconds = 1.0
    out = await tool.run({"code": "while True: pass"})
    assert "timed out" in out


async def test_python_exec_runtime_error():
    out = await PythonExecTool().run({"code": "raise ValueError('boom')"})
    assert "EXIT=" in out and "boom" in out


# ---------------------------------------------------------------------------
# Web search stub
# ---------------------------------------------------------------------------


async def test_web_search_stubbed():
    out = await WebSearchTool().run({"query": "anything"})
    assert "stubbed" in out


async def test_web_search_with_provider():
    async def provider(q, k):
        return [{"title": "t", "url": "u", "snippet": q}]

    out = await WebSearchTool(provider=provider).run({"query": "deepseek", "k": 3})
    assert "deepseek" in out


# ---------------------------------------------------------------------------
# Tool registry dispatch
# ---------------------------------------------------------------------------


async def test_registry_dispatch_unknown():
    reg = ToolRegistry.default()
    out = await reg.dispatch("does_not_exist", {})
    assert out.startswith("ERROR")


async def test_registry_dispatch_json_args():
    reg = ToolRegistry.default()
    out = await reg.dispatch("calculator", '{"expression": "10 * 5"}')
    assert out == "50"


def test_registry_schemas():
    reg = ToolRegistry.default()
    schemas = reg.to_openai_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"calculator", "python_exec", "web_search"}


# ---------------------------------------------------------------------------
# Router heuristics (no network)
# ---------------------------------------------------------------------------


def test_router_heuristic_greeting():
    r = Router(client=None, config=None)  # type: ignore[arg-type]
    assert r._heuristic("hi") == Complexity.TRIVIAL
    assert r._heuristic("สวัสดี") == Complexity.TRIVIAL
    assert r._heuristic("thanks!") == Complexity.TRIVIAL


def test_router_heuristic_short_query():
    r = Router(client=None, config=None)  # type: ignore[arg-type]
    assert r._heuristic("ping") == Complexity.TRIVIAL


def test_router_heuristic_complex_hint():
    r = Router(client=None, config=None)  # type: ignore[arg-type]
    long_complex = (
        "Please design a multi-tenant rate limiter with sliding windows "
        "for our gateway"
    )
    assert r._heuristic(long_complex) == Complexity.COMPLEX


def test_router_heuristic_ambiguous_returns_none():
    r = Router(client=None, config=None)  # type: ignore[arg-type]
    # Mid-length question with no strong signal should defer to LLM.
    assert r._heuristic("How does Python handle GIL contention?") is None


# ---------------------------------------------------------------------------
# Verifier parsing (no network)
# ---------------------------------------------------------------------------


def test_verifier_parse_strict_json():
    payload = (
        '{"correctness": 9, "completeness": 8, "clarity": 9, '
        '"overall": 9, "pass": true, "issues": []}'
    )
    v = Verifier._parse(payload)
    assert v.passed and v.overall == 9 and v.correctness == 9


def test_verifier_parse_embedded_json():
    payload = "Sure, here's my review:\n" '{"correctness": 4, "completeness": 5, "clarity": 6, "overall": 5, "pass": false, "issues": ["off by one"]}\nEnd.'
    v = Verifier._parse(payload)
    assert not v.passed and v.issues == ["off by one"]


def test_verifier_parse_garbage_defaults_neutral():
    v = Verifier._parse("nope")
    assert v.overall == 5 and v.passed is True


def test_verifier_parse_clips_out_of_range():
    payload = '{"correctness": 999, "completeness": -10, "clarity": 7, "overall": 12, "pass": true, "issues": []}'
    v = Verifier._parse(payload)
    assert v.correctness == 10 and v.completeness == 0 and v.overall == 10


# ---------------------------------------------------------------------------
# Voter scoring logic (no network)
# ---------------------------------------------------------------------------


def _exec_result() -> ExecutorResult:
    return ExecutorResult(answer="x", iterations=1, tool_calls_made=0, transcript=[])


def _verdict(score: int, passed: bool) -> VerificationResult:
    return VerificationResult(
        correctness=score,
        completeness=score,
        clarity=score,
        overall=score,
        passed=passed,
        issues=[],
        raw="",
    )


def test_candidate_score_penalizes_failure():
    passing = Candidate(answer="a", executor=_exec_result(), verdict=_verdict(7, True))
    failing = Candidate(answer="b", executor=_exec_result(), verdict=_verdict(8, False))
    # Failing candidate (8) gets -3 penalty → 5; passing 7 should still win.
    assert passing.score > failing.score


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_memory_window_trims():
    m = ConversationMemory(window_turns=4)
    for i in range(10):
        m.add_user(f"q{i}")
        m.add_assistant(f"a{i}")
    recent = m.recent_messages()
    assert len(recent) == 4
    assert recent[-1]["content"] == "a9"


def test_memory_recall_finds_relevant_past():
    m = ConversationMemory(window_turns=2)
    m.add_user("how do I use rate limiting in fastapi")
    m.add_assistant("use slowapi or starlette middleware")
    # Push the above out of the recent window.
    for i in range(5):
        m.add_user(f"unrelated chatter {i}")
        m.add_assistant(f"ok {i}")
    recalled = m.recall("rate limit fastapi", k=2)
    contents = " ".join(t.content for t in recalled)
    assert "rate limit" in contents.lower() or "slowapi" in contents.lower()


def test_memory_recall_empty_query():
    m = ConversationMemory(window_turns=2)
    m.add_user("anything")
    assert m.recall("", k=3) == []


def test_memory_persistence(tmp_path):
    p = tmp_path / "h.jsonl"
    m1 = ConversationMemory(store_path=p, window_turns=4)
    m1.add_user("hello")
    m1.add_assistant("hi")
    m2 = ConversationMemory(store_path=p, window_turns=4)
    archived = list(m2.archived())
    assert len(archived) == 2
    assert archived[0].content == "hello"


# ---------------------------------------------------------------------------
# Executor truncate
# ---------------------------------------------------------------------------


def test_truncate_short_passes_through():
    assert _truncate("hello") == "hello"


def test_truncate_long_clipped_with_marker():
    big = "x" * 20000
    out = _truncate(big, limit=1000)
    assert "[truncated" in out
    assert len(out) < len(big)
