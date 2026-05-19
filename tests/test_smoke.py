"""Network-free smoke tests for all pure-Python logic.

No API calls. Tests parsing, scoring, tool execution, routing heuristics,
memory retrieval, budget tracking, and verifier output parsing.
"""

from __future__ import annotations

import pytest

from chatbot.config import Config, ReasoningEffort, ThinkingMode
from chatbot.executor import _truncate
from chatbot.memory import ConversationMemory
from chatbot.router import Complexity, Router
from chatbot.task_budget import BudgetStatus, TaskBudgetManager
from chatbot.tools import CalculatorTool, PythonExecTool, ToolRegistry, WebSearchTool
from chatbot.verifier import Verifier, VerificationResult
from chatbot.voter import Candidate
from chatbot.executor import ExecutorResult
from chatbot.client import TokenUsage


# ═══════════════════════════════════════════════════════════════════════════════
# Calculator
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("expr,expected", [
    ("1 + 2", "3"),
    ("(2 ** 10) - 24", "1000"),
    ("7 // 2", "3"),
    ("7 % 2", "1"),
    ("-5 + 3", "-2"),
])
async def test_calculator_basic(expr, expected):
    assert (await CalculatorTool().run({"expression": expr})) == expected


async def test_calculator_rejects_unsafe():
    out = await CalculatorTool().run({"expression": "__import__('os').system('echo pwned')"})
    assert out.startswith("ERROR")


async def test_calculator_missing_arg():
    assert (await CalculatorTool().run({})).startswith("ERROR")


# ═══════════════════════════════════════════════════════════════════════════════
# Python exec
# ═══════════════════════════════════════════════════════════════════════════════

async def test_python_exec_basic():
    assert (await PythonExecTool().run({"code": "print(2 + 2)"})).strip() == "4"


async def test_python_exec_timeout():
    tool = PythonExecTool()
    tool.timeout_seconds = 1.0
    out = await tool.run({"code": "while True: pass"})
    assert "timed out" in out


async def test_python_exec_runtime_error():
    out = await PythonExecTool().run({"code": "raise ValueError('boom')"})
    assert "EXIT=" in out and "boom" in out


# ═══════════════════════════════════════════════════════════════════════════════
# Web search
# ═══════════════════════════════════════════════════════════════════════════════

async def test_web_search_stubbed():
    assert "stubbed" in (await WebSearchTool().run({"query": "anything"}))


async def test_web_search_with_provider():
    async def provider(q, k):
        return [{"title": "t", "snippet": q}]
    assert "deepseek" in (await WebSearchTool(provider=provider).run({"query": "deepseek"}))


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registry
# ═══════════════════════════════════════════════════════════════════════════════

async def test_registry_dispatch():
    reg = ToolRegistry.default()
    assert (await reg.dispatch("calculator", '{"expression": "5*5"}')) == "25"
    assert (await reg.dispatch("nope", {})).startswith("ERROR")


def test_registry_schemas():
    names = {s["function"]["name"] for s in ToolRegistry.default().to_openai_schemas()}
    assert names == {"calculator", "python_exec", "web_search"}


# ═══════════════════════════════════════════════════════════════════════════════
# Router heuristics
# ═══════════════════════════════════════════════════════════════════════════════

def test_router_trivial():
    r = Router(client=None, config=None)  # type: ignore
    assert r._heuristic("hi") == Complexity.TRIVIAL
    assert r._heuristic("สวัสดี") == Complexity.TRIVIAL
    assert r._heuristic("thanks!") == Complexity.TRIVIAL
    assert r._heuristic("ping") == Complexity.TRIVIAL


def test_router_complex():
    r = Router(client=None, config=None)  # type: ignore
    q = "Please design a multi-tenant rate limiter with sliding windows for our gateway"
    assert r._heuristic(q) == Complexity.COMPLEX


def test_router_ambiguous():
    r = Router(client=None, config=None)  # type: ignore
    assert r._heuristic("How does Python handle GIL contention?") is None


def test_router_thinking_config():
    cfg = Config.from_env.__func__.__code__  # just need a config with defaults
    # Build a minimal config manually
    from dataclasses import fields
    cfg = Config(
        api_key="test", base_url="http://x", model="m",
        effort_trivial=ThinkingMode.DISABLED,
        effort_standard=ReasoningEffort.MEDIUM,
        effort_complex=ReasoningEffort.HIGH,
        router_enabled=True, verifier_enabled=True,
        best_of_n=3, max_react_iterations=8,
        task_budget_tokens=128000, task_budget_wrapup_threshold=0.8,
        log_level="INFO",
    )
    r = Router(client=None, config=cfg)  # type: ignore
    assert r.get_thinking_config(Complexity.TRIVIAL) == (ThinkingMode.DISABLED, ReasoningEffort.LOW)
    assert r.get_thinking_config(Complexity.STANDARD) == (ThinkingMode.ENABLED, ReasoningEffort.MEDIUM)
    assert r.get_thinking_config(Complexity.COMPLEX) == (ThinkingMode.ENABLED, ReasoningEffort.HIGH)


# ═══════════════════════════════════════════════════════════════════════════════
# Task Budget
# ═══════════════════════════════════════════════════════════════════════════════

def test_budget_basic():
    b = BudgetStatus(total_budget=100000, wrapup_threshold=0.8)
    assert not b.unlimited
    assert b.tokens_remaining == 100000
    assert not b.should_wrapup
    assert not b.exhausted

    # Record 85k tokens
    b.record(TokenUsage(prompt_tokens=40000, completion_tokens=40000, reasoning_tokens=5000, total_tokens=85000))
    assert b.should_wrapup
    assert not b.exhausted
    assert b.tokens_remaining == 15000

    # Record more
    b.record(TokenUsage(prompt_tokens=10000, completion_tokens=10000, reasoning_tokens=0, total_tokens=20000))
    assert b.exhausted


def test_budget_unlimited():
    b = BudgetStatus(total_budget=0)
    assert b.unlimited
    assert not b.should_wrapup
    assert not b.exhausted
    assert b.to_system_message() is None


def test_budget_message_content():
    b = BudgetStatus(total_budget=100000, wrapup_threshold=0.8)
    msg = b.to_system_message()
    assert msg is not None
    assert "100,000" in msg
    assert "Budget is healthy" in msg

    b.record(TokenUsage(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, total_tokens=85000))
    msg = b.to_system_message()
    assert "approaching the budget limit" in msg


# ═══════════════════════════════════════════════════════════════════════════════
# Verifier
# ═══════════════════════════════════════════════════════════════════════════════

def test_verifier_parse_json():
    v = Verifier._parse(
        '{"correctness": 9, "completeness": 8, "clarity": 9, "overall": 9, "pass": true, "issues": []}',
        reasoning="test thought",
    )
    assert v.passed and v.overall == 9 and v.reasoning == "test thought"


def test_verifier_parse_embedded():
    v = Verifier._parse(
        'Review:\n{"correctness": 4, "completeness": 5, "clarity": 6, '
        '"overall": 5, "pass": false, "issues": ["wrong"]}\nEnd.',
    )
    assert not v.passed and v.issues == ["wrong"]


def test_verifier_parse_garbage():
    v = Verifier._parse("nope")
    assert v.overall == 5 and v.passed is True


def test_verifier_clips():
    v = Verifier._parse('{"correctness": 999, "completeness": -10, "clarity": 7, "overall": 12, "pass": true, "issues": []}')
    assert v.correctness == 10 and v.completeness == 0 and v.overall == 10


# ═══════════════════════════════════════════════════════════════════════════════
# Voter scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _exec_result() -> ExecutorResult:
    return ExecutorResult(answer="x", reasoning_trace=[], iterations=1, tool_calls_made=0, tokens_used=100, transcript=[])


def _verdict(score: int, passed: bool) -> VerificationResult:
    return VerificationResult(
        correctness=score, completeness=score, clarity=score, overall=score,
        passed=passed, issues=[], reasoning=None, raw="",
    )


def test_candidate_score_penalizes_failure():
    passing = Candidate(answer="a", executor=_exec_result(), verdict=_verdict(7, True))
    failing = Candidate(answer="b", executor=_exec_result(), verdict=_verdict(8, False))
    assert passing.score > failing.score


# ═══════════════════════════════════════════════════════════════════════════════
# Memory
# ═══════════════════════════════════════════════════════════════════════════════

def test_memory_window():
    m = ConversationMemory(window_turns=4)
    for i in range(10):
        m.add_user(f"q{i}")
        m.add_assistant(f"a{i}")
    assert len(m.recent_messages()) == 4


def test_memory_recall():
    m = ConversationMemory(window_turns=2)
    m.add_user("how do I rate limit fastapi")
    m.add_assistant("use slowapi")
    for i in range(5):
        m.add_user(f"unrelated {i}")
        m.add_assistant(f"ok {i}")
    recalled = m.recall("rate limit fastapi", k=2)
    assert any("rate" in t.content.lower() or "slowapi" in t.content for t in recalled)


def test_memory_persistence(tmp_path):
    p = tmp_path / "h.jsonl"
    m1 = ConversationMemory(store_path=p, window_turns=4)
    m1.add_user("hello")
    m1.add_assistant("hi")
    m2 = ConversationMemory(store_path=p, window_turns=4)
    assert len(list(m2.archived())) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Truncate
# ═══════════════════════════════════════════════════════════════════════════════

def test_truncate():
    assert _truncate("short") == "short"
    big = "x" * 20000
    out = _truncate(big, limit=1000)
    assert "[truncated" in out and len(out) < len(big)
