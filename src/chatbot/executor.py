"""ReAct-style executor with DeepSeek V4 thinking + tool use.

Loop:
    1. Call model with thinking ENABLED + tools.
    2. Model reasons (reasoning_content) then either:
       a. Returns plain text (content) → done.
       b. Returns tool_calls → execute tools, append results, repeat.
    3. Multi-turn reasoning_content handling per DeepSeek docs:
       - When tool_calls happened: pass reasoning_content back in assistant msg
       - When no tool_calls: reasoning_content ignored by API (strip to save tokens)
    4. Stop when max_react_iterations reached or task_budget exhausted.

The key insight: DeepSeek V4 "thinking with tools" is the SAME behavior as
Opus 4.7's "Interleaved Thinking" — the model reasons, decides to call tools,
gets results, reasons again. It's native to the model architecture.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .client import ChatResponse, LLMClient
from .config import Config, ReasoningEffort, ThinkingMode
from .task_budget import BudgetStatus
from .tools import ToolRegistry

log = logging.getLogger(__name__)


_EXECUTOR_SYSTEM = """You are a careful, methodical assistant.

Guidelines:
- If the question requires computation, use the calculator or python_exec tool
  rather than guessing.
- If the question depends on current facts, use web_search.
- Think step by step. Cite tool outputs explicitly when relevant.
- When you have enough information, give a clear, direct final answer.
- Never fabricate tool results."""


@dataclass
class ExecutorResult:
    answer: str
    reasoning_trace: list[str]  # collected reasoning_content from each iteration
    iterations: int
    tool_calls_made: int
    tokens_used: int
    transcript: list[dict[str, Any]] = field(default_factory=list)


class Executor:
    def __init__(
        self,
        client: LLMClient,
        config: Config,
        tools: ToolRegistry,
        *,
        system_prompt: str = _EXECUTOR_SYSTEM,
    ):
        self._client = client
        self._cfg = config
        self._tools = tools
        self._system = system_prompt

    async def run(
        self,
        query: str,
        *,
        thinking: ThinkingMode = ThinkingMode.ENABLED,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        budget: BudgetStatus | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> ExecutorResult:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system}]
        if prior_messages:
            messages.extend(prior_messages)
        messages.append({"role": "user", "content": query})

        tool_schemas = self._tools.to_openai_schemas()
        tool_calls_made = 0
        total_tokens = 0
        reasoning_trace: list[str] = []

        for iteration in range(1, self._cfg.max_react_iterations + 1):
            # Inject budget awareness message if applicable
            budget_msg = budget.to_system_message() if budget else None
            call_messages = list(messages)
            if budget_msg:
                call_messages.insert(1, {"role": "system", "content": budget_msg})

            # Check budget exhaustion BEFORE making the call
            if budget and budget.exhausted:
                log.warning("executor: budget exhausted, forcing wrap-up")
                break

            resp = await self._client.chat(
                messages=call_messages,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                tools=tool_schemas,
                max_tokens=8192,
            )

            # Record token usage
            total_tokens += resp.usage.total_tokens
            if budget:
                budget.record(resp.usage)

            # Collect reasoning trace
            if resp.reasoning_content:
                reasoning_trace.append(resp.reasoning_content)

            # No tool calls → we're done
            if not resp.tool_calls:
                log.info("executor: finished after %d iter", iteration)
                return ExecutorResult(
                    answer=resp.content,
                    reasoning_trace=reasoning_trace,
                    iterations=iteration,
                    tool_calls_made=tool_calls_made,
                    tokens_used=total_tokens,
                    transcript=messages,
                )

            # Tool calls: append assistant message WITH reasoning_content
            # (DeepSeek docs: when tool calls happen, reasoning_content
            # participates in context concatenation)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in resp.tool_calls
                ],
            }
            # Pass reasoning_content back for multi-turn tool scenarios
            if resp.reasoning_content:
                assistant_msg["reasoning_content"] = resp.reasoning_content

            messages.append(assistant_msg)

            # Execute each tool
            for tc in resp.tool_calls:
                tool_calls_made += 1
                result = await self._tools.dispatch(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _truncate(result),
                })

            # Budget wrap-up check after tool execution
            if budget and budget.should_wrapup:
                log.info("executor: budget threshold reached, requesting wrap-up")
                messages.append({
                    "role": "user",
                    "content": (
                        "Budget is running low. Please provide your best final answer "
                        "based on the information gathered so far. No more tool calls."
                    ),
                })
                # Final call without tools to force a text response
                final_resp = await self._client.chat(
                    messages=messages,
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                    max_tokens=4096,
                )
                total_tokens += final_resp.usage.total_tokens
                if budget:
                    budget.record(final_resp.usage)
                if final_resp.reasoning_content:
                    reasoning_trace.append(final_resp.reasoning_content)
                return ExecutorResult(
                    answer=final_resp.content,
                    reasoning_trace=reasoning_trace,
                    iterations=iteration,
                    tool_calls_made=tool_calls_made,
                    tokens_used=total_tokens,
                    transcript=messages,
                )

        # Hit iteration cap. Ask model to wrap up.
        log.warning("executor: hit max iterations (%d)", self._cfg.max_react_iterations)
        messages.append({
            "role": "user",
            "content": (
                "You have reached the tool-call iteration limit. "
                "Stop using tools and provide your best final answer."
            ),
        })
        final_resp = await self._client.chat(
            messages=messages,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=4096,
        )
        total_tokens += final_resp.usage.total_tokens
        if budget:
            budget.record(final_resp.usage)
        if final_resp.reasoning_content:
            reasoning_trace.append(final_resp.reasoning_content)
        return ExecutorResult(
            answer=final_resp.content,
            reasoning_trace=reasoning_trace,
            iterations=self._cfg.max_react_iterations,
            tool_calls_made=tool_calls_made,
            tokens_used=total_tokens,
            transcript=messages,
        )


def _truncate(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n... [truncated {len(text) - limit} chars] ...\n\n{tail}"
