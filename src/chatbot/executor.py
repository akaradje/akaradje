"""ReAct-style executor with tool use.

Loop:
    1. Ask the model.
    2. If it returns plain text  → done, return that.
    3. If it returns tool_calls → run each tool, append results, repeat.
    4. Stop after max_react_iterations (configurable).

Uses the OpenAI tool-calling format. DeepSeek's chat-completions endpoint
accepts the same shape, which is why we keep this provider-agnostic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .client import LLMClient
from .config import Config
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
    iterations: int
    tool_calls_made: int
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
        model: str,
        temperature: float | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> ExecutorResult:
        temp = self._cfg.temperature_generate if temperature is None else temperature

        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system}]
        if prior_messages:
            messages.extend(prior_messages)
        messages.append({"role": "user", "content": query})

        tool_schemas = self._tools.to_openai_schemas()
        tool_calls_made = 0

        for iteration in range(1, self._cfg.max_react_iterations + 1):
            resp = await self._client.chat(
                model=model,
                messages=messages,
                temperature=temp,
                tools=tool_schemas,
                max_tokens=4096,
            )
            choice = resp.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason

            tool_calls = getattr(msg, "tool_calls", None) or []

            # No tool calls — we're done.
            if not tool_calls:
                content = (msg.content or "").strip()
                log.info("executor: finished after %d iter (reason=%s)", iteration, finish_reason)
                return ExecutorResult(
                    answer=content,
                    iterations=iteration,
                    tool_calls_made=tool_calls_made,
                    transcript=messages,
                )

            # Append the assistant turn that requested the tools.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Execute each tool and append a tool-role response.
            for tc in tool_calls:
                tool_calls_made += 1
                result = await self._tools.dispatch(tc.function.name, tc.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _truncate(result),
                    }
                )

        # Hit the iteration cap. Ask the model to wrap up with what it has.
        log.warning("executor: hit max iterations (%d)", self._cfg.max_react_iterations)
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the tool-call iteration limit. "
                    "Stop using tools and provide your best final answer "
                    "based on the information gathered so far."
                ),
            }
        )
        final = await self._client.chat_text(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=2048,
        )
        return ExecutorResult(
            answer=final,
            iterations=self._cfg.max_react_iterations,
            tool_calls_made=tool_calls_made,
            transcript=messages,
        )


def _truncate(text: str, limit: int = 8000) -> str:
    """Tool outputs can be huge (think: large JSON or program stdout).
    We truncate before they get fed back to the model to keep token usage sane.
    """
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n... [truncated {len(text) - limit} chars] ...\n\n{tail}"


# Re-export json for callers that want to construct messages by hand
__all__ = ["Executor", "ExecutorResult", "json"]
