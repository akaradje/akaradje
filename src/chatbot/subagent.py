"""Sub-Agent Spawning — isolated context delegation.

Inspired by Claude Code's Task tool architecture:
- Parent agent keeps its context lean
- Sub-agent runs in its own isolated context window
- Sub-agent returns only the result, never its intermediate work
- This prevents context pollution and enables parallel execution

Use cases:
- Research a specific topic without polluting the main conversation
- Execute a multi-step tool workflow in isolation
- Run parallel explorations (e.g., try approach A and B simultaneously)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .client import LLMClient
from .config import Config, ReasoningEffort, ThinkingMode
from .tools import ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class SubAgentResult:
    """What the parent sees from a sub-agent execution."""
    answer: str
    tokens_used: int
    iterations: int
    success: bool
    error: str | None = None


_SUBAGENT_SYSTEM = """You are a focused research assistant working on a specific subtask.

Rules:
- Complete ONLY the task assigned to you
- Be thorough but concise in your response
- Use tools when needed
- Return your findings clearly — the parent agent will integrate them
- Do NOT ask follow-up questions — work with what you have"""


class SubAgent:
    """An isolated agent instance with its own context."""

    def __init__(
        self,
        client: LLMClient,
        config: Config,
        tools: ToolRegistry,
        *,
        name: str = "subagent",
        system_prompt: str = _SUBAGENT_SYSTEM,
        max_iterations: int = 5,
    ):
        self._client = client
        self._cfg = config
        self._tools = tools
        self._name = name
        self._system = system_prompt
        self._max_iter = max_iterations

    async def execute(
        self,
        task: str,
        *,
        context: str | None = None,
        thinking: ThinkingMode = ThinkingMode.ENABLED,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> SubAgentResult:
        """Execute a scoped task in isolation.

        Args:
            task: The specific task to accomplish
            context: Optional additional context from the parent
            thinking: Whether to enable thinking mode
            reasoning_effort: How hard to think
        """
        log.info("subagent[%s]: starting task: %s", self._name, task[:100])

        # Build isolated context — no parent history
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system},
        ]
        if context:
            messages.append({
                "role": "system",
                "content": f"Additional context from parent agent:\n{context}",
            })
        messages.append({"role": "user", "content": task})

        tool_schemas = self._tools.to_openai_schemas()
        total_tokens = 0

        for iteration in range(1, self._max_iter + 1):
            try:
                resp = await self._client.chat(
                    messages=messages,
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                    tools=tool_schemas,
                    max_tokens=4096,
                )
            except Exception as exc:
                log.error("subagent[%s]: API error: %s", self._name, exc)
                return SubAgentResult(
                    answer="", tokens_used=total_tokens,
                    iterations=iteration, success=False,
                    error=str(exc),
                )

            total_tokens += resp.usage.total_tokens

            # No tool calls → done
            if not resp.tool_calls:
                log.info("subagent[%s]: done in %d iterations", self._name, iteration)
                return SubAgentResult(
                    answer=resp.content,
                    tokens_used=total_tokens,
                    iterations=iteration,
                    success=True,
                )

            # Process tool calls
            messages.append({
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
            })

            for tc in resp.tool_calls:
                result = await self._tools.dispatch(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:4000],
                })

        # Hit iteration limit
        log.warning("subagent[%s]: hit max iterations", self._name)
        return SubAgentResult(
            answer="(sub-agent reached iteration limit without completing)",
            tokens_used=total_tokens,
            iterations=self._max_iter,
            success=False,
            error="max iterations reached",
        )


class SubAgentPool:
    """Manages sub-agent creation and parallel execution."""

    def __init__(self, client: LLMClient, config: Config, tools: ToolRegistry):
        self._client = client
        self._cfg = config
        self._tools = tools

    def create(
        self,
        *,
        name: str = "subagent",
        system_prompt: str = _SUBAGENT_SYSTEM,
        max_iterations: int = 5,
    ) -> SubAgent:
        """Create a new sub-agent."""
        return SubAgent(
            self._client, self._cfg, self._tools,
            name=name, system_prompt=system_prompt,
            max_iterations=max_iterations,
        )

    async def execute_parallel(
        self,
        tasks: list[dict[str, Any]],
        *,
        thinking: ThinkingMode = ThinkingMode.ENABLED,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> list[SubAgentResult]:
        """Execute multiple tasks in parallel sub-agents.

        Each task dict should have:
            - "task": str — the task description
            - "name": str (optional) — agent name for logging
            - "context": str (optional) — extra context
        """
        agents = [
            self.create(name=t.get("name", f"subagent-{i}"))
            for i, t in enumerate(tasks)
        ]

        coros = [
            agent.execute(
                task=t["task"],
                context=t.get("context"),
                thinking=thinking,
                reasoning_effort=reasoning_effort,
            )
            for agent, t in zip(agents, tasks)
        ]

        return list(await asyncio.gather(*coros))
