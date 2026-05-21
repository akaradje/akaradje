"""Multi-step Planner — structured task decomposition for complex queries.

Prevents the executor from improvising on complex tasks by generating a
step-by-step plan upfront using LLM Structured Output (JSON Mode).

Architecture:
- Planner.evaluate(query) → returns a Plan or None (for trivial queries)
- Plan contains ordered PlanStep objects with tool hints
- Steps track status: pending → running → done
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .client import LLMClient
from .config import Config, ReasoningEffort, ThinkingMode

log = logging.getLogger(__name__)

# ─── JSON schema for structured output ────────────────────────────────────────

_PLAN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "needs_planning": {
                    "type": "boolean",
                    "description": "Whether this query requires multi-step planning",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why planning is or isn't needed",
                },
                "steps": {
                    "type": "array",
                    "description": "Ordered plan steps (empty if needs_planning is false)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "1-based step number",
                            },
                            "description": {
                                "type": "string",
                                "description": "Clear description of what this step accomplishes",
                            },
                            "tool_hints": {
                                "type": "array",
                                "description": "Suggested tool names for this step (empty if none)",
                                "items": {"type": "string"},
                            },
                            "expected_output": {
                                "type": "string",
                                "description": "What information or result this step should produce",
                            },
                        },
                        "required": ["index", "description", "tool_hints", "expected_output"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["needs_planning", "reasoning", "steps"],
            "additionalProperties": False,
        },
    },
}

_PLANNER_SYSTEM = """You are a strategic task planner. Your job is to evaluate a user query
and decide whether it needs a multi-step execution plan.

Rules:
- TRIVIAL queries (greetings, simple facts, definitions, basic math, short explanations)
  do NOT need planning. Set needs_planning=false, steps=[].
- COMPLEX queries (research tasks, multi-part questions, code generation with constraints,
  data analysis, multi-step workflows) DO need planning.
- When planning: break the task into 3–7 ordered steps. Each step should have a clear goal,
  suggest relevant tools from this list, and describe the expected output.

Available tools: web_search, url_fetch, python_exec, calculator, file_read, shell_exec,
generate_artifact

Respond with a valid JSON object matching the schema exactly."""


# ─── Data types ────────────────────────────────────────────────────────────────


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class PlanStep:
    index: int
    description: str
    tool_hints: list[str] = field(default_factory=list)
    expected_output: str = ""
    status: StepStatus = StepStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "tool_hints": self.tool_hints,
            "expected_output": self.expected_output,
            "status": self.status.value,
        }


@dataclass
class Plan:
    steps: list[PlanStep]
    reasoning: str = ""

    @property
    def current_step(self) -> PlanStep | None:
        """The first non-done step (running or pending)."""
        for step in self.steps:
            if step.status != StepStatus.DONE:
                return step
        return None

    @property
    def is_complete(self) -> bool:
        return all(s.status == StepStatus.DONE for s in self.steps)

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.DONE)

    def start_step(self, index: int) -> None:
        """Mark a step as running."""
        for step in self.steps:
            if step.index == index:
                step.status = StepStatus.RUNNING
                return

    def complete_step(self, index: int) -> None:
        """Mark a step as done."""
        for step in self.steps:
            if step.index == index:
                step.status = StepStatus.DONE
                return

    def fail_step(self, index: int) -> None:
        """Mark a step as failed."""
        for step in self.steps:
            if step.index == index:
                step.status = StepStatus.FAILED
                return

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "reasoning": self.reasoning,
            "total_steps": len(self.steps),
            "completed": self.completed_count,
            "is_complete": self.is_complete,
            "current_step": self.current_step.index if self.current_step else None,
        }

    def to_context_string(self) -> str:
        """Render the plan as a system-prompt context block."""
        if not self.steps:
            return ""

        lines = ["## Execution Plan", ""]
        for step in self.steps:
            icon = {StepStatus.PENDING: "⏳", StepStatus.RUNNING: "⚡",
                    StepStatus.DONE: "✅", StepStatus.FAILED: "❌"}[step.status]
            hints = ", ".join(step.tool_hints) if step.tool_hints else "none"
            lines.append(
                f"{icon} **Step {step.index}**: {step.description} "
                f"(tools: {hints})"
            )
            if step.expected_output:
                lines.append(f"   → Expected: {step.expected_output}")

        lines.append("")
        lines.append(
            "Follow the plan step by step. Complete each step before moving to the next. "
            "Use the suggested tools where appropriate."
        )
        return "\n".join(lines)


# ─── Planner ──────────────────────────────────────────────────────────────────


class Planner:
    """Strategic planner that evaluates queries and generates structured plans.

    Uses LLM Structured Output (JSON Mode) to produce a deterministic schema.
    Trivial queries skip planning entirely. Complex queries get a 3–7 step plan.
    """

    def __init__(
        self,
        client: LLMClient,
        config: Config,
        *,
        max_steps: int = 7,
        planning_model: str | None = None,
    ):
        self._client = client
        self._cfg = config
        self._max_steps = max(1, min(max_steps, 10))
        self._planning_model = planning_model or config.model

    async def evaluate(self, query: str) -> Plan | None:
        """Evaluate a query and produce a plan if needed.

        Returns None for trivial queries, a Plan for complex ones.
        All exceptions are logged and return None (graceful degradation).
        """
        if not query.strip():
            return None

        try:
            result = await self._client.chat(
                messages=[
                    {"role": "system", "content": _PLANNER_SYSTEM},
                    {"role": "user", "content": query},
                ],
                thinking=ThinkingMode.DISABLED,
                reasoning_effort=ReasoningEffort.LOW,
                max_tokens=1024,
                response_format=_PLAN_SCHEMA,
            )

            content = result.content or ""
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                log.warning("planner: invalid JSON from model, skipping plan")
                return None

            if not data.get("needs_planning", False):
                log.debug("planner: query classified as trivial, no plan needed")
                return None

            steps_data = data.get("steps", [])
            if not steps_data:
                return None

            steps: list[PlanStep] = []
            for sd in steps_data[: self._max_steps]:
                steps.append(PlanStep(
                    index=int(sd.get("index", len(steps) + 1)),
                    description=str(sd.get("description", "")),
                    tool_hints=[str(h) for h in sd.get("tool_hints", [])],
                    expected_output=str(sd.get("expected_output", "")),
                ))

            # Re-index to ensure consistency
            for i, step in enumerate(steps, 1):
                step.index = i

            plan = Plan(steps=steps, reasoning=str(data.get("reasoning", "")))
            log.info(
                "planner: generated %d-step plan for query %r",
                len(steps), query[:80],
            )
            return plan

        except Exception as exc:
            log.error("planner: evaluation failed for query %r: %s", query[:80], exc)
            return None
