"""Top-level pipeline: query → classify → execute → verify → answer.

Architecture mirrors Claude Opus 4.7's approach but on DeepSeek V4 Pro:
- Single model with variable reasoning_effort (not multi-model)
- Task budget countdown (simulated, since DeepSeek has no native support)
- Interleaved thinking + tool use (native to V4)
- Verifier as separate judge context
- Best-of-N for COMPLEX queries (diversity via prompt variation)

Flow:
    1. Router classifies → TRIVIAL / STANDARD / COMPLEX
    2. Maps to (ThinkingMode, ReasoningEffort)
    3. Path:
       - TRIVIAL  → direct call, thinking OFF, no tools
       - STANDARD → Executor (ReAct + tools) + Verifier
       - COMPLEX  → Best-of-N voting
    4. Memory updated with final answer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .client import LLMClient
from .config import Config, ReasoningEffort, ThinkingMode
from .executor import Executor
from .memory import ConversationMemory
from .planner import Planner
from .router import Complexity, Router
from .task_budget import TaskBudgetManager
from .tools import ToolRegistry
from .verifier import VerificationResult, Verifier
from .voter import Voter, VotingResult

log = logging.getLogger(__name__)


@dataclass
class Answer:
    text: str
    complexity: Complexity
    thinking_mode: str
    reasoning_effort: str
    verdict: VerificationResult | None = None
    voting: VotingResult | None = None
    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    reasoning_trace: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


_TRIVIAL_SYSTEM = (
    "You are a concise, friendly assistant. Answer briefly and directly. "
    "No preamble, no apologies. Keep it under 3 sentences for simple questions."
)


class Orchestrator:
    def __init__(
        self,
        config: Config,
        *,
        client: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
    ):
        self._cfg = config
        self._client = client or LLMClient(config)
        self._tools = tools or ToolRegistry.default()
        self._memory = memory or ConversationMemory()
        self._budget_mgr = TaskBudgetManager(config)

        self._router = Router(self._client, config)
        self._planner = Planner(self._client, config)
        self._executor = Executor(self._client, config, self._tools)
        self._verifier = Verifier(self._client, config)
        self._voter = Voter(self._executor, self._verifier, config, self._budget_mgr)

    async def _build_plan_context(self, query: str, file_context: str | None) -> tuple[str | None, Any]:
        """Evaluate planning and merge plan context into file_context.

        Returns (merged_context, plan_or_None).
        """
        try:
            plan = await self._planner.evaluate(query)
        except Exception as exc:
            log.warning("orchestrator: planner failed (%s), continuing without plan", exc)
            return file_context, None

        if plan is None:
            return file_context, None

        plan_context = plan.to_context_string()
        merged = f"{plan_context}\n\n{file_context}" if file_context else plan_context
        return merged, plan

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    async def ask(
        self,
        query: str,
        *,
        force_complexity: Complexity | None = None,
        force_effort: ReasoningEffort | None = None,
        file_context: str | None = None,
    ) -> Answer:
        query = query.strip()
        if not query:
            return Answer(
                text="", complexity=Complexity.TRIVIAL,
                thinking_mode="disabled", reasoning_effort="low",
            )

        await self._memory.add_user(query)

        # Classify complexity
        complexity = force_complexity or await self._router.classify(query)
        thinking, effort = self._router.get_thinking_config(complexity)

        # Allow force override of effort
        if force_effort:
            effort = force_effort
            if force_effort in (ReasoningEffort.HIGH, ReasoningEffort.MAX):
                thinking = ThinkingMode.ENABLED

        log.info(
            "orchestrator: complexity=%s thinking=%s effort=%s",
            complexity.value, thinking.value, effort.value,
        )

        # Dispatch to appropriate path
        if complexity is Complexity.TRIVIAL:
            answer = await self._handle_trivial(query, file_context)
        elif complexity is Complexity.STANDARD:
            answer = await self._handle_standard(query, thinking, effort, file_context)
        else:
            answer = await self._handle_complex(query, thinking, effort, file_context)

        await self._memory.add_assistant(answer.text)
        return answer

    async def _handle_trivial(self, query: str, file_context: str | None = None) -> Answer:
        prior = self._memory.recent_messages()[:-1]
        system = _TRIVIAL_SYSTEM
        if file_context:
            system = f"{system}\n\n{file_context}"
        text = await self._client.chat_text(
            messages=[
                {"role": "system", "content": system},
                *prior,
                {"role": "user", "content": query},
            ],
            thinking=ThinkingMode.DISABLED,
            max_tokens=512,
        )
        return Answer(
            text=text,
            complexity=Complexity.TRIVIAL,
            thinking_mode="disabled",
            reasoning_effort="n/a",
            diagnostics={"path": "trivial-direct"},
        )

    async def _handle_standard(
        self, query: str, thinking: ThinkingMode, effort: ReasoningEffort,
        file_context: str | None = None,
    ) -> Answer:
        prior = await self._build_prior(query)
        budget = self._budget_mgr.create()

        # Planning phase — generate structured plan for non-trivial tasks
        merged_context, plan = await self._build_plan_context(query, file_context)

        result = await self._executor.run(
            query,
            thinking=thinking,
            reasoning_effort=effort,
            budget=budget,
            prior_messages=prior,
            file_context=merged_context,
        )
        verdict = await self._verifier.verify(query, result.answer)

        diagnostics: dict[str, Any] = {"path": "standard-executor+verifier"}
        if plan is not None:
            diagnostics["plan"] = plan.to_dict()

        return Answer(
            text=result.answer,
            complexity=Complexity.STANDARD,
            thinking_mode=thinking.value,
            reasoning_effort=effort.value,
            verdict=verdict,
            iterations=result.iterations,
            tool_calls=result.tool_calls_made,
            tokens_used=result.tokens_used,
            reasoning_trace=result.reasoning_trace,
            diagnostics=diagnostics,
        )

    async def _handle_complex(
        self, query: str, thinking: ThinkingMode, effort: ReasoningEffort,
        file_context: str | None = None,
    ) -> Answer:
        prior = await self._build_prior(query)

        # Planning phase
        merged_context, plan = await self._build_plan_context(query, file_context)

        voting = await self._voter.vote(
            query,
            thinking=thinking,
            reasoning_effort=effort,
            prior_messages=prior,
            file_context=merged_context,
        )
        winner = voting.winner

        diagnostics: dict[str, Any] = {
            "path": "complex-best-of-n",
            "n_candidates": voting.n,
            "scores": [round(c.score, 2) for c in voting.candidates],
        }
        if plan is not None:
            diagnostics["plan"] = plan.to_dict()

        return Answer(
            text=winner.answer,
            complexity=Complexity.COMPLEX,
            thinking_mode=thinking.value,
            reasoning_effort=effort.value,
            verdict=winner.verdict,
            voting=voting,
            iterations=winner.executor.iterations,
            tool_calls=winner.executor.tool_calls_made,
            tokens_used=sum(c.executor.tokens_used for c in voting.candidates),
            reasoning_trace=winner.executor.reasoning_trace,
            diagnostics=diagnostics,
        )

    async def _build_prior(self, query: str) -> list[dict[str, Any]]:
        prior = await self._memory.build_prior_messages(query)
        if prior and prior[-1].get("role") == "user" and prior[-1].get("content") == query:
            prior = prior[:-1]
        return prior
