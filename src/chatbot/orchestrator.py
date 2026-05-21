"""Top-level pipeline: query → guardrails → classify → execute → verify → correct → answer.

FULLY INTEGRATED pipeline using ALL modules. This is what makes the chatbot
feel like Opus 4.7 — not just the model, but the engineering around it.

Flow:
    1. Guardrails check input
    2. Router classifies → TRIVIAL / STANDARD / COMPLEX
    3. Structured memory injects relevant facts/preferences
    4. Progress events emitted at each stage
    5. Path:
       - TRIVIAL  → direct call, thinking OFF, no tools
       - STANDARD → Executor (ReAct + tools + scratchpad + tool_clearing)
                    → Verifier → Self-correction if failed
       - COMPLEX  → Best-of-N voting (each with full executor pipeline)
    6. Guardrails check output (PII redaction)
    7. Memory updated with final answer
    8. Cache stats recorded
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .cache import PrefixOptimizer, record_cache_usage, get_cache_stats
from .client import LLMClient
from .compaction import ContextCompactor
from .config import Config, ReasoningEffort, ThinkingMode
from .executor import Executor
from .guardrails import GuardrailsManager, SafetyLevel
from .memory import ConversationMemory
from .progress import ProgressTracker, ProgressStage
from .router import Complexity, Router
from .scratchpad import Scratchpad, ScratchpadTool
from .self_correction import SelfCorrector
from .structured_memory import StructuredMemory, MemoryTool
from .task_budget import TaskBudgetManager
from .tool_search import select_tools, tools_to_schemas
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
    corrected: bool = False
    blocked: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


_TRIVIAL_SYSTEM = (
    "You are a concise, friendly assistant. Answer briefly and directly. "
    "No preamble, no apologies. Keep it under 3 sentences for simple questions."
)


class Orchestrator:
    """The brain of the chatbot — wires ALL modules together."""

    def __init__(
        self,
        config: Config,
        *,
        client: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        structured_memory: StructuredMemory | None = None,
        progress: ProgressTracker | None = None,
    ):
        self._cfg = config
        self._client = client or LLMClient(config)
        self._tools = tools or ToolRegistry.default()
        self._memory = memory or ConversationMemory()
        self._structured_memory = structured_memory or StructuredMemory(
            store_path=".akaradje_memory/structured.json"
        )
        self._progress = progress or ProgressTracker()
        self._budget_mgr = TaskBudgetManager(config)
        self._guardrails = GuardrailsManager(strict=False, redact_pii=True)
        self._compactor = ContextCompactor(self._client, config)

        # Register memory tool into the registry
        memory_tool = MemoryTool(self._structured_memory)
        if self._tools.get("memory") is None:
            self._tools._tools["memory"] = memory_tool

        # Core pipeline components
        self._router = Router(self._client, config)
        self._executor = Executor(self._client, config, self._tools)
        self._verifier = Verifier(self._client, config)
        self._voter = Voter(self._executor, self._verifier, config, self._budget_mgr)
        self._corrector = SelfCorrector(self._client, self._verifier, config)

        # Prefix optimizer for cache hits
        self._prefix_optimizer = PrefixOptimizer(
            system_prompt=self._executor._system,
            tool_schemas=self._tools.to_openai_schemas(),
        )

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    @property
    def structured_memory(self) -> StructuredMemory:
        return self._structured_memory

    @property
    def progress(self) -> ProgressTracker:
        return self._progress

    async def ask(
        self,
        query: str,
        *,
        force_complexity: Complexity | None = None,
        force_effort: ReasoningEffort | None = None,
    ) -> Answer:
        query = query.strip()
        if not query:
            return Answer(
                text="", complexity=Complexity.TRIVIAL,
                thinking_mode="disabled", reasoning_effort="low",
            )

        # ═══ STEP 1: Input Guardrails ═══
        safety = self._guardrails.check_input(query)
        if safety.level == SafetyLevel.BLOCKED:
            await self._progress.emit(ProgressStage.ERROR, f"Blocked: {safety.reason}")
            return Answer(
                text=self._guardrails.get_blocked_response(safety),
                complexity=Complexity.TRIVIAL,
                thinking_mode="disabled",
                reasoning_effort="n/a",
                blocked=True,
                diagnostics={"path": "blocked", "reason": safety.reason},
            )

        self._memory.add_user(query)

        # ═══ STEP 2: Route ═══
        await self._progress.emit(ProgressStage.ROUTING, "Classifying query complexity...")
        complexity = force_complexity or await self._router.classify(query)
        thinking, effort = self._router.get_thinking_config(complexity)

        if force_effort:
            effort = force_effort
            if force_effort in (ReasoningEffort.HIGH, ReasoningEffort.MAX):
                thinking = ThinkingMode.ENABLED

        log.info("orchestrator: complexity=%s thinking=%s effort=%s",
                 complexity.value, thinking.value, effort.value)

        # ═══ STEP 3: Dispatch ═══
        if complexity is Complexity.TRIVIAL:
            answer = await self._handle_trivial(query)
        elif complexity is Complexity.STANDARD:
            answer = await self._handle_standard(query, thinking, effort)
        else:
            answer = await self._handle_complex(query, thinking, effort)

        # ═══ STEP 4: Output Guardrails (PII redaction) ═══
        output_safety = self._guardrails.check_output(answer.text)
        if output_safety.modified_content:
            answer.text = output_safety.modified_content
            answer.diagnostics["pii_redacted"] = True

        # ═══ STEP 5: Update memories ═══
        self._memory.add_assistant(answer.text)

        await self._progress.emit(
            ProgressStage.DONE, "Complete",
            tokens_so_far=answer.tokens_used,
        )
        return answer

    async def _handle_trivial(self, query: str) -> Answer:
        prior = self._memory.recent_messages()[:-1]

        # Inject structured memory even for trivial (e.g., preferences like language)
        mem_msg = self._structured_memory.to_system_message(query)
        messages = [{"role": "system", "content": _TRIVIAL_SYSTEM}]
        if mem_msg:
            messages.append(mem_msg)
        messages.extend(prior)
        messages.append({"role": "user", "content": query})

        text = await self._client.chat_text(
            messages=messages,
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
        self, query: str, thinking: ThinkingMode, effort: ReasoningEffort
    ) -> Answer:
        await self._progress.emit(
            ProgressStage.THINKING,
            f"Reasoning with effort={effort.value}...",
        )

        prior = self._build_prior(query)
        budget = self._budget_mgr.create()

        # Dynamic tool selection — only send relevant tools
        selected_tools = select_tools(query, self._tools, max_tools=5)
        tool_schemas = tools_to_schemas(selected_tools)

        result = await self._executor.run(
            query,
            thinking=thinking,
            reasoning_effort=effort,
            budget=budget,
            prior_messages=prior,
            tool_schemas_override=tool_schemas,
        )

        # ═══ Verify ═══
        await self._progress.emit(ProgressStage.VERIFYING, "Checking answer quality...")
        verdict = await self._verifier.verify(query, result.answer)

        # ═══ Self-correction if failed ═══
        final_answer = result.answer
        corrected = False
        if not verdict.passed:
            await self._progress.emit(
                ProgressStage.CORRECTING,
                f"Answer scored {verdict.overall}/10, attempting correction...",
            )
            correction = await self._corrector.correct_if_needed(
                query, result.answer, verdict, current_effort=effort,
            )
            if correction.improved:
                final_answer = correction.final_answer
                verdict = correction.final_verdict
                corrected = True

        return Answer(
            text=final_answer,
            complexity=Complexity.STANDARD,
            thinking_mode=thinking.value,
            reasoning_effort=effort.value,
            verdict=verdict,
            iterations=result.iterations,
            tool_calls=result.tool_calls_made,
            tokens_used=result.tokens_used,
            reasoning_trace=result.reasoning_trace,
            corrected=corrected,
            diagnostics={"path": "standard-executor+verifier+correction"},
        )

    async def _handle_complex(
        self, query: str, thinking: ThinkingMode, effort: ReasoningEffort
    ) -> Answer:
        await self._progress.emit(
            ProgressStage.VOTING,
            f"Sampling {self._cfg.best_of_n} candidates with effort={effort.value}...",
        )

        prior = self._build_prior(query)
        voting = await self._voter.vote(
            query,
            thinking=thinking,
            reasoning_effort=effort,
            prior_messages=prior,
        )
        winner = voting.winner
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
            diagnostics={
                "path": "complex-best-of-n",
                "n_candidates": voting.n,
                "scores": [round(c.score, 2) for c in voting.candidates],
            },
        )

    def _build_prior(self, query: str) -> list[dict[str, Any]]:
        """Build prior context: structured memory + conversation recall."""
        prior: list[dict[str, Any]] = []

        # Structured memory (facts, preferences, instructions)
        mem_msg = self._structured_memory.to_system_message(query)
        if mem_msg:
            prior.append(mem_msg)

        # Conversation recall + recent window
        conv_prior = self._memory.build_prior_messages(query)
        if conv_prior and conv_prior[-1].get("role") == "user" and conv_prior[-1].get("content") == query:
            conv_prior = conv_prior[:-1]
        prior.extend(conv_prior)

        return prior
