"""Top-level pipeline that turns a user query into a final answer.

Flow:
    1. Router classifies the query (TRIVIAL / STANDARD / COMPLEX).
    2. Memory builds prior context.
    3. Path selection:
        - TRIVIAL  → direct fast-model reply, no tools, no verifier.
        - STANDARD → one Executor pass on the standard model, then verify.
        - COMPLEX  → Best-of-N voting on the deep model.
    4. Update memory with the final answer.

Returns a structured `Answer` so the CLI / API layer can show diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .client import LLMClient
from .config import Config
from .executor import Executor
from .memory import ConversationMemory
from .router import Complexity, Router
from .tools import ToolRegistry
from .verifier import VerificationResult, Verifier
from .voter import Voter, VotingResult

log = logging.getLogger(__name__)


@dataclass
class Answer:
    text: str
    complexity: Complexity
    model_used: str
    verdict: VerificationResult | None = None
    voting: VotingResult | None = None
    iterations: int = 0
    tool_calls: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


_TRIVIAL_SYSTEM = (
    "You are a concise, friendly assistant. Answer briefly and directly. "
    "No preamble, no apologies."
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

        self._router = Router(self._client, config)
        self._executor = Executor(self._client, config, self._tools)
        self._verifier = Verifier(self._client, config)
        self._voter = Voter(self._executor, self._verifier, config)

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    async def ask(
        self,
        query: str,
        *,
        force_complexity: Complexity | None = None,
    ) -> Answer:
        query = query.strip()
        if not query:
            return Answer(
                text="",
                complexity=Complexity.TRIVIAL,
                model_used=self._cfg.model_fast,
            )

        # Record user turn before classification so memory recall can see it.
        self._memory.add_user(query)

        complexity = force_complexity or await self._router.classify(query)
        log.info("orchestrator: complexity=%s", complexity.value)

        if complexity is Complexity.TRIVIAL:
            answer = await self._handle_trivial(query)
        elif complexity is Complexity.STANDARD:
            answer = await self._handle_standard(query)
        else:
            answer = await self._handle_complex(query)

        self._memory.add_assistant(answer.text)
        return answer

    # ----- per-tier handlers -------------------------------------------

    async def _handle_trivial(self, query: str) -> Answer:
        prior = self._memory.recent_messages()[:-1]  # exclude the just-added user turn
        text = await self._client.chat_text(
            model=self._cfg.model_fast,
            messages=[
                {"role": "system", "content": _TRIVIAL_SYSTEM},
                *prior,
                {"role": "user", "content": query},
            ],
            temperature=0.5,
            max_tokens=512,
        )
        return Answer(
            text=text,
            complexity=Complexity.TRIVIAL,
            model_used=self._cfg.model_fast,
            diagnostics={"path": "trivial-direct"},
        )

    async def _handle_standard(self, query: str) -> Answer:
        prior = self._build_prior_excluding_current(query)
        result = await self._executor.run(
            query,
            model=self._cfg.model_standard,
            prior_messages=prior,
        )
        verdict = await self._verifier.verify(query, result.answer)
        return Answer(
            text=result.answer,
            complexity=Complexity.STANDARD,
            model_used=self._cfg.model_standard,
            verdict=verdict,
            iterations=result.iterations,
            tool_calls=result.tool_calls_made,
            diagnostics={"path": "standard-executor+verifier"},
        )

    async def _handle_complex(self, query: str) -> Answer:
        prior = self._build_prior_excluding_current(query)
        voting = await self._voter.vote(
            query,
            model=self._cfg.model_deep,
            prior_messages=prior,
        )
        winner = voting.winner
        return Answer(
            text=winner.answer,
            complexity=Complexity.COMPLEX,
            model_used=self._cfg.model_deep,
            verdict=winner.verdict,
            voting=voting,
            iterations=winner.executor.iterations,
            tool_calls=winner.executor.tool_calls_made,
            diagnostics={
                "path": "complex-best-of-n",
                "n_candidates": voting.n,
                "scores": [round(c.score, 2) for c in voting.candidates],
            },
        )

    # ----- helpers ------------------------------------------------------

    def _build_prior_excluding_current(self, query: str) -> list[dict[str, Any]]:
        """Build prior_messages for the Executor.

        We strip the last user turn (the current query) because Executor.run
        will add it back as `query`. Otherwise we'd duplicate it.
        """
        prior = self._memory.build_prior_messages(query)
        if prior and prior[-1].get("role") == "user" and prior[-1].get("content") == query:
            prior = prior[:-1]
        return prior
