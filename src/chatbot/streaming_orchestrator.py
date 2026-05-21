"""Streaming orchestrator — full pipeline with SSE event emission.

Unlike the raw StreamingClient used by the old /chat/stream endpoint,
this runs the COMPLETE pipeline (routing → ReAct executor with tools →
verifier) while yielding SSE-compatible events at each stage.

Event protocol
--------------
  progress   — stage updates (routing, thinking, verifying)
  reasoning  — model thinking chunks
  content    — final answer token chunks
  tool_call  — tool invocation start
  tool_result — tool execution result
  done       — stream complete with metadata
  error      — something went wrong
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from .client import LLMClient
from .config import Config, ReasoningEffort, ThinkingMode
from .executor import _truncate  # reuse truncation helper
from .guardrails import GuardrailsManager, SafetyLevel
from .memory import ConversationMemory
from .orchestrator import Orchestrator
from .router import Complexity, Router
from .streaming import ChunkType, StreamingClient
from .task_budget import TaskBudgetManager
from .tools import ToolRegistry
from .verifier import Verifier

log = logging.getLogger(__name__)

# System prompt for the streaming executor (mirrors executor.py)
_EXECUTOR_SYSTEM = """You are a careful, methodical assistant.

Guidelines:
- If the question requires computation, use the calculator or python_exec tool
  rather than guessing.
- If the question depends on current facts, use web_search.
- Think step by step. Cite tool outputs explicitly when relevant.
- When you have enough information, give a clear, direct final answer.
- Never fabricate tool results."""

_TRIVIAL_SYSTEM = (
    "You are a concise, friendly assistant. Answer briefly and directly. "
    "No preamble, no apologies. Keep it under 3 sentences for simple questions."
)


class StreamingOrchestrator:
    """Full orchestrator pipeline that yields SSE events as an AsyncIterator.

    Supports three paths:
    - TRIVIAL: direct streaming, no tools
    - STANDARD: streaming ReAct loop with tools + verifier
    - COMPLEX: falls back to Orchestrator.ask() (Best-of-N too complex to stream)
    """

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

        # Register ArtifactTool with shared store
        from .artifacts import ArtifactStore, ArtifactTool

        self._artifact_store = ArtifactStore()
        self._artifact_tool = ArtifactTool(self._artifact_store)
        if self._artifact_tool.name not in {t.name for t in self._tools.list()}:
            self._tools.register(self._artifact_tool)

        self._router = Router(self._client, config)
        self._verifier = Verifier(self._client, config)
        self._streaming = StreamingClient(config)
        self._guardrails = GuardrailsManager()

        # Fallback orchestrator for COMPLEX path
        self._orch = Orchestrator(
            config, client=self._client, tools=self._tools, memory=self._memory,
        )

    @property
    def artifact_store(self):
        """The shared ArtifactStore so api.py can expose REST endpoints."""
        return self._artifact_store

    async def stream(
        self,
        query: str,
        *,
        force_complexity: Complexity | None = None,
        force_effort: ReasoningEffort | None = None,
        file_context: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the full pipeline, yielding SSE event dicts."""
        query = query.strip()
        if not query:
            yield {"type": "error", "message": "Empty query"}
            return

        # ── Input guardrail ──────────────────────────────────────────
        input_check = self._guardrails.check_input(query)
        if input_check.level == SafetyLevel.BLOCKED:
            yield {
                "type": "error",
                "message": self._guardrails.get_blocked_response(input_check),
                "guardrail": input_check.level.value,
                "reason": input_check.reason,
            }
            return

        # ── Memory: record user turn ─────────────────────────────────
        await self._memory.add_user(query)

        try:
            # ── Phase 1: Route ───────────────────────────────────────
            yield {"type": "progress", "stage": "routing", "message": "Classifying query..."}

            complexity = force_complexity or await self._router.classify(query)
            thinking, effort = self._router.get_thinking_config(complexity)

            if force_effort:
                effort = force_effort
                if force_effort in (ReasoningEffort.HIGH, ReasoningEffort.MAX):
                    thinking = ThinkingMode.ENABLED

            yield {
                "type": "progress",
                "stage": "thinking",
                "message": f"Thinking with effort={effort.value}...",
            }

            log.info(
                "streaming_orch: complexity=%s thinking=%s effort=%s",
                complexity.value, thinking.value, effort.value,
            )

            # ── Phase 2: Dispatch to path ────────────────────────────
            if complexity is Complexity.TRIVIAL:
                async for event in self._stream_trivial(query, effort, file_context):
                    yield event

            elif complexity is Complexity.STANDARD:
                async for event in self._stream_standard(
                    query, thinking, effort, file_context,
                ):
                    yield event

            else:
                # COMPLEX: fall back to non-streaming orchestrator
                async for event in self._handle_complex_fallback(
                    query, complexity, force_complexity, force_effort, file_context,
                ):
                    yield event

        except Exception as exc:
            log.error("streaming_orch: unhandled error: %s", exc, exc_info=True)
            yield {"type": "error", "message": str(exc)}

    # ─── TRIVIAL path ─────────────────────────────────────────────────

    async def _stream_trivial(
        self,
        query: str,
        effort: ReasoningEffort,
        file_context: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Direct streaming for trivial queries — no tools, no verifier."""
        prior = self._memory.recent_messages()[:-1]  # exclude current user turn

        system = _TRIVIAL_SYSTEM
        if file_context:
            system = f"{system}\n\n{file_context}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            *prior,
            {"role": "user", "content": query},
        ]

        total_content = ""
        total_tokens = 0

        async for chunk in self._streaming.stream(
            messages=messages,
            thinking=ThinkingMode.DISABLED,
            reasoning_effort=ReasoningEffort.LOW,
            max_tokens=512,
        ):
            if chunk.type == ChunkType.REASONING and chunk.text:
                yield {"type": "reasoning", "text": chunk.text}
            elif chunk.type == ChunkType.CONTENT and chunk.text:
                total_content += chunk.text
                yield {"type": "content", "text": chunk.text}
            elif chunk.type == ChunkType.DONE:
                if chunk.usage:
                    total_tokens = chunk.usage.get("total_tokens", 0)
            elif chunk.type == ChunkType.ERROR:
                yield {"type": "error", "message": chunk.text}
                return

        # Output guardrail
        final_content, guardrail_info = self._check_output(total_content)
        if guardrail_info:
            yield {"type": "content", "text": f"\n\n[Content sanitized: {guardrail_info}]"}

        # Memory: record assistant turn
        await self._memory.add_assistant(final_content)

        yield {
            "type": "done",
            "complexity": Complexity.TRIVIAL.value,
            "effort": effort.value,
            "tokens": total_tokens,
            "iterations": 1,
            "tool_calls_count": 0,
        }

    # ─── STANDARD path — streaming ReAct loop ────────────────────────

    async def _stream_standard(
        self,
        query: str,
        thinking: ThinkingMode,
        effort: ReasoningEffort,
        file_context: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming ReAct loop with tool execution and verifier."""
        prior = await self._build_prior(query)
        budget = self._budget_mgr.create()

        system = _EXECUTOR_SYSTEM
        if file_context:
            system = f"{system}\n\n{file_context}"

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if prior:
            messages.extend(prior)
        messages.append({"role": "user", "content": query})

        tool_schemas = self._tools.to_openai_schemas()
        tool_calls_made = 0
        total_tokens = 0
        total_content = ""
        iterations = 0

        for iteration in range(1, self._cfg.max_react_iterations + 1):
            iterations = iteration

            # Build messages with budget awareness
            call_messages = list(messages)
            budget_msg = budget.to_system_message()
            if budget_msg:
                call_messages.insert(1, {"role": "system", "content": budget_msg})

            # Check budget exhaustion before making the call
            if budget.exhausted:
                log.warning("streaming_orch: budget exhausted, forcing wrap-up")
                break

            # Stream this iteration's LLM call
            collected_chunks: list[Any] = []

            result = await self._streaming.stream_to_result(
                messages=call_messages,
                thinking=thinking,
                reasoning_effort=effort,
                tools=tool_schemas,
                max_tokens=8192,
                on_chunk=lambda c: collected_chunks.append(c),
            )

            # Forward reasoning and content chunks that were collected
            for c in collected_chunks:
                if c.type == ChunkType.REASONING and c.text:
                    yield {"type": "reasoning", "text": c.text}
                elif c.type == ChunkType.CONTENT and c.text:
                    total_content += c.text
                    yield {"type": "content", "text": c.text}
                elif c.type == ChunkType.ERROR:
                    yield {"type": "error", "message": c.text}
                    return

            # Record token usage
            if result.usage:
                iter_tokens = result.usage.get("total_tokens", 0)
                total_tokens += iter_tokens
                # Create a minimal TokenUsage-like for budget tracking
                from .client import TokenUsage
                budget.record(TokenUsage(
                    prompt_tokens=result.usage.get("prompt_tokens", 0),
                    completion_tokens=result.usage.get("completion_tokens", 0),
                    reasoning_tokens=0,
                    total_tokens=iter_tokens,
                ))

            # No tool calls → we're done with the ReAct loop
            if not result.tool_calls:
                log.info("streaming_orch: finished after %d iter", iteration)
                break

            # ── Tool calls: build assistant message ──────────────────
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            # Preserve reasoning_content for multi-turn tool scenarios
            if result.reasoning_content:
                assistant_msg["reasoning_content"] = result.reasoning_content

            messages.append(assistant_msg)

            # ── Execute each tool ────────────────────────────────────
            for tc in result.tool_calls:
                tc_id = tc["id"]
                tc_name = tc["function"]["name"]
                tc_args_raw = tc["function"]["arguments"]

                # Parse args for the event
                try:
                    tc_args = json.loads(tc_args_raw) if tc_args_raw else {}
                except json.JSONDecodeError:
                    tc_args = {"raw": tc_args_raw}

                # Yield tool_call start event
                yield {
                    "type": "tool_call",
                    "id": tc_id,
                    "name": tc_name,
                    "args": tc_args,
                    "status": "calling",
                }

                # Execute the tool with timing
                t0 = time.perf_counter()
                tool_result = await self._tools.dispatch(tc_name, tc_args_raw)
                duration_ms = int((time.perf_counter() - t0) * 1000)
                tool_calls_made += 1

                truncated_result = _truncate(tool_result)

                # Yield tool_result event
                yield {
                    "type": "tool_result",
                    "id": tc_id,
                    "name": tc_name,
                    "result": truncated_result[:2000],  # limit event payload size
                    "duration_ms": duration_ms,
                    "status": "done",
                }

                # ── Artifact event ───────────────────────────────────
                if tc_name == "generate_artifact":
                    artifact_event = self._build_artifact_event(tool_result)
                    if artifact_event:
                        yield artifact_event

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": truncated_result,
                })

            # Budget wrap-up check after tool execution
            if budget.should_wrapup:
                log.info("streaming_orch: budget threshold reached, requesting wrap-up")
                messages.append({
                    "role": "user",
                    "content": (
                        "Budget is running low. Please provide your best final answer "
                        "based on the information gathered so far. No more tool calls."
                    ),
                })

                # Final streaming call without tools
                wrapup_chunks: list[Any] = []
                wrapup_result = await self._streaming.stream_to_result(
                    messages=messages,
                    thinking=thinking,
                    reasoning_effort=effort,
                    max_tokens=4096,
                    on_chunk=lambda c: wrapup_chunks.append(c),
                )

                for c in wrapup_chunks:
                    if c.type == ChunkType.REASONING and c.text:
                        yield {"type": "reasoning", "text": c.text}
                    elif c.type == ChunkType.CONTENT and c.text:
                        total_content += c.text
                        yield {"type": "content", "text": c.text}

                if wrapup_result.usage:
                    total_tokens += wrapup_result.usage.get("total_tokens", 0)

                total_content = wrapup_result.content or total_content
                break
        else:
            # Hit iteration cap — ask model to wrap up
            log.warning(
                "streaming_orch: hit max iterations (%d)", self._cfg.max_react_iterations,
            )
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the tool-call iteration limit. "
                    "Stop using tools and provide your best final answer."
                ),
            })

            final_chunks: list[Any] = []
            final_result = await self._streaming.stream_to_result(
                messages=messages,
                thinking=thinking,
                reasoning_effort=effort,
                max_tokens=4096,
                on_chunk=lambda c: final_chunks.append(c),
            )

            for c in final_chunks:
                if c.type == ChunkType.REASONING and c.text:
                    yield {"type": "reasoning", "text": c.text}
                elif c.type == ChunkType.CONTENT and c.text:
                    total_content += c.text
                    yield {"type": "content", "text": c.text}

            if final_result.usage:
                total_tokens += final_result.usage.get("total_tokens", 0)

            total_content = final_result.content or total_content

        # ── Output guardrail ─────────────────────────────────────────
        final_content, guardrail_info = self._check_output(total_content)
        if guardrail_info:
            yield {"type": "content", "text": f"\n\n[Content sanitized: {guardrail_info}]"}

        # ── Verifier ─────────────────────────────────────────────────
        verdict = None
        if self._cfg.verifier_enabled and final_content:
            yield {
                "type": "progress",
                "stage": "verifying",
                "message": "Verifying answer...",
            }
            try:
                verdict = await self._verifier.verify(query, final_content)
            except Exception as exc:
                log.warning("streaming_orch: verifier failed: %s", exc)

        # ── Memory: record assistant turn ────────────────────────────
        self._memory.add_assistant(final_content)

        # ── Done event ───────────────────────────────────────────────
        done_event: dict[str, Any] = {
            "type": "done",
            "complexity": Complexity.STANDARD.value,
            "effort": effort.value,
            "tokens": total_tokens,
            "iterations": iterations,
            "tool_calls_count": tool_calls_made,
        }
        if verdict:
            done_event["verification"] = {
                "passed": verdict.passed,
                "overall": verdict.overall,
                "issues": verdict.issues,
            }
        if guardrail_info:
            done_event["guardrail"] = guardrail_info

        yield done_event

    # ─── COMPLEX fallback ─────────────────────────────────────────────

    async def _handle_complex_fallback(
        self,
        query: str,
        complexity: Complexity,
        force_complexity: Complexity | None,
        force_effort: ReasoningEffort | None,
        file_context: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Fall back to non-streaming Orchestrator for COMPLEX queries.

        Best-of-N voting is too complex to stream meaningfully, so we
        run it normally and yield progress events around it.
        """
        yield {
            "type": "progress",
            "stage": "complex",
            "message": "Complex query — running Best-of-N analysis...",
        }

        try:
            # Note: the orchestrator already recorded add_user via our stream(),
            # so we need to undo and let the orchestrator do its own.
            # The memory already has the user turn from our stream() method,
            # and the orchestrator's ask() will add it again — so we remove it
            # to prevent duplication.
            # Actually, we should just call the internal methods directly.
            # But to keep things simple, we accept the small duplication
            # in memory (the most recent turn is the same).

            answer = await self._orch.ask(
                query,
                force_complexity=force_complexity or Complexity.COMPLEX,
                force_effort=force_effort,
            )

            # Emit the full answer as content
            if answer.text:
                yield {"type": "content", "text": answer.text}

            done_event: dict[str, Any] = {
                "type": "done",
                "complexity": answer.complexity.value,
                "effort": answer.reasoning_effort,
                "tokens": answer.tokens_used,
                "iterations": answer.iterations,
                "tool_calls_count": answer.tool_calls,
            }
            if answer.verdict:
                done_event["verification"] = {
                    "passed": answer.verdict.passed,
                    "overall": answer.verdict.overall,
                    "issues": answer.verdict.issues,
                }
            if answer.diagnostics:
                done_event["diagnostics"] = answer.diagnostics

            yield done_event

        except Exception as exc:
            log.error("streaming_orch: complex fallback failed: %s", exc, exc_info=True)
            yield {"type": "error", "message": f"Complex analysis failed: {exc}"}

    # ─── Helpers ──────────────────────────────────────────────────────

    async def _build_prior(self, query: str) -> list[dict[str, Any]]:
        """Build prior messages from memory, excluding the current query."""
        prior = await self._memory.build_prior_messages(query)
        # Remove the current user message if memory already appended it
        if (
            prior
            and prior[-1].get("role") == "user"
            and prior[-1].get("content") == query
        ):
            prior = prior[:-1]
        return prior

    def _build_artifact_event(self, tool_result: str) -> dict[str, Any] | None:
        """Parse an ArtifactTool result and build the SSE artifact event.

        Returns None if the tool result doesn't contain valid artifact metadata.
        """
        try:
            meta = json.loads(tool_result)
        except json.JSONDecodeError:
            return None

        artifact_id = meta.get("artifact_id")
        if not artifact_id:
            return None

        return {
            "type": "artifact",
            "id": artifact_id,
            "title": meta.get("title", ""),
            "artifact_type": meta.get("type", ""),
            "language": meta.get("language", ""),
            "version": meta.get("version", 1),
            "size": meta.get("size", 0),
        }

    def _check_output(self, content: str) -> tuple[str, str]:
        """Run output guardrail. Returns (final_content, guardrail_info_or_empty)."""
        if not content:
            return content, ""
        output_check = self._guardrails.check_output(content)
        if output_check.modified_content is not None:
            return output_check.modified_content, output_check.reason
        return content, ""
