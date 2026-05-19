"""Streaming support for DeepSeek V4 API.

DeepSeek supports `stream: true` with SSE (Server-Sent Events). When
streaming with thinking mode enabled, the response includes:
- reasoning_content chunks (model thinking, before the final answer)
- content chunks (final answer, after reasoning is done)
- tool_calls chunks (if the model wants to call tools)

This module provides:
1. StreamingClient — yields typed chunks as they arrive
2. StreamBuffer — assembles chunks into a complete response
3. Integration point for CLI (progressive rendering) and FastAPI (SSE)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from .config import Config, ReasoningEffort, ThinkingMode

log = logging.getLogger(__name__)


class ChunkType(str, Enum):
    REASONING = "reasoning"    # thinking content (may be hidden)
    CONTENT = "content"        # final answer text
    TOOL_CALL = "tool_call"    # tool call delta
    DONE = "done"              # stream finished
    ERROR = "error"            # error occurred


@dataclass
class StreamChunk:
    """A single chunk from the streaming response."""
    type: ChunkType
    text: str = ""
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args_delta: str | None = None
    finish_reason: str | None = None


@dataclass
class StreamResult:
    """Assembled result from a complete stream."""
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None

    # Accumulated tool call buffers (for assembly during streaming)
    _tool_buffers: dict[int, dict[str, Any]] = field(default_factory=dict)

    def add_chunk(self, chunk: StreamChunk) -> None:
        if chunk.type == ChunkType.CONTENT:
            self.content += chunk.text
        elif chunk.type == ChunkType.REASONING:
            self.reasoning_content += chunk.text
        elif chunk.type == ChunkType.TOOL_CALL:
            # Tool calls come as deltas indexed by position
            idx = len(self._tool_buffers)
            if chunk.tool_call_id:
                # New tool call starting
                self._tool_buffers[idx] = {
                    "id": chunk.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": chunk.tool_name or "",
                        "arguments": chunk.tool_args_delta or "",
                    },
                }
            elif self._tool_buffers:
                # Continuation of existing tool call (append args)
                last_idx = max(self._tool_buffers.keys())
                self._tool_buffers[last_idx]["function"]["arguments"] += (
                    chunk.tool_args_delta or ""
                )
        elif chunk.type == ChunkType.DONE:
            self.finish_reason = chunk.finish_reason
            # Finalize tool calls
            self.tool_calls = list(self._tool_buffers.values())


class StreamingClient:
    """Async streaming client for DeepSeek V4 API."""

    def __init__(self, config: Config):
        self._cfg = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        thinking: ThinkingMode = ThinkingMode.ENABLED,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamChunk]:
        """Yield StreamChunks as they arrive from the API."""

        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": True,
        }

        extra_body: dict[str, Any] = {
            "thinking": {"type": thinking.value},
        }
        if thinking == ThinkingMode.ENABLED:
            kwargs["reasoning_effort"] = reasoning_effort.value

        kwargs["extra_body"] = extra_body

        if tools:
            kwargs["tools"] = tools

        try:
            stream = await self._client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Reasoning content (thinking)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield StreamChunk(type=ChunkType.REASONING, text=reasoning)

                # Regular content
                if delta.content:
                    yield StreamChunk(type=ChunkType.CONTENT, text=delta.content)

                # Tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        yield StreamChunk(
                            type=ChunkType.TOOL_CALL,
                            tool_call_id=tc_delta.id if tc_delta.id else None,
                            tool_name=(
                                tc_delta.function.name
                                if tc_delta.function and tc_delta.function.name
                                else None
                            ),
                            tool_args_delta=(
                                tc_delta.function.arguments
                                if tc_delta.function and tc_delta.function.arguments
                                else None
                            ),
                        )

                # Stream finished
                if finish_reason:
                    yield StreamChunk(
                        type=ChunkType.DONE,
                        finish_reason=finish_reason,
                    )

        except Exception as exc:
            log.error("streaming: error: %s", exc)
            yield StreamChunk(type=ChunkType.ERROR, text=str(exc))

    async def stream_to_result(
        self,
        *,
        messages: list[dict[str, Any]],
        thinking: ThinkingMode = ThinkingMode.ENABLED,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        on_chunk: Any | None = None,
    ) -> StreamResult:
        """Stream and assemble into a complete result.

        Args:
            on_chunk: Optional callback(StreamChunk) for progressive rendering.
        """
        result = StreamResult()
        async for chunk in self.stream(
            messages=messages,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            tools=tools,
            max_tokens=max_tokens,
        ):
            result.add_chunk(chunk)
            if on_chunk:
                await on_chunk(chunk) if asyncio.iscoroutinefunction(on_chunk) else on_chunk(chunk)
        return result
