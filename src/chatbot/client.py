"""DeepSeek V4 Pro API client with native thinking mode support.

This wraps the OpenAI SDK pointed at DeepSeek's endpoint. The critical
difference from a vanilla OpenAI wrapper:

1. `thinking` parameter  — {"type": "enabled"} or {"type": "disabled"}
2. `reasoning_effort`    — "low" / "medium" / "high" / "max"
3. Response includes `reasoning_content` field alongside `content`
4. In thinking mode: temperature/top_p are IGNORED (DeepSeek docs say set
   to 1.0 for safety, but they won't affect output)

The response shape from DeepSeek V4:
    choice.message.content           → final answer
    choice.message.reasoning_content → chain-of-thought (may be None)
    choice.message.tool_calls        → tool call requests (standard format)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config, ReasoningEffort, ThinkingMode

log = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Parsed response from DeepSeek V4 Pro."""
    content: str
    reasoning_content: str | None
    tool_calls: list[Any]
    usage: TokenUsage
    raw: Any  # the full response object


@dataclass
class TokenUsage:
    """Token accounting for budget tracking."""
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int

    @classmethod
    def from_response(cls, resp: Any) -> TokenUsage:
        usage = resp.usage
        # DeepSeek V4 reports reasoning tokens in completion_tokens_details
        reasoning = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        return cls(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            reasoning_tokens=reasoning,
            total_tokens=usage.total_tokens or 0,
        )


class LLMClient:
    """Async client for DeepSeek V4 Pro with thinking mode support."""

    def __init__(self, config: Config):
        self._cfg = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        thinking: ThinkingMode = ThinkingMode.ENABLED,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int = 8192,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Single chat completion with thinking mode control.

        Key differences from vanilla OpenAI:
        - `extra_body` carries the thinking config
        - `reasoning_effort` controls depth of CoT
        - temperature is fixed at 1.0 (DeepSeek ignores it in thinking mode)
        """
        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1.0,  # DeepSeek recommendation; ignored in thinking mode
            "top_p": 1.0,        # DeepSeek recommendation; ignored in thinking mode
        }

        # Thinking mode config via extra_body
        extra_body: dict[str, Any] = {
            "thinking": {"type": thinking.value},
        }
        if thinking == ThinkingMode.ENABLED:
            kwargs["reasoning_effort"] = reasoning_effort.value

        kwargs["extra_body"] = extra_body

        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        log.debug(
            "chat: model=%s thinking=%s effort=%s msgs=%d",
            self._cfg.model, thinking.value,
            reasoning_effort.value if thinking == ThinkingMode.ENABLED else "n/a",
            len(messages),
        )

        resp = await self._client.chat.completions.create(**kwargs)
        return self._parse_response(resp)

    def _parse_response(self, resp: Any) -> ChatResponse:
        choice = resp.choices[0]
        msg = choice.message

        return ChatResponse(
            content=(msg.content or "").strip(),
            reasoning_content=getattr(msg, "reasoning_content", None),
            tool_calls=list(msg.tool_calls) if msg.tool_calls else [],
            usage=TokenUsage.from_response(resp),
            raw=resp,
        )

    async def chat_text(
        self,
        *,
        messages: list[dict[str, Any]],
        thinking: ThinkingMode = ThinkingMode.ENABLED,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        max_tokens: int = 8192,
    ) -> str:
        """Convenience: return only the final answer text."""
        resp = await self.chat(
            messages=messages,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        return resp.content

    async def chat_many(
        self,
        *,
        messages_variants: list[list[dict[str, Any]]],
        thinking: ThinkingMode = ThinkingMode.ENABLED,
        reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH,
        max_tokens: int = 8192,
    ) -> list[ChatResponse]:
        """Sample N answers in parallel using message variants for diversity.

        Since DeepSeek V4 ignores temperature in thinking mode, we achieve
        diversity through slightly different system prompts per sample
        (handled by the Voter, not here).
        """
        tasks = [
            self.chat(
                messages=msgs,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
            )
            for msgs in messages_variants
        ]
        return list(await asyncio.gather(*tasks))
