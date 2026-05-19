"""Thin wrapper around the OpenAI-compatible DeepSeek API.

We keep this layer deliberately small. Everything else in the codebase calls
into `LLMClient.chat()` so swapping providers is a one-file change.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config

log = logging.getLogger(__name__)


class LLMClient:
    """Async wrapper that talks to any OpenAI-compatible endpoint."""

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
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        """Single chat completion. Returns the raw OpenAI response object."""

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        log.debug("chat: model=%s, msgs=%d, temp=%.2f", model, len(messages), temperature)
        return await self._client.chat.completions.create(**kwargs)

    async def chat_text(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Convenience: return only the first choice's text content."""
        resp = await self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    async def chat_many(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        n: int,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> list[str]:
        """Sample N answers in parallel. Used by Best-of-N voting.

        Note: we issue independent requests rather than relying on the API's
        `n` parameter, because not every OpenAI-compatible backend supports
        it and independent requests give us better diversity anyway.
        """
        tasks: Iterable[asyncio.Task[str]] = [
            asyncio.create_task(
                self.chat_text(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            for _ in range(n)
        ]
        return list(await asyncio.gather(*tasks))
