"""Context Caching — maximize DeepSeek prefix cache hit rate.

DeepSeek V4 automatically caches common prefixes on disk. Cache hit pricing
is 1/10 of cache miss ($0.028 vs $0.28 per 1M tokens for V4-Flash).

The trick: ensure the SAME prefix is sent on EVERY call. That means:
1. System prompt must be identical across calls (no dynamic content in it)
2. Tool definitions must be in the same order every time
3. Any "stable" context (e.g., persona, instructions) comes before dynamic content

This module reorders messages to maximize prefix stability.

Cache detection: the API response includes `prompt_cache_hit_tokens` and
`prompt_cache_miss_tokens` in usage. We track these for observability.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Accumulated cache statistics for observability."""
    total_calls: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        if total == 0:
            return 0.0
        return self.cache_hit_tokens / total

    @property
    def savings_estimate_usd(self) -> float:
        """Estimated savings from cache hits (V4-Flash pricing)."""
        # Cache hit: $0.028/M, Cache miss: $0.28/M → savings = hit_tokens * (0.28 - 0.028) / 1M
        return self.cache_hit_tokens * (0.28 - 0.028) / 1_000_000

    def record(self, usage: Any) -> None:
        """Extract cache stats from API response usage object."""
        self.total_calls += 1
        # DeepSeek reports these in usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens
        if hasattr(usage, "prompt_cache_hit_tokens"):
            self.cache_hit_tokens += getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        if hasattr(usage, "prompt_cache_miss_tokens"):
            self.cache_miss_tokens += getattr(usage, "prompt_cache_miss_tokens", 0) or 0


class PrefixOptimizer:
    """Ensures messages are ordered for maximum cache prefix reuse.

    Strategy:
    - "Stable prefix" = system prompt + tool schemas (same every call)
    - "Semi-stable" = recalled memory context (changes slowly)
    - "Dynamic" = recent turns + current query (changes every call)

    By keeping the stable prefix identical and at the front, DeepSeek's
    disk-based caching can skip recomputation for that entire block.
    """

    def __init__(self, system_prompt: str, tool_schemas: list[dict[str, Any]]):
        self._system_prompt = system_prompt
        self._tool_schemas = tool_schemas
        # Compute a hash of the stable prefix for debugging
        prefix_content = system_prompt + str(tool_schemas)
        self._prefix_hash = hashlib.md5(prefix_content.encode()).hexdigest()[:8]
        log.debug("cache: stable prefix hash=%s", self._prefix_hash)

    def optimize_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reorder messages to maximize prefix cache hits.

        Input messages may have system prompts scattered throughout.
        Output ensures:
        1. First message: the canonical system prompt (STABLE)
        2. Any additional system messages (semi-stable, e.g., memory recall)
        3. User/assistant turns (dynamic)

        The canonical system prompt is ALWAYS the same text, regardless of
        what was passed in messages[0]. This ensures prefix stability.
        """
        # Separate message types
        system_msgs: list[dict[str, Any]] = []
        conversation_msgs: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                conversation_msgs.append(msg)

        # Build optimized sequence:
        # 1. Canonical system prompt (replaces whatever was first)
        # 2. Additional system context (memory recall, budget, etc.)
        # 3. Conversation turns
        optimized: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]

        # Add any extra system messages that aren't the main prompt
        for sys_msg in system_msgs:
            if sys_msg.get("content") != self._system_prompt:
                optimized.append(sys_msg)

        # Add conversation in order
        optimized.extend(conversation_msgs)

        return optimized

    @property
    def prefix_hash(self) -> str:
        """Hash of the stable prefix for debugging cache behavior."""
        return self._prefix_hash


# Global cache stats singleton (per-process)
_global_cache_stats = CacheStats()


def get_cache_stats() -> CacheStats:
    return _global_cache_stats


def record_cache_usage(usage: Any) -> None:
    """Called by LLMClient after every API response."""
    _global_cache_stats.record(usage)
