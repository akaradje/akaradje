"""Context Compaction — prevent context rot in long conversations.

When the conversation grows too long, we need to compress older turns into
a summary. This prevents:
1. Token budget exhaustion (1M context is large but not infinite)
2. "Context rot" — model quality degrades as context grows because
   attention gets diluted across irrelevant old content
3. Increased latency and cost from processing long prefixes

Strategy (same as Anthropic/OpenAI recommendations):
- Keep the most recent N turns verbatim (the "hot window")
- Summarize everything older into a compressed "memory summary"
- The summary replaces the old turns in the context
- Track total estimated tokens and trigger compaction at threshold

This is NOT the same as memory.py's keyword retrieval. Memory does
selective recall of specific past facts. Compaction does wholesale
compression of the entire old conversation to keep context size manageable.
"""

from __future__ import annotations

import logging
from typing import Any

from .client import LLMClient
from .config import Config, ThinkingMode

log = logging.getLogger(__name__)


# Rough token estimation (for deciding when to compact)
def estimate_tokens(text: str) -> int:
    """Rough token count estimate. ~4 chars per token for English, ~2 for code."""
    return max(1, len(text) // 3)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens in a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total += estimate_tokens(item["text"])
        # Account for role tokens, tool calls, etc.
        total += 4  # overhead per message
        if msg.get("tool_calls"):
            total += estimate_tokens(str(msg["tool_calls"]))
    return total


_COMPACTION_PROMPT = """You are a conversation summarizer. Condense the following conversation history into a concise but complete summary that preserves:
1. Key decisions made
2. Important facts established
3. Current state of any ongoing tasks
4. Any constraints or preferences the user stated

Be thorough but concise. Use bullet points. Keep technical details accurate.
Do NOT lose any actionable information.

CONVERSATION TO SUMMARIZE:
{conversation}

SUMMARY:"""


class ContextCompactor:
    """Manages context size through intelligent compaction.

    Compaction triggers when estimated tokens exceed the threshold.
    The compactor:
    1. Splits messages into "cold" (old, to be summarized) and "hot" (recent, kept verbatim)
    2. Sends cold messages to the LLM for summarization
    3. Replaces cold messages with a single system message containing the summary
    """

    def __init__(
        self,
        client: LLMClient,
        config: Config,
        *,
        max_context_tokens: int = 100_000,
        hot_window_turns: int = 10,
        compaction_ratio: float = 0.7,
    ):
        self._client = client
        self._cfg = config
        self._max_tokens = max_context_tokens
        self._hot_window = hot_window_turns
        self._compaction_ratio = compaction_ratio  # compact when exceeding this % of max
        self._compaction_count = 0

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    def needs_compaction(self, messages: list[dict[str, Any]]) -> bool:
        """Check if messages need compaction."""
        estimated = estimate_messages_tokens(messages)
        threshold = int(self._max_tokens * self._compaction_ratio)
        return estimated > threshold

    async def compact(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compact messages if they exceed the threshold.

        Returns the (possibly compacted) message list.
        """
        if not self.needs_compaction(messages):
            return messages

        log.info(
            "compaction: triggered (estimated %d tokens, threshold %d)",
            estimate_messages_tokens(messages),
            int(self._max_tokens * self._compaction_ratio),
        )

        # Separate system messages, cold turns, and hot turns
        system_msgs: list[dict[str, Any]] = []
        conversation_msgs: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                conversation_msgs.append(msg)

        # Keep the last N turns as "hot" (verbatim)
        hot_count = min(self._hot_window * 2, len(conversation_msgs))  # *2 for user+assistant pairs
        cold_msgs = conversation_msgs[:-hot_count] if hot_count > 0 else conversation_msgs
        hot_msgs = conversation_msgs[-hot_count:] if hot_count > 0 else []

        if not cold_msgs:
            return messages  # nothing to compact

        # Summarize cold messages
        summary = await self._summarize(cold_msgs)
        self._compaction_count += 1

        # Rebuild: system + summary + hot
        compacted: list[dict[str, Any]] = list(system_msgs)
        compacted.append({
            "role": "system",
            "content": f"[CONVERSATION SUMMARY — compaction #{self._compaction_count}]\n{summary}",
        })
        compacted.extend(hot_msgs)

        log.info(
            "compaction: %d msgs → %d msgs (saved ~%d tokens)",
            len(messages),
            len(compacted),
            estimate_messages_tokens(cold_msgs) - estimate_tokens(summary),
        )

        return compacted

    async def _summarize(self, messages: list[dict[str, Any]]) -> str:
        """Use the LLM to summarize old conversation turns."""
        # Format messages for summarization
        conversation_text = "\n".join(
            f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')[:500]}"
            for msg in messages
            if msg.get("content")
        )

        # Truncate if conversation text itself is too long
        if len(conversation_text) > 20000:
            conversation_text = conversation_text[:10000] + "\n...[truncated]...\n" + conversation_text[-10000:]

        try:
            summary = await self._client.chat_text(
                messages=[{
                    "role": "user",
                    "content": _COMPACTION_PROMPT.format(conversation=conversation_text),
                }],
                thinking=ThinkingMode.DISABLED,  # fast, no CoT needed for summarization
                max_tokens=2000,
            )
            return summary
        except Exception as exc:
            log.warning("compaction: summarization failed (%s), using truncation fallback", exc)
            # Fallback: just keep first and last few messages as text
            return self._fallback_summary(messages)

    @staticmethod
    def _fallback_summary(messages: list[dict[str, Any]]) -> str:
        """Non-LLM fallback: extract key lines."""
        lines = []
        for msg in messages[:5]:
            content = msg.get("content", "")[:200]
            if content:
                lines.append(f"- [{msg.get('role')}] {content}")
        if len(messages) > 10:
            lines.append(f"  ... ({len(messages) - 10} more turns) ...")
        for msg in messages[-5:]:
            content = msg.get("content", "")[:200]
            if content:
                lines.append(f"- [{msg.get('role')}] {content}")
        return "\n".join(lines)
