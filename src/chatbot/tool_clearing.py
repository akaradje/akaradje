"""Tool-Result Clearing — remove stale tool outputs from context.

Problem: In long agent loops, tool results accumulate in the context window.
Old results from iterations 1-2 become irrelevant noise by iteration 7-8,
diluting attention and causing "context rot" — the model gets dumber.

Solution (same as Claude Code's approach):
1. After each iteration, score tool results by recency and relevance
2. Replace old results with a 1-line summary: "[tool_result cleared: calculator returned 42]"
3. Keep only the N most recent tool results in full
4. Never clear system messages or the current query

This is different from compaction.py which summarizes entire conversation
turns. Tool clearing operates at the granularity of individual tool_call
results within a single agent loop.

Research: "Chroma found focused 300-token prompts beat full prompts of
~113,000 tokens on LongMemEval" — less is more for recall accuracy.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def clear_stale_tool_results(
    messages: list[dict[str, Any]],
    *,
    keep_recent_n: int = 3,
    max_result_chars: int = 500,
) -> list[dict[str, Any]]:
    """Replace old tool results with compact summaries.

    Strategy:
    - Identify all tool-role messages
    - Keep the last `keep_recent_n` tool results verbatim
    - Replace older ones with a 1-line summary
    - Never touch system/user/assistant messages

    Args:
        messages: The full message list
        keep_recent_n: How many recent tool results to preserve in full
        max_result_chars: Max chars for preserved results (truncate beyond this)

    Returns:
        New message list with stale tool results cleared
    """
    # Find indices of all tool messages
    tool_indices: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            tool_indices.append(i)

    if len(tool_indices) <= keep_recent_n:
        # Nothing to clear — all results are "recent"
        return messages

    # Split into stale (to be cleared) and recent (to be preserved)
    stale_indices = set(tool_indices[:-keep_recent_n])

    cleared: list[dict[str, Any]] = []
    cleared_count = 0

    for i, msg in enumerate(messages):
        if i in stale_indices:
            # Replace with compact summary
            content = msg.get("content", "")
            summary = _summarize_tool_result(content)
            cleared.append({
                "role": msg["role"],
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": summary,
            })
            cleared_count += 1
        elif msg.get("role") == "tool":
            # Recent tool result — keep but truncate if huge
            content = msg.get("content", "")
            if len(content) > max_result_chars:
                content = content[:max_result_chars] + "\n[...truncated...]"
            cleared.append({**msg, "content": content})
        else:
            cleared.append(msg)

    if cleared_count > 0:
        log.debug("tool_clearing: cleared %d stale tool results", cleared_count)

    return cleared


def _summarize_tool_result(content: str) -> str:
    """Create a 1-line summary of a tool result.

    Heuristics:
    - If it starts with ERROR → keep the error line
    - If it's a number → keep the number
    - If it's JSON → note the shape
    - Otherwise → first 80 chars
    """
    content = content.strip()
    if not content:
        return "[cleared: empty result]"

    if content.startswith("ERROR"):
        first_line = content.split("\n")[0][:100]
        return f"[cleared: {first_line}]"

    # Pure number
    try:
        float(content)
        return f"[cleared: returned {content}]"
    except ValueError:
        pass

    # JSON object
    if content.startswith("{") or content.startswith("["):
        lines = content.count("\n") + 1
        chars = len(content)
        return f"[cleared: JSON result, {lines} lines, {chars} chars]"

    # Generic — first meaningful line
    first_line = content.split("\n")[0][:80]
    total_lines = content.count("\n") + 1
    if total_lines > 1:
        return f"[cleared: \"{first_line}...\" ({total_lines} lines total)]"
    return f"[cleared: \"{first_line}\"]"


def estimate_savings(
    messages: list[dict[str, Any]],
    *,
    keep_recent_n: int = 3,
) -> dict[str, int]:
    """Estimate how many tokens would be saved by clearing.

    Returns dict with 'before_chars', 'after_chars', 'saved_chars'.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent_n:
        return {"before_chars": 0, "after_chars": 0, "saved_chars": 0}

    stale = tool_indices[:-keep_recent_n]
    before = sum(len(messages[i].get("content", "")) for i in stale)
    # Each cleared result becomes ~50-100 chars
    after = len(stale) * 75
    return {
        "before_chars": before,
        "after_chars": after,
        "saved_chars": max(0, before - after),
    }
