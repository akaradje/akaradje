"""Dynamic Tool Search — load only relevant tools per query.

Problem: Sending all tool schemas in every API call wastes tokens.
With 6 tools, that's ~1500 tokens of schemas per call. For 8 iterations,
that's 12,000 wasted tokens — $0.02 per query just for tool definitions.

Solution (inspired by Claude Code's "tool search" feature):
- Score each tool's relevance to the current query
- Only include tools likely to be needed
- Always include a "meta" tool that can request additional tools if needed

From official docs: "Tool search enables your agent to work with hundreds
of tools by dynamically discovering and loading them on demand."
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .tools import Tool, ToolRegistry

log = logging.getLogger(__name__)


# Keyword → tool name mapping for fast matching
_TOOL_KEYWORDS: dict[str, list[str]] = {
    "calculator": [
        "calculate", "compute", "math", "arithmetic", "sum", "multiply",
        "divide", "subtract", "factorial", "power", "sqrt", "percentage",
        "คำนวณ", "บวก", "ลบ", "คูณ", "หาร",
    ],
    "python_exec": [
        "code", "python", "script", "program", "execute", "run",
        "algorithm", "implement", "function", "class", "loop",
        "data", "parse", "sort", "filter", "regex",
        "โค้ด", "เขียนโปรแกรม",
    ],
    "web_search": [
        "search", "find", "latest", "current", "news", "today",
        "who is", "what is", "when did", "how many", "price",
        "update", "recent", "2024", "2025", "2026",
        "ค้นหา", "ล่าสุด", "ข่าว",
    ],
    "url_fetch": [
        "url", "website", "page", "link", "article", "read",
        "http", "https", "fetch", "content of",
        "เว็บ", "ลิงก์",
    ],
    "file_read": [
        "file", "read", "open", "content", "source", "config",
        "log", ".py", ".js", ".ts", ".json", ".yaml", ".md",
        "ไฟล์", "อ่าน",
    ],
    "shell_exec": [
        "run command", "terminal", "shell", "bash", "ls", "git",
        "npm", "pip", "docker", "test", "build", "deploy",
        "grep", "find", "curl", "mkdir",
        "คำสั่ง", "เทอร์มินัล",
    ],
    "scratchpad": [
        "plan", "note", "remember", "track", "step",
        "progress", "status", "todo", "checklist",
        "จด", "แผน", "บันทึก",
    ],
}

# Tools that should ALWAYS be included (very cheap, always useful)
_ALWAYS_INCLUDE = {"scratchpad"}

# Minimum relevance score to include a tool
_MIN_RELEVANCE = 0.1


def score_tool_relevance(query: str, tool_name: str) -> float:
    """Score how relevant a tool is to the query (0.0 - 1.0)."""
    keywords = _TOOL_KEYWORDS.get(tool_name, [])
    if not keywords:
        return 0.0

    query_lower = query.lower()
    hits = sum(1 for kw in keywords if kw.lower() in query_lower)

    if hits == 0:
        return 0.0

    # Normalize by keyword list length (so tools with more keywords
    # aren't unfairly penalized)
    score = min(1.0, hits / max(1, len(keywords) * 0.3))
    return score


def select_tools(
    query: str,
    registry: ToolRegistry,
    *,
    max_tools: int = 4,
    min_relevance: float = _MIN_RELEVANCE,
) -> list[Tool]:
    """Select the most relevant tools for a given query.

    Args:
        query: The user's question
        registry: Full tool registry
        max_tools: Maximum number of tools to include
        min_relevance: Minimum score to include a tool

    Returns:
        List of Tool objects to include in the API call
    """
    all_tools = registry.list()

    # Score each tool
    scored: list[tuple[float, Tool]] = []
    for tool in all_tools:
        if tool.name in _ALWAYS_INCLUDE:
            scored.append((1.0, tool))  # always include
            continue
        relevance = score_tool_relevance(query, tool.name)
        if relevance >= min_relevance:
            scored.append((relevance, tool))

    # Sort by relevance (descending), take top max_tools
    scored.sort(key=lambda x: -x[0])
    selected = [tool for _, tool in scored[:max_tools]]

    # If nothing was selected (very generic query), include defaults
    if not selected:
        defaults = ["web_search", "python_exec", "scratchpad"]
        selected = [t for t in all_tools if t.name in defaults]

    tool_names = [t.name for t in selected]
    log.debug("tool_search: selected %s for query '%s'", tool_names, query[:50])

    return selected


def tools_to_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert selected tools to OpenAI function schemas."""
    return [t.to_openai_schema() for t in tools]
