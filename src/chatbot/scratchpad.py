"""Agent Scratchpad — persistent notepad for maintaining plans across iterations.

Problem: In multi-turn tool-use loops, the model "forgets" its original plan
by iteration 5+ because the plan was stated in reasoning_content (which gets
dropped or summarized). It starts making redundant tool calls or losing track.

Solution (inspired by Claude Code's scratchpad.md pattern):
- Give the model a `scratchpad` tool it can read/write
- The scratchpad content is injected as a system message each iteration
- The model can update it to track: current plan, completed steps, findings
- This keeps the "executive summary" always visible regardless of context length

Key insight from research: "Claude Code pairs a 1M-token window with commands
for compact, clear, rewind and subagents" — the scratchpad is the mechanism
that lets compaction work without losing state.
"""

from __future__ import annotations

import logging
from typing import Any

from .tools import Tool

log = logging.getLogger(__name__)


class Scratchpad:
    """In-memory scratchpad that persists across iterations within a single task."""

    def __init__(self, max_chars: int = 4000):
        self._content: str = ""
        self._max_chars = max_chars
        self._version: int = 0

    @property
    def content(self) -> str:
        return self._content

    @property
    def version(self) -> int:
        return self._version

    @property
    def is_empty(self) -> bool:
        return not self._content.strip()

    def write(self, content: str) -> str:
        """Overwrite the scratchpad with new content."""
        self._content = content[:self._max_chars]
        self._version += 1
        log.debug("scratchpad: write v%d (%d chars)", self._version, len(self._content))
        return f"Scratchpad updated (v{self._version}, {len(self._content)} chars)"

    def append(self, text: str) -> str:
        """Append text to the scratchpad."""
        new_content = self._content + "\n" + text if self._content else text
        if len(new_content) > self._max_chars:
            # Trim from the beginning to make room
            overflow = len(new_content) - self._max_chars
            new_content = "...[trimmed]...\n" + new_content[overflow + 50:]
        self._content = new_content
        self._version += 1
        log.debug("scratchpad: append v%d (%d chars)", self._version, len(self._content))
        return f"Appended to scratchpad (v{self._version}, {len(self._content)} chars)"

    def read(self) -> str:
        """Read the current scratchpad content."""
        if self.is_empty:
            return "(scratchpad is empty)"
        return self._content

    def clear(self) -> str:
        """Clear the scratchpad."""
        self._content = ""
        self._version += 1
        return "Scratchpad cleared"

    def to_system_message(self) -> dict[str, Any] | None:
        """Generate a system message containing the scratchpad.

        Returns None if scratchpad is empty (no need to inject).
        This message should be included in every iteration so the model
        always sees its own notes.
        """
        if self.is_empty:
            return None
        return {
            "role": "system",
            "content": (
                f"[SCRATCHPAD v{self._version}] Your working notes (update with scratchpad tool):\n"
                f"---\n{self._content}\n---"
            ),
        }


class ScratchpadTool(Tool):
    """Tool that lets the model read/write its own scratchpad.

    Operations:
    - read: view current scratchpad
    - write: overwrite scratchpad with new content
    - append: add text to scratchpad
    - clear: empty the scratchpad
    """

    name = "scratchpad"
    description = (
        "Your personal notepad for tracking plans, progress, and key findings. "
        "Use this to maintain state across multiple tool calls. "
        "Operations: 'read', 'write', 'append', 'clear'. "
        "BEST PRACTICE: Write your plan at the start, update after each step."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "append", "clear"],
                "description": "What to do with the scratchpad",
            },
            "content": {
                "type": "string",
                "description": "Text to write/append (ignored for read/clear)",
            },
        },
        "required": ["operation"],
    }

    def __init__(self, scratchpad: Scratchpad):
        self._pad = scratchpad

    async def run(self, args: dict[str, Any]) -> str:
        op = str(args.get("operation", "read")).lower()
        content = str(args.get("content", ""))

        if op == "read":
            return self._pad.read()
        elif op == "write":
            if not content:
                return "ERROR: 'content' is required for write operation"
            return self._pad.write(content)
        elif op == "append":
            if not content:
                return "ERROR: 'content' is required for append operation"
            return self._pad.append(content)
        elif op == "clear":
            return self._pad.clear()
        else:
            return f"ERROR: unknown operation '{op}'. Use: read, write, append, clear"
