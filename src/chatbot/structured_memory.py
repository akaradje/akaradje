"""Structured Persistent Memory — typed entries that survive across sessions.

Inspired by Anthropic's memory tool: "Claude can create, read, update, and
delete files that persist between sessions, allowing it to build knowledge
over time without keeping everything in the context window."

This is different from memory.py (conversation history + keyword recall).
Structured memory stores SEMANTIC facts, not raw conversation turns:

Types of entries:
- FACT: something the user stated ("I use Python 3.11", "My name is Alex")
- PREFERENCE: user's preferences ("prefer concise answers", "use Thai")
- TASK: ongoing tasks / context ("working on rate limiter project")
- DECISION: decisions made ("chose PostgreSQL over MongoDB")
- INSTRUCTION: standing instructions ("always include type hints")

Each entry has:
- id, type, content, created_at, last_accessed, relevance_score, tags

The memory is:
- Persisted to disk as JSON (not JSONL — we need random access)
- Injected as a system message each turn (only relevant entries)
- Searchable by keyword and type
- Automatically decays unused entries (relevance_score decreases over time)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"
    DECISION = "decision"
    INSTRUCTION = "instruction"


@dataclass
class MemoryEntry:
    """A single structured memory entry."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: MemoryType = MemoryType.FACT
    content: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    relevance_score: float = 1.0  # decays over time

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        d = dict(d)
        d["type"] = MemoryType(d.get("type", "fact"))
        return cls(**d)

    def touch(self) -> None:
        """Mark as accessed — resets relevance decay."""
        self.last_accessed = time.time()
        self.access_count += 1
        self.relevance_score = min(1.0, self.relevance_score + 0.1)


class StructuredMemory:
    """Persistent structured memory store."""

    def __init__(self, store_path: str | Path | None = None):
        self._entries: dict[str, MemoryEntry] = {}
        self._store_path = Path(store_path) if store_path else None
        if self._store_path:
            self._load()

    # ─── CRUD ───────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.FACT,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Add a new memory entry."""
        entry = MemoryEntry(
            type=type,
            content=content,
            tags=tags or [],
        )
        self._entries[entry.id] = entry
        self._save()
        log.info("memory+: added [%s] %s (%s)", entry.type.value, entry.content[:50], entry.id)
        return entry

    def get(self, entry_id: str) -> MemoryEntry | None:
        entry = self._entries.get(entry_id)
        if entry:
            entry.touch()
        return entry

    def update(self, entry_id: str, content: str) -> bool:
        """Update an existing entry's content."""
        entry = self._entries.get(entry_id)
        if not entry:
            return False
        entry.content = content
        entry.touch()
        self._save()
        return True

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save()
            return True
        return False

    # ─── Retrieval ──────────────────────────────────────────────────────

    def search(
        self,
        query: str = "",
        *,
        type_filter: MemoryType | None = None,
        tag_filter: str | None = None,
        k: int = 10,
    ) -> list[MemoryEntry]:
        """Search memory entries by keyword, type, or tag.

        Results are scored by: keyword overlap + recency + relevance_score.
        """
        candidates = list(self._entries.values())

        # Type filter
        if type_filter:
            candidates = [e for e in candidates if e.type == type_filter]

        # Tag filter
        if tag_filter:
            candidates = [e for e in candidates if tag_filter.lower() in [t.lower() for t in e.tags]]

        if not query:
            # No query — return most relevant
            candidates.sort(key=lambda e: -e.relevance_score)
            return candidates[:k]

        # Keyword scoring
        query_words = set(query.lower().split())
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in candidates:
            entry_words = set(entry.content.lower().split())
            tag_words = set(t.lower() for t in entry.tags)

            keyword_score = len(query_words & entry_words) / max(1, len(query_words))
            tag_score = len(query_words & tag_words) * 0.5

            # Recency bonus (decays over days)
            age_days = (time.time() - entry.last_accessed) / 86400
            recency_score = max(0, 1.0 - age_days * 0.05)  # loses 5% per day

            total_score = (
                keyword_score * 0.5
                + tag_score * 0.2
                + recency_score * 0.15
                + entry.relevance_score * 0.15
            )

            if total_score > 0.05:
                scored.append((total_score, entry))

        scored.sort(key=lambda x: -x[0])
        results = [entry for _, entry in scored[:k]]

        # Touch accessed entries
        for entry in results:
            entry.touch()

        return results

    def get_all(self, type_filter: MemoryType | None = None) -> list[MemoryEntry]:
        """Get all entries, optionally filtered by type."""
        entries = list(self._entries.values())
        if type_filter:
            entries = [e for e in entries if e.type == type_filter]
        return sorted(entries, key=lambda e: -e.last_accessed)

    # ─── Context injection ──────────────────────────────────────────────

    def to_system_message(self, query: str = "", max_entries: int = 8) -> dict[str, Any] | None:
        """Generate a system message with relevant memory for the current query.

        Returns None if memory is empty.
        """
        if not self._entries:
            return None

        # Always include instructions and active tasks
        always_show = [
            e for e in self._entries.values()
            if e.type in (MemoryType.INSTRUCTION, MemoryType.TASK)
        ]

        # Search for query-relevant entries
        relevant = self.search(query, k=max_entries - len(always_show)) if query else []

        # Combine and deduplicate
        shown_ids: set[str] = set()
        entries_to_show: list[MemoryEntry] = []

        for entry in always_show + relevant:
            if entry.id not in shown_ids:
                shown_ids.add(entry.id)
                entries_to_show.append(entry)

        if not entries_to_show:
            return None

        # Format
        lines: list[str] = ["[MEMORY] Known facts and preferences:"]
        for entry in entries_to_show[:max_entries]:
            type_emoji = {
                MemoryType.FACT: "📌",
                MemoryType.PREFERENCE: "⭐",
                MemoryType.TASK: "📋",
                MemoryType.DECISION: "✅",
                MemoryType.INSTRUCTION: "⚠️",
            }.get(entry.type, "•")
            lines.append(f"  {type_emoji} [{entry.type.value}] {entry.content}")

        return {"role": "system", "content": "\n".join(lines)}

    # ─── Decay ──────────────────────────────────────────────────────────

    def decay_all(self, amount: float = 0.02) -> int:
        """Apply relevance decay to all entries.

        Call this periodically (e.g., once per session or daily).
        Entries that decay below threshold get auto-deleted.
        """
        to_delete: list[str] = []
        for entry in self._entries.values():
            entry.relevance_score = max(0, entry.relevance_score - amount)
            if entry.relevance_score <= 0 and entry.type not in (MemoryType.INSTRUCTION,):
                to_delete.append(entry.id)

        for entry_id in to_delete:
            del self._entries[entry_id]

        if to_delete:
            log.info("memory: decayed and removed %d entries", len(to_delete))
            self._save()

        return len(to_delete)

    # ─── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._store_path or not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            for entry_dict in data.get("entries", []):
                entry = MemoryEntry.from_dict(entry_dict)
                self._entries[entry.id] = entry
            log.info("structured_memory: loaded %d entries", len(self._entries))
        except Exception as exc:
            log.warning("structured_memory: load failed: %s", exc)

    def _save(self) -> None:
        if not self._store_path:
            return
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "entries": [e.to_dict() for e in self._entries.values()],
            }
            self._store_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("structured_memory: save failed: %s", exc)

    def __len__(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Tool — lets the model manage its own memory
# ═══════════════════════════════════════════════════════════════════════════════

from .tools import Tool


class MemoryTool(Tool):
    """Tool that lets the model store and retrieve structured memories.

    Operations:
    - add: store a new fact/preference/instruction
    - search: find relevant memories
    - list: show all memories of a type
    - delete: remove a memory by ID
    """

    name = "memory"
    description = (
        "Store and retrieve persistent memories across conversations. "
        "Use this to remember user preferences, project context, decisions, "
        "and standing instructions. Memories persist between sessions. "
        "Operations: 'add', 'search', 'list', 'delete'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "search", "list", "delete"],
                "description": "What to do",
            },
            "content": {
                "type": "string",
                "description": "For 'add': the fact/preference to remember. For 'search': the query.",
            },
            "type": {
                "type": "string",
                "enum": ["fact", "preference", "task", "decision", "instruction"],
                "description": "Memory type (for 'add' and 'list')",
            },
            "entry_id": {
                "type": "string",
                "description": "For 'delete': the memory ID to remove",
            },
        },
        "required": ["operation"],
    }

    def __init__(self, memory: StructuredMemory):
        self._mem = memory

    async def run(self, args: dict[str, Any]) -> str:
        op = str(args.get("operation", "")).lower()
        content = str(args.get("content", ""))
        mem_type = args.get("type")
        entry_id = args.get("entry_id")

        if op == "add":
            if not content:
                return "ERROR: 'content' required for add"
            t = MemoryType(mem_type) if mem_type else MemoryType.FACT
            entry = self._mem.add(content, type=t)
            return f"Stored memory [{t.value}] id={entry.id}: {content[:100]}"

        elif op == "search":
            results = self._mem.search(content or "", k=5)
            if not results:
                return "No matching memories found."
            lines = [f"Found {len(results)} memories:"]
            for e in results:
                lines.append(f"  [{e.type.value}] {e.content[:150]} (id={e.id})")
            return "\n".join(lines)

        elif op == "list":
            t_filter = MemoryType(mem_type) if mem_type else None
            entries = self._mem.get_all(type_filter=t_filter)
            if not entries:
                return "No memories stored."
            lines = [f"All memories ({len(entries)}):"]
            for e in entries[:20]:
                lines.append(f"  [{e.type.value}] {e.content[:100]} (id={e.id})")
            if len(entries) > 20:
                lines.append(f"  ... and {len(entries)-20} more")
            return "\n".join(lines)

        elif op == "delete":
            if not entry_id:
                return "ERROR: 'entry_id' required for delete"
            if self._mem.delete(entry_id):
                return f"Deleted memory {entry_id}"
            return f"Memory {entry_id} not found"

        return f"ERROR: unknown operation '{op}'"
