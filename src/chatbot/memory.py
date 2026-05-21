"""Conversation memory — short-term window + semantic long-term recall.

Two layers:
1. **Recent window** — last K turns verbatim, always included in context.
2. **Long-term recall** — semantic search via VectorMemory (Qdrant + embeddings).
   Falls back to keyword-based recall if no VectorMemory is configured.

Auto-indexing: every turn added to memory is asynchronously vectorized and
stored in Qdrant for future semantic recall.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class Turn:
    role: str
    content: str
    ts: float = field(default_factory=time.time)

    def to_message(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """Conversation memory with short-term window and optional semantic long-term recall.

    Parameters:
        window_turns: Number of recent turns kept verbatim.
        store_path: Optional JSONL file for durable archive persistence.
        recall_k: Number of historical turns to recall.
        vector_memory: Optional VectorMemory instance for semantic search.
            When provided, `recall()` uses Qdrant cosine-similarity search.
            When None, falls back to legacy keyword-overlap scoring.
        auto_index: If True (default), every `add_user`/`add_assistant` call
            also indexes the turn into VectorMemory asynchronously.
    """

    def __init__(
        self,
        *,
        window_turns: int = 10,
        store_path: str | Path | None = None,
        recall_k: int = 3,
        vector_memory: Any | None = None,  # VectorMemory | None
        auto_index: bool = True,
    ):
        self._window = max(2, window_turns)
        self._recall_k = recall_k
        self._turns: list[Turn] = []
        self._archive: list[Turn] = []
        self._store_path = Path(store_path) if store_path else None
        self._vector_memory = vector_memory
        self._auto_index = auto_index and vector_memory is not None
        self._pending_index: list[Turn] = []
        self._index_lock = asyncio.Lock()

        if self._store_path is not None:
            self._load()

    # ─── Public API ──────────────────────────────────────────────────────────

    async def add_user(self, content: str) -> None:
        await self._add(Turn(role="user", content=content))

    async def add_assistant(self, content: str) -> None:
        await self._add(Turn(role="assistant", content=content))

    async def aadd_user(self, content: str) -> None:
        """Alias for add_user (async)."""
        await self.add_user(content)

    async def aadd_assistant(self, content: str) -> None:
        """Alias for add_assistant (async)."""
        await self.add_assistant(content)

    def add_user_sync(self, content: str) -> None:
        """Synchronous add — does NOT index into vector memory. Use in non-async contexts only."""
        self._add_sync(Turn(role="user", content=content))

    def add_assistant_sync(self, content: str) -> None:
        """Synchronous add — does NOT index into vector memory. Use in non-async contexts only."""
        self._add_sync(Turn(role="assistant", content=content))

    def reset(self) -> None:
        self._turns.clear()

    def recent_messages(self) -> list[dict[str, Any]]:
        return [t.to_message() for t in self._turns]

    async def recall(self, query: str, k: int | None = None) -> list[Turn]:
        """Recall historically relevant turns for the given query.

        If VectorMemory is configured, uses semantic cosine-similarity search.
        Otherwise falls back to keyword-overlap scoring.
        """
        k = k if k is not None else self._recall_k
        if k <= 0 or not self._archive:
            return []

        # Flush any pending vector indexes before recall
        await self._flush_pending()

        recent_ids = {id(t) for t in self._turns}

        if self._vector_memory is not None:
            return await self._semantic_recall(query, k, recent_ids)
        return self._keyword_recall(query, k, recent_ids)

    async def build_prior_messages(self, query: str) -> list[dict[str, Any]]:
        """Build the prior-messages block for system prompt injection.

        Includes semantically recalled historical turns (prefixed with a
        system preamble) followed by the recent window verbatim.
        """
        msgs: list[dict[str, Any]] = []
        recalled = await self.recall(query)
        if recalled:
            preamble = "Earlier in this conversation:\n" + "\n".join(
                f"- [{t.role}] {_one_line(t.content)}" for t in recalled
            )
            msgs.append({"role": "system", "content": preamble})
        msgs.extend(self.recent_messages())
        return msgs

    # ─── Internal ────────────────────────────────────────────────────────────

    async def _add(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._archive.append(turn)
        while len(self._turns) > self._window:
            self._turns.pop(0)
        if self._store_path is not None:
            self._append_to_disk(turn)
        if self._auto_index:
            self._pending_index.append(turn)
            # Fire-and-forget: don't block the conversation on indexing
            asyncio.ensure_future(self._flush_pending())

    def _add_sync(self, turn: Turn) -> None:
        """Synchronous add for non-async callers (no vector indexing)."""
        self._turns.append(turn)
        self._archive.append(turn)
        while len(self._turns) > self._window:
            self._turns.pop(0)
        if self._store_path is not None:
            self._append_to_disk(turn)

    async def _flush_pending(self) -> None:
        """Index all pending turns into VectorMemory."""
        if not self._pending_index or self._vector_memory is None:
            return

        async with self._index_lock:
            batch = list(self._pending_index)
            self._pending_index.clear()

        for turn in batch:
            try:
                await self._vector_memory.add_memory(
                    text=turn.content,
                    metadata={"role": turn.role, "ts": turn.ts},
                )
            except Exception as exc:
                log.warning("memory: vector index failed for turn: %s", exc)
                # Re-queue on failure so it's retried next recall
                async with self._index_lock:
                    self._pending_index.insert(0, turn)

    async def _semantic_recall(
        self, query: str, k: int, recent_ids: set[int],
    ) -> list[Turn]:
        """Semantic recall via VectorMemory."""
        try:
            results = await self._vector_memory.recall(query, k=k + len(self._turns))
        except Exception as exc:
            log.warning("memory: semantic recall failed (%s), falling back to keyword", exc)
            return self._keyword_recall(query, k, recent_ids)

        # Map results back to Turn objects by matching text
        recalled: list[Turn] = []
        seen_texts: set[str] = set()
        for r in results:
            text = r.get("text", "")
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            # Find matching turn in archive
            for turn in reversed(self._archive):
                if turn.content == text and id(turn) not in recent_ids:
                    recalled.append(turn)
                    break
            else:
                # Reconstructed turn from vector payload
                recalled.append(Turn(
                    role=r.get("role", "unknown"),
                    content=text,
                    ts=r.get("ts", 0.0),
                ))
            if len(recalled) >= k:
                break

        return recalled

    def _keyword_recall(
        self, query: str, k: int, recent_ids: set[int],
    ) -> list[Turn]:
        """Legacy keyword-overlap recall (used when VectorMemory is absent)."""
        pool = [t for t in self._archive if id(t) not in recent_ids]
        if not pool:
            return []
        q_tokens = Counter(_tokenize(query))
        if not q_tokens:
            return []
        scored: list[tuple[float, Turn]] = []
        for turn in pool:
            t_tokens = Counter(_tokenize(turn.content))
            if not t_tokens:
                continue
            overlap = sum((q_tokens & t_tokens).values())
            if overlap == 0:
                continue
            length_penalty = max(1.0, len(t_tokens) ** 0.5)
            scored.append((overlap / length_penalty, turn))
        scored.sort(key=lambda sv: sv[0], reverse=True)
        return [t for _, t in scored[:k]]

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        assert self._store_path is not None
        if not self._store_path.exists():
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            return
        try:
            with self._store_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self._archive.append(Turn(
                        role=obj["role"], content=obj["content"],
                        ts=float(obj.get("ts", time.time())),
                    ))
            log.info("memory: loaded %d turns from %s", len(self._archive), self._store_path)
        except Exception as exc:
            log.warning("memory: load failed (%s)", exc)

    def _append_to_disk(self, turn: Turn) -> None:
        assert self._store_path is not None
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            with self._store_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(
                    {"role": turn.role, "content": turn.content, "ts": turn.ts}
                ) + "\n")
        except Exception as exc:
            log.warning("memory: persist failed (%s)", exc)

    # ─── Utility ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._turns)

    def archived(self) -> Iterable[Turn]:
        return iter(self._archive)

    @property
    def vector_memory(self) -> Any | None:
        return self._vector_memory

    @property
    def using_semantic_search(self) -> bool:
        return self._vector_memory is not None


def _one_line(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
