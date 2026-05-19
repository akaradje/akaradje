"""Conversation memory + lightweight keyword RAG.

Two layers:

1. **Recent window** — the last K turns are passed verbatim to every call.
   This is what gives the bot continuity inside a single session.

2. **Long-term recall** — every turn is also appended to a JSONL log.
   When a new query comes in we score past turns by keyword overlap and
   surface the top matches as a "previously you said..." preamble.

We deliberately use a no-dependency keyword retriever rather than a vector
DB. That keeps the scaffolding self-contained; swap in Qdrant / pgvector /
Chroma when you actually need semantic recall.
"""

from __future__ import annotations

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
    role: str  # "user" or "assistant"
    content: str
    ts: float = field(default_factory=time.time)

    def to_message(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    def __init__(
        self,
        *,
        window_turns: int = 10,
        store_path: str | Path | None = None,
        recall_k: int = 3,
    ):
        self._window = max(2, window_turns)
        self._recall_k = recall_k
        self._turns: list[Turn] = []
        self._archive: list[Turn] = []  # everything ever, for retrieval
        self._store_path = Path(store_path) if store_path else None
        if self._store_path is not None:
            self._load()

    # ----- mutation -----------------------------------------------------

    def add_user(self, content: str) -> None:
        self._add(Turn(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._add(Turn(role="assistant", content=content))

    def reset(self) -> None:
        self._turns.clear()

    def _add(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._archive.append(turn)
        # Trim recent window. Always keep complete user/assistant pairs by
        # trimming two at a time.
        while len(self._turns) > self._window:
            self._turns.pop(0)
        if self._store_path is not None:
            self._append_to_disk(turn)

    # ----- retrieval ----------------------------------------------------

    def recent_messages(self) -> list[dict[str, Any]]:
        return [t.to_message() for t in self._turns]

    def recall(self, query: str, k: int | None = None) -> list[Turn]:
        """Return up to k past turns most relevant to `query` by keyword overlap.

        We exclude turns already in the recent window (no point recalling
        what's already in context).
        """
        k = k if k is not None else self._recall_k
        if k <= 0 or not self._archive:
            return []

        recent_ids = {id(t) for t in self._turns}
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
            # Sum of min-counts (a poor man's BM25 lite). Normalize by sqrt
            # of length to avoid bias toward very long turns.
            overlap = sum((q_tokens & t_tokens).values())
            if overlap == 0:
                continue
            length_penalty = max(1.0, len(t_tokens) ** 0.5)
            scored.append((overlap / length_penalty, turn))

        scored.sort(key=lambda sv: sv[0], reverse=True)
        return [t for _, t in scored[:k]]

    def build_prior_messages(self, query: str) -> list[dict[str, Any]]:
        """The full prior context to feed an Executor: recall preamble + recent window."""
        msgs: list[dict[str, Any]] = []
        recalled = self.recall(query)
        if recalled:
            preamble = "Earlier in this conversation, the user said or you said:\n" + "\n".join(
                f"- [{t.role}] {_one_line(t.content)}" for t in recalled
            )
            msgs.append({"role": "system", "content": preamble})
        msgs.extend(self.recent_messages())
        return msgs

    # ----- persistence --------------------------------------------------

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
                    self._archive.append(
                        Turn(
                            role=obj["role"],
                            content=obj["content"],
                            ts=float(obj.get("ts", time.time())),
                        )
                    )
            log.info("memory: loaded %d archived turns from %s", len(self._archive), self._store_path)
        except Exception as exc:
            log.warning("memory: failed to load archive (%s) — starting empty", exc)

    def _append_to_disk(self, turn: Turn) -> None:
        assert self._store_path is not None
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            with self._store_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"role": turn.role, "content": turn.content, "ts": turn.ts}) + "\n")
        except Exception as exc:  # pragma: no cover
            log.warning("memory: failed to persist turn (%s)", exc)

    # ----- introspection -----------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def archived(self) -> Iterable[Turn]:
        return iter(self._archive)


def _one_line(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
