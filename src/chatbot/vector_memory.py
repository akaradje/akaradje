"""Vector memory — semantic long-term recall via Qdrant + embeddings.

Architecture:
- Qdrant local mode (in-memory or persistent `.akaradje_data/qdrant/`)
- Embeddings via DeepSeek API (OpenAI-compatible /v1/embeddings endpoint)
- Cosine-similarity search for top-K relevant history chunks
- Async with tenacity retry backoff on embedding API calls
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

log = logging.getLogger(__name__)

DEFAULT_COLLECTION = "conversation_memory"
DEFAULT_EMBEDDING_MODEL = "deepseek-chat"
DEFAULT_VECTOR_SIZE = 1536  # will be auto-detected on first embedding
DEFAULT_STORAGE_DIR = ".akaradje_data/qdrant"


class VectorMemory:
    """Semantic long-term memory backed by Qdrant vector search.

    Usage::

        vm = VectorMemory(api_key="...", base_url="https://api.deepseek.com")
        await vm.add_memory("User asked about Python generators", {"role": "user"})
        results = await vm.recall("Python generators", k=5)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        storage_path: str | Path | None = None,
        persist: bool = True,
    ):
        if not HAS_OPENAI:
            raise ImportError("openai package is required for VectorMemory embeddings")
        if not HAS_QDRANT:
            raise ImportError("qdrant-client package is required for VectorMemory")

        api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        self._embedding_model = embedding_model or os.getenv(
            "DEEPSEEK_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self._vector_size_str = os.getenv("EMBEDDING_DIMS", "")
        self._vector_size = int(self._vector_size_str) if self._vector_size_str else vector_size
        self._collection_name = collection_name
        self._persist = persist

        # Embedding client
        self._emb_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Qdrant client
        if persist and storage_path is not None:
            storage_dir = str(storage_path)
        elif persist:
            storage_dir = str(Path(DEFAULT_STORAGE_DIR).resolve())
        else:
            storage_dir = ":memory:"

        Path(storage_dir).mkdir(parents=True, exist_ok=True) if storage_dir != ":memory:" else None

        self._qdrant = QdrantClient(path=storage_dir) if storage_dir != ":memory:" else QdrantClient(location=":memory:")
        self._ensure_collection()

    # ─── Collection setup ──────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        try:
            self._qdrant.get_collection(self._collection_name)
            log.debug("vector_memory: collection %r exists", self._collection_name)
        except Exception:
            self._qdrant.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            log.info(
                "vector_memory: created collection %r (dim=%d, distance=cosine)",
                self._collection_name,
                self._vector_size,
            )

    # ─── Embedding ─────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        reraise=True,
    )
    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for one or more texts via the DeepSeek API.

        Rate-limiting and transient errors are handled by tenacity retry
        with exponential backoff (1s → 2s → 4s ... max 15s, 3 attempts).
        """
        try:
            resp = await self._emb_client.embeddings.create(
                model=self._embedding_model,
                input=texts,
            )
            return [d.embedding for d in resp.data]
        except Exception as exc:
            log.error(
                "vector_memory: embedding failed for %d texts (model=%s): %s",
                len(texts), self._embedding_model, exc,
            )
            raise

    async def _embed_single(self, text: str) -> list[float]:
        embeddings = await self._embed([text])
        vector = embeddings[0]
        # Auto-detect actual vector size and update if different
        if len(vector) != self._vector_size:
            log.info(
                "vector_memory: detected dims=%d (was %d), updating",
                len(vector), self._vector_size,
            )
            self._vector_size = len(vector)
        return vector

    # ─── Public API ────────────────────────────────────────────────────────

    async def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        *,
        doc_id: str | None = None,
    ) -> str:
        """Index a text chunk into the vector store.

        Args:
            text: The conversation turn or knowledge chunk to index.
            metadata: Arbitrary key-value payload stored alongside the vector.
            doc_id: Optional unique ID. Generated if not provided.

        Returns:
            The document ID of the upserted point.
        """
        if not text.strip():
            return ""

        doc_id = doc_id or f"mem-{uuid.uuid4().hex[:12]}"
        meta = dict(metadata or {})
        meta["indexed_at"] = time.time()

        try:
            vector = await self._embed_single(text)
            self._qdrant.upsert(
                collection_name=self._collection_name,
                points=[
                    PointStruct(
                        id=doc_id,
                        vector=vector,
                        payload={
                            "text": text,
                            **meta,
                        },
                    )
                ],
            )
            log.debug("vector_memory: indexed %r (len=%d)", doc_id, len(text))
            return doc_id
        except Exception as exc:
            log.error("vector_memory: add_memory failed for %r: %s", doc_id, exc)
            raise

    async def recall(
        self,
        query: str,
        k: int = 5,
        *,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over indexed memories.

        Args:
            query: The search query (typically the user's current message).
            k: Number of results to return.
            score_threshold: Optional minimum cosine similarity (0-1).
                Results below this threshold are excluded.

        Returns:
            List of dicts with keys: id, text, score, plus any stored metadata.
        """
        if not query.strip():
            return []

        try:
            vector = await self._embed_single(query)
            results = self._qdrant.search(
                collection_name=self._collection_name,
                query_vector=vector,
                limit=k,
                score_threshold=score_threshold,
            )
            return [
                {
                    "id": hit.id,
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    **{k: v for k, v in (hit.payload or {}).items() if k != "text"},
                }
                for hit in results
            ]
        except Exception as exc:
            log.error("vector_memory: recall failed for query %r: %s", query[:100], exc)
            return []

    # ─── Maintenance ───────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of indexed points."""
        try:
            info = self._qdrant.get_collection(self._collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all points from the collection (keeps schema)."""
        try:
            self._qdrant.delete_collection(self._collection_name)
            self._ensure_collection()
            log.info("vector_memory: cleared collection %r", self._collection_name)
        except Exception as exc:
            log.error("vector_memory: clear failed: %s", exc)

    def close(self) -> None:
        """Release Qdrant client resources."""
        try:
            self._qdrant.close()
        except Exception:
            pass

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def vector_size(self) -> int:
        return self._vector_size
