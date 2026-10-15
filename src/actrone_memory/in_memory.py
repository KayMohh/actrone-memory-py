from __future__ import annotations

import math
import time
from datetime import datetime

from actrone_memory.exceptions import MemoryNotFoundError
from actrone_memory.logging import bind_logger
from actrone_memory.models import ContentType, MemoryEntry, SessionMetadata, Turn

log = bind_logger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. Returns 0.0 when either is a zero vector.

    Tolerates length mismatch by comparing over the shorter prefix, matching the
    TS ``cosineSimilarity`` helper so both OSS libs behave identically.
    """
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        av = a[i]
        bv = b[i]
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _session_key(agent_id: str, session_id: str) -> str:
    return f"{agent_id}::{session_id}"


class InMemoryStore:
    """Zero-dependency, in-process store implementing **both** memory tiers.

    This is the local-first default: it needs no Redis and no Qdrant, so
    ``MemoryManager.create()`` runs with zero external services, parity with the
    TypeScript ``actrone-memory`` on-ramp. Data lives for the lifetime of the
    process; swap in :class:`RedisStore` + :class:`QdrantStore` for durability and
    horizontal scale.

    Concurrency: every method is synchronous internally (no ``await`` points), so
    operations are atomic with respect to the asyncio event loop, concurrent
    background tasks (e.g. summarisation) cannot interleave a partial mutation.
    """

    def __init__(
        self,
        max_turns: int = 50,
        *,
        relevance_weight: float = 0.7,
        recency_weight: float = 0.3,
    ) -> None:
        self._max_turns = max_turns
        self._relevance_w = relevance_weight
        self._recency_w = recency_weight
        # L1 state
        self._turns: dict[str, list[Turn]] = {}
        self._created_at: dict[str, datetime] = {}
        self._summary_locks: dict[str, float] = {}  # session_key -> monotonic expiry
        # L2 state, memories keyed by agent_id
        self._memories: dict[str, list[MemoryEntry]] = {}

    # ------------------------------------------------------------------
    # L1, hot session tier
    # ------------------------------------------------------------------

    async def append_turn(self, agent_id: str, session_id: str, turn: Turn) -> None:
        key = _session_key(agent_id, session_id)
        bucket = self._turns.setdefault(key, [])
        if not bucket:
            self._created_at[key] = turn.timestamp
        bucket.append(turn)
        # Cap the retained window (newest kept), mirroring L1's ltrim.
        if len(bucket) > self._max_turns:
            del bucket[: len(bucket) - self._max_turns]

    async def get_recent_turns(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[Turn]:
        bucket = self._turns.get(_session_key(agent_id, session_id), [])
        if n is None or n >= len(bucket):
            return list(bucket)
        return bucket[len(bucket) - n :]

    async def turn_count(self, agent_id: str, session_id: str) -> int:
        return len(self._turns.get(_session_key(agent_id, session_id), []))

    async def clear_session(self, agent_id: str, session_id: str) -> None:
        key = _session_key(agent_id, session_id)
        self._turns.pop(key, None)
        self._created_at.pop(key, None)
        self._summary_locks.pop(key, None)

    async def get_session_metadata(self, agent_id: str, session_id: str) -> SessionMetadata | None:
        key = _session_key(agent_id, session_id)
        bucket = self._turns.get(key)
        if not bucket:
            return None
        return SessionMetadata(
            agent_id=agent_id,
            session_id=session_id,
            turn_count=len(bucket),
            created_at=self._created_at.get(key),
            last_active=bucket[-1].timestamp,
        )

    async def try_acquire_summary_lock(
        self, agent_id: str, session_id: str, ttl_seconds: int
    ) -> bool:
        """SET-NX-EX equivalent. Returns True for the first caller within a TTL window.

        Deduplicates concurrent summarisations within this process and throttles
        re-summarisation, exactly like the Redis ``SET key NX EX ttl`` lock, but
        scoped to this process (in-memory backends are single-instance by design).
        """
        key = _session_key(agent_id, session_id)
        now = time.monotonic()
        expiry = self._summary_locks.get(key)
        if expiry is not None and expiry > now:
            return False
        self._summary_locks[key] = now + ttl_seconds
        return True

    # ------------------------------------------------------------------
    # L2, cold semantic tier
    # ------------------------------------------------------------------

    async def upsert(self, entry: MemoryEntry) -> None:
        if entry.embedding is None:
            raise ValueError(f"MemoryEntry {entry.id} has no embedding, embed before upserting.")
        bucket = self._memories.setdefault(entry.agent_id, [])
        for i, existing in enumerate(bucket):
            if existing.id == entry.id:
                bucket[i] = entry
                return
        bucket.append(entry)

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        threshold: float,
        limit: int = 20,
        content_types: list[ContentType] | None = None,
        query_text: str | None = None,
    ) -> list[MemoryEntry]:
        """Hybrid relevance search, dense + lexical + recency fused with RRF.

        Only memories with embedding cosine ≥ ``threshold`` are admitted; among those, the ranking
        fuses embedding cosine, BM25 over the content (when ``query_text`` is given), and recency
        via Reciprocal Rank Fusion. Without ``query_text`` (or with no shared terms) it degrades to
        the classic ``relevance × cosine + recency × recency`` blend, matching :class:`QdrantStore`.

        Time complexity: O(n) similarity scoring + O(n log n) rank over the agent's memories,
        acceptable for the in-process on-ramp (bounded by ``max_episodic_memories``).
        """
        from actrone_memory.retrieval import hybrid_rank

        bucket = self._memories.get(agent_id, [])
        if not bucket:
            return []

        allowed: set[str] | None = set(content_types) if content_types else None
        candidates = [entry for entry in bucket if allowed is None or entry.content_type in allowed]
        return hybrid_rank(
            candidates,
            query_embedding,
            query_text,
            threshold=threshold,
            relevance_weight=self._relevance_w,
            recency_weight=self._recency_w,
            limit=limit,
        )

    async def delete(self, memory_id: str) -> None:
        for agent_id, bucket in self._memories.items():
            for i, entry in enumerate(bucket):
                if entry.id == memory_id:
                    del bucket[i]
                    log.debug("memory.inmemory.deleted", memory_id=memory_id, agent_id=agent_id)
                    return
        raise MemoryNotFoundError(memory_id)

    async def delete_agent_memories(self, agent_id: str) -> None:
        """Delete every memory for an agent. Irreversible (mirrors QdrantStore)."""
        self._memories.pop(agent_id, None)

    async def close(self) -> None:
        """No-op, the in-memory store holds no external connections."""
