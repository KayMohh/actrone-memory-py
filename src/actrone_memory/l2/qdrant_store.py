from __future__ import annotations

from datetime import UTC, datetime

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from actrone_memory.exceptions import MemoryNotFoundError, StoreConnectionError
from actrone_memory.logging import bind_logger
from actrone_memory.metrics import track_store_op
from actrone_memory.models import ContentType, MemoryEntry

log = bind_logger(__name__)

_COLLECTION = "agent_memories"

# Retry only TRANSIENT Qdrant errors. Retrying everything re-attempts programmer
# bugs and permanent 4xx responses, which wastes time and masks the root cause,
# so the transient set is an explicit allowlist.
_QDRANT_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ResponseHandlingException,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
)

# Retryable HTTP status codes: server-side faults plus explicit rate limiting.
# Every other 4xx (wrong vector width, malformed filter, bad API key) is a
# permanent error and must surface on the first attempt.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_transient_qdrant_failure(exc: BaseException) -> bool:
    """Retry predicate that looks through the store's own error wrapper.

    Every method below converts a client error into ``StoreConnectionError``
    before returning, so a predicate matching only raw client types would never
    match and the retry policy would silently never fire. The original client
    exception is preserved on ``StoreConnectionError.cause``.
    """
    cause = exc.cause if isinstance(exc, StoreConnectionError) else exc
    if isinstance(cause, UnexpectedResponse):
        return cause.status_code in _RETRYABLE_STATUS
    return isinstance(cause, _QDRANT_TRANSIENT_EXCEPTIONS)


_QDRANT_RETRY = retry(
    retry=retry_if_exception(_is_transient_qdrant_failure),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    reraise=True,
)


class QdrantStore:
    """Qdrant L2 cold episodic memory store.

    Target read latency: ~10 ms P99.
    All write and search operations retry up to 3 times on transient failures
    with exponential backoff + jitter.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        dimensions: int,
        relevance_weight: float,
        recency_weight: float,
    ) -> None:
        self._client = client
        self._collection = collection
        self._dimensions = dimensions
        self._relevance_w = relevance_weight
        self._recency_w = recency_weight

    # ------------------------------------------------------------------
    # Collection bootstrap
    # ------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        try:
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(
                        size=self._dimensions,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                log.info("qdrant.collection.created", collection=self._collection)
        except Exception as exc:
            raise StoreConnectionError("Qdrant", exc) from exc

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @_QDRANT_RETRY
    @track_store_op("l2", "upsert")
    async def upsert(self, entry: MemoryEntry) -> None:
        """Write a single MemoryEntry to Qdrant. The entry must have an embedding."""
        if entry.embedding is None:
            raise ValueError(f"MemoryEntry {entry.id} has no embedding, embed before upserting.")

        point = qmodels.PointStruct(
            id=entry.id,
            vector=entry.embedding,
            payload={
                "agent_id": entry.agent_id,
                "session_id": entry.session_id,
                "content": entry.content,
                "content_type": entry.content_type,
                "importance_score": entry.importance_score,
                "topic_tags": entry.topic_tags,
                "token_count": entry.token_count,
                "timestamp": entry.timestamp.isoformat(),
                "source_turn_ids": entry.source_turn_ids,
                "source": entry.source,
                "sensitivity": entry.sensitivity,
            },
        )
        try:
            await self._client.upsert(collection_name=self._collection, points=[point])
        except Exception as exc:
            log.error("qdrant.upsert.failed", entry_id=entry.id, error=str(exc))
            raise StoreConnectionError("Qdrant", exc) from exc

    @_QDRANT_RETRY
    @track_store_op("l2", "upsert_batch")
    async def upsert_batch(self, entries: list[MemoryEntry]) -> None:
        """Write multiple MemoryEntries in one Qdrant request; skips embeddingless entries."""
        if not entries:
            return
        points = [
            qmodels.PointStruct(
                id=e.id,
                vector=e.embedding,
                payload={
                    "agent_id": e.agent_id,
                    "session_id": e.session_id,
                    "content": e.content,
                    "content_type": e.content_type,
                    "importance_score": e.importance_score,
                    "topic_tags": e.topic_tags,
                    "token_count": e.token_count,
                    "timestamp": e.timestamp.isoformat(),
                    "source_turn_ids": e.source_turn_ids,
                    "source": e.source,
                    "sensitivity": e.sensitivity,
                },
            )
            for e in entries
            if e.embedding is not None
        ]
        if not points:
            return
        try:
            await self._client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:
            raise StoreConnectionError("Qdrant", exc) from exc

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @_QDRANT_RETRY
    @track_store_op("l2", "search")
    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        threshold: float,
        limit: int = 20,
        content_types: list[ContentType] | None = None,
        query_text: str | None = None,
    ) -> list[MemoryEntry]:
        """Hybrid semantic search, dense + lexical + recency fused with RRF.

        Qdrant returns the dense (cosine) candidates over ``threshold``; when ``query_text`` is
        given, those candidates are re-ranked by fusing the server cosine score with BM25 over the
        content and recency via Reciprocal Rank Fusion. Without ``query_text`` (or no shared terms)
        it degrades to the classic ``relevance × cosine + recency × recency`` blend. Only memories
        with cosine ≥ ``threshold`` are ever returned (admission unchanged).

        Time complexity: O(n log n) for re-ranking n Qdrant results.
        """
        from actrone_memory.retrieval import fuse_channels

        filters = [qmodels.FieldCondition(key="agent_id", match=qmodels.MatchValue(value=agent_id))]
        if content_types:
            filters.append(
                qmodels.FieldCondition(
                    key="content_type",
                    match=qmodels.MatchAny(any=content_types),
                )
            )

        try:
            # search() was removed in qdrant-client 1.10; use query_points() per
            # the qdrant-client 1.x migration guide.
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query_embedding,
                limit=limit * 3,  # over-fetch so re-ranking has candidates to choose from
                score_threshold=threshold,
                query_filter=qmodels.Filter(must=filters),
                with_payload=True,
            )
        except Exception as exc:
            raise StoreConnectionError("Qdrant", exc) from exc

        now_ts = datetime.now(UTC).timestamp()
        entries: dict[str, MemoryEntry] = {}
        dense: list[tuple[float, str]] = []
        recency: dict[str, float] = {}
        documents: dict[str, str] = {}

        for r in response.points:
            payload = r.payload or {}
            ts_str: str = payload.get("timestamp", datetime.now(UTC).isoformat())
            entry_ts = datetime.fromisoformat(ts_str).timestamp()
            age_seconds = max(0.0, now_ts - entry_ts)

            point_id = str(r.id)
            recency[point_id] = max(0.0, 1.0 - age_seconds / (30 * 86400))
            dense.append((r.score, point_id))
            documents[point_id] = payload.get("content", "")
            entries[point_id] = MemoryEntry(
                id=point_id,
                agent_id=payload.get("agent_id", agent_id),
                session_id=payload.get("session_id", ""),
                content=payload.get("content", ""),
                content_type=payload.get("content_type", "turn"),
                importance_score=payload.get("importance_score", 0.5),
                topic_tags=payload.get("topic_tags", []),
                token_count=payload.get("token_count", 0),
                timestamp=datetime.fromisoformat(ts_str),
                source_turn_ids=payload.get("source_turn_ids", []),
                source=payload.get("source", "unknown"),
                sensitivity=payload.get("sensitivity", "none"),
            )

        if not entries:
            return []

        dense_ranking = [pid for _, pid in sorted(dense, key=lambda x: x[0], reverse=True)]
        fused_order = fuse_channels(
            ids=list(entries.keys()),
            dense_ranking=dense_ranking,
            documents=documents,
            recency=recency,
            query_text=query_text,
            relevance_weight=self._relevance_w,
            recency_weight=self._recency_w,
        )
        if fused_order is None:
            # No lexical signal, classic relevance × cosine + recency × recency blend.
            blended = sorted(
                dense,
                key=lambda x: self._relevance_w * x[0] + self._recency_w * recency[x[1]],
                reverse=True,
            )
            return [entries[pid] for _, pid in blended[:limit]]
        return [entries[pid] for pid in fused_order[:limit]]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    @_QDRANT_RETRY
    @track_store_op("l2", "delete")
    async def delete(self, memory_id: str) -> None:
        """Delete a memory by ID.

        Raises ``MemoryNotFoundError`` when Qdrant reports the point is absent.
        Note that Qdrant treats deleting an unknown point id as a success, so a
        missing id is usually a silent no-op here; callers that need a hard
        guarantee should read the memory back. ``InMemoryStore`` always raises.
        """
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.PointIdsList(points=[memory_id]),
            )
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                raise MemoryNotFoundError(memory_id) from exc
            raise StoreConnectionError("Qdrant", exc) from exc
        except Exception as exc:
            raise StoreConnectionError("Qdrant", exc) from exc

    @_QDRANT_RETRY
    @track_store_op("l2", "delete_agent_memories")
    async def delete_agent_memories(self, agent_id: str) -> None:
        """Delete all memories for a given agent. Irreversible."""
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="agent_id", match=qmodels.MatchValue(value=agent_id)
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            raise StoreConnectionError("Qdrant", exc) from exc

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def from_url(
        cls,
        url: str,
        *,
        api_key: str | None = None,
        collection: str = _COLLECTION,
        dimensions: int = 1536,
        relevance_weight: float = 0.7,
        recency_weight: float = 0.3,
        timeout: float = 10.0,  # noqa: ASYNC109, passthrough to Qdrant client, not an asyncio timeout
        grpc_port: int = 6334,
    ) -> QdrantStore:
        """Connect to Qdrant and ensure the collection exists.

        Args:
            url: Qdrant server URL (e.g. http://localhost:6333).
            api_key: API key for Qdrant Cloud; None for self-hosted.
            collection: Name of the Qdrant collection to use.
            dimensions: Embedding vector size. Must match your embedder.
            timeout: Per-request timeout in seconds.
        """
        try:
            # Qdrant's client takes integer-second timeouts; round the float param.
            client = AsyncQdrantClient(url=url, api_key=api_key, timeout=int(timeout))
            store = cls(client, collection, dimensions, relevance_weight, recency_weight)
            await store.ensure_collection()
            return store
        except Exception as exc:
            raise StoreConnectionError("Qdrant", exc) from exc

    async def close(self) -> None:
        await self._client.close()
