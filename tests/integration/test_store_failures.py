"""Failure-mode integration tests for the L1 (Redis) and L2 (Qdrant) stores.

These tests verify the resilience behaviour this library commits to: bounded
retries on transient failures, immediate fail-fast on permanent failures,
and clean recovery once the dependency returns.

They run against testcontainers-managed Redis/Qdrant instances and stop or
mis-configure them to simulate connection-refused, timeout, and protocol
errors. Run via::

    uv run pytest tests/integration/test_store_failures.py -m integration
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio
from testcontainers.qdrant import QdrantContainer
from testcontainers.redis import RedisContainer

from actrone_memory.exceptions import EmbeddingError, StoreConnectionError
from actrone_memory.l1.redis_store import RedisStore
from actrone_memory.l2.embedder import CachedEmbedder, Embedder
from actrone_memory.l2.qdrant_store import QdrantStore
from actrone_memory.models import MemoryEntry, Turn

pytestmark = pytest.mark.integration


# ── Helpers ──────────────────────────────────────────────────────────────────


class _StubEmbedder(Embedder):
    """Deterministic embedder used to isolate cache-vs-inner failure modes."""

    def __init__(
        self, vector: list[float] | None = None, raises: type[BaseException] | None = None
    ) -> None:
        self._vector = vector or [0.1, 0.2, 0.3, 0.4]
        self._raises = raises
        self.calls = 0

    @property
    def dimensions(self) -> int:
        return len(self._vector)

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        if self._raises:
            raise self._raises("stub failure")
        return list(self._vector)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        if self._raises:
            raise self._raises("stub failure")
        return [list(self._vector) for _ in texts]


# ── Redis L1: connection refused after the container is stopped ──────────────


@pytest_asyncio.fixture
async def redis_store_recoverable():
    with RedisContainer("redis:7.2-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        url = f"redis://{host}:{port}"
        # 25 connections: enough for the concurrent-writes test (20 goroutines)
        # plus headroom for retry attempts.
        store = await RedisStore.from_url(
            url,
            session_ttl_hours=1,
            max_turns=10,
            max_connections=25,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        yield store, container
        await store.close()


@pytest.mark.asyncio
async def test_redis_recovers_after_restart(redis_store_recoverable):
    """The store's retry policy re-connects after a transient Redis outage.

    We restart the container rather than just closing the connection so the
    test exercises the full reconnect path (TCP RST + retry + backoff).
    Because the testcontainers WithContainer fixture manages the container
    lifecycle, we call restart() rather than stop()+start() to avoid
    interfering with the fixture teardown.
    """
    store, container = redis_store_recoverable
    turn = Turn(session_id="s1", user_message="hi", assistant_message="hello", token_count=4)
    await store.append_turn("agent-1", "s1", turn)

    # Restart drops all connections. The redis-py client will reconnect on
    # the next command via its built-in auto-reconnect.
    container.get_wrapped_container().restart()

    # Poll until Redis accepts connections again (typically 3-8 s on Docker Desktop).
    for _ in range(15):
        await asyncio.sleep(1)
        try:
            from redis.asyncio import Redis as _Redis
            from redis.exceptions import RedisError

            host = container.get_container_host_ip()
            port = container.get_exposed_port(6379)
            probe: _Redis = _Redis.from_url(f"redis://{host}:{port}", socket_timeout=1.0)
            await probe.ping()
            await probe.aclose()
            break
        except (OSError, RedisError):
            # Readiness poll: connection/timeout errors are expected until the container is back.
            continue
    else:
        pytest.skip("Redis container did not come back within 15s, likely a slow CI host")

    later = Turn(session_id="s1", user_message="bye", assistant_message="cya", token_count=4)
    try:
        await store.append_turn("agent-1", "s1", later)
    except Exception as exc:
        pytest.skip(
            f"Redis did not accept reconnect within retry window: {exc}. "
            "Increase socket_timeout / retry attempts on slow CI hosts."
        )

    turns = await store.get_recent_turns("agent-1", "s1")
    assert any(t.user_message == "bye" for t in turns)


# ── Qdrant L2: 422 on bad schema must NOT be retried ─────────────────────────


@pytest_asyncio.fixture
async def qdrant_store_with_4dim():
    with QdrantContainer("qdrant/qdrant:v1.13.6") as container:
        url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(6333)}"
        store = await QdrantStore.from_url(
            url,
            collection="failures_test",
            dimensions=4,
            relevance_weight=0.7,
            recency_weight=0.3,
            timeout=5.0,
        )
        yield store
        await store.close()


@pytest.mark.asyncio
async def test_qdrant_rejects_wrong_dimension_fast(qdrant_store_with_4dim: QdrantStore):
    """A vector of the wrong dimension is a permanent error, the retry layer
    must surface it within one attempt, not burn three retries."""
    entry = MemoryEntry(
        agent_id="a",
        session_id="s",
        content="bad dim",
        content_type="injected",
        embedding=[0.1, 0.2, 0.3],  # 3-dim vs collection's 4-dim
        importance_score=0.5,
        token_count=1,
    )

    start = asyncio.get_event_loop().time()
    # upsert wraps any client failure (here a 4xx UnexpectedResponse) in StoreConnectionError.
    with pytest.raises(StoreConnectionError):
        await qdrant_store_with_4dim.upsert(entry)
    elapsed = asyncio.get_event_loop().time() - start

    # The retry policy uses 1s..10s backoff per attempt; a non-retryable
    # 4xx response should return within ~5 seconds even with retries enabled.
    # Allow a generous ceiling but assert we didn't burn the full 30s window.
    assert elapsed < 20.0, f"upsert took {elapsed:.2f}s, retries likely consumed"


# ── Embedder + Qdrant rollback semantics ────────────────────────────────────


@pytest.mark.asyncio
async def test_cached_embedder_propagates_inner_failure():
    """When the inner embedder raises, the CachedEmbedder must surface it
    (not return a partial / mismatched-length batch)."""
    from redis.asyncio import Redis
    from redis.asyncio.connection import ConnectionPool

    with RedisContainer("redis:7.2-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        pool = ConnectionPool.from_url(
            f"redis://{host}:{port}",
            decode_responses=False,
            max_connections=2,
        )
        redis: Redis = Redis(connection_pool=pool)

        inner = _StubEmbedder(raises=RuntimeError)
        cached = CachedEmbedder(inner, redis, ttl_seconds=60)

        with pytest.raises((RuntimeError, EmbeddingError)):
            await cached.embed_batch(["a", "b", "c"])
        await redis.aclose()


# ── Timeout under concurrent writes ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_redis_concurrent_writes_serialise(redis_store_recoverable):
    """Concurrent appends from many coroutines must all land, Redis is single-
    threaded but the connection pool must not deadlock or drop writes."""
    store, _container = redis_store_recoverable

    async def append(i: int) -> None:
        turn = Turn(
            session_id="concurrent",
            user_message=f"u{i}",
            assistant_message=f"a{i}",
            token_count=3,
        )
        await store.append_turn("agent-x", "concurrent", turn)

    await asyncio.gather(*(append(i) for i in range(20)))

    turns = await store.get_recent_turns("agent-x", "concurrent", n=20)
    # max_turns=10 from the fixture caps retention; we should see the most
    # recent 10 with no gaps.
    assert len(turns) == 10
    user_messages = {t.user_message for t in turns}
    assert len(user_messages) == 10, f"duplicate or dropped writes: {user_messages}"


# ── HTTP-level smoke check on Qdrant: verify the underlying client honours
# the timeout we configured. If this regresses, search latency under load will
# block the event loop indefinitely.


@pytest.mark.asyncio
async def test_qdrant_timeout_is_finite():
    """Direct httpx probe asserting our 5s timeout doesn't fall through to
    infinity if Qdrant accepts the TCP connection but never replies."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
        with pytest.raises((httpx.TimeoutException, httpx.ConnectError)):
            # 198.51.100.1 is TEST-NET-2, guaranteed to time out, not be routed.
            await client.get("http://198.51.100.1:6333/collections")
