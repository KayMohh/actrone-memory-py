"""Retry-policy regression tests for the L1 (Redis) and L2 (Qdrant) stores.

Both stores wrap their driver exceptions in ``StoreConnectionError`` before the
exception reaches the retry decorator. A predicate that matches only the raw
driver types therefore never fires, which silently reduces the documented
"3 attempts with exponential backoff + jitter" to a single attempt. These tests
pin the observable behaviour: transient failures are retried, permanent ones
fail fast.

Backoff waits are patched to zero so the assertions stay fast and deterministic.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from redis import exceptions as redis_exc
from tenacity import wait_none

from actrone_memory.exceptions import StoreConnectionError
from actrone_memory.l1.redis_store import RedisStore
from actrone_memory.l2.qdrant_store import QdrantStore
from actrone_memory.models import MemoryEntry, Turn

_ATTEMPTS = 3


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the backoff wait from every retried store method under test."""
    for method in (
        RedisStore.append_turn,
        RedisStore.get_recent_turns,
        RedisStore.turn_count,
        QdrantStore.upsert,
        QdrantStore.search,
    ):
        monkeypatch.setattr(method.retry, "wait", wait_none())  # type: ignore[attr-defined]


def _turn() -> Turn:
    return Turn(session_id="s1", user_message="hi", assistant_message="hello", token_count=4)


def _entry() -> MemoryEntry:
    return MemoryEntry(
        agent_id="a1",
        session_id="s1",
        content="fact",
        content_type="injected",
        embedding=[0.1, 0.2, 0.3, 0.4],
        token_count=1,
    )


def _unexpected(status_code: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status_code,
        reason_phrase="simulated",
        content=b"{}",
        headers=httpx.Headers(),
    )


class _CountingRedis:
    """Minimal Redis double that raises ``error`` and counts real attempts."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.attempts = 0

    def pipeline(self) -> _CountingRedis:
        return self

    def rpush(self, *_: object, **__: object) -> None: ...
    def ltrim(self, *_: object, **__: object) -> None: ...
    def expire(self, *_: object, **__: object) -> None: ...
    def hset(self, *_: object, **__: object) -> None: ...
    def hincrby(self, *_: object, **__: object) -> None: ...

    async def execute(self) -> None:
        self.attempts += 1
        raise self._error

    async def hget(self, *_: object, **__: object) -> None:
        return None

    async def lrange(self, *_: object, **__: object) -> list[bytes]:
        self.attempts += 1
        raise self._error

    async def llen(self, *_: object, **__: object) -> int:
        self.attempts += 1
        raise self._error


class _CountingQdrant:
    """Minimal AsyncQdrantClient double that raises ``error`` and counts attempts."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.attempts = 0

    async def upsert(self, **_: object) -> None:
        self.attempts += 1
        raise self._error

    async def query_points(self, **_: object) -> Any:
        self.attempts += 1
        raise self._error


def _redis_store(client: _CountingRedis) -> RedisStore:
    return RedisStore(client, session_ttl_hours=1, max_turns=10)  # type: ignore[arg-type]


def _qdrant_store(client: _CountingQdrant) -> QdrantStore:
    return QdrantStore(client, "c", 4, 0.7, 0.3)  # type: ignore[arg-type]


# ── Redis L1: transient failures are retried ────────────────────────────────


@pytest.mark.parametrize(
    "error",
    [
        redis_exc.ConnectionError("connection refused"),
        redis_exc.TimeoutError("read timed out"),
        redis_exc.BusyLoadingError("loading dataset in memory"),
    ],
    ids=["connection", "timeout", "busy_loading"],
)
@pytest.mark.asyncio
async def test_redis_append_turn_retries_transient(error: BaseException) -> None:
    client = _CountingRedis(error)
    store = _redis_store(client)

    with pytest.raises(StoreConnectionError):
        await store.append_turn("a1", "s1", _turn())

    assert client.attempts == _ATTEMPTS


@pytest.mark.asyncio
async def test_redis_reads_retry_transient() -> None:
    for call in ("get_recent_turns", "turn_count"):
        client = _CountingRedis(redis_exc.ConnectionError("connection refused"))
        store = _redis_store(client)

        with pytest.raises(StoreConnectionError):
            await getattr(store, call)("a1", "s1")

        assert client.attempts == _ATTEMPTS, f"{call} did not retry"


@pytest.mark.asyncio
async def test_redis_succeeds_on_a_later_attempt() -> None:
    """A transient blip must not surface to the caller once a retry succeeds."""
    calls = {"n": 0}

    class _FlakyPipe:
        def rpush(self, *_: object, **__: object) -> None: ...
        def ltrim(self, *_: object, **__: object) -> None: ...
        def expire(self, *_: object, **__: object) -> None: ...
        def hset(self, *_: object, **__: object) -> None: ...
        def hincrby(self, *_: object, **__: object) -> None: ...

        async def execute(self) -> list[object]:
            calls["n"] += 1
            if calls["n"] < 2:
                raise redis_exc.ConnectionError("connection refused")
            return [1, None, True, None, 1, True]

    class _FlakyRedis:
        def pipeline(self) -> _FlakyPipe:
            return _FlakyPipe()

        async def hget(self, *_: object, **__: object) -> None:
            return None

    store = RedisStore(_FlakyRedis(), session_ttl_hours=1, max_turns=10)  # type: ignore[arg-type]
    await store.append_turn("a1", "s1", _turn())

    assert calls["n"] == 2


# ── Redis L1: permanent failures fail fast ──────────────────────────────────


@pytest.mark.parametrize(
    "error",
    [
        redis_exc.ResponseError("unknown command"),
        redis_exc.DataError("invalid input type"),
        redis_exc.AuthenticationError("invalid password"),
    ],
    ids=["response", "data", "auth"],
)
@pytest.mark.asyncio
async def test_redis_does_not_retry_permanent(error: BaseException) -> None:
    client = _CountingRedis(error)
    store = _redis_store(client)

    with pytest.raises(StoreConnectionError):
        await store.append_turn("a1", "s1", _turn())

    assert client.attempts == 1


# ── Qdrant L2: transient retried, permanent 4xx fails fast ──────────────────


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("read timed out"),
        _unexpected(503),
        _unexpected(429),
    ],
    ids=["connect", "timeout", "server_error", "rate_limited"],
)
@pytest.mark.asyncio
async def test_qdrant_upsert_retries_transient(error: BaseException) -> None:
    client = _CountingQdrant(error)
    store = _qdrant_store(client)

    with pytest.raises(StoreConnectionError):
        await store.upsert(_entry())

    assert client.attempts == _ATTEMPTS


@pytest.mark.parametrize(
    "status_code",
    [400, 403, 404, 422],
    ids=["bad_request", "forbidden", "not_found", "unprocessable"],
)
@pytest.mark.asyncio
async def test_qdrant_does_not_retry_client_errors(status_code: int) -> None:
    """A 4xx is a permanent error (bad vector width, bad filter): fail on attempt one."""
    client = _CountingQdrant(_unexpected(status_code))
    store = _qdrant_store(client)

    with pytest.raises(StoreConnectionError):
        await store.upsert(_entry())

    assert client.attempts == 1


@pytest.mark.asyncio
async def test_qdrant_search_retries_transient() -> None:
    client = _CountingQdrant(httpx.ConnectError("connection refused"))
    store = _qdrant_store(client)

    with pytest.raises(StoreConnectionError):
        await store.search(agent_id="a1", query_embedding=[0.1, 0.2, 0.3, 0.4], threshold=0.5)

    assert client.attempts == _ATTEMPTS
