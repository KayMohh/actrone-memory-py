from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

from actrone_memory.exceptions import StoreConnectionError
from actrone_memory.l1.redis_store import RedisStore
from actrone_memory.models import Turn


def _make_store(mock_redis: MagicMock) -> RedisStore:
    return RedisStore(client=mock_redis, session_ttl_hours=1, max_turns=5)


def _make_redis() -> MagicMock:
    """AsyncMock for the Redis client, pipeline() is sync, all others are async."""
    redis = MagicMock()
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock()
    redis.hincrby = AsyncMock()
    redis.expire = AsyncMock()
    redis.lrange = AsyncMock(return_value=[])
    redis.llen = AsyncMock(return_value=0)
    redis.delete = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.ping = AsyncMock()
    return redis


def _turn(session_id: str = "s1") -> Turn:
    return Turn(session_id=session_id, user_message="hi", assistant_message="hello", token_count=10)


# ------------------------------------------------------------------
# Key helpers
# ------------------------------------------------------------------


def test_turns_key():
    assert RedisStore._turns_key("agent-1", "sess-1") == "agent:agent-1:session:sess-1:turns"


def test_meta_key():
    assert RedisStore._meta_key("agent-1", "sess-1") == "agent:agent-1:session:sess-1:meta"


# ------------------------------------------------------------------
# append_turn
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_turn_executes_pipeline():
    # pipeline() is synchronous in redis-py, use MagicMock, not AsyncMock
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, None, True, None, 1, True])

    redis = _make_redis()
    redis.pipeline.return_value = pipe

    store = _make_store(redis)
    await store.append_turn("agent-1", "s1", _turn())

    pipe.rpush.assert_called_once()
    pipe.ltrim.assert_called_once()
    pipe.execute.assert_called_once()


@pytest.mark.asyncio
async def test_append_turn_raises_on_redis_error():
    redis = _make_redis()
    redis.pipeline.side_effect = RedisError("connection refused")

    store = _make_store(redis)
    with pytest.raises(StoreConnectionError) as exc_info:
        await store.append_turn("agent-1", "s1", _turn())

    assert exc_info.value.store == "Redis"


# ------------------------------------------------------------------
# get_recent_turns
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_recent_turns_deserialises_correctly():
    turn = _turn()
    redis = _make_redis()
    redis.lrange.return_value = [turn.model_dump_json().encode()]

    store = _make_store(redis)
    turns = await store.get_recent_turns("agent-1", "s1")

    assert len(turns) == 1
    assert turns[0].user_message == "hi"
    assert turns[0].session_id == "s1"


@pytest.mark.asyncio
async def test_get_recent_turns_returns_empty_on_no_data():
    redis = _make_redis()
    store = _make_store(redis)
    turns = await store.get_recent_turns("agent-1", "s1")
    assert turns == []


@pytest.mark.asyncio
async def test_get_recent_turns_raises_on_redis_error():
    redis = _make_redis()
    redis.lrange.side_effect = RedisError("timeout")

    store = _make_store(redis)
    with pytest.raises(StoreConnectionError):
        await store.get_recent_turns("agent-1", "s1")


# ------------------------------------------------------------------
# clear_session
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_session_deletes_both_keys():
    redis = _make_redis()
    store = _make_store(redis)
    await store.clear_session("agent-1", "s1")

    redis.delete.assert_called_once_with(
        "agent:agent-1:session:s1:turns",
        "agent:agent-1:session:s1:meta",
    )


@pytest.mark.asyncio
async def test_clear_session_raises_on_redis_error():
    redis = _make_redis()
    redis.delete.side_effect = RedisError("error")

    store = _make_store(redis)
    with pytest.raises(StoreConnectionError):
        await store.clear_session("agent-1", "s1")


# ------------------------------------------------------------------
# get_session_metadata
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_metadata_returns_none_when_empty():
    redis = _make_redis()
    store = _make_store(redis)
    meta = await store.get_session_metadata("agent-1", "s1")
    assert meta is None


@pytest.mark.asyncio
async def test_get_session_metadata_parses_correctly():
    redis = _make_redis()
    redis.hgetall.return_value = {
        b"agent_id": b"agent-1",
        b"session_id": b"s1",
        b"turn_count": b"3",
        b"created_at": b"2026-01-01T00:00:00+00:00",
        b"last_active": b"2026-01-01T01:00:00+00:00",
    }

    store = _make_store(redis)
    meta = await store.get_session_metadata("agent-1", "s1")

    assert meta is not None
    assert meta.turn_count == 3
    assert meta.agent_id == "agent-1"


# ------------------------------------------------------------------
# turn_count
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_count_returns_llen():
    redis = _make_redis()
    redis.llen.return_value = 7

    store = _make_store(redis)
    count = await store.turn_count("agent-1", "s1")
    assert count == 7


@pytest.mark.asyncio
async def test_turn_count_raises_on_redis_error():
    redis = _make_redis()
    redis.llen.side_effect = RedisError("timeout")

    store = _make_store(redis)
    with pytest.raises(StoreConnectionError):
        await store.turn_count("agent-1", "s1")


# ------------------------------------------------------------------
# get_session_metadata error path
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_metadata_raises_on_redis_error():
    redis = _make_redis()
    redis.hgetall.side_effect = RedisError("connection lost")

    store = _make_store(redis)
    with pytest.raises(StoreConnectionError):
        await store.get_session_metadata("agent-1", "s1")


# ------------------------------------------------------------------
# close
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_calls_aclose():
    redis = _make_redis()
    redis.aclose = AsyncMock()

    store = _make_store(redis)
    await store.close()
    redis.aclose.assert_called_once()


# ------------------------------------------------------------------
# from_url factory (mocked at Redis level)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_from_url_raises_on_ping_failure():
    with (
        patch("actrone_memory.l1.redis_store.Redis") as mock_redis_cls,
        patch("actrone_memory.l1.redis_store.ConnectionPool") as mock_pool_cls,
    ):
        mock_client = AsyncMock()
        mock_client.ping.side_effect = RedisError("refused")
        mock_redis_cls.return_value = mock_client
        mock_pool_cls.from_url.return_value = MagicMock()

        with pytest.raises(StoreConnectionError) as exc_info:
            await RedisStore.from_url("redis://badhost:6379", 1, 5)
        assert exc_info.value.store == "Redis"
