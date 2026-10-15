"""Metrics-wiring tests.

The Prometheus collectors are only useful if something records them. Declaring a
metric and never observing it is worse than omitting it, because a ``/metrics``
scrape then reports a confident zero. These tests assert each collector actually
moves when the code path it measures runs.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis import exceptions as redis_exc

from actrone_memory.config import MemoryConfig
from actrone_memory.exceptions import StoreConnectionError
from actrone_memory.l1.redis_store import RedisStore
from actrone_memory.l2.embedder import CachedEmbedder
from actrone_memory.manager import MemoryManager
from actrone_memory.metrics import REGISTRY
from tests.conftest import ConstantEmbedder


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


@pytest.mark.asyncio
async def test_retrieve_context_records_duration() -> None:
    before = _sample("actrone_memory_retrieve_context_duration_seconds_count")

    mm = await MemoryManager.create(MemoryConfig(backend="memory", embedding_provider="hashing"))
    try:
        await mm.store_turn("a1", "s1", "hello", "hi there")
        await mm.retrieve_context("a1", "s1", "hello", token_budget=1024)
    finally:
        await mm.close()

    assert _sample("actrone_memory_retrieve_context_duration_seconds_count") == before + 1


@pytest.mark.asyncio
async def test_store_op_records_success() -> None:
    labels = {"tier": "l1", "operation": "turn_count", "outcome": "success"}
    before = _sample("actrone_memory_store_op_total", labels)

    client = MagicMock()
    client.llen = AsyncMock(return_value=3)
    store = RedisStore(client, session_ttl_hours=1, max_turns=10)

    assert await store.turn_count("a1", "s1") == 3
    assert _sample("actrone_memory_store_op_total", labels) == before + 1
    assert _sample("actrone_memory_store_op_duration_seconds_count", labels) == before + 1


@pytest.mark.asyncio
async def test_store_op_records_error_outcome() -> None:
    labels = {"tier": "l1", "operation": "clear_session", "outcome": "error"}
    before = _sample("actrone_memory_store_op_total", labels)

    client = MagicMock()
    client.delete = AsyncMock(side_effect=redis_exc.ResponseError("unknown command"))
    store = RedisStore(client, session_ttl_hours=1, max_turns=10)

    with pytest.raises(StoreConnectionError):
        await store.clear_session("a1", "s1")

    assert _sample("actrone_memory_store_op_total", labels) == before + 1


@pytest.mark.asyncio
async def test_embedding_cache_counters_move() -> None:
    hits_before = _sample("actrone_memory_embedding_cache_hits_total")
    misses_before = _sample("actrone_memory_embedding_cache_misses_total")

    cache = AsyncMock()
    cache.get.return_value = None
    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    await embedder.embed("a miss")

    assert _sample("actrone_memory_embedding_cache_misses_total") == misses_before + 1

    cache.get.return_value = json.dumps([0.9, 0.8, 0.7, 0.6]).encode()
    await embedder.embed("a hit")

    assert _sample("actrone_memory_embedding_cache_hits_total") == hits_before + 1


@pytest.mark.asyncio
async def test_embedding_cache_counters_count_a_whole_batch() -> None:
    hits_before = _sample("actrone_memory_embedding_cache_hits_total")
    misses_before = _sample("actrone_memory_embedding_cache_misses_total")

    vector = json.dumps([0.9, 0.8, 0.7, 0.6]).encode()
    cache = AsyncMock()
    cache.mget.return_value = [vector, None, None]
    pipe = MagicMock()
    pipe.set = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    cache.pipeline = MagicMock(return_value=pipe)

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    await embedder.embed_batch(["hit", "miss one", "miss two"])

    assert _sample("actrone_memory_embedding_cache_hits_total") == hits_before + 1
    assert _sample("actrone_memory_embedding_cache_misses_total") == misses_before + 2
