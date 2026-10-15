"""Redis-protocol compatibility: Valkey runs the L1 store unmodified.

There is no separate Valkey adapter and there should not be one. ``RedisStore`` uses only
standard Redis commands (RPUSH, LTRIM, LRANGE, LLEN, EXPIRE, HSET, HGET, HGETALL, HINCRBY,
DEL, SET NX EX and pipelines), all of which Valkey implements, so the existing adapter works
as-is. The same reasoning covers other protocol-compatible servers such as DragonflyDB,
ElastiCache and Upstash.

That is a claim worth testing rather than asserting, which is what this file does: it points
the real ``RedisStore`` at a real Valkey container and runs the published conformance suite
plus the behaviours that depend on Redis-specific semantics (TTL, list trimming, SET NX).

Run with: uv run pytest tests/integration/test_valkey_compatibility.py -m integration
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from actrone_memory.models import Turn
from actrone_memory.testing import check_l1_store

pytestmark = pytest.mark.integration

pytest.importorskip("redis", reason="redis extra not installed")
redis_module = pytest.importorskip("testcontainers.redis", reason="testcontainers not installed")

from actrone_memory.l1.redis_store import RedisStore  # noqa: E402

# Valkey is the Redis fork the Linux Foundation took on after the licence change; the 8.x
# line is wire-compatible with Redis 7.2, which is what the adapter targets.
_VALKEY_IMAGE = "valkey/valkey:8-alpine"


@pytest_asyncio.fixture
async def valkey_url():
    with redis_module.RedisContainer(_VALKEY_IMAGE) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"


def _turn(user: str) -> Turn:
    return Turn(session_id="s1", user_message=user, assistant_message="ok", token_count=3)


@pytest.mark.asyncio
async def test_valkey_passes_the_published_l1_conformance_suite(valkey_url: str) -> None:
    """The unmodified Redis adapter satisfies the full hot-tier contract on Valkey."""

    async def factory() -> RedisStore:
        return await RedisStore.from_url(valkey_url, session_ttl_hours=1, max_turns=50)

    await check_l1_store(factory)


@pytest.mark.asyncio
async def test_valkey_honours_list_trimming(valkey_url: str) -> None:
    """LTRIM, so retention is enforced by the server and not in Python."""
    store = await RedisStore.from_url(valkey_url, session_ttl_hours=1, max_turns=3)
    try:
        for i in range(6):
            await store.append_turn("agent-1", "trim", _turn(f"msg-{i}"))

        turns = await store.get_recent_turns("agent-1", "trim")

        assert [t.user_message for t in turns] == ["msg-3", "msg-4", "msg-5"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_valkey_honours_set_nx_for_the_summary_lock(valkey_url: str) -> None:
    """SET key NX EX, the primitive that keeps summarisation to one holder fleet-wide."""
    store = await RedisStore.from_url(valkey_url, session_ttl_hours=1, max_turns=10)
    try:
        results = await asyncio.gather(
            *(store.try_acquire_summary_lock("agent-1", "race", 60) for _ in range(12))
        )

        assert sum(1 for won in results if won) == 1, f"expected one winner, got {results}"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_valkey_honours_key_expiry(valkey_url: str) -> None:
    """EXPIRE, so an abandoned session ages out without an application sweep."""
    store = await RedisStore.from_url(valkey_url, session_ttl_hours=1, max_turns=10)
    try:
        await store.append_turn("agent-1", "ttl", _turn("will expire"))

        # Reach past the adapter to assert the TTL was actually applied server-side.
        ttl = await store._r.ttl("agent:agent-1:session:ttl:turns")  # noqa: SLF001
        assert 0 < ttl <= 3600
    finally:
        await store.close()
