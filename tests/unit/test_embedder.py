from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from actrone_memory.exceptions import EmbeddingError
from actrone_memory.l2.embedder import CachedEmbedder, HashingEmbedder, OpenAIEmbedder
from tests.conftest import ConstantEmbedder


@pytest.mark.asyncio
async def test_cached_embedder_returns_vector():
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.return_value = None

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    result = await embedder.embed("hello world")
    assert result == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_cached_embedder_uses_cache_on_hit():
    cached_value = json.dumps([0.9, 0.8, 0.7, 0.6]).encode()
    cache = AsyncMock()
    cache.get.return_value = cached_value

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    result = await embedder.embed("hello world")
    # Should return cached value, not inner embedder's value
    assert result == [0.9, 0.8, 0.7, 0.6]


def _batch_cache(*values: bytes | None) -> AsyncMock:
    """Cache double for ``embed_batch``: one MGET read plus a pipelined write."""
    cache = AsyncMock()
    cache.mget.return_value = list(values)
    pipe = MagicMock()
    pipe.set = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    cache.pipeline = MagicMock(return_value=pipe)
    return cache


@pytest.mark.asyncio
async def test_cached_embedder_batch_partial_cache_hit():
    cache = _batch_cache(json.dumps([0.9, 0.8, 0.7, 0.6]).encode(), None)

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    results = await embedder.embed_batch(["cached text", "uncached text"])
    assert len(results) == 2
    assert results[0] == [0.9, 0.8, 0.7, 0.6]
    assert results[1] == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_cached_embedder_batch_reads_in_one_round_trip():
    """All lookups go out as a single MGET, never one GET per text."""
    cache = _batch_cache(None, None, None)

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    await embedder.embed_batch(["a", "b", "c"])

    cache.mget.assert_awaited_once()
    assert len(cache.mget.await_args.args[0]) == 3
    cache.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_cached_embedder_batch_writes_misses_in_one_pipeline():
    cache = _batch_cache(json.dumps([0.9, 0.8, 0.7, 0.6]).encode(), None, None)

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    await embedder.embed_batch(["cached", "miss one", "miss two"])

    pipe = cache.pipeline.return_value
    assert pipe.set.call_count == 2, "only the misses should be written back"
    pipe.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_cached_embedder_batch_all_hits_skips_inner_and_pipeline():
    vector = json.dumps([0.9, 0.8, 0.7, 0.6]).encode()
    cache = _batch_cache(vector, vector)
    inner = ConstantEmbedder()

    embedder = CachedEmbedder(inner, cache)
    results = await embedder.embed_batch(["one", "two"])

    assert results == [[0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6]]
    cache.pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_cached_embedder_batch_empty_input_short_circuits():
    cache = _batch_cache()

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    assert await embedder.embed_batch([]) == []
    cache.mget.assert_not_awaited()


@pytest.mark.asyncio
async def test_cached_embedder_writes_to_cache_on_miss():
    cache = AsyncMock()
    cache.get.return_value = None

    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    await embedder.embed("new text")
    cache.set.assert_called_once()


def test_cached_embedder_dimensions_delegates_to_inner():
    cache = AsyncMock()
    embedder = CachedEmbedder(ConstantEmbedder(), cache)
    assert embedder.dimensions == 4


# ------------------------------------------------------------------
# OpenAIEmbedder (mocked openai client)
# ------------------------------------------------------------------


def _make_openai_embedder() -> tuple[OpenAIEmbedder, MagicMock]:
    """Construct OpenAIEmbedder then swap out the client with a mock."""
    embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small", dimensions=1536)
    mock_client = MagicMock()
    embedder._client = mock_client
    return embedder, mock_client


def test_openai_embedder_dimensions():
    embedder, _ = _make_openai_embedder()
    assert embedder.dimensions == 1536


@pytest.mark.asyncio
async def test_openai_embedder_embed_returns_vector():
    embedder, mock_client = _make_openai_embedder()

    mock_item = MagicMock()
    mock_item.embedding = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.data = [mock_item]
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    result = await embedder.embed("hello world")
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_openai_embedder_embed_batch_returns_vectors():
    embedder, mock_client = _make_openai_embedder()

    items = [MagicMock(embedding=[float(i)]) for i in range(3)]
    mock_response = MagicMock()
    mock_response.data = items
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    results = await embedder.embed_batch(["a", "b", "c"])
    assert len(results) == 3


@pytest.mark.asyncio
async def test_openai_embedder_embed_batch_empty_returns_empty():
    embedder, _ = _make_openai_embedder()
    results = await embedder.embed_batch([])
    assert results == []


@pytest.mark.asyncio
async def test_openai_embedder_raises_embedding_error_on_failure():
    embedder, mock_client = _make_openai_embedder()
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=Exception("API error"))

    with pytest.raises(EmbeddingError, match="OpenAI embedding failed"):
        await embedder.embed("fail")


# ------------------------------------------------------------------
# HashingEmbedder (dependency-free, deterministic default)
# ------------------------------------------------------------------


def test_hashing_embedder_rejects_bad_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        HashingEmbedder(dimensions=0)


def test_hashing_embedder_dimensions_property():
    assert HashingEmbedder(dimensions=128).dimensions == 128
    assert HashingEmbedder().dimensions == 256  # default matches TS LocalEmbedder


@pytest.mark.asyncio
async def test_hashing_embedder_is_deterministic():
    emb = HashingEmbedder()
    v1 = await emb.embed("the quick brown fox")
    v2 = await emb.embed("the quick brown fox")
    assert v1 == v2
    assert len(v1) == 256


@pytest.mark.asyncio
async def test_hashing_embedder_is_l2_normalised():
    emb = HashingEmbedder()
    vec = await emb.embed("hello world hello")
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_hashing_embedder_empty_text_is_zero_vector():
    emb = HashingEmbedder(dimensions=16)
    vec = await emb.embed("!!!")  # no word characters
    assert vec == [0.0] * 16


@pytest.mark.asyncio
async def test_hashing_embedder_word_overlap_scores_higher():
    from actrone_memory.in_memory import cosine_similarity

    emb = HashingEmbedder()
    query = await emb.embed("database connection pool settings")
    related = await emb.embed("connection pool settings for the database")
    unrelated = await emb.embed("the weather in Paris is sunny today")
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


@pytest.mark.asyncio
async def test_hashing_embedder_batch_matches_single():
    emb = HashingEmbedder()
    batch = await emb.embed_batch(["alpha beta", "gamma delta"])
    assert batch[0] == await emb.embed("alpha beta")
    assert batch[1] == await emb.embed("gamma delta")
