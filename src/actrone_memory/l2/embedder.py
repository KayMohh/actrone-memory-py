from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from actrone_memory.exceptions import EmbeddingError
from actrone_memory.logging import bind_logger
from actrone_memory.metrics import embedding_cache_hits_total, embedding_cache_misses_total

if TYPE_CHECKING:
    # `redis` is an OPTIONAL extra: only the CachedEmbedder (redis_qdrant backend) uses it,
    # and then only as a constructor type hint. Keeping it out of runtime imports means the
    # local-first install (`pip install actrone-memory`) pulls no redis.
    from redis.asyncio import Redis

log = bind_logger(__name__)

_CACHE_PREFIX = "embed:"
_WORD_RE = re.compile(r"[a-z0-9]+")


_RetryableFn = TypeVar("_RetryableFn", bound=Callable[..., Awaitable[Any]])


def _build_openai_retry() -> Callable[[_RetryableFn], _RetryableFn]:
    """Construct an OpenAI-specific retry decorator at import time.

    Importing ``openai`` lazily lets the rest of the module load when the
    optional dependency is missing (e.g. local-embedder-only deployments).
    The decorator allowlists only transient OpenAI client errors so that
    permanent failures (bad model name, invalid API key, schema mismatch)
    surface immediately instead of burning three retries.
    """
    transient: tuple[type[BaseException], ...]
    try:
        import openai

        transient = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
    except ImportError:
        # Fall back to network-level transients only. The decorator is still
        # required by the abstract Embedder API.
        transient = (ConnectionError, TimeoutError)

    # UNSAFE: `Any` on purpose. tenacity 9.2 types `retry(...)` as returning a
    # `_RetryDecorated[P, R]` wrapper, while 9.1 and earlier returned the function type unchanged.
    # A direct return fails mypy on 9.2, and a cast of the typed value is a `redundant-cast` error
    # on 9.1, so the value goes through `Any` to keep one declared type across the supported range
    # (`tenacity>=8.3,<10`). At runtime it is the same decorator, which hands back a callable of
    # the original signature.
    decorator: Any = retry(
        retry=retry_if_exception_type(transient),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        reraise=True,
    )
    return cast("Callable[[_RetryableFn], _RetryableFn]", decorator)


_EMBED_RETRY = _build_openai_retry()


class Embedder(ABC):
    """Abstract interface all embedding providers must implement."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single text string."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts.

        The returned list is always the same length as `texts`.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors this provider produces."""

    @property
    def relevance_threshold(self) -> float | None:
        """The cosine similarity at which this model's results turn from unrelated to relevant.

        Used as the admission threshold when ``MemoryConfig.relevance_threshold`` is not set.
        Override it in a custom embedder once you have measured it; the default None means the
        library default applies.
        """
        return None


# Calibrated admission thresholds for the built-in embedders, kept identical to the TypeScript
# library. Measured on a labelled set of 48 relevant and 528 unrelated query and memory pairs
# (2026-09-22). The lexical embedder only matches shared words, so no threshold makes it semantic:
# 0.30 is where it still recalls about 42% of relevant memories while admitting about 7% of
# unrelated ones. bge-small-en-v1.5 at 0.63 recalls about 88% with about 84% precision, and
# all-MiniLM-L6-v2 at 0.40 recalls about 88% with about 91% precision. The old single default
# (0.72) recalled 65% with bge-small, 23% with MiniLM and 4% with the lexical embedder.
LEXICAL_RELEVANCE_THRESHOLD = 0.3
BGE_SMALL_RELEVANCE_THRESHOLD = 0.63
MINILM_RELEVANCE_THRESHOLD = 0.4

# FastEmbedEmbedder thresholds by fastembed model id; a model not listed declares none.
_FASTEMBED_THRESHOLDS: dict[str, float] = {
    "BAAI/bge-small-en-v1.5": BGE_SMALL_RELEVANCE_THRESHOLD,
    "sentence-transformers/all-MiniLM-L6-v2": MINILM_RELEVANCE_THRESHOLD,
}


class CachedEmbedder(Embedder):
    """Wraps any Embedder with a Redis-backed cache (default TTL: 7 days).

    On a cache hit, returns the stored vector in < 1 ms with no API call.
    On a miss, calls the inner embedder and writes the result to the cache.
    """

    def __init__(self, inner: Embedder, cache: Redis, ttl_seconds: int = 604800) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl = ttl_seconds

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def relevance_threshold(self) -> float | None:
        # A cache changes nothing about the vectors, so the inner model's calibration still holds.
        return self._inner.relevance_threshold

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return f"{_CACHE_PREFIX}{digest}"

    async def embed(self, text: str) -> list[float]:
        key = self._cache_key(text)
        cached = await self._cache.get(key)
        if cached:
            embedding_cache_hits_total.inc()
            return json.loads(cached)  # type: ignore[no-any-return]

        embedding_cache_misses_total.inc()
        vector = await self._inner.embed(text)
        await self._cache.set(key, json.dumps(vector), ex=self._ttl)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed against the cache: one MGET for lookups, one pipeline for writes.

        Reading each key with its own round trip made a batch of n texts cost n
        sequential round trips, which dominates the call once n grows. MGET collapses
        that into one, and the misses are written back in a single pipeline.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        # One round trip for every lookup. Duplicate texts share a key, which MGET
        # returns once per requested position, so the mapping stays positional.
        cached_values = await self._cache.mget([self._cache_key(t) for t in texts])
        for i, (text, cached) in enumerate(zip(texts, cached_values, strict=True)):
            if cached:
                results[i] = json.loads(cached)
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        embedding_cache_hits_total.inc(len(texts) - len(miss_texts))
        embedding_cache_misses_total.inc(len(miss_texts))

        if miss_texts:
            vectors = await self._inner.embed_batch(miss_texts)

            if len(vectors) != len(miss_texts):
                raise EmbeddingError(
                    f"Embedder returned {len(vectors)} vectors for {len(miss_texts)} texts."
                )

            pipe = self._cache.pipeline()
            for idx, text, vector in zip(miss_indices, miss_texts, vectors, strict=True):
                results[idx] = vector
                pipe.set(self._cache_key(text), json.dumps(vector), ex=self._ttl)
            await pipe.execute()

        # Contract: returned list is always the same length as `texts`.
        # Any None slot indicates a corrupt cache entry or inner-embedder bug, fail loudly.
        for i, slot in enumerate(results):
            if slot is None:
                raise EmbeddingError(
                    f"CachedEmbedder produced no vector for index {i} "
                    f"(text length {len(texts[i])})."
                )

        return cast("list[list[float]]", results)


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-small (1536-dim) with automatic retry.

    Retries up to 3 times on any exception using exponential backoff with
    jitter, starting at 1 second and capped at 10 seconds.
    """

    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", dimensions: int = 1536
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    @_EMBED_RETRY
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            log.error(
                "openai.embed.failed", error=str(exc), model=self._model, batch_size=len(texts)
            )
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc


def _fnv1a_32(s: str) -> int:
    """FNV-1a 32-bit hash → non-negative int. Matches the TS ``hash32`` helper."""
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        # 32-bit FNV prime multiply, kept in uint32.
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free hashing embedder (a "hashing vectorizer").

    Hashes words into a fixed-dimension bag-of-words vector and L2-normalises it,
    so cosine similarity reflects word overlap. This is the **zero-dependency,
    zero-API-key default**, parity with the TypeScript ``LocalEmbedder``, which
    makes the whole library run fully offline with no model download and no
    external service. It is not semantically rich (no synonymy); for production
    recall quality pass an :class:`OpenAIEmbedder` or the sentence-transformers
    :class:`LocalEmbedder`.

    Deterministic: identical text always yields an identical vector, so no cache
    is needed.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def relevance_threshold(self) -> float | None:
        return LEXICAL_RELEVANCE_THRESHOLD

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        for word in _WORD_RE.findall(text.lower()):
            bucket = _fnv1a_32(word) % self._dimensions
            vec[bucket] += 1.0
        # L2-normalise so cosine similarity is a plain dot product.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    async def embed(self, text: str) -> list[float]:
        return self._embed_one(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class FastEmbedEmbedder(Embedder):
    """In-process ONNX dense embedder via ``fastembed`` (onnxruntime, no torch, no GPU).

    The default "real" dense tier: local-first, zero-egress after a one-time model download,
    no API key. The default model ``BAAI/bge-small-en-v1.5`` (384-dim) is small (~130 MB ONNX),
    lazily downloaded + cached on first construction, and **offline thereafter**, so the
    library's "no-egress" promise holds for every call after the initial fetch.

    Requires: ``pip install actrone-memory[onnx]``. CPU-bound encoding is offloaded to a thread
    pool so it never blocks the event loop.
    """

    _DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = _DEFAULT_MODEL, cache_dir: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError("Install onnx extras: pip install actrone-memory[onnx]") from exc

        self._model_name = model_name
        # Model files download on first construction (cached under ``cache_dir``); the object is
        # offline-only afterwards. A fetch failure (air-gapped first run) raises here so the
        # caller (build_local_embedder) can fall back to hashing and keep "no-egress" literal.
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._dimensions = self._resolve_dimensions(TextEmbedding, model_name, self._model)

    @staticmethod
    def _resolve_dimensions(text_embedding_cls: Any, model_name: str, model: Any) -> int:  # noqa: ANN401
        """Resolve the model's vector width from fastembed's catalogue, probing as a fallback."""
        try:
            catalogue = list(text_embedding_cls.list_supported_models())
        except Exception:  # catalogue availability/shape varies by version, probe instead.
            catalogue = []
        for desc in catalogue:
            dim = desc.get("dim") if isinstance(desc, dict) else None
            if desc.get("model") == model_name and isinstance(dim, int) and dim > 0:
                return dim
        first = next(iter(model.embed(["probe"])))
        return len(first)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def relevance_threshold(self) -> float | None:
        # Only measured models carry a threshold; for any other, set one in MemoryConfig.
        return _FASTEMBED_THRESHOLDS.get(self._model_name)

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import asyncio

        loop = asyncio.get_running_loop()

        def _encode() -> list[list[float]]:
            # fastembed's .embed() yields one numpy array per input, in order.
            return [vector.tolist() for vector in self._model.embed(texts)]

        return await loop.run_in_executor(None, _encode)


class LocalEmbedder(Embedder):
    """sentence-transformers/all-MiniLM-L6-v2 for offline / free usage (384-dim).

    Requires: pip install actrone-memory[local]
    CPU-bound encoding is offloaded to a thread pool to avoid blocking the event loop.
    """

    _MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    _DIMENSIONS = 384

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install local extras: pip install actrone-memory[local]") from exc

        self._model = SentenceTransformer(self._MODEL_NAME)

    @property
    def dimensions(self) -> int:
        return self._DIMENSIONS

    @property
    def relevance_threshold(self) -> float | None:
        return MINILM_RELEVANCE_THRESHOLD

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        from functools import partial

        # functools.partial keeps the bound encode() call typed by the
        # SentenceTransformer stubs, a plain lambda would erase the return
        # type to Any and fail mypy --strict downstream.
        loop = asyncio.get_running_loop()
        encode = partial(self._model.encode, texts, convert_to_numpy=True)
        embeddings = await loop.run_in_executor(None, encode)
        return [e.tolist() for e in embeddings]


def build_local_embedder(
    *, hashing_dimensions: int = 256, model_name: str | None = None
) -> Embedder:
    """Return the best available **local, offline, zero-egress** embedder, degrading gracefully.

    Tier order, the graceful chain behind ``embedding_provider="local"`` (the default):

    1. **In-process ONNX** (:class:`FastEmbedEmbedder`, ``[onnx]`` extra), dense, no torch. The
       preferred "real" recall tier.
    2. **sentence-transformers** (:class:`LocalEmbedder`, ``[local]`` extra), dense, torch-backed.
    3. **Hashing** (:class:`HashingEmbedder`), dependency-free lexical fallback that always works.

    Each tier is tried in turn; an ``ImportError`` (library absent) *or* a runtime model-fetch
    failure (air-gapped first run) falls through to the next. This guarantees a working embedder
    with **no API key and no unavoidable network call**, the model download is one-time and the
    hashing tier needs none, so the "local-first / no-egress" promise always holds.
    """
    try:
        onnx = FastEmbedEmbedder(model_name) if model_name else FastEmbedEmbedder()
        log.info(
            "memory.embedder.local",
            tier="fastembed-onnx",
            model=onnx.model_name,
            dim=onnx.dimensions,
        )
        return onnx
    except Exception as exc:  # ImportError (library absent) OR model-fetch failure (air-gapped)
        log.info("memory.embedder.local.skip", tier="fastembed-onnx", reason=str(exc))

    try:
        st = LocalEmbedder()
        log.info("memory.embedder.local", tier="sentence-transformers", dim=st.dimensions)
        return st
    except Exception as exc:  # ImportError (library absent) OR model-fetch failure (air-gapped)
        log.info("memory.embedder.local.skip", tier="sentence-transformers", reason=str(exc))

    log.warning(
        "memory.embedder.local.hashing",
        note="no local dense embedder available; using lexical hashing "
        '(pip install "actrone-memory[onnx]" for dense recall)',
    )
    return HashingEmbedder(dimensions=hashing_dimensions)
