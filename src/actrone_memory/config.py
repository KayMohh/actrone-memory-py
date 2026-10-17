from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from actrone_memory.exceptions import ConfigurationError
from actrone_memory.models import Sensitivity

# Admission threshold for an embedder that declares no calibrated ``relevance_threshold``, such as
# the OpenAI embedder or a custom one you pass in yourself.
DEFAULT_RELEVANCE_THRESHOLD = 0.72


def resolve_relevance_threshold(configured: float | None, declared: float | None) -> float:
    """Return the admission threshold a manager applies.

    Args:
        configured: ``MemoryConfig.relevance_threshold``, or None when unset.
        declared: The embedder's calibrated ``relevance_threshold``, or None if it declares none.

    Returns:
        The configured value when set, else the embedder's, else DEFAULT_RELEVANCE_THRESHOLD.
    """
    if configured is not None:
        return configured
    if declared is not None:
        return declared
    return DEFAULT_RELEVANCE_THRESHOLD


class MemoryConfig(BaseSettings):
    """All configuration for actrone-memory.

    Values are read from environment variables with the ``ACTRONE_`` prefix.
    You can also pass values directly when constructing this object.

    Required only when ``embedding_provider="openai"``:
        ACTRONE_OPENAI_API_KEY

    All other settings have sensible defaults. The default provider is ``"local"``, a
    local-first, zero-egress dense embedder that needs no API key (see below).
    """

    model_config = SettingsConfigDict(env_prefix="ACTRONE_", env_file=".env", extra="ignore")

    # ── Backend ──────────────────────────────────────────────────────────
    # "memory" (default) is the zero-service, in-process local-first backend:
    # no Redis, no Qdrant, no API key, parity with the TypeScript on-ramp.
    # "redis_qdrant" is the durable, horizontally-scalable production backend.
    backend: Literal["memory", "redis_qdrant"] = "memory"

    # ── Store connections (used only when backend="redis_qdrant") ─────────
    redis_url: str = "redis://localhost:6379"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    # ── Redis connection pool ────────────────────────────────────────────
    # Raise max_connections under high concurrency (many concurrent agents).
    redis_max_connections: int = 10
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0

    # ── Qdrant client ────────────────────────────────────────────────────
    qdrant_timeout: float = 10.0

    # ── Embedding ────────────────────────────────────────────────────────
    # "local" (default): best available local, offline, zero-egress dense embedder, degrading
    #   gracefully, in-process ONNX (fastembed, [onnx] extra) → sentence-transformers ([local]
    #   extra) → dependency-free lexical hashing. No API key; one-time model download, then offline.
    # "openai": text-embedding-3-small (needs ACTRONE_OPENAI_API_KEY).
    # "hashing": force the dependency-free, deterministic, fully-offline lexical embedder.
    embedding_provider: Literal["openai", "local", "hashing"] = "local"
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    # Vector width for the hashing embedder (matches the TS LocalEmbedder default).
    hashing_dimensions: int = 256
    embedding_cache_ttl_seconds: int = 604800  # 7 days

    # ── Summarisation ────────────────────────────────────────────────────
    # Model used for LLM-based conversation summarisation (OpenAI only).
    # GPT-4o-mini is cheap (~$0.00003 per call) and fast enough for background use.
    summarisation_model: str = "gpt-4o-mini"

    # ── Session (L1 Redis) ───────────────────────────────────────────────
    session_ttl_hours: int = 24
    max_session_turns: int = 50

    # ── Episodic memory (L2 Qdrant) ──────────────────────────────────────
    qdrant_collection: str = "agent_memories"
    max_episodic_memories: int = 500
    # Minimum cosine similarity for a long-term memory to be admitted. Leave unset (None) to use the
    # threshold the embedder was calibrated for (``Embedder.relevance_threshold``), falling back to
    # DEFAULT_RELEVANCE_THRESHOLD for an embedder that declares none. Similarity scales differ by
    # model, so one fixed number cannot suit them all: the lexical hashing embedder scores relevant
    # text near 0.24, while bge-small scores unrelated text near 0.48.
    relevance_threshold: float | None = None
    relevance_weight: float = 0.7  # recency_weight = 1 - relevance_weight
    recency_weight: float = 0.3
    # Hybrid retrieval: among the threshold-admitted candidates, fuse the embedding
    # (dense) ranking with a BM25 (lexical) ranking and recency via Reciprocal Rank Fusion, so an
    # exact-keyword match the embedder under-ranks still surfaces. Admission (cosine ≥ threshold) is
    # unchanged. Set False to force the classic single-channel dense+recency blend.
    hybrid_retrieval: bool = True

    # Cross-encoder reranking (opt-in). A cross-encoder rescores (query, memory) pairs
    # jointly, more precise than the embedder, but O(K) inferences, so it only reorders the top
    # ``rerank_top_k`` of an over-fetched set. Needs the [onnx] extra; degrades to the un-reranked
    # order if unavailable. Off by default (adds a model download + per-query latency).
    rerank_enabled: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20

    # ── Fact extraction (LLM-gated, opt-in) ──────────────────────────────
    # When True *and* an LLM embedder/summariser is configured (OpenAI provider),
    # durable atomic facts are extracted from a session's turns and stored as
    # first-class memories (content_type="fact", source="extracted"). Off by
    # default because it costs one LLM call per extraction. Rides the same cadence
    # as auto-summarisation to stay cost-bounded.
    extract_facts: bool = False

    # ── Redaction on read (opt-in) ───────────────────────────────────────
    # Sensitivity classes whose memories come back masked from retrieve_context() and
    # search_memories(), and so from every framework adapter built on them: the content becomes
    # "[redacted: <class>]" and the embedding is dropped. Stored memories are not changed. Tags come
    # from the fact extractor or from inject_memory(); raw turns carry none, so they are never
    # redacted. A per-call ``redact=`` overrides this. Env: ACTRONE_REDACT_SENSITIVITIES='["pii"]'.
    redact_sensitivities: list[Sensitivity] = Field(default_factory=list)

    # ── Auto-summarisation ───────────────────────────────────────────────
    auto_summarise: bool = True
    summarise_after_turns: int = 20
    # L1 caps the turn list at max_session_turns, so once a session reaches
    # summarise_after_turns the count stays above the threshold and would
    # otherwise spawn a summarisation on *every* subsequent turn. This cooldown
    # is the TTL of a Redis SET-NX lock that both deduplicates concurrent/fleet
    # summarisations of the same session and throttles re-summarisation.
    summarise_cooldown_seconds: int = 300

    # ── Background-task error handling ────────────────────────────────────
    # When True, background-task failures (e.g. _summarise_session) re-raise
    # so test suites catch regressions deterministically. Production code
    # should leave this False and rely on metrics / structured logs.
    strict_background_errors: bool = False

    # ── Graceful shutdown ─────────────────────────────────────────────────
    # Maximum time MemoryManager.close() waits for in-flight background
    # tasks to drain before cancelling them. A configurable grace period
    # keeps shutdown bounded; 30 s is a sane default for most deployments.
    shutdown_grace_seconds: float = 30.0

    # ── Token counting ───────────────────────────────────────────────────
    # "heuristic" (default): about 4 characters per token, the same counter as the TypeScript
    #   library, with no dependency and no network access.
    # "tiktoken": exact cl100k_base counts. Needs the [tiktoken] extra, and tiktoken downloads the
    #   encoding file once on first use unless TIKTOKEN_CACHE_DIR already holds it.
    token_counter: Literal["heuristic", "tiktoken"] = "heuristic"  # noqa: S105  (a counter name, not a secret)

    # ── Token budget fractions (must sum to 1.0) ─────────────────────────
    budget_fraction_system: float = 0.30
    budget_fraction_episodic: float = 0.25
    budget_fraction_session: float = 0.35
    budget_fraction_current_turn: float = 0.10

    def validate_runtime(self) -> None:
        """Validate settings that require cross-field checks.

        Call this before opening connections. Raises ConfigurationError with a
        clear message if any required setting is missing or inconsistent.
        """
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "ACTRONE_OPENAI_API_KEY is required when embedding_provider='openai'.",
                details={"embedding_provider": self.embedding_provider},
            )

        budget_total = (
            self.budget_fraction_system
            + self.budget_fraction_episodic
            + self.budget_fraction_session
            + self.budget_fraction_current_turn
        )
        if abs(budget_total - 1.0) > 1e-6:
            raise ConfigurationError(
                f"Token budget fractions must sum to 1.0, got {budget_total:.4f}.",
                details={"sum": budget_total},
            )

        if self.relevance_threshold is not None and not (0.0 < self.relevance_threshold < 1.0):
            raise ConfigurationError(
                f"relevance_threshold must be between 0 and 1, got {self.relevance_threshold}.",
                details={"relevance_threshold": self.relevance_threshold},
            )

        if self.redis_max_connections < 1:
            raise ConfigurationError(
                "redis_max_connections must be ≥ 1.",
                details={"redis_max_connections": self.redis_max_connections},
            )

        # The ranking blend is relevance_weight × cosine + recency_weight × recency.
        # Weights that do not sum to 1.0 silently rescale every score, which shifts
        # the ordering against the documented formula instead of failing.
        rank_total = self.relevance_weight + self.recency_weight
        if abs(rank_total - 1.0) > 1e-6:
            raise ConfigurationError(
                f"relevance_weight + recency_weight must sum to 1.0, got {rank_total:.4f}.",
                details={
                    "relevance_weight": self.relevance_weight,
                    "recency_weight": self.recency_weight,
                },
            )

        for name in ("max_session_turns", "max_episodic_memories", "rerank_top_k"):
            value = getattr(self, name)
            if value < 1:
                raise ConfigurationError(f"{name} must be ≥ 1, got {value}.", details={name: value})
