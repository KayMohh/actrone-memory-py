from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from actrone_memory.config import MemoryConfig, resolve_relevance_threshold
from actrone_memory.exceptions import (
    ConfigurationError,
    StoreConnectionError,
    TokenBudgetError,
    ValidationError,
)
from actrone_memory.extraction import FactExtractor, OpenAIFactExtractor
from actrone_memory.in_memory import InMemoryStore
from actrone_memory.l2.embedder import (
    CachedEmbedder,
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    build_local_embedder,
)
from actrone_memory.logging import bind_logger
from actrone_memory.metrics import (
    background_task_failed_total,
    background_task_in_flight,
    retrieve_context_duration_seconds,
)
from actrone_memory.models import (
    MemoryEntry,
    RetrievedContext,
    Sensitivity,
    SessionMetadata,
    ToolResult,
    Turn,
)
from actrone_memory.protocols import L1Store, L2Store
from actrone_memory.redaction import redact_memories, resolve_redaction
from actrone_memory.rerank import CrossEncoderReranker, build_reranker
from actrone_memory.tokens import TokenCounter, build_token_counter

if TYPE_CHECKING:
    # `redis` / `qdrant-client` are OPTIONAL extras: only the durable `redis_qdrant` backend
    # imports them, and it does so lazily inside `create()`. Keeping them out of module-load imports
    # means the local-first install (`pip install actrone-memory`) needs neither Redis nor Qdrant.
    from redis.asyncio import Redis

log = bind_logger(__name__)

# Input length limits enforced at the public API boundary.
_MAX_ID_LEN = 256
_MAX_MESSAGE_LEN = 100_000
_MAX_CONTENT_LEN = 100_000
_MAX_QUERY_LEN = 10_000
_MAX_TAGS = 50


def _validate_id(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ValidationError(field, "must not be empty")
    if len(value) > _MAX_ID_LEN:
        raise ValidationError(field, f"must be ≤ {_MAX_ID_LEN} characters")


def _validate_text(value: str, field: str, max_len: int) -> None:
    if len(value) > max_len:
        raise ValidationError(field, f"must be ≤ {max_len} characters")


class MemoryManager:
    """Two-tier persistent agent memory: Redis L1 (hot) + Qdrant L2 (cold semantic).

    Do not instantiate directly. Use ``MemoryManager.create()`` or the
    ``create_memory_manager()`` context manager.

    Usage::

        mm = await MemoryManager.create()
        await mm.store_turn(agent_id, session_id, user_msg, assistant_msg)
        ctx = await mm.retrieve_context(agent_id, session_id, query, token_budget=4096)
        await mm.close()

    Or let ``create_memory_manager()`` close it for you::

        async with create_memory_manager() as mm:
            ...
    """

    def __init__(
        self,
        l1: L1Store,
        l2: L2Store,
        embedder: Embedder,
        config: MemoryConfig,
        summariser: _Summariser | None = None,
        extractor: FactExtractor | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._embedder = embedder
        self._cfg = config
        self._summariser = summariser
        self._extractor = extractor
        self._reranker = reranker
        self._threshold = resolve_relevance_threshold(
            config.relevance_threshold, embedder.relevance_threshold
        )
        self._count_tokens: TokenCounter = build_token_counter(config.token_counter)
        # The sensitivity classes masked on read unless a call passes its own ``redact=``.
        self._redact_default = resolve_redaction(config.redact_sensitivities)
        # Strong references to background tasks. Without these, asyncio's GC
        # may collect the underlying task object mid-execution and silently
        # drop the work. close() drains this set before shutting down stores.
        self._background_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def relevance_threshold(self) -> float:
        """The admission threshold applied to long-term memories.

        ``config.relevance_threshold`` when set, else the embedder's calibrated threshold, else
        the library default.
        """
        return self._threshold

    async def store_turn(
        self,
        agent_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        tool_results: list[ToolResult] | None = None,
    ) -> str:
        """Persist a conversation turn to Redis L1.

        Also triggers background summarisation to Qdrant L2 once the session
        reaches ``summarise_after_turns`` turns.

        Args:
            agent_id: Unique identifier for the agent (e.g. ``"support-bot"``).
            session_id: Unique identifier for the conversation session.
            user_message: The user's raw message text.
            assistant_message: The agent's response text.
            tool_results: Optional tool call results from this turn.

        Returns:
            The turn ID (UUID string).

        Raises:
            ValidationError: If any argument fails length or format validation.
            StoreConnectionError: If the Redis write fails after retries.
        """
        _validate_id(agent_id, "agent_id")
        _validate_id(session_id, "session_id")
        _validate_text(user_message, "user_message", _MAX_MESSAGE_LEN)
        _validate_text(assistant_message, "assistant_message", _MAX_MESSAGE_LEN)

        token_count = self._count_tokens(f"{user_message}\n{assistant_message}")

        turn = Turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            tool_results=tool_results or [],
            token_count=token_count,
        )

        await self._l1.append_turn(agent_id, session_id, turn)
        log.debug("memory.turn.stored", agent_id=agent_id, session_id=session_id, turn_id=turn.id)

        if self._cfg.auto_summarise:
            count = await self._l1.turn_count(agent_id, session_id)
            # L1 caps the list at max_session_turns, so once count crosses the
            # threshold it stays there. Guard with a Redis SET-NX cooldown lock so
            # we summarise the session at most once per cooldown window and never
            # spawn duplicate/concurrent summaries (across the whole fleet), rather
            # than firing an LLM summary + L2 upsert on every subsequent turn.
            if count >= self._cfg.summarise_after_turns and await self._l1.try_acquire_summary_lock(
                agent_id, session_id, self._cfg.summarise_cooldown_seconds
            ):
                task = asyncio.create_task(
                    self._summarise_session(agent_id, session_id),
                    name=f"summarise:{agent_id}:{session_id}",
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        return turn.id

    async def retrieve_context(
        self,
        agent_id: str,
        session_id: str,
        query: str,
        token_budget: int,
        *,
        redact: Iterable[Sensitivity] | None = None,
    ) -> RetrievedContext:
        """Fetch relevant context for the next LLM call using the 4-phase pipeline.

        Phase 1, Parallel fetch: the L1 read (recent turns) runs concurrently with
        the whole L2 path (embed the query, search, optional rerank), so the fetch
        takes as long as the slower of the two rather than their sum.

        Phase 2, Budget allocation: the token budget is divided between system
        prompt, episodic memory, session turns, and the current turn using the
        fractions in MemoryConfig.

        Phase 3, Relevance ranking: L2 results are filtered at the configured
        threshold and ranked by reciprocal rank fusion of dense, lexical and
        recency signals (or ``0.7 × cosine_similarity + 0.3 × recency_score``
        when ``hybrid_retrieval`` is off).

        Phase 4, Priority pruning: if results exceed the allocated budget,
        oldest session turns are dropped first, then lowest-ranked memories.
        The system-prompt budget is never consumed by this method.

        Args:
            agent_id: Agent identifier.
            session_id: Session identifier.
            query: The user's current message; drives L2 semantic search.
            token_budget: Total tokens available for context. Must be > 0.
            redact: Sensitivity classes to mask in the returned memories, such as
                ``["pii", "sensitive"]``: each matching memory's content becomes
                ``"[redacted: <class>]"``. ``None`` uses ``config.redact_sensitivities``;
                an empty list masks nothing for this call. Turns are never masked.

        Returns:
            RetrievedContext with recent_turns, episodic_memories, and budget stats.

        Raises:
            TokenBudgetError: If ``token_budget`` is 0 or negative.
            ValidationError: If agent_id, session_id or redact are invalid.
            StoreConnectionError: If Redis or Qdrant fail after retries.
            EmbeddingError: If the embedding call fails after retries.
        """
        _validate_id(agent_id, "agent_id")
        _validate_id(session_id, "session_id")
        _validate_text(query, "query", _MAX_QUERY_LEN)
        if token_budget <= 0:
            raise TokenBudgetError(
                f"token_budget must be > 0, got {token_budget}",
                details={"token_budget": token_budget},
            )
        redact_classes = self._redaction_for(redact)

        start = time.monotonic()

        # Phase 1, parallel fetch: the L1 read runs alongside the whole L2 path, which has to
        # embed the query before it can search, so neither waits for the other.
        recent_turns, episodic_memories = await asyncio.gather(
            self._l1.get_recent_turns(agent_id, session_id),
            self._fetch_long_term(agent_id, query),
        )
        # Mask before pruning, so the budget counts the placeholder that is actually sent.
        episodic_memories = redact_memories(episodic_memories, redact_classes, self._count_tokens)

        # Phase 2, budget allocation.
        episodic_budget = int(token_budget * self._cfg.budget_fraction_episodic)
        session_budget = int(token_budget * self._cfg.budget_fraction_session)

        # Phase 4, priority-weighted pruning. O(n) greedy passes.
        pruned_turns = self._prune_turns(recent_turns, session_budget)
        pruned_memories = self._prune_memories(episodic_memories, episodic_budget)

        total_tokens = sum(t.token_count for t in pruned_turns) + sum(
            m.token_count for m in pruned_memories
        )
        elapsed_seconds = time.monotonic() - start
        elapsed_ms = elapsed_seconds * 1000
        retrieve_context_duration_seconds.observe(elapsed_seconds)

        log.info(
            "memory.context.retrieved",
            agent_id=agent_id,
            session_id=session_id,
            turns=len(pruned_turns),
            memories=len(pruned_memories),
            tokens_used=total_tokens,
            token_budget=token_budget,
            duration_ms=round(elapsed_ms, 2),
        )

        return RetrievedContext(
            recent_turns=pruned_turns,
            episodic_memories=pruned_memories,
            total_tokens_used=total_tokens,
            token_budget=token_budget,
            retrieval_duration_ms=elapsed_ms,
        )

    async def inject_memory(
        self,
        agent_id: str,
        content: str,
        importance: float = 0.8,
        session_id: str = "injected",
        topic_tags: list[str] | None = None,
        source: str = "injected",
        sensitivity: Sensitivity = "none",
    ) -> str:
        """Write a fact directly into L2 long-term memory.

        Use this to seed an agent with background knowledge before a conversation
        starts, e.g. user preferences, company policies, domain facts.

        Args:
            agent_id: The agent that should have access to this memory.
            content: Text of the memory. Write it as a clear, self-contained sentence.
            importance: Score from 0.0 to 1.0. Higher values surface this memory
                more readily. Recommended ≥ 0.8 for facts you always want recalled.
            session_id: Label for where this memory came from. Defaults to ``"injected"``.
            topic_tags: Optional keywords describing the memory topic.
            source: Provenance attribution, where this fact originated (e.g.
                ``"injected"``, ``"import:crm"``, ``"tool:web_search"``).
            sensitivity: PII/sensitivity classification for governance and
                right-to-erasure (``"none"`` | ``"low"`` | ``"pii"`` | ``"sensitive"``).

        Returns:
            The memory ID (UUID). Save this to delete the memory later.

        Raises:
            ValidationError: If any argument fails validation.
            StoreConnectionError: If the Qdrant write fails after retries.
            EmbeddingError: If embedding generation fails after retries.
        """
        _validate_id(agent_id, "agent_id")
        _validate_text(content, "content", _MAX_CONTENT_LEN)
        if not content.strip():
            raise ValidationError("content", "must not be blank")
        if not 0.0 <= importance <= 1.0:
            raise ValidationError("importance", "must be between 0.0 and 1.0")
        tags = topic_tags or []
        if len(tags) > _MAX_TAGS:
            raise ValidationError("topic_tags", f"must contain ≤ {_MAX_TAGS} tags")
        _validate_text(source, "source", _MAX_ID_LEN)

        embedding = await self._embedder.embed(content)
        entry = MemoryEntry(
            agent_id=agent_id,
            session_id=session_id,
            content=content,
            content_type="injected",
            embedding=embedding,
            importance_score=importance,
            topic_tags=tags,
            token_count=self._count_tokens(content),
            source=source,
            sensitivity=sensitivity,
        )
        await self._l2.upsert(entry)
        log.info(
            "memory.injected",
            agent_id=agent_id,
            memory_id=entry.id,
            importance=importance,
            source=source,
            sensitivity=sensitivity,
        )
        return entry.id

    async def delete_memory(self, agent_id: str, memory_id: str) -> None:
        """Permanently remove a memory from Qdrant L2 by ID.

        Args:
            agent_id: Agent that owns the memory (used for audit logging).
            memory_id: ID returned by ``inject_memory()`` or from ``search_memories()``.

        Raises:
            MemoryNotFoundError: If the memory_id does not exist in Qdrant.
            StoreConnectionError: If the Qdrant delete fails after retries.
        """
        _validate_id(agent_id, "agent_id")
        _validate_id(memory_id, "memory_id")
        await self._l2.delete(memory_id)
        log.info("memory.deleted", agent_id=agent_id, memory_id=memory_id)

    async def erase_agent_memories(self, agent_id: str, session_id: str | None = None) -> None:
        """Local right-to-erasure, irreversibly delete an agent's long-term memories.

        This is the governance seed that graduates to hosted *provable* erasure: in
        the OSS library it performs a hard local delete. If ``session_id`` is given,
        the session's short-term (L1) turns are cleared too; otherwise only the
        durable L2 store is wiped (L1 turns are ephemeral and expire on their TTL).

        Args:
            agent_id: The agent whose long-term memories to erase.
            session_id: Optional session whose short-term turns to also clear.

        Raises:
            ValidationError: If arguments are invalid.
            StoreConnectionError: If the underlying delete fails after retries.
        """
        _validate_id(agent_id, "agent_id")
        await self._l2.delete_agent_memories(agent_id)
        if session_id is not None:
            _validate_id(session_id, "session_id")
            await self._l1.clear_session(agent_id, session_id)
        log.info("memory.agent.erased", agent_id=agent_id, session_id=session_id)

    async def clear_session(self, agent_id: str, session_id: str) -> None:
        """Delete all Redis L1 turns for a session.

        Long-term Qdrant memories (summaries, injected facts) are NOT deleted, they persist
        across sessions by design.

        Raises:
            ValidationError: If arguments are invalid.
            StoreConnectionError: If the Redis delete fails after retries.
        """
        _validate_id(agent_id, "agent_id")
        _validate_id(session_id, "session_id")
        await self._l1.clear_session(agent_id, session_id)
        log.info("memory.session.cleared", agent_id=agent_id, session_id=session_id)

    async def search_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 10,
        *,
        redact: Iterable[Sensitivity] | None = None,
    ) -> list[MemoryEntry]:
        """Semantic search over L2 long-term memory for a given agent.

        Results are ranked the same way as in ``retrieve_context()``: reciprocal rank fusion of
        dense, lexical and recency signals, or ``0.7 × cosine_similarity + 0.3 × recency_score``
        when ``hybrid_retrieval`` is off.

        Args:
            agent_id: Agent whose memories to search.
            query: Search query in plain text.
            limit: Maximum number of results to return.
            redact: Sensitivity classes to mask in the results, as for ``retrieve_context()``.
                ``None`` uses ``config.redact_sensitivities``.

        Returns:
            List of MemoryEntry objects, sorted by descending relevance score.

        Raises:
            ValidationError: If arguments are invalid.
            StoreConnectionError: If Qdrant is unreachable.
            EmbeddingError: If embedding generation fails.
        """
        _validate_id(agent_id, "agent_id")
        _validate_text(query, "query", _MAX_QUERY_LEN)
        if limit < 1:
            raise ValidationError("limit", "must be ≥ 1")
        redact_classes = self._redaction_for(redact)

        embedding = await self._embedder.embed(query)
        # Over-fetch when reranking so the cross-encoder has a candidate pool to reorder.
        fetch_limit = max(limit, self._cfg.rerank_top_k) if self._reranker else limit
        candidates = await self._l2.search(
            agent_id=agent_id,
            query_embedding=embedding,
            threshold=self._threshold,
            limit=fetch_limit,
            query_text=query if self._cfg.hybrid_retrieval else None,
        )
        reranked = await self._maybe_rerank(query, candidates)
        return redact_memories(reranked[:limit], redact_classes, self._count_tokens)

    def _redaction_for(self, redact: Iterable[Sensitivity] | None) -> frozenset[Sensitivity]:
        """The classes to mask for one read: the call's own list, else the configured default."""
        return self._redact_default if redact is None else resolve_redaction(redact)

    async def _fetch_long_term(self, agent_id: str, query: str) -> list[MemoryEntry]:
        """The L2 half of the parallel fetch: embed the query, search, then the optional rerank.

        Kept as one coroutine so ``retrieve_context`` can run it concurrently with the L1 read.
        """
        query_embedding = await self._embedder.embed(query)
        candidates = await self._l2.search(
            agent_id=agent_id,
            query_embedding=query_embedding,
            threshold=self._threshold,
            limit=self._cfg.max_episodic_memories,
            query_text=query if self._cfg.hybrid_retrieval else None,
        )
        # Optional cross-encoder rerank over the top-K (precision lift; A4).
        return await self._maybe_rerank(query, candidates)

    async def _maybe_rerank(self, query: str, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Apply the optional cross-encoder rerank over the top-K candidates. No-op when the
        reranker is disabled/unavailable, so callers always get a valid ordering."""
        if self._reranker is None or not entries:
            return entries
        return await self._reranker.rerank(query, entries, top_k=self._cfg.rerank_top_k)

    async def get_session_metadata(self, agent_id: str, session_id: str) -> SessionMetadata | None:
        """Return basic stats about a session (turn count, created_at, last_active).

        Returns None if the session does not exist or has expired from Redis.
        """
        _validate_id(agent_id, "agent_id")
        _validate_id(session_id, "session_id")
        return await self._l1.get_session_metadata(agent_id, session_id)

    async def get_recent_turns(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[Turn]:
        """Return recent session turns, oldest first, capped at ``n``.

        The raw history read that framework memory adapters build on (for example a LangChain
        ``BaseChatMessageHistory`` or a LlamaIndex memory), so they do not have to reach into
        the L1 store directly. Defaults to everything L1 still retains.

        Raises:
            ValidationError: If arguments are invalid.
            StoreConnectionError: If the underlying read fails after retries.
        """
        _validate_id(agent_id, "agent_id")
        _validate_id(session_id, "session_id")
        if n is not None and n < 1:
            raise ValidationError("n", "must be ≥ 1")
        return await self._l1.get_recent_turns(agent_id, session_id, n=n)

    async def extract_memories(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[str]:
        """Extract durable facts from a session's recent turns and store them.

        Turns → atomic facts (``content_type="fact"``, ``source="extracted"``),
        each with an LLM-classified sensitivity. This is the "credible beyond turn
        storage" capability; it is **LLM-gated**, a ``FactExtractor`` must be
        configured (``MemoryConfig.extract_facts=True`` with the OpenAI provider),
        otherwise a ``ConfigurationError`` is raised.

        Args:
            agent_id: Agent identifier.
            session_id: Session whose turns to mine for facts.
            n: How many recent turns to consider (defaults to the summarise window).

        Returns:
            The memory IDs of the stored facts (empty if nothing durable was found).

        Raises:
            ConfigurationError: If no fact extractor is configured.
            ValidationError: If arguments are invalid.
        """
        _validate_id(agent_id, "agent_id")
        _validate_id(session_id, "session_id")
        if self._extractor is None:
            raise ConfigurationError(
                "Fact extraction is not enabled. Set MemoryConfig.extract_facts=True "
                "with the OpenAI embedding provider.",
                details={"extract_facts": self._cfg.extract_facts},
            )
        turns = await self._l1.get_recent_turns(
            agent_id, session_id, n=n or self._cfg.summarise_after_turns
        )
        return await self._store_extracted_facts(agent_id, session_id, turns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _store_extracted_facts(
        self, agent_id: str, session_id: str, turns: list[Turn]
    ) -> list[str]:
        """Run the fact extractor over ``turns`` and upsert each fact to L2.

        Best-effort: with no extractor or no turns it returns ``[]``. Extractor
        failures surface as an empty fact list (the extractor swallows its own
        errors), so this never breaks the caller.
        """
        if self._extractor is None or not turns:
            return []

        combined = "\n".join(
            f"User: {t.user_message}\nAssistant: {t.assistant_message}" for t in turns
        )
        facts = await self._extractor.extract(combined)
        if not facts:
            return []

        source_ids = [t.id for t in turns]
        stored: list[str] = []
        for fact in facts:
            embedding = await self._embedder.embed(fact.content)
            entry = MemoryEntry(
                agent_id=agent_id,
                session_id=session_id,
                content=fact.content,
                content_type="fact",
                embedding=embedding,
                importance_score=fact.importance,
                topic_tags=fact.topic_tags,
                token_count=self._count_tokens(fact.content),
                source_turn_ids=source_ids,
                source="extracted",
                sensitivity=fact.sensitivity,
            )
            await self._l2.upsert(entry)
            stored.append(entry.id)

        log.info(
            "memory.facts.extracted",
            agent_id=agent_id,
            session_id=session_id,
            facts=len(stored),
        )
        return stored

    def _prune_turns(self, turns: list[Turn], budget: int) -> list[Turn]:
        """Trim a session-turn list to fit within ``budget`` tokens.

        Strategy: walk newest → oldest and admit each turn while it fits.
        The first turn that overflows terminates the loop, older turns are
        guaranteed to drop because the next LLM call cares about *recent*
        context above all else.

        Invariants:
            * Returned list is in chronological order (oldest first).
            * Every admitted turn has token_count > 0.
            * Total tokens never exceed ``budget``.

        Complexity: O(n) over the input length; no allocations beyond the
        result list.

        Args:
            turns:  Session turns ordered oldest → newest, as returned by L1.
            budget: Maximum tokens allowed in the pruned set (≥ 0).

        Returns:
            A new list (the input is not mutated) containing the most recent
            turns that fit. Empty when ``budget == 0`` or all turns oversize.
        """
        result: list[Turn] = []
        remaining = budget
        for turn in reversed(turns):
            if turn.token_count <= remaining:
                result.append(turn)
                remaining -= turn.token_count
            else:
                break
        result.reverse()
        return result

    def _prune_memories(self, memories: list[MemoryEntry], budget: int) -> list[MemoryEntry]:
        """Trim a relevance-ranked memory list to fit within ``budget`` tokens.

        Strategy: walk highest-ranked → lowest, skipping individual memories
        that would overflow but continuing to consider smaller ones. This
        differs from ``_prune_turns`` (which stops on the first overflow)
        because the input here is pre-sorted by *relevance*, not recency, skipping one
        oversized memory to admit several smaller, equally relevant ones is the right trade-off.

        Invariants:
            * Returned list preserves the input's relative order.
            * Every admitted memory has token_count > 0.
            * Total tokens never exceed ``budget``.

        Complexity: O(n) over the input length; no allocations beyond the
        result list.

        Args:
            memories: Pre-sorted memories (highest relevance first) as
                returned by ``QdrantStore.search``.
            budget:   Maximum tokens allowed in the pruned set (≥ 0).

        Returns:
            A new list (the input is not mutated) containing the highest-
            ranked memories that fit the budget.
        """
        result: list[MemoryEntry] = []
        remaining = budget
        for mem in memories:
            if mem.token_count <= remaining:
                result.append(mem)
                remaining -= mem.token_count
        return result

    async def _summarise_session(self, agent_id: str, session_id: str) -> None:
        """Background task, compress recent turns into a Qdrant L2 summary.

        Uses an LLM when the OpenAI provider is configured (produces a genuine
        abstractive summary). Falls back to extractive summarisation (first +
        last quarter of text) when running in local-embedding mode.

        Failures are recorded on ``background_task_failed_total`` and
        re-raised when ``MemoryConfig.strict_background_errors`` is True
        (test/dev mode). In production the metric drives alerting; the task
        does not propagate the exception to its parent because asyncio
        background tasks have no caller to receive it.
        """
        gauge = background_task_in_flight.labels(task="summarise_session")
        gauge.inc()
        try:
            turns = await self._l1.get_recent_turns(
                agent_id, session_id, n=self._cfg.summarise_after_turns
            )
            if not turns:
                return

            combined = "\n".join(
                f"User: {t.user_message}\nAssistant: {t.assistant_message}" for t in turns
            )

            if self._summariser is not None:
                summary = await self._summariser.summarise(combined)
            else:
                # Extractive fallback: first + last quarter of the conversation text.
                # Used when no summariser is configured (local embedding mode).
                summary = _extractive_summary(combined)

            embedding = await self._embedder.embed(summary)
            entry = MemoryEntry(
                agent_id=agent_id,
                session_id=session_id,
                content=summary,
                content_type="summary",
                embedding=embedding,
                importance_score=0.6,
                token_count=self._count_tokens(summary),
                source_turn_ids=[t.id for t in turns],
                source="summary",
            )
            await self._l2.upsert(entry)
            log.info(
                "memory.session.summarised",
                agent_id=agent_id,
                session_id=session_id,
                memory_id=entry.id,
                turns_compressed=len(turns),
            )
            # Opt-in fact extraction rides the summarise cadence (already deduped by
            # the summary lock), so it stays cost-bounded and never per-turn.
            if self._extractor is not None:
                await self._store_extracted_facts(agent_id, session_id, turns)
        except asyncio.CancelledError:
            # Shutdown drain cancelled us; do not record as a failure.
            raise
        except Exception as exc:
            background_task_failed_total.labels(
                task="summarise_session",
                exception=type(exc).__name__,
            ).inc()
            log.exception(
                "memory.summarise.failed",
                agent_id=agent_id,
                session_id=session_id,
                exception=type(exc).__name__,
            )
            if self._cfg.strict_background_errors:
                raise
        finally:
            gauge.dec()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        config: MemoryConfig | None = None,
        *,
        l1: L1Store | None = None,
        l2: L2Store | None = None,
        embedder: Embedder | None = None,
        extractor: FactExtractor | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> MemoryManager:
        """Build a ready MemoryManager for the configured backend.

        The default backend is ``"memory"``, a zero-service, in-process store, and
        the default embedding provider is ``"local"``, which picks the best local
        embedder available (in-process ONNX, then sentence-transformers, then a
        dependency-free lexical hashing fallback). So ``create()`` needs no Redis,
        no Qdrant, and no API key (parity with the TypeScript on-ramp). Pass
        ``embedding_provider="hashing"`` to force the fallback and skip any model
        download. Set
        ``backend="redis_qdrant"`` (env ``ACTRONE_BACKEND=redis_qdrant``) for the
        durable, horizontally-scalable production path.

        Reads configuration from environment variables when config is None.

        Args:
            config: Settings to use. Read from the environment when None.
            l1: Custom hot-tier store satisfying the :class:`L1Store` protocol.
            l2: Custom long-term store satisfying the :class:`L2Store` protocol.
            embedder: Custom :class:`Embedder`, used instead of the configured provider.
            extractor: Custom :class:`FactExtractor`, used instead of the OpenAI one the
                config would build. Lets you extract facts without an OpenAI key.
            reranker: Custom reranker, used instead of the one ``rerank_enabled`` builds.

        Only Redis and Qdrant adapters ship with this package. ``l1`` / ``l2`` are the
        supported way to run any other engine (pgvector, Weaviate, Valkey, and so on)
        without constructing the manager by hand: anything satisfying the protocol
        works. Whatever you inject is used as-is, and the corresponding backend is not
        built or connected, so injecting both stores never opens a Redis or Qdrant
        connection. You own the lifecycle of an injected store; ``close()`` still calls
        its ``close()``.

        Raises:
            ConfigurationError: If required settings are missing or invalid.
            StoreConnectionError: If Redis or Qdrant cannot be reached
                (``redis_qdrant`` backend only).
        """
        cfg = config or MemoryConfig()
        cfg.validate_runtime()

        raw_embedder, summariser = cls._build_embedder(cfg)
        # An injected collaborator wins over the one the config would construct, so a
        # caller can supply a deterministic extractor (or one that needs no OpenAI key)
        # without dropping to the constructor the docs tell you not to call.
        resolved_extractor = extractor if extractor is not None else cls._build_extractor(cfg)
        resolved_reranker = (
            reranker
            if reranker is not None
            else build_reranker(enabled=cfg.rerank_enabled, model_name=cfg.rerank_model)
        )
        if embedder is not None:
            raw_embedder = embedder

        if l1 is not None and l2 is not None:
            # Fully injected: build no backend at all, so no connection is opened.
            log.info("memory.backend.injected", embedding_provider=cfg.embedding_provider)
            return cls(l1, l2, raw_embedder, cfg, summariser, resolved_extractor, resolved_reranker)

        if cfg.backend == "memory":
            # Zero-service local-first path. The hashing embedder is deterministic,
            # so no cache layer is needed; a real embedder is still fine here (just
            # uncached) for a local run against OpenAI/sentence-transformers.
            store = InMemoryStore(
                max_turns=cfg.max_session_turns,
                relevance_weight=cfg.relevance_weight,
                recency_weight=cfg.recency_weight,
            )
            log.info("memory.backend.local", embedding_provider=cfg.embedding_provider)
            return cls(
                l1 or store,
                l2 or store,
                raw_embedder,
                cfg,
                summariser,
                resolved_extractor,
                resolved_reranker,
            )

        # ── Durable production backend: Redis L1 + Qdrant L2 ──────────────
        # `redis` + `qdrant-client` are optional extras, imported lazily here so the
        # local-first install never pulls them. A clear message points at the extra when missing.
        try:
            from redis.asyncio import Redis
            from redis.asyncio.connection import ConnectionPool
            from redis.exceptions import RedisError

            from actrone_memory.l1.redis_store import RedisStore
            from actrone_memory.l2.qdrant_store import QdrantStore
        except ImportError as exc:  # pragma: no cover - exercised via import guard test
            raise ConfigurationError(
                "backend='redis_qdrant' requires the redis + qdrant extras: "
                "pip install actrone-memory[redis,qdrant]  (or [production])"
            ) from exc

        # One pool serves both the session store and the embedding cache. Building a
        # second pool here would double the connection count against the same server
        # and leave the cache's connections unowned by close().
        pool = ConnectionPool.from_url(
            cfg.redis_url,
            decode_responses=False,
            max_connections=cfg.redis_max_connections,
            socket_timeout=cfg.redis_socket_timeout,
            socket_connect_timeout=cfg.redis_socket_connect_timeout,
        )
        redis_client: Redis = Redis(connection_pool=pool)

        # Fail fast on an unreachable Redis rather than surfacing it on first write,
        # releasing the pool we just opened so a failed create() leaks nothing.
        try:
            await redis_client.ping()
        except RedisError as exc:
            await redis_client.aclose()
            raise StoreConnectionError("Redis", exc) from exc

        cached_embedder = CachedEmbedder(
            raw_embedder, redis_client, ttl_seconds=cfg.embedding_cache_ttl_seconds
        )

        resolved_l1: L1Store = (
            l1
            if l1 is not None
            else RedisStore(redis_client, cfg.session_ttl_hours, cfg.max_session_turns)
        )
        try:
            resolved_l2: L2Store = (
                l2
                if l2 is not None
                else await QdrantStore.from_url(
                    cfg.qdrant_url,
                    api_key=cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None,
                    collection=cfg.qdrant_collection,
                    # Size the collection to the embedder actually in use, not a fixed
                    # constant, the hashing/local embedders have their own dimensions.
                    dimensions=raw_embedder.dimensions,
                    relevance_weight=cfg.relevance_weight,
                    recency_weight=cfg.recency_weight,
                    timeout=cfg.qdrant_timeout,
                )
            )
        except BaseException:
            # A half-built manager has no close(), so release Redis before propagating.
            await redis_client.aclose()
            raise

        return cls(
            resolved_l1,
            resolved_l2,
            cached_embedder,
            cfg,
            summariser,
            resolved_extractor,
            resolved_reranker,
        )

    @staticmethod
    def _build_extractor(cfg: MemoryConfig) -> FactExtractor | None:
        """Construct the LLM fact extractor when opt-in *and* an LLM is available.

        Extraction requires an OpenAI key (the only LLM provider wired in OSS), so
        it is silently skipped for the hashing/local embedding providers even if
        ``extract_facts`` is set, extraction is best-effort enrichment.
        """
        if not cfg.extract_facts or cfg.embedding_provider != "openai":
            return None
        return OpenAIFactExtractor(api_key=_resolve_openai_key(cfg), model=cfg.summarisation_model)

    @staticmethod
    def _build_embedder(cfg: MemoryConfig) -> tuple[Embedder, _Summariser | None]:
        """Construct the embedder (and matching summariser) for the config.

        Returns the raw (uncached) embedder plus an LLM summariser when the OpenAI
        provider is configured, else ``None`` (the extractive fallback is used).
        """
        if cfg.embedding_provider == "openai":
            api_key = _resolve_openai_key(cfg)
            return (
                OpenAIEmbedder(
                    api_key=api_key,
                    model=cfg.embedding_model,
                    dimensions=cfg.embedding_dimensions,
                ),
                _OpenAISummariser(api_key=api_key, model=cfg.summarisation_model),
            )
        if cfg.embedding_provider == "local":
            # Graceful local chain (default): in-process ONNX → sentence-transformers → hashing.
            return build_local_embedder(hashing_dimensions=cfg.hashing_dimensions), None
        # "hashing", the dependency-free, deterministic, fully-offline lexical embedder (explicit).
        return HashingEmbedder(dimensions=cfg.hashing_dimensions), None

    async def close(self) -> None:
        """Close all connections to Redis and Qdrant.

        Drains in-flight background tasks (e.g. session summarisation) up to
        ``MemoryConfig.shutdown_grace_seconds``. Tasks that have not finished
        within the grace window are cancelled so the process can exit.
        """
        if self._background_tasks:
            grace = self._cfg.shutdown_grace_seconds
            pending = list(self._background_tasks)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=grace,
                )
            except TimeoutError:
                log.warning(
                    "memory.shutdown.background_drain_timeout",
                    pending=len(self._background_tasks),
                    grace_seconds=grace,
                )
                for task in self._background_tasks:
                    task.cancel()
                # Give cancelled tasks a brief window to clean up.
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
        await self._l1.close()
        await self._l2.close()

    async def __aenter__(self) -> MemoryManager:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


# ------------------------------------------------------------------
# Summariser protocol and implementations
# ------------------------------------------------------------------


class _Summariser:
    """Internal protocol, summarise a block of conversation text into a concise paragraph."""

    async def summarise(self, text: str) -> str:
        raise NotImplementedError


class _OpenAISummariser(_Summariser):
    """Abstractive summariser using GPT-4o-mini.

    Produces a 2-3 sentence summary capturing key topics and facts.
    Cost: ~$0.00003 per summarisation at typical conversation lengths.
    """

    _SYSTEM_PROMPT = (
        "You are a precise conversation summariser. "
        "Summarise the following conversation in 2-3 sentences. "
        "Capture the key topics discussed and any important facts or decisions. "
        "Be concise. Do not add opinions or information not present in the conversation."
    )

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def summarise(self, text: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return response.choices[0].message.content or _extractive_summary(text)
        except Exception as exc:
            log.warning("summariser.llm.failed", error=str(exc), model=self._model)
            return _extractive_summary(text)


def _resolve_openai_key(cfg: MemoryConfig) -> str:
    """Return the OpenAI API key as a plain string, failing loudly if missing.

    Centralises the None-narrowing that ``MemoryConfig.openai_api_key: SecretStr | None``
    requires when ``embedding_provider == "openai"``. ``validate_runtime()``
    asserts presence at startup, but a defensive check here lets mypy --strict
    erase the Optional cleanly without a ``# type: ignore``.
    """
    if cfg.openai_api_key is None:
        raise ConfigurationError(
            "ACTRONE_OPENAI_API_KEY is required when embedding_provider='openai'.",
            details={"embedding_provider": cfg.embedding_provider},
        )
    return cfg.openai_api_key.get_secret_value()


def _extractive_summary(text: str, max_chars: int = 800) -> str:
    """Fallback: return first + last quarter of the text, capped at max_chars.

    Used when no LLM summariser is available (local embedding mode) or when
    the LLM call fails. Not as useful as an abstractive summary but ensures
    L2 always gets some signal.
    """
    if len(text) <= max_chars:
        return text
    quarter = max_chars // 4
    return text[: quarter * 2] + "\n...\n" + text[-quarter:]


# ------------------------------------------------------------------
# Convenience context manager
# ------------------------------------------------------------------


@asynccontextmanager
async def create_memory_manager(config: MemoryConfig | None = None) -> AsyncIterator[MemoryManager]:
    """Async context manager that creates and automatically closes a MemoryManager.

    Usage::

        async with create_memory_manager() as mm:
            await mm.store_turn(...)
    """
    mm = await MemoryManager.create(config)
    try:
        yield mm
    finally:
        await mm.close()
