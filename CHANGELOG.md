# Changelog

All notable changes to `actrone-memory` follow [Semantic Versioning](https://semver.org/).

---

## [0.3.0] - 2026-10-17

### Added

- **Redaction on read.** `retrieve_context()` and `search_memories()` take `redact=["pii", "sensitive"]`,
  and `MemoryConfig.redact_sensitivities` (`ACTRONE_REDACT_SENSITIVITIES`) sets it for every read, so
  every framework adapter built on those methods is covered. A matching memory comes back with its
  content replaced by `[redacted: <class>]` and its embedding dropped; the stored memory is not changed,
  and `redact=[]` turns it off for one call. It acts only on sensitivity tags: raw turns carry none, so
  they are never masked. `redacted_content()` and `SENSITIVITIES` are exported.

### Changed

- **`retrieve_context` reads L1 while L2 embeds and searches.** The recent-turn read used to run
  only alongside the query embedding, and the long-term search waited for both. It now runs
  alongside the whole long-term path (embed, search, optional rerank), so a retrieval waits for
  the slower of the two tiers instead of their sum, as the docs describe. Results are unchanged.

## [0.2.1] - 2026-10-01

Framework integration fixes found by running every integration against the real framework at the
oldest and newest version we support, fact extraction that works on small local models, plus PyPI
listing updates.

### Fixed

- **Fact extraction returned nothing on small models.** With `qwen2.5:3b` on Ollama, extraction
  came back empty on 12 of 15 real exchanges, because the model read the assistant's reply as part
  of what to mine. Extraction spec 1.1 frames the conversation, says whose facts to record, asks
  for a JSON-schema structured output and gives two worked examples (one with facts, one with
  none). On the same model it found 14 of 14 expected facts and stored nothing for small talk.
  `EXTRACTION_SPEC_VERSION` is now `"1.1"`; the prompt, `EXTRACTION_RESPONSE_SCHEMA` and
  `format_extraction_input` are exported for custom extractors.
- **Microsoft Agent Framework.** The adapter targeted the pre-1.0 API (`ChatAgent`,
  `ContextProvider.invoking`, `Context`), which Agent Framework 1.x removed. It now implements the
  1.x `ContextProvider` (`before_run` adds the recalled context through
  `context.extend_instructions`, `after_run` saves the turn from the response), so
  `Agent(client, context_providers=[memory.as_context_provider()])` works.
- **LangGraph.** `ActroneCheckpointer` was a plain class, so `graph.compile(checkpointer=...)`
  rejected it. It is now a real `BaseCheckpointSaver` that wraps a saver of your choice
  (`saver=`, an in-memory saver by default) for graph state and also records each completed
  human and AI exchange as memory, once per exchange.
- **Haystack.** `ActroneRetriever` and `ActroneWriter` were not registered Haystack components,
  so `Pipeline.add_component` refused them. They are now real `@component` classes with declared
  inputs and outputs, and support both `run` and `run_async`.
- **LlamaIndex.** `ActroneLlamaMemory` now recalls against the user message a chat engine has just
  stored when `get()` is called without input, keeps that message in the history it returns, and
  implements `set`/`aset` and the tokenizer that chat engines expect.
- **AutoGen.** `update_context` now returns an `UpdateContextResult` wrapping a
  `MemoryQueryResult`, as `AssistantAgent` requires, and `query` accepts a string or a
  `MemoryContent`.
- **DSPy.** `ActroneRM` now returns passages with `long_text` and a `passages` attribute, the shape
  `dspy.Retrieve` reads.

### Added

- **`OpenAIFactExtractor` works with local and self-hosted models.** It takes `base_url` (for
  example `http://localhost:11434/v1` for Ollama) or a ready `client`, so any OpenAI-compatible
  server can do extraction, as the TypeScript library already allowed. It asks for a JSON schema
  and falls back to JSON mode, for good, only when a server rejects the schema.

### Changed

- **Newer framework majors are supported.** Haystack 3, Agno 2 and 3, Pydantic AI 2 and Google
  ADK 2 passed the same checks as the older releases, so the extras now allow them:
  `haystack-ai>=2.0,<4`, `agno>=1.0,<4`, `pydantic-ai>=1.32,<3` and `google-adk>=1.2,<3`.
- **Supported framework versions now match what was tested.** Each extra's lower bound is the
  oldest release the integration was verified against: `crewai>=0.95`, `autogen-agentchat` and
  `autogen-core>=0.4.3`, `smolagents>=1.5.1`, `openai-agents>=0.2`, `pydantic-ai>=1.32`,
  `semantic-kernel>=1.16` and `google-adk>=1.2`. Older releases fail to install or import with
  current dependencies, or lack the API the integration uses. The `dspy` extra now installs `dspy` (the package
  was renamed from `dspy-ai`) and allows 3.x: `dspy>=2.5,<4`.
- **PyPI listing.** Keywords now cover what people search for (agent-memory, long-term-memory,
  semantic-search, rag, llm, pgvector, agno), and new classifiers state that the package is typed,
  asyncio-based, built on Pydantic 2, Python 3 only and OS independent.

## [0.2.0] - 2026-09-25

The first release published to PyPI. It includes everything built since 0.1.0, which was an internal
milestone and was never published.

### Added

- **Postgres for both tiers, so no new service is needed.** `PostgresStore`
  (`actrone_memory.l1.postgres_store`) holds recent turns and `PgVectorStore`
  (`actrone_memory.l2.pgvector_store`) holds long-term memories via
  [pgvector](https://github.com/pgvector/pgvector). Install `actrone-memory[pgvector]`.
  Both accept a DSN (`from_dsn`) or a pool you already own (`from_pool`), so one Postgres
  pool can serve the whole library. Postgres has no TTL or list trimming, so the L1 store
  implements expiry and the retention cap explicitly, and orders turns by a `bigserial`
  rather than the clock so ordering is stable when timestamps collide.
- **Custom stores are now a supported path, not just a possible one.**
  `MemoryManager.create()` accepts `l1`, `l2` and `embedder`. Whatever you inject is used
  as-is and the matching built-in backend is never constructed, so injecting both stores
  opens no Redis or Qdrant connection. Previously a custom store meant calling the
  constructor the docs tell you not to call.
- **Published store conformance suite** (`actrone_memory.testing`). `check_l1_store` and
  `check_l2_store` assert the behaviours the `Protocol` cannot: turns come back
  oldest-first, `n` windows from the end, a search never returns another agent's memories,
  `threshold` and `limit` are honoured, an upsert replaces rather than duplicates, and
  erasure is scoped. Both built-in stores and both Postgres stores pass it. The TypeScript
  library ships the same suite as `actrone-memory/testing`.
- **`MemoryManager.get_recent_turns()`**, the raw history read framework memory adapters
  need, matching the TypeScript `getRecentTurns`.
- **`ActroneChatMessageHistory`** for LangChain, implementing the `BaseChatMessageHistory`
  surface, which is unchanged across LangChain 0.x and 1.x.
- **Valkey coverage.** No new adapter: `RedisStore` uses only standard Redis commands, so
  Valkey (and DragonflyDB, ElastiCache, Upstash) work with it unmodified. Now proven by an
  integration suite that runs the conformance suite against a real Valkey container.

- **AutoGen 0.4 adapter** (`integrations/autogen.py`), implements `autogen_core.memory.Memory` protocol.
  - `add()` stores text content as an injected memory entry.
  - `query()` performs Qdrant L2 semantic search and returns `MemoryQueryResult`.
  - `update_context()` injects episodic memories into the AutoGen model context as system messages.
  - Install: `pip install actrone-memory[autogen]`

- **LlamaIndex adapter** (`integrations/llamaindex.py`), implements the `BaseMemory` interface.
  - `aget()` / `get()` returns recent turns + episodic memories as `ChatMessage` list.
  - `aput()` / `put()` buffers user messages and flushes to Redis L1 on assistant response.
  - `areset()` / `reset()` clears session memory.
  - Install: `pip install actrone-memory[llamaindex]`

- **Haystack v2 adapters** (`integrations/haystack.py`), two `@component`-compatible classes.
  - `ActroneRetriever`, retrieves memories as Haystack `Document` objects with metadata.
  - `ActroneWriter`, stores conversation turns after LLM inference.
  - Install: `pip install actrone-memory[haystack]`

- **DSPy adapter** (`integrations/dspy.py`), `ActroneRM` is a DSPy `Retrieve`-compatible module.
  - `forward()` performs semantic search and returns `dspy.Prediction(passages=[...])`.
  - Supports single query and multi-query (results interleaved by rank position).
  - Works from both sync and async contexts via thread-pool isolation.
  - `store_turn()` persists turns after DSPy program inference.
  - Install: `pip install actrone-memory[dspy]`

- **New optional extras**: `autogen`, `llamaindex`, `haystack`, `dspy` (also included in `all`).
- **Contract test suite** (`tests/contract/`), each adapter has a full contract test file.

### Changed

- **The `langchain` extra now spans both majors (`>=0.2,<3`).** LangChain removed
  `langchain_core.memory` in 1.x, retiring the `BaseMemory` abstraction, so the old `<1`
  cap held users on a 0.3.x line whose published advisories are only fixed in 1.x. Tier 2
  moves to `ActroneChatMessageHistory`; `ActroneMemory` remains for 0.x and raises on 1.x
  with a pointer to the alternatives.

- **Local-first, zero-service default backend.** `MemoryManager.create()` now runs
  fully in-process with **no Redis, no Qdrant, and no API key**, parity with the
  TypeScript `actrone-memory` on-ramp. "Memory that never phones home."
  - New `InMemoryStore` (`actrone_memory.in_memory`) implements **both** the L1
    (hot session) and L2 (cold semantic) tiers with the same blended
    relevance+recency ranking as the Qdrant backend.
  - New dependency-free `HashingEmbedder` (`actrone_memory.l2.embedder`), a
    deterministic hashing vectorizer (`embedding_provider="hashing"`), and the
    final fallback of the default `local` provider. No model download, no
    external call.
  - New `L1Store` / `L2Store` `Protocol`s (`actrone_memory.protocols`), the
    manager now depends on the store seam, not concrete backends.
  - New config: `backend: "memory" | "redis_qdrant"` and `hashing_dimensions`.

- **Provenance-typed facts v1**, every stored memory now carries **`source`**
  (attribution: `user`/`assistant`/`tool`/`summary`/`injected`/`extracted`/
  `reflection`/`imported`/`unknown`, or a namespaced string like `"import:crm"`)
  and **`sensitivity`** (`none`/`low`/`pii`/`sensitive`). `inject_memory()` accepts
  both; summaries are tagged `source="summary"`. Defaults are backwards-compatible
  (`unknown`/`none`). New `MemorySource`/`Sensitivity` types exported.
- **Local right-to-erasure**, `MemoryManager.erase_agent_memories(agent_id,
  session_id=None)` hard-deletes an agent's long-term memories (and optionally a
  session's turns). The governance seed that graduates to hosted *provable*
  erasure. `L2Store` gains `delete_agent_memories`.

- **Optional LLM fact extraction** (turns → durable facts), to the shared
  extraction spec v1 (`docs/memory-spec/extraction.v1.md`). New
  `actrone_memory.extraction` (`FactExtractor` seam, `OpenAIFactExtractor`,
  `ExtractedFact`, `parse_facts`, `EXTRACTION_SPEC_VERSION`). Opt-in via
  `MemoryConfig.extract_facts=True` (OpenAI provider); stores each fact as
  `content_type="fact"`, `source="extracted"` with a classified `sensitivity`.
  Exposed as `MemoryManager.extract_memories()` and auto-run on the summarise
  cadence (cost-bounded). New `"fact"` content type.

- **Default `backend` is now `"memory"`** (was implicitly Redis + Qdrant) and the
  **default `embedding_provider` is now `"local"`** (was `"openai"`): the best local embedder
  available, fastembed, then sentence-transformers, then lexical hashing. Existing
  deployments must set `ACTRONE_BACKEND=redis_qdrant` and
  `ACTRONE_EMBEDDING_PROVIDER=openai` (+ `ACTRONE_OPENAI_API_KEY`) to keep the
  previous behaviour. No code changes are required, the switches are read at
  startup.
- Qdrant collection dimensionality is now sized from the embedder in use
  (`embedder.dimensions`) rather than a fixed `embedding_dimensions`, so the
  hashing/local embedders provision correctly.

- `pyproject.toml` keywords expanded to include all supported frameworks.
- `integrations/__init__.py` updated with install instructions for all extras.
- **Token counting no longer needs the network.** The default counter is a dependency-free
  heuristic (about 4 characters per token), identical to the TypeScript library's, so both
  allocate a budget the same way. Exact `cl100k_base` counts are opt-in with
  `token_counter="tiktoken"` and the new `[tiktoken]` extra; `tiktoken` is no longer a base
  dependency, which takes the bare install from 18 packages to 11.

### Fixed

- **Recall used one similarity threshold for every embedder.** Scores are not comparable across
  models, so the fixed `0.72` recalled 4% of relevant memories with the lexical embedder, 23% with
  MiniLM and 65% with bge-small, measured on a labelled set of 48 relevant and 528 unrelated
  pairs. Each built-in embedder now declares its calibrated threshold (`0.3`, `0.4`, `0.63`),
  `relevance_threshold` defaults to that value, and `MemoryManager.relevance_threshold` reports the
  one in use. An explicit setting still wins; OpenAI and custom embedders keep `0.72`.
- The first `store_turn` downloaded `tiktoken`'s encoding file, contradicting the no-egress
  promise and failing on an air-gapped first run.
- The library printed its own debug and info events to stdout in any program that imported it.
  Without a logging setup it is now quiet, and warnings go to stderr.
- `actrone-memory add <framework> --write` produced a file that could not be imported: the core
  recipe used top-level `await` and names it never defined. Every recipe is now the code of a
  mypy-checked example under `examples/frameworks/` (seven new ones), and a written file runs
  with `python <file>`. Extras in install commands are quoted, since unquoted brackets fail in zsh.
- The CrewAI docs showed a `memory_backend` argument CrewAI does not have; they now show the
  adapter's real path, `build_context` into the task text.
- `PgVectorStore` rejects a non-integer or non-positive `dimensions` instead of interpolating it
  into DDL.

---

## [0.1.0] - 2026-05-18 (internal, not published)

### Added

- Two-tier persistent memory: Redis L1 (hot, <1 ms) + Qdrant L2 (semantic, ~10 ms).
- `MemoryManager` public API: `store_turn`, `retrieve_context`, `inject_memory`, `delete_memory`, `clear_session`, `search_memories`.
- 4-phase retrieval pipeline: parallel L1+L2 fetch → budget allocation → relevance filter → pruning.
- Auto-summarisation: background compression after N turns (default 20) via GPT-4o-mini.
- LangChain adapter (`integrations/langchain.py`), `BaseChatMemory` drop-in.
- LangGraph checkpointer (`integrations/langgraph.py`), `BaseCheckpointSaver` implementation.
- CrewAI backend (`integrations/crewai.py`), `MemoryBackend` implementation.
- OpenAI embedder (`text-embedding-3-small`) and local embedder (`all-MiniLM-L6-v2`).
- Embedding cache: Redis key `embed:{sha256(text)}` with 7-day TTL.
- Full test suite: unit (≥80%), integration (testcontainers), contract.
- CI: ruff, mypy --strict, pip-audit, Docker build, PyPI publish on tag.
