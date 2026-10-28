# API Reference

> Complete documentation for every public method, data type, and error in `actrone-memory`.
> Each section explains what a method does in plain English before showing the technical details.

---

## Quick Navigation

- [MemoryManager](#memorymanager), the main object you'll use
  - [create()](#create), connect to the configured backend
  - [store_turn()](#store_turn), save a conversation message
  - [retrieve_context()](#retrieve_context), fetch relevant memory for the next reply
  - [get_recent_turns()](#get_recent_turns), read raw recent conversation turns
  - [inject_memory()](#inject_memory), manually add a fact to long-term memory
  - [extract_memories()](#extract_memories), extract durable facts from conversation turns
  - [delete_memory()](#delete_memory), remove a specific memory
  - [erase_agent_memories()](#erase_agent_memories), wipe all long-term memories for an agent
  - [clear_session()](#clear_session), wipe a session's short-term memory
  - [search_memories()](#search_memories), search long-term memory directly
  - [get_session_metadata()](#get_session_metadata), check session stats
- [Data Types](#data-types), the objects these methods return
- [MemoryConfig](#memoryconfig), all configuration options
- [Errors](#errors), what can go wrong and how to handle it
- [Framework Integrations](#framework-integrations), LangChain, LangGraph, CrewAI

> This page focuses on the core API and a few representative framework integrations. The full adapter list, version ranges, and optional extras are in the project README compatibility matrix.

---

## MemoryManager

This is the main class you'll interact with. It manages both the short-term (hot session) tier and the long-term (semantic) tier, and handles all the logic of keeping them in sync.

By default both tiers are served by an in-process store, so nothing external is required. Set `backend="redis_qdrant"` to put the hot tier in Redis and the long-term tier in Qdrant.

**Never create one directly with `MemoryManager(...)`.** Always use `MemoryManager.create()` or the `create_memory_manager()` context manager shown below, they wire up the stores and the embedding provider for you.

---

### `create()`

**What it does:** Builds the stores for the configured backend (in-process by default, or Redis + Qdrant when `backend="redis_qdrant"`), sets up the embedding provider, and returns a ready-to-use `MemoryManager`. Think of it as "opening" the memory system.

`MemoryManager.create()` is an `async def`, so it returns a coroutine and is not itself an async context manager. Use `create_memory_manager()` for `async with` or await it manually and then close it yourself.

```python
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
) -> MemoryManager
```

**Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `config` | `MemoryConfig` or `None` | `None` | Your settings. If left blank, reads from environment variables automatically. |
| `l1` | `L1Store` or `None` | `None` | Custom hot-tier store implementing the `L1Store` protocol (e.g. `PostgresStore`). |
| `l2` | `L2Store` or `None` | `None` | Custom long-term store implementing the `L2Store` protocol (e.g. `PgVectorStore`, `MyWeaviateStore`). |
| `embedder` | `Embedder` or `None` | `None` | Custom embedding provider implementing the `Embedder` protocol. |
| `extractor` | `FactExtractor` or `None` | `None` | Custom fact extraction provider. |
| `reranker` | `CrossEncoderReranker` or `None` | `None` | Optional cross-encoder reranker. |

**Returns:** A `MemoryManager` instance, ready to use.

**Can fail with:**

- `ConfigurationError`, if required settings are missing or invalid (for example, when `embedding_provider="openai"` and `ACTRONE_OPENAI_API_KEY` is unset)
- `StoreConnectionError`, if Redis or Qdrant isn't running or can't be reached when the durable backend is selected

**The two recommended ways to use it:**

```python
from actrone_memory import MemoryManager, create_memory_manager

# Option A, context manager (recommended)
# The manager closes automatically when the block ends, even if an error occurs.
async with create_memory_manager() as memory:
    await memory.store_turn(...)
    context = await memory.retrieve_context(...)

# Option B, manual open/close
memory = await MemoryManager.create()
try:
    await memory.store_turn(...)
finally:
    await memory.close()  # always close when you're done
```

---

### `store_turn()`

**What it does:** Saves one exchange between the user and the agent (one "turn") into the short-term tier. If auto-summarisation is enabled, it also quietly triggers a background compression to long-term memory after enough turns accumulate.

```python
async def store_turn(
    agent_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
    tool_results: list[ToolResult] | None = None,
) -> str
```

**Parameters:**

| Parameter | Type | Description |
| --- | --- | --- |
| `agent_id` | `str` | A unique name for your agent, e.g. `"customer-support-bot"`. This scopes the memory so different agents don't share memories. |
| `session_id` | `str` | A unique ID for this conversation, e.g. the user's ID or a UUID. Each new conversation should get a new session ID. |
| `user_message` | `str` | What the user said. |
| `assistant_message` | `str` | What the agent replied. |
| `tool_results` | list or `None` | Optional. If the agent called any external tools (like a web search), you can store the results here for future reference. |

**Returns:** `str`, the ID of the saved turn (useful if you need to reference it later).

**Can fail with:** `StoreConnectionError`, if the backing store is unreachable.

```python
turn_id = await memory.store_turn(
    agent_id          = "support-bot",
    session_id        = "user-abc-123",
    user_message      = "My order hasn't arrived yet.",
    assistant_message = "I'm sorry to hear that! Can you share your order number?",
)
```

---

### `retrieve_context()`

**What it does:** Before your agent replies, call this to get all the relevant memory, recent messages from this session plus any related memories from past sessions, pre-packaged and ready to inject into your LLM prompt.

It runs a 4-phase pipeline (fetch both tiers in parallel → allocate the token budget → rank by relevance → prune to fit) and returns only what fits in your budget.

```python
async def retrieve_context(
    agent_id: str,
    session_id: str,
    query: str,
    token_budget: int,
    *,
    redact: Iterable[Sensitivity] | None = None,
) -> RetrievedContext
```

**Parameters:**

| Parameter | Type | Description |
| --- | --- | --- |
| `agent_id` | `str` | Same agent ID you used in `store_turn()`. |
| `session_id` | `str` | Same session ID you used in `store_turn()`. |
| `query` | `str` | The user's current message. This is used to search long-term memory for relevant past conversations. |
| `token_budget` | `int` | How many tokens you can spare for memory context. Must be greater than 0. A safe default is `4096`. |
| `redact` | `list[Sensitivity]` or `None` | Sensitivity classes to mask in the returned memories, for example `['pii', 'sensitive']`. Each match comes back as `"[redacted: pii]"` with its embedding dropped. `None` (the default) uses `redact_sensitivities` from the config; `[]` masks nothing for this call. Turns are never masked. |

**Returns:** `RetrievedContext`, see [Data Types](#data-types) below.

**Can fail with:**

- `TokenBudgetError`, if you pass `token_budget=0` or less
- `ValidationError`, if `redact` names an unknown sensitivity class
- `StoreConnectionError`, if the backing store is unreachable
- `EmbeddingError`, if the embedding API call fails

```python
context = await memory.retrieve_context(
    agent_id     = "support-bot",
    session_id   = "user-abc-123",
    query        = "I still haven't received my order.",
    token_budget = 4096,
)

# Use it to build your LLM prompt
for turn in context.recent_turns:
    print(f"User: {turn.user_message}")
    print(f"Agent: {turn.assistant_message}")

for mem in context.episodic_memories:
    print(f"Relevant memory: {mem.content}")

print(f"Used {context.total_tokens_used} of {context.token_budget} tokens")
```

---

### `get_recent_turns()`

**What it does:** Reads raw recent conversation turns from the short-term tier (L1) without running semantic search or token budget allocation. Used by framework chat-history adapters.

```python
async def get_recent_turns(
    agent_id: str,
    session_id: str,
    *,
    limit: int | None = None,
) -> list[Turn]
```

**Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent_id` | `str` | - | The agent whose turns to fetch. |
| `session_id` | `str` | - | The conversation session ID. |
| `limit` | `int` or `None` | `None` | Max number of most recent turns to return (newest window). Defaults to `max_session_turns`. |

**Returns:** `list[Turn]`, oldest first.

**Can fail with:** `ValidationError`, `StoreConnectionError`

```python
turns = await memory.get_recent_turns(agent_id="support-bot", session_id="user-abc-123", limit=10)
for turn in turns:
    print(f"{turn.user_message} -> {turn.assistant_message}")
```

---

### `inject_memory()`

**What it does:** Manually writes a piece of information directly into the long-term tier. Use this to seed an agent with background knowledge before a conversation starts, such as user preferences, company policies, or facts you've collected elsewhere.

```python
async def inject_memory(
    agent_id: str,
    content: str,
    importance: float = 0.8,
    session_id: str = "injected",
    topic_tags: list[str] | None = None,
    source: str = "injected",
    sensitivity: Sensitivity = "none",
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent_id` | `str` | - | The agent that should have access to this memory. |
| `content` | `str` | - | The text to remember. Write it as a clear, self-contained sentence. |
| `importance` | `float` | `0.8` | A score from 0.0 to 1.0. Higher importance means this memory surfaces more readily. Most injected facts should be 0.8 or higher. |
| `session_id` | `str` | `"injected"` | A label for where this memory came from. Can be anything, useful for filtering or debugging. |
| `topic_tags` | `list[str]` or `None` | `None` | Optional keywords that describe this memory, e.g. `["user-profile", "preferences"]`. |
| `source` | `str` | `"injected"` | Provenance attribution (e.g. `"injected"`, `"import:crm"`, `"tool:web_search"`). |
| `sensitivity` | `Sensitivity` | `"none"` | Classification for governance and right-to-erasure (`"none"`, `"low"`, `"pii"`, `"sensitive"`). |

**Returns:** `str`, the memory ID (save this if you want to delete the memory later).

**Can fail with:** `ValidationError`, `StoreConnectionError`, `EmbeddingError`

```python
memory_id = await memory.inject_memory(
    agent_id   = "support-bot",
    content    = "This user is a Premium subscriber and prefers email communication over chat.",
    importance = 0.95,
    topic_tags = ["user-profile", "communication-preferences"],
    source     = "import:crm",
    sensitivity = "low",
)
```

---

### `extract_memories()`

**What it does:** Mines a session's conversation turns for durable atomic facts and stores them in long-term memory (L2) as `content_type="fact"` and `source="extracted"`. Requires an LLM fact extractor (e.g. OpenAI provider).

```python
async def extract_memories(
    agent_id: str,
    session_id: str,
    *,
    turns: list[Turn] | None = None,
) -> list[str]
```

**Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent_id` | `str` | - | The agent to attribute extracted facts to. |
| `session_id` | `str` | - | The conversation session to mine. |
| `turns` | `list[Turn]` or `None` | `None` | Optional turns to extract from. Defaults to fetching recent turns from L1. |

**Returns:** `list[str]`, IDs of the newly stored fact memories.

**Can fail with:** `ValidationError`, `StoreConnectionError`, `EmbeddingError`

```python
fact_ids = await memory.extract_memories(
    agent_id   = "support-bot",
    session_id = "user-abc-123",
)
print(f"Extracted {len(fact_ids)} facts into long-term memory")
```

---

### `delete_memory()`

**What it does:** Permanently removes a specific memory from long-term storage. Useful if you injected incorrect information, or if a user asks you to forget something.

```python
async def delete_memory(agent_id: str, memory_id: str) -> None
```

**Parameters:**

| Parameter | Type | Description |
| --- | --- | --- |
| `agent_id` | `str` | The agent that owns this memory (used for audit logging). |
| `memory_id` | `str` | The ID returned when you called `inject_memory()`, or from a `search_memories()` result. |

**Can fail with:** `ValidationError`, `MemoryNotFoundError` (if the ID doesn't exist), `StoreConnectionError`

```python
await memory.delete_memory(
    agent_id  = "support-bot",
    memory_id = memory_id,  # the ID you got from inject_memory()
)
```

---

### `erase_agent_memories()`

**What it does:** Permanently deletes memories for an agent across storage tiers (local right-to-erasure / GDPR compliance). Always deletes all L2 episodic memories for `agent_id`. If `session_id` is also provided, clears that session's L1 turns as well.

```python
async def erase_agent_memories(
    agent_id: str,
    session_id: str | None = None,
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent_id` | `str` | - | The agent whose memories to erase. |
| `session_id` | `str` or `None` | `None` | Optional session ID to also clear from short-term memory (L1). |

**Can fail with:** `ValidationError`, `StoreConnectionError`

```python
# Hard delete all long-term memories for this agent:
await memory.erase_agent_memories(agent_id="support-bot")

# Or erase long-term memories AND a specific session's turns:
await memory.erase_agent_memories(agent_id="support-bot", session_id="user-abc-123")
```

---

### `clear_session()`

**What it does:** Deletes all messages for one session in the short-term tier. Useful when a conversation is fully resolved and you want to free up memory.

> **Note:** This only clears short-term memory. Long-term memories (summaries, injected facts, extracted facts) are **not** deleted; they persist across sessions by design.

```python
async def clear_session(agent_id: str, session_id: str) -> None
```

**Can fail with:**

- `ValidationError`, if `agent_id` or `session_id` is invalid
- `StoreConnectionError`, if the underlying store cannot be reached

```python
await memory.clear_session(agent_id="support-bot", session_id="user-abc-123")
```

---

### `search_memories()`

**What it does:** Searches long-term memory directly using a text query. Returns memories ranked by how closely they match the meaning of your query. Useful for building admin UIs, debugging memory issues, or building features like "show me everything this agent knows about topic X".

```python
async def search_memories(
    agent_id: str,
    query: str,
    limit: int = 10,
    *,
    redact: Iterable[Sensitivity] | None = None,
) -> list[MemoryEntry]
```

**Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent_id` | `str` | - | Which agent's memories to search. |
| `query` | `str` | - | What you're looking for, in plain English. |
| `limit` | `int` | `10` | Maximum number of results to return. |
| `redact` | `list[Sensitivity]` or `None` | `None` | Sensitivity classes to mask in the results, as for `retrieve_context()`. `None` uses the config default. |

**Returns:** `list[MemoryEntry]`, sorted by relevance, filtered at the configured threshold.

**Can fail with:**

- `ValidationError`, if `agent_id`, `query`, or `limit` is invalid
- `StoreConnectionError`, if the backing store is unreachable
- `EmbeddingError`, if embedding generation fails

```python
results = await memory.search_memories(
    agent_id = "support-bot",
    query    = "shipping and delivery complaints",
    limit    = 5,
)

for result in results:
    print(f"[{result.content_type}] {result.content}")
```

---

### `get_session_metadata()`

**What it does:** Returns basic stats about a session, when it was created, when it was last active, and how many turns it has. Returns `None` if the session doesn't exist or has expired.

```python
async def get_session_metadata(agent_id: str, session_id: str) -> SessionMetadata | None
```

**Can fail with:**

- `ValidationError`, if `agent_id` or `session_id` is invalid
- `StoreConnectionError`, if the underlying store cannot be reached

```python
meta = await memory.get_session_metadata("support-bot", "user-abc-123")
if meta:
    print(f"Session has {meta.turn_count} turns")
    print(f"Last active: {meta.last_active}")
```

---

## Data Types

These are the objects returned by the methods above.

---

### `RetrievedContext`

Returned by `retrieve_context()`. Contains everything you need to build your LLM prompt.

```python
class RetrievedContext:
    recent_turns: list[Turn]          # Messages from this session, newest last
    episodic_memories: list[MemoryEntry]  # Relevant memories from past sessions
    total_tokens_used: int            # How many tokens all this context uses
    token_budget: int                 # The budget you passed in
    retrieval_duration_ms: float      # How long the retrieval took, in milliseconds

    # Computed property:
    budget_utilisation: float         # total_tokens_used / token_budget (e.g. 0.73 = 73%)
```

**How to use it:**

```python
context = await memory.retrieve_context(...)

# Inject recent messages into your prompt
messages = [{"role": "system", "content": "You are a helpful assistant."}]

for turn in context.recent_turns:
    messages.append({"role": "user",      "content": turn.user_message})
    messages.append({"role": "assistant", "content": turn.assistant_message})

# Inject relevant long-term memories as a system note
if context.episodic_memories:
    memory_text = "\n".join(f"- {m.content}" for m in context.episodic_memories)
    messages.insert(1, {
        "role": "system",
        "content": f"Relevant memories:\n{memory_text}"
    })

# Current user message
messages.append({"role": "user", "content": current_user_message})
```

---

### `Turn`

A single back-and-forth exchange between user and agent.

```python
class Turn:
    id: str                         # Unique ID for this turn
    session_id: str
    user_message: str               # What the user said
    assistant_message: str          # What the agent replied
    tool_results: list[ToolResult]  # Any tool calls made during this turn
    timestamp: datetime             # When this turn happened (UTC)
    token_count: int                # Total tokens in this turn
```

---

### `MemoryEntry`

A single item stored in long-term memory. It can hold a compressed summary, an injected fact, a tool result, or a fact extracted during a session.

```python
class MemoryEntry:
    id: str                # Unique ID, use this with delete_memory()
    agent_id: str
    session_id: str
    content: str           # The actual text of the memory
    content_type: str      # One of: "turn", "summary", "tool_result", "injected", "fact"
    embedding: list[float] | None
    importance_score: float   # 0.0-1.0, how important this memory is
    topic_tags: list[str]  # Keywords describing this memory
    token_count: int
    timestamp: datetime    # When this memory was created (UTC)
    source_turn_ids: list[str]  # Which turns this memory was derived from
    source: str           # Where the fact came from, e.g. "user", "assistant", "summary", "extracted"
    sensitivity: str      # One of: "none", "low", "pii", "sensitive"
```

---

### `ToolResult`

Represents the result of an external tool call (e.g. a web search, database query, or API call).

```python
class ToolResult:
    tool_name: str             # Name of the tool that was called
    params: dict               # What was passed to the tool
    result: object             # What the tool returned
    success: bool              # Whether the call succeeded
    duration_ms: int | None    # How long the call took
```

---

### `SessionMetadata`

Basic stats about a session, returned by `get_session_metadata()`.

```python
class SessionMetadata:
    agent_id: str
    session_id: str
    turn_count: int            # How many turns this session has
    created_at: datetime | None
    last_active: datetime | None
```

---

### `Sensitivity`, `MemorySource`, `ContentType`

Vocabularies for memory classification, provenance, and governance:

```python
ContentType = Literal["turn", "summary", "tool_result", "injected", "fact"]

MemorySource = Literal[
    "user",        # stated by the user
    "assistant",   # asserted by the agent
    "tool",        # produced by a tool call
    "summary",     # distilled from conversation summarisation
    "injected",    # seeded directly via inject_memory()
    "extracted",   # derived by fact extraction
    "reflection",  # synthesised by a reflection pass
    "imported",    # loaded from an external system
    "unknown",     # provenance not recorded
]  # Or any custom namespaced string such as "import:crm"

Sensitivity = Literal[
    "none",       # non-personal, freely retained
    "low",        # mildly personal / preference data
    "pii",        # personally identifiable information
    "sensitive",  # special-category / regulated (health, financial, credentials)
]
```

---

## MemoryConfig

All configuration is read from environment variables automatically. You can also create a `MemoryConfig` object explicitly if you prefer not to use environment variables.

```python
from actrone_memory import MemoryConfig, create_memory_manager

config = MemoryConfig(
    backend            = "redis_qdrant",
    redis_url          = "redis://localhost:6379",
    qdrant_url         = "http://localhost:6333",
    embedding_provider = "openai",
    openai_api_key     = "sk-...",
    session_ttl_hours  = 48,
    max_session_turns  = 100,
)

async with create_memory_manager(config) as memory:
    ...
```

**All available settings:**

| Setting | Environment Variable | Default | Description |
| --- | --- | --- | --- |
| `backend` | `ACTRONE_BACKEND` | `memory` | `memory` runs fully in-process. Set `redis_qdrant` for the durable backend. |
| `redis_url` | `ACTRONE_REDIS_URL` | `redis://localhost:6379` | Where your Redis server is running |
| `qdrant_url` | `ACTRONE_QDRANT_URL` | `http://localhost:6333` | Where your Qdrant server is running |
| `qdrant_api_key` | `ACTRONE_QDRANT_API_KEY` | `None` | API key for Qdrant Cloud (not needed for self-hosted) |
| `qdrant_collection` | `ACTRONE_QDRANT_COLLECTION` | `agent_memories` | Collection name in Qdrant |
| `embedding_provider` | `ACTRONE_EMBEDDING_PROVIDER` | `local` | `local` (best offline embedder available: ONNX, then sentence-transformers, then hashing), `hashing`, or `openai` (needs an API key) |
| `openai_api_key` | `ACTRONE_OPENAI_API_KEY` | `None` | Required when `embedding_provider = "openai"` |
| `session_ttl_hours` | `ACTRONE_SESSION_TTL_HOURS` | `24` | How many hours before short-term memory expires |
| `max_session_turns` | `ACTRONE_MAX_SESSION_TURNS` | `50` | Maximum messages stored in short-term memory per session |
| `relevance_threshold` | `ACTRONE_RELEVANCE_THRESHOLD` | *(calibrated)* | How similar a memory must be to the query before it's included (0.0-1.0). Unset, the embedder's calibrated value applies: `0.3` lexical, `0.63` bge-small, `0.4` MiniLM, `0.72` for OpenAI or a custom embedder. Raise it to be more selective. |
| `hybrid_retrieval` | `ACTRONE_HYBRID_RETRIEVAL` | `true` | Combine dense vector search with BM25 lexical ranking via Reciprocal Rank Fusion |
| `rerank_enabled` | `ACTRONE_RERANK_ENABLED` | `false` | Enable cross-encoder reranking over retrieved candidate memories (requires `[onnx]` extra) |
| `extract_facts` | `ACTRONE_EXTRACT_FACTS` | `false` | Enable LLM fact extraction from conversation turns (OpenAI provider) |
| `token_counter` | `ACTRONE_TOKEN_COUNTER` | `heuristic` | `heuristic` (about 4 characters per token, no network) or `tiktoken` (exact counts, needs the `[tiktoken]` extra). |
| `auto_summarise` | `ACTRONE_AUTO_SUMMARISE` | `true` | Automatically compress old conversations into long-term memory |
| `summarise_after_turns` | `ACTRONE_SUMMARISE_AFTER_TURNS` | `20` | Trigger summarisation after this many turns |
| `redact_sensitivities` | `ACTRONE_REDACT_SENSITIVITIES` | `[]` | Sensitivity classes masked whenever memories are read, for example `'["pii", "sensitive"]'`. Matching memories come back as `[redacted: <class>]`. Stored memories are not changed. |

**Token budget fractions** (these must add up to exactly 1.0):

| Setting | Default | Description |
| --- | --- | --- |
| `budget_fraction_system` | `0.30` | Fraction reserved for the system prompt |
| `budget_fraction_episodic` | `0.25` | Fraction for long-term memories |
| `budget_fraction_session` | `0.35` | Fraction for recent session messages |
| `budget_fraction_current_turn` | `0.10` | Fraction for the current user message |

---

## Errors

All errors extend `ActroneMemoryError`, so you can catch everything with one handler or be specific:

```python
from actrone_memory.exceptions import (
    ActroneMemoryError,    # catch-all base class
    ConfigurationError,    # bad or missing settings
    StoreConnectionError,  # Redis or Qdrant unreachable
    EmbeddingError,        # embedding API failed
    MemoryNotFoundError,   # delete_memory() got a non-existent ID
    TokenBudgetError,      # token_budget was 0 or negative
    ValidationError,       # invalid API input
)

# Catch everything:
try:
    context = await memory.retrieve_context(...)
except ActroneMemoryError as e:
    print(f"Memory error [{e.code}]: {e.message}")

# Or be specific and handle each case:
try:
    context = await memory.retrieve_context(...)
except StoreConnectionError as e:
    print(f"Could not reach {e.store} ({e.code}): {e.message}")
    print(f"Details: {e.details}")
    context = empty_context
except EmbeddingError:
    print("Embedding failed, retrying without semantic search")
```

**Every error includes:**

- `e.code`, a machine-readable identifier (e.g. `"ERR_STORE_CONNECTION"`), useful for logging
- `e.message`, a human-readable description of what went wrong
- `e.details`, a structured dictionary of extra context
- some error classes also expose convenience attributes such as `e.store` or `e.field`

| Error | Code | When it's raised |
| --- | --- | --- |
| `ConfigurationError` | `ERR_CONFIGURATION` | A required setting is missing or invalid when you call `MemoryManager.create()` |
| `StoreConnectionError` | `ERR_STORE_CONNECTION` | Redis or Qdrant can't be reached |
| `EmbeddingError` | `ERR_EMBEDDING` | The embedding API returned an error |
| `MemoryNotFoundError` | `ERR_MEMORY_NOT_FOUND` | You called `delete_memory()` with an ID that doesn't exist |
| `TokenBudgetError` | `ERR_TOKEN_BUDGET` | You passed `token_budget=0` or a negative number |
| `ValidationError` | `ERR_VALIDATION` | A public API argument was invalid (for example, a bad sensitivity tag or empty ID) |

---

## Framework Integrations

### LangChain

Two adapters, because LangChain 1.x removed the `BaseMemory` abstraction. Use
`ActroneChatMessageHistory` unless you are pinned to 0.x: it targets `BaseChatMessageHistory`,
which is unchanged across both majors. Both are async-only, so drive LangChain with `ainvoke`
or `astream`.

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from actrone_memory.integrations.langchain import ActroneChatMessageHistory

history = ActroneChatMessageHistory(
    agent_id     = "my-agent",
    session_id   = "user-session-1",
    token_budget = 4096,            # optional, default 4096
)

# Use it anywhere LangChain expects a chat message history:
chain = RunnableWithMessageHistory(runnable, lambda _: history)
response = await chain.ainvoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "user-session-1"}},
)

# Or read and write it directly:
await history.aadd_messages([HumanMessage(content="hi"), AIMessage(content="hello")])
messages = await history.aget_messages()
```

On **LangChain 0.x only**, `ActroneMemory` is the `ConversationBufferMemory` drop-in and the
rest of your chain stays the same. It raises on 1.x with a pointer to the adapter above.

```python
from actrone_memory.integrations.langchain import ActroneMemory

memory = ActroneMemory(agent_id="my-agent", session_id="user-session-1")
chain = ConversationChain(llm=llm, memory=memory)
```

### LangGraph

```python
from actrone_memory.integrations.langgraph import ActroneCheckpointer

checkpointer = ActroneCheckpointer(
    agent_id     = "my-agent",
    token_budget = 4096,
)

graph = graph_builder.compile(checkpointer=checkpointer)
# Your graph now persists state across sessions and restarts
```

### CrewAI

```python
from actrone_memory.integrations.crewai import ActroneCrewMemory
from crewai import Task

# Agents sharing the same agent_id and session_id share memory
memory = ActroneCrewMemory(
    agent_id   = "research-crew",
    session_id = "project-alpha",
)

context = await memory.build_context("competitor pricing")
task = Task(
    description=f"{context}\n\nSummarise competitor pricing.",
    expected_output="A short summary of competitor pricing and strategic implications.",
    agent=researcher,
)
# ...crew.kickoff(), then store what the crew concluded:
await memory.save(result, {"task_input": "Summarise competitor pricing."})
```

The adapter feeds memory into the task text rather than replacing CrewAI's own memory storage,
so it works the same across CrewAI versions.

See the [`examples/`](../examples/) folder for fully working scripts.
