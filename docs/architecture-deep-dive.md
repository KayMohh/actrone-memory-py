# Architecture Deep Dive

> This document explains how `actrone-memory` works under the hood.
> No prior knowledge assumed, if you know what an API is, you'll follow along.

---

## The Core Problem: AI Agents Have No Memory

When you talk to an AI, everything you say is sent to the language model as a single block of text called the **context window**. Think of it as a piece of paper, the AI can only see what's written on that one page.

Two problems come from this:

1. **The page has a size limit.** Most models cap out at 8,000 to 128,000 tokens (roughly 6,000-100,000 words). In a long conversation, old messages fall off the page.
2. **The page is thrown away after every session.** Start a new conversation and the AI has no idea who you are.

`actrone-memory` fixes both problems by giving agents two types of memory, one for what just happened, one for everything ever said.

---

## The Two-Tier Memory System

Inspired by how computers use RAM and a hard drive, we use two different storage systems with different trade-offs.

> **Note:** The diagrams in this document illustrate the architecture using the durable production backend (`backend="redis_qdrant"`, Redis for L1 and Qdrant for L2). By default (`backend="memory"`), `actrone-memory` runs completely in-process with zero external services and zero network egress, using in-memory stores and local embeddings. Postgres + pgvector is also supported via `actrone-memory[pgvector]`.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   SHORT-TERM MEMORY                    LONG-TERM MEMORY             │
│   ─────────────────                    ────────────────             │
│   Redis (in-memory database)           Qdrant (vector database)     │
│                                                                     │
│   Like RAM in a computer:              Like a hard drive:           │
│   • Lightning fast (< 1ms)             • Fast (~10ms)               │
│   • Stores raw recent messages         • Stores compressed history  │
│   • Expires after 24 hours             • Keeps data forever         │
│   • Holds last 50 messages             • Searches by meaning        │
│                                                                     │
│   "What did we just talk about?"       "Have we ever talked         │
│                                         about this topic before?"   │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Two Tiers?

You could store everything in one place, but no single system is perfect for both jobs:

- **Redis** is extremely fast but stores data in raw text and doesn't understand meaning. You can't ask it "find everything we talked about related to trading", you'd have to search message by message.
- **Qdrant** understands meaning through something called **vector embeddings** (explained below), but it's slower and is designed for search, not for storing ordered lists of recent messages.

Using both gives you speed *and* intelligence.

---

## Vector Embeddings: Search by Meaning, Not Keywords

This is the magic that makes long-term memory useful.

When a new message comes in, we convert it into a list of numbers (called an **embedding vector**) that captures the *meaning* of the text. Similar topics produce similar number patterns, even if the exact words are different.

```text
"I'm building a stock trading bot"
        │
        ▼  (OpenAI text-embedding-3-small)
        │
[0.021, -0.418, 0.093, 0.671, ...]  ← 1,536 numbers representing the meaning
        │
        ▼
Stored in Qdrant alongside the original text
```

Later, when the user asks "What are the risks of algorithmic trading?", we embed *that* question and find the stored memories whose number patterns are closest, even though the exact words were different. This is called **cosine similarity**.

```text
Query:   "What are the risks of algorithmic trading?"
         [0.019, -0.401, 0.087, 0.658, ...]

Memory:  "I'm building a stock trading bot"
         [0.021, -0.418, 0.093, 0.671, ...]

Similarity score: 0.94  ✓  (above the threshold, included)


Query:   "What are the risks of algorithmic trading?"
         [0.019, -0.401, 0.087, 0.658, ...]

Memory:  "My cat's name is Whiskers"
         [-0.312, 0.891, -0.445, 0.123, ...]

Similarity score: 0.11  ✗  (below threshold, excluded)
```

The **threshold** is a quality gate: only memories that are genuinely relevant to the current question get included. Each
embedder carries the threshold it was calibrated for, because similarity scores are not comparable across models: `0.63`
for the default bge-small model, `0.4` for MiniLM, `0.3` for the keyword-only hashing embedder, and `0.72` for OpenAI or
a custom embedder. `ACTRONE_RELEVANCE_THRESHOLD` overrides it.

---

## The Retrieval Pipeline: Step by Step

Every time your agent needs to reply, you call `retrieve_context()`. Here is exactly what happens inside:

```text
You call: retrieve_context(query="What risks should I worry about?", token_budget=4096)
                │
                │
    ┌───────────▼───────────┐
    │   STEP 1: Fetch       │  Both happen at the same time (in parallel)
    │   (takes ~10ms)       │  so you don't wait twice.
    │                       │
    │  Redis ──────────────►│  "Give me the last 50 messages"
    │  Qdrant ─────────────►│  "Find memories similar to this query"
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │   STEP 2: Allocate    │  Decide how to divide the token budget.
    │   the budget          │
    │                       │
    │  System prompt   30%  │  ← Your agent's instructions. Sacred, never cut.
    │  Long-term mem   25%  │  ← Relevant memories from past conversations.
    │  Recent messages 35%  │  ← What was said in this session.
    │  Current message 10%  │  ← What the user just sent.
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │   STEP 3: Rank        │  Score each memory using two signals:
    │                       │
    │  Score = (0.7 × how   │  • Relevance: how closely it matches the query
    │           relevant)   │  • Recency: how recently it was created
    │        + (0.3 × how   │
    │           recent)     │  A memory from today scores higher than an equally
    │                       │  relevant memory from 3 weeks ago.
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │   STEP 4: Prune       │  If we still have too much to fit in the budget,
    │                       │  cut the lowest-priority items first:
    │  1. Oldest messages   │  ← dropped first (least relevant to now)
    │  2. Low-ranked mem    │  ← dropped next (least relevant to query)
    │  3. System prompt     │  ← NEVER dropped
    └───────────┬───────────┘
                │
                ▼
    RetrievedContext ready to inject into your LLM prompt ✓
```

---

## Auto-Summarisation: Compressing Old Conversations

After every 20 messages (configurable), a **background process** kicks in and compresses the conversation into a short summary, which gets saved to Qdrant. The process runs silently, users never wait for it.

```text
Session: 20 messages exchanged
              │
              ▼
    ┌─────────────────────┐
    │  Background task    │   Runs after the reply is sent, no delay for users
    │  kicks in           │
    └────────┬────────────┘
             │
             ▼
    Fetch the 20 messages from Redis
             │
             ▼
    Compress them into a summary paragraph
             │
             ▼
    Convert summary to embedding vector
             │
             ▼
    Save to Qdrant as a "summary" memory entry
             │
             ▼
    ✓  Now searchable in all future sessions
```

This means that even if the Redis short-term memory expires (after 24 hours), the compressed knowledge from that conversation lives on in Qdrant forever.

---

## The Embedding Cache: Avoiding Redundant API Calls

Converting text to an embedding vector costs money (a fraction of a cent per call) and takes ~100ms. If we embed the same text twice, we're wasting both.

The library caches every embedding in Redis with a 7-day expiry:

```text
embed("What is the capital of France?")
         │
         ├──► Check Redis cache
         │         │
         │    HIT ─┤─► Return cached vector instantly  (< 1ms, costs $0)
         │         │
         │   MISS ─┤─► Call OpenAI API                 (~100ms, ~$0.00002)
         │              │
         │              ▼
         │         Save to Redis cache (expires in 7 days)
         │              │
         └──────────────►  Return vector
```

---

## Data Flow: The Full Picture

Here's everything together, from user message to stored memory and back:

```text
  USER SENDS A MESSAGE
          │
          ▼
  ┌───────────────────────────────────────────────────────────┐
  │                    YOUR APPLICATION                        │
  │                                                           │
  │  1. Call retrieve_context()  ─────────────────────────►  │
  │                                                           │
  │     Redis  ─► recent messages  ┐                         │
  │     Qdrant ─► relevant history ┘ combined & pruned       │
  │                                                           │
  │  2. Inject context into LLM prompt                        │
  │                                                           │
  │  3. Call LLM → get response                               │
  │                                                           │
  │  4. Call store_turn()  ───────────────────────────────►  │
  │                                                           │
  │     Redis  ◄─ save raw message pair                       │
  │     Qdrant ◄─ (background) save compressed summary       │
  │                                                           │
  └───────────────────────────────────────────────────────────┘
          │
          ▼
  AGENT REPLIES, with full memory context ✓
```

---

## Concurrency and Safety

The library is safe to use in a high-traffic application:

- **Multiple users simultaneously**, each `(agent_id, session_id)` pair is completely isolated. User A's memory never leaks into User B's context.
- **Multiple async tasks**, built on Python's `asyncio`. You can call `store_turn()` and `retrieve_context()` from many coroutines at the same time without conflicts.
- **Background tasks**, auto-summarisation runs as a fire-and-forget background task. If it fails (e.g. a network blip), the error is logged and silently swallowed. It never crashes your main application.

---

## Further Reading

- [API Reference](api-reference.md), every method documented with examples
- [Examples / Cookbook](../examples/), copy-paste scripts to get started fast
