# Security policy

`actrone-memory` stores raw conversation history, semantic memory vectors, and session metadata. A vulnerability in this library could expose real user data, so we take security reports seriously and act on them quickly. The library is maintained by Apocalypse Technologies ("we", "us"), the company behind Actrone.

---

## How to report a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.** A public issue lets everyone see the problem before a fix exists, which puts all users at risk.

Instead, email us privately at **[security@actrone.com](mailto:security@actrone.com)** with:

1. A clear description of what the vulnerability is
2. Which part of the library is affected (e.g. `RedisStore`, `QdrantStore`, the LangChain adapter, etc.)
3. Steps to reproduce it, a minimal code example helps enormously
4. Your assessment of the impact (e.g. "an attacker could read another user's conversation history")
5. Your name or handle if you'd like credit in the release notes, or let us know if you prefer to stay anonymous

**What happens next:**

| Timeline | What we do |
| --- | --- |
| Within 72 hours | We acknowledge your report and confirm we received it |
| Days 1-14 | We investigate, develop a fix, and review it internally |
| Day 15 | We publish a patched release and a public advisory |

We will not take legal action against researchers who follow this policy. We consider responsible disclosure a service to the community.

---

## Known areas of risk

These are the parts of the library where security matters most. Understanding them helps you deploy safely.

### Conversation data stored in plain text

Everything you pass to `store_turn()`, user messages, assistant responses, tool results, is stored as plain text in Redis. Redis does not encrypt data at rest by default.

**What to do:**

- Use a `rediss://` URL (note the double `s`) to enable TLS in transit
- Run Redis inside a private network, not exposed to the internet
- Set Redis ACLs to restrict which clients can read session keys
- For regulated industries (healthcare, finance), consider encrypting the data before storing it

### Long-term memories are searchable by anyone with Qdrant access

Vector payloads in Qdrant include the original text of the memory. Anyone who can query your Qdrant instance can read your users' conversation summaries.

**What to do:**

- Always set `ACTRONE_QDRANT_API_KEY` when using Qdrant Cloud
- For self-hosted Qdrant, put it behind a firewall or VPN
- Use separate Qdrant collections per tenant if you're serving multiple customers

### Injected memories appear in LLM prompts

If an attacker can call `inject_memory()`, or write malicious content into a turn, that content will be embedded and surfaced in future prompts. This is a form of **prompt injection via memory**.

**What to do:**

- Treat `inject_memory()` as an admin operation, protect it behind authentication
- Validate and sanitise all user input before storing it
- Audit injected memories periodically using `search_memories()`

### The embedding cache uses text as a key

Embeddings are cached in Redis under a SHA-256 hash of the original text. If you update your embedding model, old cached embeddings become stale but will continue to be returned. SHA-256 collisions are not a practical concern, but stale embeddings are.

**What to do:**

- After upgrading your embedding model, flush the embedding cache by deleting all `embed:*` keys from Redis

### Dependencies

All dependencies are pinned in `uv.lock`. We run `pip-audit` in CI and block any release that has a HIGH or CRITICAL finding in its dependency tree. If you find a vulnerability in one of our dependencies before we do, please report it here and we will release a patch immediately.

---

## Supported versions

We only maintain the latest release. Older versions do not receive security patches.

| Version | Security fixes |
| --- | --- |
| 0.3.x (current) | Yes |
| Older than 0.3.0 | No |
