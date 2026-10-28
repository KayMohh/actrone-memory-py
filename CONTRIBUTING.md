# Contributing to actrone-memory

First off, thank you for wanting to help. Whether it's fixing a typo, improving a code example, or building a new integration, every contribution matters.

This document walks you through everything you need to get set up and submit a pull request.

---

## What You'll Need Before You Start

- **Python 3.11 or newer**, check with `python --version`
- **[uv](https://docs.astral.sh/uv/)**, the package manager we use
- **[Docker](https://docs.docker.com/get-docker/)**, needed to run the Redis/Valkey, Qdrant, and Postgres integration suites locally

Install uv with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setting Up Your Development Environment

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/actrone-memory-py
cd actrone-memory-py

# 2. Create an isolated Python environment and install everything
uv venv .venv
uv pip install -e ".[dev,langchain,langgraph,crewai]"

# 3. Ensure Docker is running (for integration tests)
# The integration suite automatically manages ephemeral Redis, Valkey, Qdrant,
# and Postgres test containers via testcontainers, so you do not need to manually
# run them. If you want manual standalone containers for local development:
# docker run -d --name actrone-redis    -p 6379:6379 redis:7.2-alpine
# docker run -d --name actrone-qdrant   -p 6333:6333 qdrant/qdrant:v1.9.2
# docker run -d --name actrone-postgres -p 5432:5432 pgvector/pgvector:pg16

# 4. Set your environment variables
cp .env.example .env
# The defaults need no API key. Set ACTRONE_OPENAI_API_KEY only if you use ACTRONE_EMBEDDING_PROVIDER=openai
```

You're ready.

---

## Running the Tests

### Unit tests: no infrastructure needed, run these constantly

```bash
pytest tests/unit/ -v
```

These test all the logic using mock objects, so you don't need Redis, Qdrant, Postgres, or Valkey running. Run them every time you change something.

### Integration tests: Redis / Qdrant / Postgres / Valkey

```bash
pytest tests/integration/ -v
```

These run the real adapters against live containers. The Postgres suite covers both the hot tier (`PostgresStore`) and the pgvector long-term tier (`PgVectorStore`), and the Valkey suite checks that the Redis adapter still works against a Redis-compatible fork.

### Full suite with coverage report

```bash
pytest tests/ -v --cov=actrone_memory --cov-report=term-missing
```

**All pull requests must pass the unit suite and maintain ≥ 80% coverage.** CI will check this automatically and block the PR if it drops.

### Checking code style

```bash
ruff check src/ tests/      # find style issues
ruff format --check src/ tests/  # fail if formatting would change files
mypy src/actrone_memory   # type check
```

All three must pass with zero errors before your PR will be reviewed.

---

## How the Codebase is Structured

If you're new to the project, here's where things live:

```text
src/actrone_memory/
│
├── __init__.py          ← Public package exports.
├── cli.py               ← CLI entry points.
├── config.py            ← All configuration settings.
├── exceptions.py        ← Custom error classes.
├── extraction.py        ← Fact extraction pipeline.
├── in_memory.py         ← In-process L1/L2 store implementation.
├── manager.py           ← Main public interface. Start here.
├── models.py            ← Data types (MemoryEntry, Turn, etc.).
├── protocols.py         ← L1/L2 store protocols.
├── redaction.py         ← Sensitivity redaction helpers.
├── retrieval.py         ← Retrieval/ranking helpers.
├── testing.py           ← Conformance testing utilities.
├── tokens.py            ← Token counting helpers.
├── benchmark/          ← Benchmarking and comparison helpers.
├── l1/
│   ├── __init__.py
│   ├── postgres_store.py ← Short-term memory in Postgres.
│   └── redis_store.py   ← Short-term memory (Redis operations).
│
├── l2/
│   ├── __init__.py
│   ├── embedder.py       ← Text-to-vector embedding layer.
│   ├── pgvector_store.py ← Long-term memory in Postgres + pgvector.
│   └── qdrant_store.py   ← Long-term memory (Qdrant operations).
│
└── integrations/      ← Framework adapters and helper integrations.

tests/
├── __init__.py
├── conftest.py
├── contract/            ← Framework contract tests.
├── integration/         ← Real Redis / Qdrant / Postgres / Valkey tests.
└── unit/                ← Logic tests using mocks (fast, no infrastructure).
```

---

## Code Style Rules

We keep things consistent so the codebase stays readable for everyone:

- **Type annotate everything.** Every function parameter and return value needs a type hint. `mypy` will catch missing ones.
- **Anything that touches the network must be async; no blocking calls inside async functions.**
- **No mutable default arguments.** Use `None` and set the default inside the function instead.
- **No bare `except: pass`.** If you catch an exception, either handle it properly or re-raise it.
- **Line length is 100 characters.** `ruff format --check` catches drift; `ruff format` rewrites files.

---

## Submitting a Pull Request

```text
1. Create a branch from main
   git checkout -b feat/your-feature-name

2. Make your changes and write tests for them

3. Run the test suite and confirm it passes
   pytest tests/unit/ -v

4. Commit using Conventional Commits format:
   feat: add Weaviate L2 store
   fix: handle Redis timeout in append_turn
   docs: add example for multi-agent memory sharing
   test: add coverage for qdrant_store error paths

5. Push and open a pull request against main
```

**In your PR description, explain *why* the change is needed**, not just what you changed. "Fixes a bug" tells us nothing; "Redis was not releasing the pipeline object on timeout, causing connection leaks" tells us everything.

### What we look for in review

- Does this solve a real problem without adding unnecessary complexity?
- Are all new code paths covered by tests, including error cases?
- Does it follow the existing async patterns in `manager.py`?
- Are any new dependencies justified, stable, and pinned in `pyproject.toml`?

---

## Reporting a Bug

Open a [GitHub Issue](https://github.com/actrone/actrone-memory-py/issues) with:

- Your Python version (`python --version`) and OS
- A minimal code snippet that reproduces the problem
- What you expected to happen vs. what actually happened
- Any relevant error messages or logs (redact API keys and personal data before pasting)

For security vulnerabilities, **do not open a public issue**, see [SECURITY.md](SECURITY.md) for the private reporting process.

---

## Questions?

Open a [Discussion](https://github.com/actrone/actrone-memory-py/discussions) on GitHub, no question is too basic.
