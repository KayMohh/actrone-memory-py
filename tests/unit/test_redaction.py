"""Redaction on read: the pure helpers and the manager's read paths."""

from __future__ import annotations

import pytest

from actrone_memory import SENSITIVITIES, redacted_content
from actrone_memory.config import MemoryConfig
from actrone_memory.exceptions import ValidationError
from actrone_memory.manager import MemoryManager
from actrone_memory.models import MemoryEntry, Sensitivity
from actrone_memory.redaction import redact_memories, resolve_redaction
from tests.conftest import ConstantEmbedder


def _entry(content: str, sensitivity: Sensitivity) -> MemoryEntry:
    return MemoryEntry(
        agent_id="a1",
        session_id="s1",
        content=content,
        content_type="fact",
        embedding=[0.1, 0.2],
        token_count=99,
        sensitivity=sensitivity,
    )


def _count(text: str) -> int:
    return len(text)


# ── resolve_redaction ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ([], frozenset()),
        (["pii"], frozenset({"pii"})),
        (["pii", "sensitive", "pii"], frozenset({"pii", "sensitive"})),
        (list(SENSITIVITIES), frozenset(SENSITIVITIES)),
    ],
)
def test_resolve_redaction_accepts_known_classes(
    given: list[str], expected: frozenset[Sensitivity]
) -> None:
    assert resolve_redaction(given) == expected


@pytest.mark.parametrize(
    ("given", "message"),
    [
        (["PII"], "unknown sensitivity 'PII'"),
        (["pii", "secret"], "unknown sensitivity 'secret'"),
        ("pii", "pass a list of sensitivity classes"),
    ],
)
def test_resolve_redaction_rejects_bad_input(given: list[str] | str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        resolve_redaction(given)


# ── redact_memories ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("classes", "expected_contents"),
    [
        (frozenset(), ["plain fact", "jane@example.com", "has asthma"]),
        (frozenset({"pii"}), ["plain fact", "[redacted: pii]", "has asthma"]),
        (
            frozenset({"pii", "sensitive"}),
            ["plain fact", "[redacted: pii]", "[redacted: sensitive]"],
        ),
    ],
)
def test_redact_memories_masks_only_the_listed_classes(
    classes: frozenset[Sensitivity], expected_contents: list[str]
) -> None:
    entries = [
        _entry("plain fact", "none"),
        _entry("jane@example.com", "pii"),
        _entry("has asthma", "sensitive"),
    ]
    assert [e.content for e in redact_memories(entries, classes, _count)] == expected_contents


def test_a_masked_memory_keeps_its_identity_and_loses_its_vector() -> None:
    original = _entry("jane@example.com", "pii")

    (masked,) = redact_memories([original], frozenset({"pii"}), _count)

    assert masked.id == original.id
    assert masked.sensitivity == "pii"
    assert masked.source == original.source
    assert masked.embedding is None
    assert masked.token_count == len(redacted_content("pii"))
    # The caller's object, like the stored memory, is not changed.
    assert original.content == "jane@example.com"
    assert original.embedding == [0.1, 0.2]


def test_no_classes_returns_the_same_list() -> None:
    entries = [_entry("jane@example.com", "pii")]
    assert redact_memories(entries, frozenset(), _count) is entries


# ── MemoryManager read paths ─────────────────────────────────────────────────


async def _manager(**config: object) -> MemoryManager:
    cfg = MemoryConfig(embedding_provider="hashing", relevance_threshold=0.01, **config)  # type: ignore[arg-type]
    mm = await MemoryManager.create(cfg, embedder=ConstantEmbedder())
    await mm.store_turn("a1", "s1", "my email is jane@example.com", "noted")
    await mm.inject_memory("a1", "prefers concise answers", sensitivity="low")
    await mm.inject_memory("a1", "email is jane@example.com", sensitivity="pii")
    await mm.inject_memory("a1", "has a peanut allergy", sensitivity="sensitive")
    return mm


def _contents(memories: list[MemoryEntry]) -> set[str]:
    return {m.content for m in memories}


@pytest.mark.asyncio
async def test_retrieve_context_redacts_on_request() -> None:
    mm = await _manager()

    ctx = await mm.retrieve_context("a1", "s1", "email", 4000, redact=["pii", "sensitive"])

    assert _contents(ctx.episodic_memories) == {
        "prefers concise answers",
        "[redacted: pii]",
        "[redacted: sensitive]",
    }
    # Turns carry no sensitivity tag, so they are never masked.
    assert "jane@example.com" in ctx.recent_turns[0].user_message
    await mm.close()


@pytest.mark.asyncio
async def test_the_config_default_covers_both_read_paths() -> None:
    mm = await _manager(redact_sensitivities=["pii"])

    ctx = await mm.retrieve_context("a1", "s1", "email", 4000)
    found = await mm.search_memories("a1", "email")

    for memories in (ctx.episodic_memories, found):
        assert "[redacted: pii]" in _contents(memories)
        assert "email is jane@example.com" not in _contents(memories)
        assert "has a peanut allergy" in _contents(memories)
    await mm.close()


@pytest.mark.asyncio
async def test_an_empty_list_turns_redaction_off_for_one_call() -> None:
    mm = await _manager(redact_sensitivities=["pii", "sensitive"])

    found = await mm.search_memories("a1", "email", redact=[])

    assert "email is jane@example.com" in _contents(found)
    await mm.close()


@pytest.mark.asyncio
async def test_redaction_never_changes_what_is_stored() -> None:
    mm = await _manager()

    await mm.retrieve_context("a1", "s1", "email", 4000, redact=["pii"])
    found = await mm.search_memories("a1", "email")

    assert "email is jane@example.com" in _contents(found)
    await mm.close()


@pytest.mark.asyncio
async def test_an_unknown_class_fails_before_any_read() -> None:
    mm = await _manager()

    with pytest.raises(ValidationError, match="unknown sensitivity 'PII'"):
        await mm.retrieve_context("a1", "s1", "email", 4000, redact=["PII"])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="unknown sensitivity 'PII'"):
        await mm.search_memories("a1", "email", redact=["PII"])  # type: ignore[list-item]
    await mm.close()


def test_the_config_reads_classes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTRONE_REDACT_SENSITIVITIES", '["pii", "sensitive"]')
    assert MemoryConfig().redact_sensitivities == ["pii", "sensitive"]


def test_the_config_rejects_an_unknown_class() -> None:
    with pytest.raises(ValueError, match="redact_sensitivities"):
        MemoryConfig(redact_sensitivities=["secret"])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_the_config_default_reaches_the_adapter_context_string() -> None:
    """Framework adapters render prompts through governed_context, so they inherit the default."""
    from actrone_memory.integrations._context import governed_context

    cfg = MemoryConfig(  # type: ignore[call-arg]
        embedding_provider="hashing", relevance_threshold=0.01, redact_sensitivities=["pii"]
    )
    mm = await MemoryManager.create(cfg, embedder=ConstantEmbedder())
    await mm.store_turn("a1", "s1", "what is on file?", "let me check")
    await mm.inject_memory("a1", "email is jane@example.com", sensitivity="pii")

    prompt = await governed_context(mm, "a1", "s1", "email")

    assert "[redacted: pii]" in prompt
    assert "jane@example.com" not in prompt
    await mm.close()
