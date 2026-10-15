from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.exceptions import ConfigurationError
from actrone_memory.extraction import (
    EXTRACTION_RESPONSE_SCHEMA,
    EXTRACTION_SPEC_VERSION,
    EXTRACTION_SYSTEM_PROMPT,
    ExtractedFact,
    FactExtractor,
    OpenAIFactExtractor,
    format_extraction_input,
    parse_facts,
)
from actrone_memory.in_memory import InMemoryStore
from actrone_memory.l2.embedder import HashingEmbedder
from actrone_memory.manager import MemoryManager


class FakeExtractor:
    """Deterministic FactExtractor for tests, no LLM."""

    def __init__(self, facts: list[ExtractedFact]) -> None:
        self.facts = facts
        self.calls = 0

    async def extract(self, text: str) -> list[ExtractedFact]:
        self.calls += 1
        return self.facts


def _manager(extractor: FactExtractor | None = None, **cfg_over: object) -> MemoryManager:
    store = InMemoryStore()
    cfg = MemoryConfig(auto_summarise=False, relevance_threshold=0.1, **cfg_over)  # type: ignore[arg-type]
    return MemoryManager(store, store, HashingEmbedder(), cfg, extractor=extractor)


def test_fake_extractor_satisfies_protocol():
    assert isinstance(FakeExtractor([]), FactExtractor)


def test_spec_version_is_pinned():
    assert EXTRACTION_SPEC_VERSION == "1.1"


def test_prompt_matches_the_shared_spec():
    # The same hash is pinned in actrone-memory-ts and in the spec doc (extraction.v1.md), so a
    # prompt edit in one library fails here until the other library and the spec catch up.
    digest = hashlib.sha256(EXTRACTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert digest == "c25f946a45428b327a6983f8d5e7f8052a06736ac6cb1795824fb3e564593b09"


# ------------------------------------------------------------------
# OpenAIFactExtractor (any OpenAI-compatible client)
# ------------------------------------------------------------------

_ONE_FACT = (
    '{"facts": [{"content": "The user\'s name is Alex.", "sensitivity": "pii", '
    '"topic_tags": ["identity"], "importance": 0.8}]}'
)


class _RejectedError(Exception):
    """What the openai SDK raises when a server refuses a request (a 4xx status)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _FakeCompletions:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def _extractor(replies: list[str | Exception]) -> tuple[OpenAIFactExtractor, _FakeCompletions]:
    completions = _FakeCompletions(replies)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIFactExtractor(model="small-local-model", client=client), completions  # type: ignore[arg-type]


def test_format_extraction_input_frames_the_conversation():
    framed = format_extraction_input("User: hi\nAssistant: hello")
    assert framed.startswith("Conversation:\n\nUser: hi\nAssistant: hello")
    assert framed.endswith("Extract the durable facts from this conversation.")


@pytest.mark.asyncio
async def test_openai_extractor_sends_the_spec_request_with_a_json_schema():
    extractor, completions = _extractor([_ONE_FACT])
    facts = await extractor.extract("User: I'm Alex.\nAssistant: Hi Alex!")

    assert [(f.content, f.sensitivity) for f in facts] == [("The user's name is Alex.", "pii")]
    request = completions.requests[0]
    assert request["model"] == "small-local-model"
    assert request["messages"][0] == {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}
    conversation = "User: I'm Alex.\nAssistant: Hi Alex!"
    assert request["messages"][1]["content"] == format_extraction_input(conversation)
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["schema"] == EXTRACTION_RESPONSE_SCHEMA


@pytest.mark.asyncio
async def test_openai_extractor_falls_back_to_json_mode_when_the_schema_is_rejected():
    extractor, completions = _extractor([_RejectedError(400), _ONE_FACT, _ONE_FACT])

    assert len(await extractor.extract("User: I'm Alex.")) == 1
    assert len(await extractor.extract("User: I'm Alex.")) == 1
    # The rejected schema request, then JSON mode for this call and every later one.
    assert [r["response_format"]["type"] for r in completions.requests] == [
        "json_schema",
        "json_object",
        "json_object",
    ]


@pytest.mark.asyncio
async def test_openai_extractor_keeps_the_schema_after_a_transient_error():
    extractor, completions = _extractor([_RejectedError(503), _ONE_FACT])

    assert await extractor.extract("User: I'm Alex.") == []  # best effort: nothing, no raise
    assert len(await extractor.extract("User: I'm Alex.")) == 1
    formats = [r["response_format"]["type"] for r in completions.requests]
    assert formats == ["json_schema", "json_schema"]


@pytest.mark.asyncio
async def test_openai_extractor_returns_nothing_when_both_formats_fail():
    extractor, _ = _extractor([_RejectedError(400), _RejectedError(400)])
    assert await extractor.extract("User: I'm Alex.") == []


def test_openai_extractor_builds_a_client_for_a_base_url():
    pytest.importorskip("openai")
    extractor = OpenAIFactExtractor(
        api_key="local", model="m", base_url="http://localhost:11434/v1"
    )
    assert str(extractor._client.base_url).startswith("http://localhost:11434/v1")


# ------------------------------------------------------------------
# parse_facts
# ------------------------------------------------------------------


def test_parse_facts_envelope():
    raw = '{"facts": [{"content": "User is named Alex.", "sensitivity": "pii", "importance": 0.9}]}'
    facts = parse_facts(raw)
    assert len(facts) == 1
    assert facts[0].content == "User is named Alex."
    assert facts[0].sensitivity == "pii"
    assert facts[0].importance == 0.9


def test_parse_facts_bare_list():
    facts = parse_facts('[{"content": "User prefers email."}]')
    assert len(facts) == 1
    assert facts[0].sensitivity == "none"  # default
    assert facts[0].importance == 0.6  # default


def test_parse_facts_invalid_json_returns_empty():
    assert parse_facts("not json at all") == []


def test_parse_facts_non_list_returns_empty():
    assert parse_facts('{"facts": "oops"}') == []


def test_parse_facts_skips_malformed_entries():
    raw = '{"facts": [{"content": ""}, {"noContent": true}, "string", {"content": "Good fact."}]}'
    facts = parse_facts(raw)
    assert [f.content for f in facts] == ["Good fact."]


def test_parse_facts_skips_bad_sensitivity_but_keeps_others():
    raw = (
        '{"facts": [{"content": "bad", "sensitivity": "nonsense"}, '
        '{"content": "ok", "sensitivity": "low"}]}'
    )
    facts = parse_facts(raw)
    assert [f.content for f in facts] == ["ok"]


def test_parse_facts_clamps_content_and_count():
    long = "x" * 5000
    many = ",".join(f'{{"content": "fact {i}"}}' for i in range(30))
    raw = f'{{"facts": [{{"content": "{long}"}}, {many}]}}'
    facts = parse_facts(raw)
    assert len(facts) <= 20
    assert len(facts[0].content) <= 2000


# ------------------------------------------------------------------
# extract_memories (public API)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_memories_stores_facts_with_provenance():
    extractor = FakeExtractor(
        [
            ExtractedFact(content="User's email is alex@example.com.", sensitivity="pii"),
            ExtractedFact(content="User prefers concise answers.", sensitivity="low"),
        ]
    )
    mm = _manager(extractor)
    try:
        await mm.store_turn("a1", "s1", "I'm Alex, email alex@example.com", "Noted.")
        ids = await mm.extract_memories("a1", "s1")
        assert len(ids) == 2

        hits = await mm.search_memories("a1", "user email alex example concise answers", limit=10)
        facts = [h for h in hits if h.content_type == "fact"]
        assert facts
        assert all(f.source == "extracted" for f in facts)
        assert {f.sensitivity for f in facts} <= {"pii", "low"}
    finally:
        await mm.close()


@pytest.mark.asyncio
async def test_extract_memories_raises_without_extractor():
    mm = _manager(extractor=None)
    try:
        with pytest.raises(ConfigurationError, match="not enabled"):
            await mm.extract_memories("a1", "s1")
    finally:
        await mm.close()


@pytest.mark.asyncio
async def test_extract_memories_empty_when_no_turns():
    mm = _manager(FakeExtractor([ExtractedFact(content="ignored")]))
    try:
        ids = await mm.extract_memories("a1", "s-empty")
        assert ids == []
    finally:
        await mm.close()


@pytest.mark.asyncio
async def test_extraction_rides_summarise_path():
    extractor = FakeExtractor([ExtractedFact(content="User is building a trading bot.")])
    mm = _manager(extractor)
    try:
        await mm.store_turn("a1", "s1", "I'm building a trading bot", "Cool!")
        await mm._summarise_session("a1", "s1")
        assert extractor.calls == 1  # extraction ran alongside summarisation
        hits = await mm.search_memories("a1", "user building trading bot", limit=10)
        assert any(h.content_type == "fact" for h in hits)
    finally:
        await mm.close()


# ------------------------------------------------------------------
# _build_extractor gating
# ------------------------------------------------------------------


def test_build_extractor_off_by_default():
    cfg = MemoryConfig()
    assert MemoryManager._build_extractor(cfg) is None


def test_build_extractor_skipped_for_non_openai_provider():
    # extract_facts on, but hashing provider has no LLM → skipped.
    cfg = MemoryConfig(extract_facts=True, embedding_provider="hashing")
    assert MemoryManager._build_extractor(cfg) is None


def test_build_extractor_built_for_openai():
    cfg = MemoryConfig(
        extract_facts=True,
        embedding_provider="openai",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )
    extractor = MemoryManager._build_extractor(cfg)
    assert extractor is not None
    assert isinstance(extractor, FactExtractor)
