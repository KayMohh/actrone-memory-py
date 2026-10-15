"""Tests for the logging helpers.

These paths had no coverage, which is how a broken pairing shipped: the stdlib
``add_logger_name`` / ``filter_by_level`` processors were combined with
``PrintLoggerFactory``, whose logger has no ``.name`` and no level, so calling
``configure_json_logging()`` raised ``AttributeError`` on the first line emitted.
Every test here actually emits a line and inspects what reached stdout.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest
import structlog

from actrone_memory.logging import (
    bind_logger,
    clear_log_context,
    configure_dev_logging,
    configure_json_logging,
    set_request_id,
    set_trace_id,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Restore global structlog/stdlib state so tests cannot leak configuration."""
    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    previous_level = root.level
    try:
        yield
    finally:
        clear_log_context()
        structlog.reset_defaults()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)


def _lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return [line for line in capsys.readouterr().out.splitlines() if line.strip()]


def _emit_json(capsys: pytest.CaptureFixture[str], **fields: object) -> dict[str, object]:
    log = bind_logger("actrone_memory.test")
    log.info("memory.test.event", **fields)
    lines = _lines(capsys)
    assert lines, "no log line was emitted"
    parsed: dict[str, object] = json.loads(lines[-1])
    return parsed


def test_configure_json_logging_emits_parseable_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_json_logging()

    payload = _emit_json(capsys, agent_id="a1")

    assert payload["event"] == "memory.test.event"
    assert payload["service"] == "actrone-memory"
    assert payload["agent_id"] == "a1"
    assert payload["level"] == "info"
    assert payload["logger"] == "actrone_memory.test"
    assert "timestamp" in payload


def test_configure_dev_logging_emits_a_line(capsys: pytest.CaptureFixture[str]) -> None:
    """The console renderer must not raise, and must carry the event name."""
    configure_dev_logging()
    bind_logger("actrone_memory.test").info("memory.test.event")

    lines = _lines(capsys)

    assert lines, "no log line was emitted"
    assert "memory.test.event" in lines[-1]


def test_request_and_trace_ids_reach_the_log_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_json_logging()
    set_request_id("req-1")
    set_trace_id("trace-1")

    payload = _emit_json(capsys)

    assert payload["request_id"] == "req-1"
    assert payload["trace_id"] == "trace-1"


def test_ids_apply_to_a_logger_created_before_they_were_set(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Module-level loggers are built at import time, before any request exists."""
    configure_json_logging()
    log = bind_logger("actrone_memory.test")  # created first

    set_request_id("req-later")  # set afterwards, as a request handler would
    log.info("memory.test.event")

    assert json.loads(_lines(capsys)[-1])["request_id"] == "req-later"


def test_clear_log_context_stops_ids_leaking(capsys: pytest.CaptureFixture[str]) -> None:
    configure_json_logging()
    set_request_id("req-1")
    set_trace_id("trace-1")
    clear_log_context()

    payload = _emit_json(capsys)

    assert "request_id" not in payload
    assert "trace_id" not in payload


def test_setting_an_id_to_none_clears_it(capsys: pytest.CaptureFixture[str]) -> None:
    configure_json_logging()
    set_request_id("req-1")
    set_request_id(None)

    assert "request_id" not in _emit_json(capsys)


def test_json_logging_honours_the_level_argument(capsys: pytest.CaptureFixture[str]) -> None:
    """The level argument must actually filter, not be silently ignored."""
    configure_json_logging(level=logging.WARNING)
    log = bind_logger("actrone_memory.test")

    log.info("memory.test.suppressed")
    assert _lines(capsys) == []

    log.warning("memory.test.kept")
    assert json.loads(_lines(capsys)[-1])["event"] == "memory.test.kept"


def test_bind_logger_attaches_extra_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_json_logging()
    bind_logger("actrone_memory.test", agent_id="a-42").info("memory.test.event")

    assert json.loads(_lines(capsys)[-1])["agent_id"] == "a-42"


def test_library_does_not_configure_logging_on_import(capsys: pytest.CaptureFixture[str]) -> None:
    """A library must leave logging configuration to the application."""
    structlog.reset_defaults()
    assert not structlog.is_configured()
