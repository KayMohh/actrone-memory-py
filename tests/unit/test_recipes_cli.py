"""Unit tests for the recipe registry + the ``actrone-memory`` CLI.

Mirrors the TS ``recipes``/``cli`` tests: every recipe is well-formed (install + hosted-upgrade
seed), and the CLI's ``add`` / ``list`` / ``--write`` / error paths behave, all against an injected
IO so nothing touches real stdout or disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from actrone_memory._example_snippets import EXAMPLE_SNIPPETS
from actrone_memory.cli import run_cli
from actrone_memory.recipes import (
    HOSTED_UPGRADE_HINT,
    RECIPE_FRAMEWORKS,
    RECIPES,
    get_recipe,
    list_frameworks,
    render_recipe,
    render_standalone_file,
)


class _FakeIO:
    """Records log/error output and simulates a filesystem for ``--write`` tests."""

    def __init__(self, existing: set[str] | None = None) -> None:
        self.logs: list[str] = []
        self.errors: list[str] = []
        self.written: dict[str, str] = {}
        self._existing = existing or set()

    def log(self, message: str) -> None:
        self.logs.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def file_exists(self, path: str) -> bool:
        return path in self._existing or path in self.written

    def write_file(self, path: str, content: str) -> None:
        self.written[path] = content


# ── recipes registry ──────────────────────────────────────────────────────────


def test_every_recipe_is_well_formed() -> None:
    assert set(RECIPES) == set(list_frameworks())
    for slug, recipe in RECIPES.items():
        assert recipe.framework == slug
        assert recipe.label
        # Extras are quoted: unquoted brackets are a glob in zsh (the macOS default shell).
        assert recipe.install == "pip install actrone-memory" or recipe.install.startswith(
            'pip install "actrone-memory['
        )
        # Every recipe ends with the one-import hosted-upgrade seed.
        assert recipe.snippet.rstrip().endswith(HOSTED_UPGRADE_HINT)


def test_new_adapter_recipes_are_backed_by_mypy_checked_examples() -> None:
    """Trust gate (mirrors the TS recipes-examples test): every adapter added this cycle ships a
    real ``examples/frameworks/<slug>.py`` whose ``# region`` snippet is extracted into
    snippets.json. ``mypy examples/frameworks/`` type-checks those sources in CI, so a recipe whose
    wiring stops compiling fails the build, the recipes are verified, not just strings."""
    snippets_path = Path(__file__).resolve().parents[2] / "examples" / "snippets.json"
    snippets = json.loads(snippets_path.read_text(encoding="utf-8"))
    new_this_cycle = [
        "core",
        "openai_agents",
        "pydantic_ai",
        "claude_agent_sdk",
        "semantic_kernel",
        "google_adk",
        "microsoft_agent_framework",
        "agno",
        "smolagents",
        "aws_strands",
    ]
    for slug in new_this_cycle:
        assert slug in RECIPES, f"missing recipe for {slug}"
        assert slug in snippets, f"missing examples/frameworks/{slug}.py #region {slug}"
        assert "actrone_memory" in snippets[slug]
        assert "MemoryManager.create()" in snippets[slug]


def test_the_six_new_frameworks_have_recipes() -> None:
    for slug in (
        "openai_agents",
        "pydantic_ai",
        "claude_agent_sdk",
        "semantic_kernel",
        "google_adk",
        "microsoft_agent_framework",
    ):
        assert slug in RECIPES


def test_get_recipe_is_case_insensitive() -> None:
    assert get_recipe("LangGraph") is RECIPES["langgraph"]
    assert get_recipe("  crewai  ") is RECIPES["crewai"]
    assert get_recipe("nope") is None


def test_render_helpers_include_install_and_snippet() -> None:
    recipe = RECIPES["core"]
    rendered = render_recipe(recipe)
    assert recipe.install in rendered
    assert "Paste into your agent" in rendered
    standalone = render_standalone_file(recipe)
    assert standalone.startswith("# actrone-memory")
    assert recipe.install in standalone


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_list_prints_all_frameworks() -> None:
    io = _FakeIO()
    assert run_cli(["list"], io) == 0
    printed = io.logs[0].splitlines()
    assert printed == list_frameworks()


def test_add_prints_recipe() -> None:
    io = _FakeIO()
    assert run_cli(["add", "openai_agents"], io) == 0
    assert "OpenAI Agents SDK" in io.logs[0]
    assert 'pip install "actrone-memory[openai_agents]"' in io.logs[0]


def test_add_unknown_framework_errors() -> None:
    io = _FakeIO()
    assert run_cli(["add", "nope"], io) == 1
    assert "Unknown framework" in io.errors[0]


def test_add_write_creates_a_new_file() -> None:
    io = _FakeIO()
    assert run_cli(["add", "langgraph", "--write", "memory.py"], io) == 0
    assert "memory.py" in io.written
    assert io.written["memory.py"].startswith("# actrone-memory")


def test_add_write_refuses_to_overwrite() -> None:
    io = _FakeIO(existing={"memory.py"})
    assert run_cli(["add", "langgraph", "--write", "memory.py"], io) == 1
    assert "Refusing to overwrite" in io.errors[0]
    assert "memory.py" not in io.written


def test_no_command_prints_usage_and_returns_1() -> None:
    io = _FakeIO()
    assert run_cli([], io) == 1
    assert "Usage" in io.logs[0]


def test_unknown_command_errors() -> None:
    io = _FakeIO()
    assert run_cli(["frobnicate"], io) == 1
    assert "Unknown command" in io.errors[0]


# ── The CLI prints exactly the mypy-checked example code ─────────────────────
# Before this gate the recipes were hand-typed strings: the core recipe put `await` at module top
# level, a SyntaxError in a normal Python file, and used names it never defined, so a `--write`
# file could not even be imported.


def test_every_framework_has_a_recipe_none_dropped_for_a_missing_example() -> None:
    assert sorted(RECIPES) == sorted(RECIPE_FRAMEWORKS)


def test_each_recipe_is_its_examples_code() -> None:
    snippets_path = Path(__file__).resolve().parents[2] / "examples" / "snippets.json"
    snippets = json.loads(snippets_path.read_text(encoding="utf-8"))
    for framework, recipe in RECIPES.items():
        assert EXAMPLE_SNIPPETS[framework] == snippets[framework]
        assert recipe.snippet.startswith(snippets[framework].rstrip()), framework


def test_every_written_file_is_valid_python_with_no_top_level_await() -> None:
    import ast

    for framework, recipe in RECIPES.items():
        tree = ast.parse(render_standalone_file(recipe), filename=f"{framework}.py")
        top_level_awaits = [
            node
            for stmt in tree.body
            if not isinstance(stmt, ast.AsyncFunctionDef)
            for node in ast.walk(stmt)
            if isinstance(node, ast.Await)
        ]
        assert top_level_awaits == [], framework


def test_the_written_core_file_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    monkeypatch.setenv("ACTRONE_EMBEDDING_PROVIDER", "hashing")  # deterministic, no model download
    target = tmp_path / "memory_core.py"
    target.write_text(render_standalone_file(RECIPES["core"]), encoding="utf-8")
    runpy.run_path(str(target), run_name="__main__")  # raises if the example fails
