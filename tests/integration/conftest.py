"""Auto-apply the integration marker to every test in this directory.

Tests here require real Redis and Qdrant instances started by testcontainers.
Docker must be running. To run only integration tests:

    uv run pytest -m integration

To exclude them (the default for unit-test runs):

    uv run pytest -m "not integration"
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test collected from the integration directory as 'integration'.

    pytest's module-level ``pytestmark`` only marks tests defined *in* the
    conftest, not sibling test files. This hook applies the mark to all tests
    whose node ID contains '/integration/' regardless of file.
    """
    integration_mark = pytest.mark.integration
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(integration_mark, append=False)
