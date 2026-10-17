"""Redaction on read: mask memories by sensitivity before their text reaches a model.

The library never detects personal data itself. A memory's ``sensitivity`` comes from the fact
extractor or from what you pass to ``inject_memory()``, and this module acts only on that tag. Raw
conversation turns carry no tag, so they are never redacted.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from typing import get_args

from actrone_memory.exceptions import ValidationError
from actrone_memory.models import MemoryEntry, Sensitivity

#: Every sensitivity class, least to most sensitive.
SENSITIVITIES: tuple[Sensitivity, ...] = get_args(Sensitivity)


def redacted_content(sensitivity: Sensitivity) -> str:
    """Return the placeholder that replaces a redacted memory's content.

    Args:
        sensitivity: The memory's sensitivity class.

    Returns:
        The placeholder text, for example ``"[redacted: pii]"``.
    """
    return f"[redacted: {sensitivity}]"


def resolve_redaction(classes: Iterable[str]) -> frozenset[Sensitivity]:
    """Validate the sensitivity classes to redact and return them as a set.

    Args:
        classes: Sensitivity class names, such as ``["pii", "sensitive"]``.

    Returns:
        The classes as a frozen set, empty when nothing should be redacted.

    Raises:
        ValidationError: If ``classes`` is a single string, or names an unknown class.
    """
    if isinstance(classes, str):
        raise ValidationError(
            "redact", "pass a list of sensitivity classes, such as ['pii', 'sensitive']"
        )
    resolved: set[Sensitivity] = set()
    for value in classes:
        if value not in SENSITIVITIES:
            raise ValidationError(
                "redact",
                f"unknown sensitivity {value!r}; expected one of {', '.join(SENSITIVITIES)}",
            )
        resolved.add(value)
    return frozenset(resolved)


def redact_memories(
    entries: list[MemoryEntry],
    classes: Collection[Sensitivity],
    count_tokens: Callable[[str], int],
) -> list[MemoryEntry]:
    """Return ``entries`` with every memory whose sensitivity is in ``classes`` masked.

    A masked memory keeps its id, sensitivity, source and position in the ranking. Its content
    becomes the placeholder, its embedding is dropped because a vector can leak the text it
    encodes, and its token count is recounted so budget pruning sees the size actually sent. The
    stored memory is not changed. O(n) in the number of entries.

    Args:
        entries: Memories as returned by a search, in ranked order.
        classes: Sensitivity classes to mask. Empty means mask nothing.
        count_tokens: The manager's token counter.

    Returns:
        A new list, or ``entries`` itself when ``classes`` is empty.
    """
    if not classes:
        return entries
    masked: list[MemoryEntry] = []
    for entry in entries:
        if entry.sensitivity in classes:
            placeholder = redacted_content(entry.sensitivity)
            entry = entry.model_copy(
                update={
                    "content": placeholder,
                    "embedding": None,
                    "token_count": count_tokens(placeholder),
                }
            )
        masked.append(entry)
    return masked
