"""Google ADK memory adapter for actrone-memory (Tier 1 + Tier 2).

Google's Agent Development Kit governs long-term memory through ``BaseMemoryService``
(``add_session_to_memory`` + ``search_memory``). This adapter offers:

- **Tier 1**, :meth:`ActroneGoogleADKMemory.build_context`: the governed memory as a string
  to prepend to an agent's instruction for a turn (inherited from the shared base).
- **Tier 2**, :meth:`ActroneGoogleADKMemory.as_memory_service`: a real ADK
  ``BaseMemoryService`` backed by the Actrone store, so it drops straight into a
  ``Runner(memory_service=…)``. ``google-adk`` is imported lazily inside that method only, so
  the module imports (and the Tier-1 path runs) without ADK installed.

Then persist a completed turn with :meth:`remember`.
"""

from __future__ import annotations

from typing import Any

from actrone_memory.integrations._context import BaseActroneMemory


class ActroneGoogleADKMemory(BaseActroneMemory):
    """Governed memory for a Google ADK agent.

    Usage (Tier 2, native ``BaseMemoryService``)::

        from google.adk.runners import Runner
        from actrone_memory.integrations.google_adk import ActroneGoogleADKMemory

        memory = ActroneGoogleADKMemory(agent_id="support-bot", session_id="s1")
        runner = Runner(agent=agent, app_name="support", memory_service=memory.as_memory_service())
    """

    def as_memory_service(self) -> Any:
        """Return an ADK ``BaseMemoryService`` backed by the Actrone store (Tier 2).

        Requires ``google-adk`` (``pip install actrone-memory[google_adk]``); imported lazily
        here so the module and the Tier-1 path work without it.
        """
        try:
            from google.adk.memory import BaseMemoryService
            from google.adk.memory.base_memory_service import SearchMemoryResponse
            from google.adk.memory.memory_entry import MemoryEntry as ADKMemoryEntry
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - exercised via import guard test
            raise ImportError(
                "Install google-adk extras: pip install actrone-memory[google_adk]"
            ) from exc

        outer = self

        class _ActroneADKMemoryService(BaseMemoryService):  # type: ignore[misc]
            """ADK ``BaseMemoryService`` that stores into / searches Actrone memory."""

            async def add_session_to_memory(self, session: Any) -> None:
                mm = await outer.manager()
                for event in getattr(session, "events", []) or []:
                    content = getattr(event, "content", None)
                    text = _adk_content_text(content)
                    if text:
                        await mm.inject_memory(outer.agent_id, text, 0.6)

            async def search_memory(self, *, app_name: str, user_id: str, query: str) -> Any:
                mm = await outer.manager()
                results = await mm.search_memories(outer.agent_id, query)
                memories = [
                    ADKMemoryEntry(
                        content=types.Content(parts=[types.Part(text=r.content)]),
                        author="actrone-memory",
                        timestamp=str(getattr(r, "created_at", "")),
                    )
                    for r in results
                ]
                return SearchMemoryResponse(memories=memories)

        return _ActroneADKMemoryService()


def _adk_content_text(content: Any) -> str:
    """Flatten an ADK ``types.Content`` (``parts=[Part(text=…)]``) into plain text. Pure."""
    if content is None:
        return ""
    parts = getattr(content, "parts", None)
    if not parts:
        return str(getattr(content, "text", "") or "")
    texts = [str(getattr(p, "text", "") or "") for p in parts]
    return " ".join(t for t in texts if t).strip()
