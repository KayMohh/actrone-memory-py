"""actrone-memory, Two-tier persistent memory for AI agents.

Local-first by default: ``MemoryManager.create()`` runs with zero external
services and no API key (in-process store, plus the best local embedder
available: in-process ONNX if the ``onnx`` extra is installed, then
sentence-transformers, then a dependency-free lexical hashing fallback).
Set ``backend="redis_qdrant"`` and an embedding provider for the durable path.
"""

from actrone_memory.config import MemoryConfig
from actrone_memory.exceptions import (
    ActroneMemoryError,
    ConfigurationError,
    EmbeddingError,
    MemoryNotFoundError,
    StoreConnectionError,
    TokenBudgetError,
    ValidationError,
)
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
from actrone_memory.in_memory import InMemoryStore, cosine_similarity
from actrone_memory.l2.embedder import (
    Embedder,
    HashingEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
)
from actrone_memory.manager import MemoryManager, create_memory_manager
from actrone_memory.models import (
    ContentType,
    MemoryEntry,
    MemorySource,
    RetrievedContext,
    Sensitivity,
    SessionMetadata,
    ToolResult,
    Turn,
)
from actrone_memory.protocols import L1Store, L2Store
from actrone_memory.redaction import SENSITIVITIES, redacted_content

__version__ = "0.2.1"
__all__ = [
    "MemoryManager",
    "MemoryConfig",
    "MemoryEntry",
    "ContentType",
    "MemorySource",
    "Sensitivity",
    "SENSITIVITIES",
    "redacted_content",
    "RetrievedContext",
    "SessionMetadata",
    "ToolResult",
    "Turn",
    "create_memory_manager",
    "InMemoryStore",
    "cosine_similarity",
    "ExtractedFact",
    "FactExtractor",
    "OpenAIFactExtractor",
    "parse_facts",
    "format_extraction_input",
    "EXTRACTION_SPEC_VERSION",
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_RESPONSE_SCHEMA",
    "Embedder",
    "HashingEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "L1Store",
    "L2Store",
    "ActroneMemoryError",
    "ConfigurationError",
    "EmbeddingError",
    "MemoryNotFoundError",
    "StoreConnectionError",
    "TokenBudgetError",
    "ValidationError",
]
