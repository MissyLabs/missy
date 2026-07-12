"""Conversation memory subsystem for Missy.

Public API
----------
- :class:`SQLiteMemoryStore` — SQLite-backed store with FTS5 search. The
  production memory backend; all built-in call sites use this.
- :class:`MemoryStore` — legacy JSON-backed store, zero-dependency but
  not used by any production code path as of SR-3.1/3.5 (non-atomic
  full-file rewrites on every write). Retained for embedders who want a
  dependency-free store; new code should use ``SQLiteMemoryStore``.
- :class:`ResilientMemoryStore` — Wraps any store with in-memory fallback.
- :class:`ConversationTurn` — Single turn dataclass (JSON store variant).
- :class:`MemorySynthesizer` — Unified memory synthesis across subsystems.
- :class:`GraphMemoryStore` — SQLite-backed entity-relationship graph.
- :class:`Entity` — Graph entity dataclass.
- :class:`Relationship` — Graph relationship dataclass.
- :class:`GraphQuery` — Graph traversal result dataclass.
- :class:`EntityExtractor` — Rule-based entity/relationship extractor.
"""

from missy.memory.graph_store import (
    Entity,
    EntityExtractor,
    GraphMemoryStore,
    GraphQuery,
    Relationship,
)
from missy.memory.resilient import ResilientMemoryStore
from missy.memory.sqlite_store import ConversationTurn as SQLiteConversationTurn
from missy.memory.sqlite_store import LargeContentRecord, SQLiteMemoryStore, SummaryRecord
from missy.memory.store import ConversationTurn, MemoryStore
from missy.memory.synthesizer import MemoryFragment, MemorySynthesizer

__all__ = [
    "ConversationTurn",
    "Entity",
    "EntityExtractor",
    "GraphMemoryStore",
    "GraphQuery",
    "LargeContentRecord",
    "MemoryFragment",
    "MemoryStore",
    "MemorySynthesizer",
    "Relationship",
    "ResilientMemoryStore",
    "SQLiteConversationTurn",
    "SQLiteMemoryStore",
    "SummaryRecord",
]
