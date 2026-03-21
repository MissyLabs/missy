"""Conversation memory subsystem for Missy.

Public API
----------
- :class:`MemoryStore` — JSON-backed store (default, zero-dependency).
- :class:`SQLiteMemoryStore` — SQLite-backed store with FTS5 search.
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
