"""Conversation memory subsystem for Missy.

Public API
----------
- :class:`MemoryStore` — JSON-backed store (default, zero-dependency).
- :class:`SQLiteMemoryStore` — SQLite-backed store with FTS5 search.
- :class:`ResilientMemoryStore` — Wraps any store with in-memory fallback.
- :class:`ConversationTurn` — Single turn dataclass (JSON store variant).
- :class:`MemorySynthesizer` — Unified memory synthesis across subsystems.
"""

from missy.memory.resilient import ResilientMemoryStore
from missy.memory.sqlite_store import ConversationTurn as SQLiteConversationTurn
from missy.memory.sqlite_store import LargeContentRecord, SQLiteMemoryStore, SummaryRecord
from missy.memory.store import ConversationTurn, MemoryStore
from missy.memory.synthesizer import MemoryFragment, MemorySynthesizer

__all__ = [
    "ConversationTurn",
    "LargeContentRecord",
    "MemoryFragment",
    "MemoryStore",
    "MemorySynthesizer",
    "ResilientMemoryStore",
    "SQLiteConversationTurn",
    "SQLiteMemoryStore",
    "SummaryRecord",
]
