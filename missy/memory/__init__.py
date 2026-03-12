"""Conversation memory subsystem for Missy.

Public API
----------
- :class:`MemoryStore` — JSON-backed store (default, zero-dependency).
- :class:`SQLiteMemoryStore` — SQLite-backed store with FTS5 search.
- :class:`ResilientMemoryStore` — Wraps any store with in-memory fallback.
- :class:`ConversationTurn` — Single turn dataclass (JSON store variant).
"""

from missy.memory.resilient import ResilientMemoryStore
from missy.memory.sqlite_store import ConversationTurn as SQLiteConversationTurn
from missy.memory.sqlite_store import SQLiteMemoryStore
from missy.memory.store import ConversationTurn, MemoryStore

__all__ = [
    "ConversationTurn",
    "MemoryStore",
    "ResilientMemoryStore",
    "SQLiteConversationTurn",
    "SQLiteMemoryStore",
]
