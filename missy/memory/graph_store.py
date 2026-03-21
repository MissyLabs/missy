"""SQLite-backed entity-relationship graph memory for Missy.

Stores entities and their relationships extracted from conversation turns using
rule-based pattern matching (no NLP dependencies required).  Uses the same
``memory.db`` database file as :class:`~missy.memory.sqlite_store.SQLiteMemoryStore`.

Example::

    from missy.memory.graph_store import GraphMemoryStore

    store = GraphMemoryStore()
    entities, rels = store.ingest_turn("I used file_read on ~/.missy/config.yaml", "user", "sess-1")
    ctx = store.get_context_subgraph("config file")
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = frozenset(
    {"person", "tool", "file", "project", "concept", "location", "organization"}
)

VALID_RELATION_TYPES = frozenset(
    {"uses", "creates", "modifies", "depends_on", "related_to", "owns", "triggers"}
)


@dataclass
class Entity:
    """A named entity tracked in the graph.

    Attributes:
        id: Unique identifier prefixed with ``ent_``.
        name: Normalised (lowercase) entity name.
        entity_type: One of ``person``, ``tool``, ``file``, ``project``,
            ``concept``, ``location``, ``organization``.
        properties: Arbitrary metadata stored as a JSON-serialisable dict.
        first_seen: ISO-8601 UTC timestamp of first observation.
        last_seen: ISO-8601 UTC timestamp of most recent observation.
        mention_count: Number of times the entity has been observed.
    """

    id: str
    name: str
    entity_type: str
    properties: dict = field(default_factory=dict)
    first_seen: str = ""
    last_seen: str = ""
    mention_count: int = 1

    @classmethod
    def new(
        cls,
        name: str,
        entity_type: str,
        properties: dict | None = None,
    ) -> Entity:
        """Construct a new entity with a generated id and current UTC timestamp."""
        now = datetime.now(UTC).isoformat()
        return cls(
            id=f"ent_{uuid.uuid4().hex[:16]}",
            name=name.lower().strip(),
            entity_type=entity_type,
            properties=properties or {},
            first_seen=now,
            last_seen=now,
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "mention_count": self.mention_count,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Entity:
        """Deserialise from a SQLite row."""
        return cls(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            properties=json.loads(row["properties"] or "{}"),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            mention_count=row["mention_count"],
        )


@dataclass
class Relationship:
    """A directed edge between two :class:`Entity` objects.

    Attributes:
        id: Unique identifier prefixed with ``rel_``.
        source_id: :attr:`Entity.id` of the source node.
        target_id: :attr:`Entity.id` of the target node.
        relation_type: One of ``uses``, ``creates``, ``modifies``,
            ``depends_on``, ``related_to``, ``owns``, ``triggers``.
        weight: Strength of the relationship in ``[0.0, 1.0]``.  Increases
            with co-occurrence frequency.
        context: The sentence where the relationship was observed.
        created_at: ISO-8601 UTC timestamp of first observation.
        last_seen: ISO-8601 UTC timestamp of most recent observation.
    """

    id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 0.5
    context: str = ""
    created_at: str = ""
    last_seen: str = ""

    @classmethod
    def new(
        cls,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 0.5,
        context: str = "",
    ) -> Relationship:
        """Construct a new relationship with a generated id and current timestamp."""
        now = datetime.now(UTC).isoformat()
        return cls(
            id=f"rel_{uuid.uuid4().hex[:16]}",
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=min(1.0, max(0.0, weight)),
            context=context,
            created_at=now,
            last_seen=now,
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "context": self.context,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Relationship:
        """Deserialise from a SQLite row."""
        return cls(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            weight=row["weight"],
            context=row["context"] or "",
            created_at=row["created_at"],
            last_seen=row["last_seen"],
        )


@dataclass
class GraphQuery:
    """Result of a graph traversal.

    Attributes:
        entities: Entities visited during the traversal.
        relationships: Relationships traversed.
        paths: Each path is a list of :attr:`Entity.id` strings from a
            seed entity to the entity at the end of the path.
    """

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity extractor
# ---------------------------------------------------------------------------

# Known built-in tool names used by Missy.
_BUILTIN_TOOLS = frozenset(
    {
        "shell_exec",
        "file_read",
        "file_write",
        "web_fetch",
        "vision_capture",
        "vision_burst",
        "vision_analyze",
        "vision_devices",
        "vision_scene",
    }
)

# Relationship verb patterns: (regex, relation_type)
_VERB_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmodif(?:y|ies|ied)\b", re.IGNORECASE), "modifies"),
    (re.compile(r"\bcreate[sd]?\b", re.IGNORECASE), "creates"),
    (re.compile(r"\buse[sd]?\b|\busing\b|\butilize[sd]?\b", re.IGNORECASE), "uses"),
    (re.compile(r"\bdepends?\s+on\b", re.IGNORECASE), "depends_on"),
    (re.compile(r"\bown[sd]?\b", re.IGNORECASE), "owns"),
    (re.compile(r"\btrigger[sd]?\b|\binvoke[sd]?\b", re.IGNORECASE), "triggers"),
]

_PROXIMITY_CHARS = 50  # characters between two entities to infer co-occurrence


class EntityExtractor:
    """Extract entities and relationships from plain text using pattern matching.

    No NLP or machine-learning libraries are required.  Extraction is purely
    rule-based: compiled regular expressions for tools, file paths, URLs, and
    simple heuristics for projects and persons.

    Example::

        extractor = EntityExtractor()
        entities = extractor.extract_entities("I ran file_read on ~/.missy/config.yaml")
        # [Entity(name="file_read", entity_type="tool"), Entity(name="~/.missy/config.yaml", entity_type="file")]
    """

    # -- compiled patterns ---------------------------------------------------

    _TOOL_RE = re.compile(r"\b(shell_exec|file_read|file_write|web_fetch|vision_\w+)\b")
    _FILE_RE = re.compile(r"(?:^|(?<=\s))(?:~\/[\w./\-]+|\/[\w./\-]{2,}(?:\.[\w]{1,10})?)")
    _URL_RE = re.compile(r"https?://[\w./\-?&=%#@+]+")
    # Person heuristic: two or more capitalised words not at sentence start
    _PERSON_RE = re.compile(r"(?<!\.\s)(?<!\n)(?<!\A)\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b")
    # Project heuristic: word immediately following "project", "repo", "repository"
    _PROJECT_RE = re.compile(
        r"(?:project|repo(?:sitory)?)\s+[\"']?([\w\-\.]+)[\"']?",
        re.IGNORECASE,
    )

    def extract_entities(self, text: str) -> list[Entity]:
        """Return a deduplicated list of entities found in *text*.

        Args:
            text: Input text to scan.

        Returns:
            A list of :class:`Entity` objects.  Each entity appears at most
            once (deduplication is by ``(name, entity_type)``).
        """
        seen: dict[tuple[str, str], Entity] = {}

        def _add(name: str, entity_type: str, properties: dict | None = None) -> None:
            key = (name.lower().strip(), entity_type)
            if key not in seen:
                seen[key] = Entity.new(name, entity_type, properties)

        # Tools
        for m in self._TOOL_RE.finditer(text):
            _add(m.group(1), "tool")

        # File paths
        for m in self._FILE_RE.finditer(text):
            path = m.group(0).strip()
            if len(path) >= 3:
                _add(path, "file")

        # URLs → treat as concepts (external resources)
        for m in self._URL_RE.finditer(text):
            _add(m.group(0), "concept", {"url": m.group(0)})

        # Projects
        for m in self._PROJECT_RE.finditer(text):
            _add(m.group(1), "project")

        # Persons (heuristic — avoid false positives on sentence beginnings)
        for m in self._PERSON_RE.finditer(text):
            candidate = m.group(1)
            # Filter out common non-person capitalised phrases
            if candidate.lower() not in {
                "true",
                "false",
                "none",
                "null",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            }:
                _add(candidate, "person")

        return list(seen.values())

    def _detect_relation_type(self, sentence: str) -> str:
        """Return the most specific relation type found in *sentence*."""
        for pattern, rel_type in _VERB_PATTERNS:
            if pattern.search(sentence):
                return rel_type
        return "related_to"

    def extract_relationships(
        self,
        text: str,
        entities: list[Entity],
    ) -> list[Relationship]:
        """Return relationships inferred between *entities* in *text*.

        Two strategies are applied in order:

        1. **Verb-based**: if a sentence contains two entities and a known
           action verb between them, a typed relationship is created.
        2. **Proximity-based**: if two entities appear within
           ``_PROXIMITY_CHARS`` characters of each other in the same sentence,
           a ``related_to`` relationship is added as a fallback.

        Args:
            text: Source text (used for proximity and verb detection).
            entities: Entities previously extracted from *text*.

        Returns:
            A deduplicated list of :class:`Relationship` objects.
        """
        if len(entities) < 2:
            return []

        rels: dict[tuple[str, str, str], Relationship] = {}

        def _add_rel(src: Entity, tgt: Entity, rel_type: str, ctx: str) -> None:
            key = (src.id, tgt.id, rel_type)
            if key not in rels:
                rels[key] = Relationship.new(
                    source_id=src.id,
                    target_id=tgt.id,
                    relation_type=rel_type,
                    context=ctx[:200],
                )

        # Split into sentences for verb-based detection
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Find which entities appear in this sentence
            present: list[Entity] = [e for e in entities if e.name in sentence_lower]
            if len(present) < 2:
                continue

            rel_type = self._detect_relation_type(sentence)

            # For tool+file combinations, prefer specific verb types
            for i, src in enumerate(present):
                for tgt in present[i + 1 :]:
                    # Tool acting on file → modifies/creates/uses
                    if src.entity_type == "tool" and tgt.entity_type == "file":
                        inferred = rel_type if rel_type != "related_to" else "modifies"
                        _add_rel(src, tgt, inferred, sentence)
                    elif tgt.entity_type == "tool" and src.entity_type == "file":
                        inferred = rel_type if rel_type != "related_to" else "modifies"
                        _add_rel(tgt, src, inferred, sentence)
                    else:
                        _add_rel(src, tgt, rel_type, sentence)

        # Proximity-based: scan all entity-pair positions in the full text
        text_lower = text.lower()
        entity_positions: list[tuple[int, Entity]] = []
        for entity in entities:
            start = 0
            while True:
                pos = text_lower.find(entity.name, start)
                if pos == -1:
                    break
                entity_positions.append((pos, entity))
                start = pos + 1

        entity_positions.sort(key=lambda x: x[0])

        for i, (pos_a, ent_a) in enumerate(entity_positions):
            for pos_b, ent_b in entity_positions[i + 1 :]:
                if ent_a.id == ent_b.id:
                    continue
                gap = pos_b - (pos_a + len(ent_a.name))
                if gap > _PROXIMITY_CHARS:
                    break
                key = (ent_a.id, ent_b.id, "related_to")
                if key not in rels:
                    ctx = text[max(0, pos_a - 10) : pos_b + len(ent_b.name) + 10]
                    rels[key] = Relationship.new(
                        source_id=ent_a.id,
                        target_id=ent_b.id,
                        relation_type="related_to",
                        weight=0.3,
                        context=ctx[:200],
                    )

        return list(rels.values())

    def extract_from_turn(
        self,
        turn_content: str,
        role: str,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from a single conversation turn.

        Args:
            turn_content: Text content of the turn.
            role: Speaker role (``"user"``, ``"assistant"``, or ``"tool"``).

        Returns:
            A two-tuple of ``(entities, relationships)``.
        """
        entities = self.extract_entities(turn_content)
        relationships = self.extract_relationships(turn_content, entities)
        return entities, relationships


# ---------------------------------------------------------------------------
# GraphMemoryStore
# ---------------------------------------------------------------------------


class GraphMemoryStore:
    """SQLite-backed entity-relationship graph memory.

    Uses the **same** ``memory.db`` file as
    :class:`~missy.memory.sqlite_store.SQLiteMemoryStore` — just opens
    additional tables (``entities``, ``relationships``) in that database.
    Thread-safety is provided by the same thread-local connection pattern.

    Args:
        db_path: Path to the SQLite database.  Tilde-expansion is performed
            automatically.  Defaults to ``~/.missy/memory.db``.

    Example::

        store = GraphMemoryStore()
        entities, rels = store.ingest_turn(
            "I used file_write to create ~/workspace/notes.txt",
            role="user",
            session_id="sess-1",
        )
        print(store.get_entity_summary("file_write"))
    """

    def __init__(self, db_path: str = "~/.missy/memory.db") -> None:
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._local = threading.local()
        self._extractor = EntityExtractor()
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection, creating it if needed."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    def _ensure_tables(self) -> None:
        """Create graph tables and indexes if they do not already exist."""
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id             TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                entity_type    TEXT NOT NULL,
                properties     TEXT NOT NULL DEFAULT '{}',
                first_seen     TEXT NOT NULL,
                last_seen      TEXT NOT NULL,
                mention_count  INTEGER NOT NULL DEFAULT 1,
                UNIQUE(name, entity_type)
            );

            CREATE INDEX IF NOT EXISTS idx_entities_name
                ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type
                ON entities(entity_type);

            CREATE TABLE IF NOT EXISTS relationships (
                id            TEXT PRIMARY KEY,
                source_id     TEXT NOT NULL REFERENCES entities(id),
                target_id     TEXT NOT NULL REFERENCES entities(id),
                relation_type TEXT NOT NULL,
                weight        REAL NOT NULL DEFAULT 0.5,
                context       TEXT NOT NULL DEFAULT '',
                created_at    TEXT NOT NULL,
                last_seen     TEXT NOT NULL,
                UNIQUE(source_id, target_id, relation_type)
            );

            CREATE INDEX IF NOT EXISTS idx_rel_source
                ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target
                ON relationships(target_id);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> str:
        """Upsert an entity by ``(name, entity_type)``.

        If an entity with the same normalised name and type already exists,
        its ``last_seen`` timestamp and ``mention_count`` are updated and
        the existing ``id`` is returned.

        Args:
            entity: The entity to insert or update.

        Returns:
            The ``id`` of the stored entity (may differ from ``entity.id``
            when an existing row was found).
        """
        conn = self._conn()
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """INSERT INTO entities
               (id, name, entity_type, properties, first_seen, last_seen, mention_count)
               VALUES (?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT(name, entity_type) DO UPDATE SET
                   last_seen     = excluded.last_seen,
                   mention_count = mention_count + 1,
                   properties    = CASE
                       WHEN excluded.properties != '{}'
                       THEN excluded.properties
                       ELSE entities.properties
                   END""",
            (
                entity.id,
                entity.name,
                entity.entity_type,
                json.dumps(entity.properties),
                entity.first_seen,
                now,
            ),
        )
        conn.commit()
        # Return the canonical id (may be the existing one on conflict)
        row = conn.execute(
            "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
            (entity.name, entity.entity_type),
        ).fetchone()
        return row["id"] if row else entity.id

    def add_relationship(self, rel: Relationship) -> str:
        """Upsert a relationship by ``(source_id, target_id, relation_type)``.

        When a matching relationship already exists, ``last_seen`` is updated
        and ``weight`` is nudged upward (capped at 1.0) to reflect increased
        co-occurrence confidence.

        Args:
            rel: The relationship to insert or update.

        Returns:
            The ``id`` of the stored relationship.
        """
        conn = self._conn()
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """INSERT INTO relationships
               (id, source_id, target_id, relation_type, weight, context,
                created_at, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                   last_seen = excluded.last_seen,
                   weight    = MIN(1.0, weight + 0.05),
                   context   = CASE
                       WHEN excluded.context != ''
                       THEN excluded.context
                       ELSE relationships.context
                   END""",
            (
                rel.id,
                rel.source_id,
                rel.target_id,
                rel.relation_type,
                rel.weight,
                rel.context,
                rel.created_at,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            """SELECT id FROM relationships
               WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
            (rel.source_id, rel.target_id, rel.relation_type),
        ).fetchone()
        return row["id"] if row else rel.id

    def ingest_turn(
        self,
        turn_content: str,
        role: str,
        session_id: str,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from a turn and persist them.

        Entities are upserted so that repeated mentions increment
        ``mention_count``.  Relationships are upserted with weight bumping.

        Args:
            turn_content: The text content of the turn.
            role: Speaker role — ``"user"``, ``"assistant"``, or ``"tool"``.
            session_id: Session identifier (stored in entity properties for
                traceability).

        Returns:
            A two-tuple of ``(entities, relationships)`` that were found.
            The objects carry the canonical ``id`` values after upsert.
        """
        raw_entities, raw_rels = self._extractor.extract_from_turn(turn_content, role)
        if not raw_entities:
            return [], []

        # Upsert entities and remap ids
        old_to_new: dict[str, str] = {}
        final_entities: list[Entity] = []
        for ent in raw_entities:
            canonical_id = self.add_entity(ent)
            old_to_new[ent.id] = canonical_id
            # Re-fetch to get the persisted state
            persisted = self.get_entity(canonical_id)
            if persisted:
                final_entities.append(persisted)

        # Remap relationship ids and upsert
        final_rels: list[Relationship] = []
        for rel in raw_rels:
            new_src = old_to_new.get(rel.source_id)
            new_tgt = old_to_new.get(rel.target_id)
            if not new_src or not new_tgt:
                continue
            remapped = Relationship.new(
                source_id=new_src,
                target_id=new_tgt,
                relation_type=rel.relation_type,
                weight=rel.weight,
                context=rel.context,
            )
            canonical_rel_id = self.add_relationship(remapped)
            persisted_rel = self._get_relationship_by_id(canonical_rel_id)
            if persisted_rel:
                final_rels.append(persisted_rel)

        return final_entities, final_rels

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Entity | None:
        """Return a single entity by id, or ``None`` if not found.

        Args:
            entity_id: The ``Entity.id`` to look up.
        """
        conn = self._conn()
        row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return Entity.from_row(row) if row else None

    def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        """Search entities by name substring and/or type.

        Args:
            name: Optional substring to match against ``Entity.name``
                (case-insensitive LIKE search).
            entity_type: Optional exact entity type filter.
            limit: Maximum number of results.

        Returns:
            Entities ordered by ``mention_count`` descending.
        """
        conn = self._conn()
        clauses: list[str] = []
        params: list[object] = []

        if name:
            clauses.append("name LIKE ?")
            params.append(f"%{name.lower()}%")
        if entity_type:
            clauses.append("entity_type = ?")
            params.append(entity_type)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM entities {where} ORDER BY mention_count DESC LIMIT ?",
            params,
        ).fetchall()
        return [Entity.from_row(r) for r in rows]

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> list[Relationship]:
        """Return relationships for an entity.

        Args:
            entity_id: The entity whose edges to retrieve.
            direction: ``"outbound"`` (entity is source), ``"inbound"``
                (entity is target), or ``"both"`` (default).

        Returns:
            Relationships ordered by ``weight`` descending.
        """
        conn = self._conn()
        if direction == "outbound":
            where = "WHERE source_id = ?"
            params: list[object] = [entity_id]
        elif direction == "inbound":
            where = "WHERE target_id = ?"
            params = [entity_id]
        else:
            where = "WHERE source_id = ? OR target_id = ?"
            params = [entity_id, entity_id]

        rows = conn.execute(
            f"SELECT * FROM relationships {where} ORDER BY weight DESC",
            params,
        ).fetchall()
        return [Relationship.from_row(r) for r in rows]

    def get_neighbors(
        self,
        entity_id: str,
        max_depth: int = 2,
    ) -> GraphQuery:
        """BFS traversal from *entity_id* up to *max_depth* hops.

        Args:
            entity_id: Starting entity.
            max_depth: Maximum traversal depth (1 = direct neighbours only).

        Returns:
            A :class:`GraphQuery` containing all reachable entities,
            traversed relationships, and the shortest path (as entity-id
            lists) from the seed to each visited entity.
        """
        visited_entities: dict[str, Entity] = {}
        visited_rels: dict[str, Relationship] = {}
        paths: dict[str, list[str]] = {entity_id: [entity_id]}

        seed = self.get_entity(entity_id)
        if not seed:
            return GraphQuery()

        visited_entities[entity_id] = seed
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for rel in self.get_relationships(current_id, direction="both"):
                visited_rels[rel.id] = rel
                neighbour_id = rel.target_id if rel.source_id == current_id else rel.source_id
                if neighbour_id not in visited_entities:
                    neighbour = self.get_entity(neighbour_id)
                    if neighbour:
                        visited_entities[neighbour_id] = neighbour
                        paths[neighbour_id] = paths[current_id] + [neighbour_id]
                        queue.append((neighbour_id, depth + 1))

        return GraphQuery(
            entities=list(visited_entities.values()),
            relationships=list(visited_rels.values()),
            paths=list(paths.values()),
        )

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def find_related(self, query: str, limit: int = 10) -> GraphQuery:
        """Find entities related to *query* text by name-matching then BFS.

        Steps:

        1. Extract candidate entities from *query* using
           :class:`EntityExtractor`.
        2. For each candidate, search the store for matching stored entities.
        3. BFS-expand each matched entity up to depth 1.
        4. Deduplicate and return up to *limit* entities.

        Args:
            query: Free-form text to drive the lookup.
            limit: Maximum number of entities in the result.

        Returns:
            A :class:`GraphQuery` with the most relevant context.
        """
        query_entities = self._extractor.extract_entities(query)
        seed_ids: list[str] = []

        # Direct name lookups for extracted entities
        for qe in query_entities:
            for stored in self.find_entities(name=qe.name, limit=5):
                if stored.id not in seed_ids:
                    seed_ids.append(stored.id)

        # Also do a simple keyword scan across entity names
        words = [w for w in query.lower().split() if len(w) > 3]
        for word in words:
            for stored in self.find_entities(name=word, limit=3):
                if stored.id not in seed_ids:
                    seed_ids.append(stored.id)

        if not seed_ids:
            return GraphQuery()

        all_entities: dict[str, Entity] = {}
        all_rels: dict[str, Relationship] = {}
        all_paths: list[list[str]] = []

        for seed_id in seed_ids[:5]:  # cap seed expansion
            sub = self.get_neighbors(seed_id, max_depth=1)
            for e in sub.entities:
                all_entities[e.id] = e
            for r in sub.relationships:
                all_rels[r.id] = r
            all_paths.extend(sub.paths)

        # Sort entities by mention_count descending, truncate to limit
        sorted_entities = sorted(
            all_entities.values(),
            key=lambda e: e.mention_count,
            reverse=True,
        )[:limit]
        kept_ids = {e.id for e in sorted_entities}
        filtered_rels = [
            r for r in all_rels.values() if r.source_id in kept_ids and r.target_id in kept_ids
        ]

        return GraphQuery(
            entities=sorted_entities,
            relationships=filtered_rels,
            paths=[p for p in all_paths if all(n in kept_ids for n in p)],
        )

    def get_context_subgraph(self, query: str, max_entities: int = 15) -> str:
        """Return a formatted context block of relevant entities and edges.

        Suitable for injection into a system prompt.  Format::

            Entity Graph:
            - file_read (tool) --[uses]--> ~/.missy/config.yaml (file)
            - missy (project) --[related_to]--> file_read (tool)

        Args:
            query: Text used to select relevant entities.
            max_entities: Maximum number of entity relationships to include.

        Returns:
            A multi-line string, or an empty string when no entities match.
        """
        result = self.find_related(query, limit=max_entities)
        if not result.entities:
            return ""

        # Build a lookup for entity display
        eid_to_entity = {e.id: e for e in result.entities}

        lines: list[str] = ["Entity Graph:"]
        seen_lines: set[str] = set()

        for rel in sorted(result.relationships, key=lambda r: r.weight, reverse=True):
            src = eid_to_entity.get(rel.source_id)
            tgt = eid_to_entity.get(rel.target_id)
            if not src or not tgt:
                continue
            line = (
                f"- {src.name} ({src.entity_type})"
                f" --[{rel.relation_type}]--> "
                f"{tgt.name} ({tgt.entity_type})"
            )
            if line not in seen_lines:
                lines.append(line)
                seen_lines.add(line)

        # Include any entities with no relationships
        rel_entity_ids = {r.source_id for r in result.relationships} | {
            r.target_id for r in result.relationships
        }
        for entity in result.entities:
            if entity.id not in rel_entity_ids:
                line = f"- {entity.name} ({entity.entity_type})"
                if line not in seen_lines:
                    lines.append(line)
                    seen_lines.add(line)

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def get_entity_summary(self, entity_name: str) -> str:
        """Return a human-readable summary of an entity and its relationships.

        Args:
            entity_name: Name to look up (case-insensitive).

        Returns:
            A formatted string, or an empty string when the entity is unknown.
        """
        matches = self.find_entities(name=entity_name, limit=1)
        if not matches:
            return ""

        entity = matches[0]
        rels = self.get_relationships(entity.id, direction="both")

        lines = [
            f"{entity.name} ({entity.entity_type})",
            f"  Seen {entity.mention_count} time(s), last: {entity.last_seen[:10]}",
        ]

        # Build entity-id to name mapping for readable output
        neighbour_ids = {r.target_id if r.source_id == entity.id else r.source_id for r in rels}
        neighbours: dict[str, Entity] = {}
        for nid in neighbour_ids:
            n = self.get_entity(nid)
            if n:
                neighbours[nid] = n

        if rels:
            lines.append("  Relationships:")
            for rel in sorted(rels, key=lambda r: r.weight, reverse=True):
                if rel.source_id == entity.id:
                    other = neighbours.get(rel.target_id)
                    if other:
                        lines.append(
                            f"    --[{rel.relation_type}]--> {other.name} ({other.entity_type})"
                        )
                else:
                    other = neighbours.get(rel.source_id)
                    if other:
                        lines.append(
                            f"    <--[{rel.relation_type}]-- {other.name} ({other.entity_type})"
                        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune(
        self,
        min_mentions: int = 1,
        older_than_days: int = 90,
    ) -> int:
        """Delete stale entities and their relationships.

        An entity is removed when it has fewer than *min_mentions* mentions
        **and** has not been seen within *older_than_days* days.  Orphaned
        relationships are removed automatically via ON DELETE CASCADE (or
        explicitly when FK enforcement is not active).

        Args:
            min_mentions: Minimum mention count below which old entities
                are eligible for deletion.
            older_than_days: Age threshold in days for last_seen.

        Returns:
            Number of entities deleted.
        """
        conn = self._conn()
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        # Find stale entity ids first so we can clean up relationships too
        stale_rows = conn.execute(
            """SELECT id FROM entities
               WHERE mention_count < ? AND last_seen < ?""",
            (min_mentions + 1, cutoff),
        ).fetchall()
        stale_ids = [r["id"] for r in stale_rows]
        if not stale_ids:
            return 0

        placeholders = ",".join("?" for _ in stale_ids)
        # Remove relationships referencing stale entities
        conn.execute(
            f"DELETE FROM relationships WHERE source_id IN ({placeholders})"
            f" OR target_id IN ({placeholders})",
            stale_ids + stale_ids,
        )
        cur = conn.execute(f"DELETE FROM entities WHERE id IN ({placeholders})", stale_ids)
        conn.commit()
        return cur.rowcount

    def merge_entities(self, keep_id: str, merge_id: str) -> None:
        """Merge *merge_id* into *keep_id*, re-pointing all relationships.

        After merging, ``keep_id``'s ``mention_count`` is incremented by the
        count from the merged entity, and *merge_id* is deleted.

        Args:
            keep_id: The entity to preserve.
            merge_id: The entity to absorb and delete.
        """
        if keep_id == merge_id:
            return
        conn = self._conn()

        keep = self.get_entity(keep_id)
        going = self.get_entity(merge_id)
        if not keep or not going:
            return

        # Reassign relationships
        conn.execute(
            "UPDATE relationships SET source_id = ? WHERE source_id = ?",
            (keep_id, merge_id),
        )
        conn.execute(
            "UPDATE relationships SET target_id = ? WHERE target_id = ?",
            (keep_id, merge_id),
        )

        # Update mention count on the keeper
        combined = keep.mention_count + going.mention_count
        conn.execute(
            "UPDATE entities SET mention_count = ? WHERE id = ?",
            (combined, keep_id),
        )

        # Remove the merged entity
        conn.execute("DELETE FROM entities WHERE id = ?", (merge_id,))

        # Remove any self-loops created by the merge
        conn.execute(
            "DELETE FROM relationships WHERE source_id = ? AND target_id = ?",
            (keep_id, keep_id),
        )
        conn.commit()

    def stats(self) -> dict:
        """Return aggregate statistics about the graph.

        Returns:
            A dict with keys:

            * ``entity_count`` — total number of entities
            * ``relationship_count`` — total number of relationships
            * ``entity_types`` — dict mapping type name to count
            * ``relation_types`` — dict mapping relation type to count
        """
        conn = self._conn()
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        type_rows = conn.execute(
            "SELECT entity_type, COUNT(*) as n FROM entities GROUP BY entity_type"
        ).fetchall()
        rel_type_rows = conn.execute(
            "SELECT relation_type, COUNT(*) as n FROM relationships GROUP BY relation_type"
        ).fetchall()
        return {
            "entity_count": entity_count,
            "relationship_count": rel_count,
            "entity_types": {r["entity_type"]: r["n"] for r in type_rows},
            "relation_types": {r["relation_type"]: r["n"] for r in rel_type_rows},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_relationship_by_id(self, rel_id: str) -> Relationship | None:
        """Return a relationship by id, or None."""
        conn = self._conn()
        row = conn.execute("SELECT * FROM relationships WHERE id = ?", (rel_id,)).fetchone()
        return Relationship.from_row(row) if row else None
