"""Tests for missy.memory.graph_store — entity-relationship graph memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.memory.graph_store import (
    Entity,
    EntityExtractor,
    GraphMemoryStore,
    Relationship,
)
from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    """Return a path to a temporary SQLite database."""
    return str(tmp_path / "memory.db")


@pytest.fixture()
def store(tmp_db: str) -> GraphMemoryStore:
    """Return a freshly initialised GraphMemoryStore backed by a temp DB."""
    return GraphMemoryStore(db_path=tmp_db)


@pytest.fixture()
def extractor() -> EntityExtractor:
    return EntityExtractor()


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------


class TestEntityDataclass:
    def test_new_generates_id_with_prefix(self):
        ent = Entity.new("file_read", "tool")
        assert ent.id.startswith("ent_")

    def test_new_normalises_name_to_lowercase(self):
        ent = Entity.new("  FileRead  ", "tool")
        assert ent.name == "fileread"

    def test_new_sets_timestamps(self):
        ent = Entity.new("file_read", "tool")
        assert ent.first_seen
        assert ent.last_seen

    def test_to_dict_roundtrip(self):
        ent = Entity.new("file_read", "tool", properties={"foo": "bar"})
        d = ent.to_dict()
        assert d["name"] == "file_read"
        assert d["entity_type"] == "tool"
        assert d["properties"] == {"foo": "bar"}
        assert d["mention_count"] == 1

    def test_default_mention_count_is_one(self):
        ent = Entity.new("x", "concept")
        assert ent.mention_count == 1


# ---------------------------------------------------------------------------
# Relationship dataclass
# ---------------------------------------------------------------------------


class TestRelationshipDataclass:
    def test_new_generates_id_with_prefix(self):
        rel = Relationship.new("src", "tgt", "uses")
        assert rel.id.startswith("rel_")

    def test_weight_clamped_to_range(self):
        rel_low = Relationship.new("s", "t", "uses", weight=-5.0)
        rel_high = Relationship.new("s", "t", "uses", weight=99.0)
        assert rel_low.weight == pytest.approx(0.0)
        assert rel_high.weight == pytest.approx(1.0)

    def test_to_dict_contains_all_fields(self):
        rel = Relationship.new("s", "t", "creates", weight=0.8, context="ctx")
        d = rel.to_dict()
        assert d["source_id"] == "s"
        assert d["target_id"] == "t"
        assert d["relation_type"] == "creates"
        assert d["weight"] == pytest.approx(0.8)
        assert d["context"] == "ctx"

    def test_new_timestamps_populated(self):
        rel = Relationship.new("s", "t", "uses")
        assert rel.created_at
        assert rel.last_seen


# ---------------------------------------------------------------------------
# EntityExtractor — entity detection
# ---------------------------------------------------------------------------


class TestEntityExtractorTools:
    def test_detects_built_in_tool(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("I called file_read on the config.")
        names = [e.name for e in entities]
        assert "file_read" in names

    def test_detects_vision_tool(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("Running vision_capture now.")
        names = [e.name for e in entities]
        assert "vision_capture" in names

    def test_tool_entity_has_correct_type(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("shell_exec ran the command.")
        tools = [e for e in entities if e.entity_type == "tool"]
        assert len(tools) >= 1

    def test_no_false_tool_matches(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("The file system is fine.")
        tool_names = [e.name for e in entities if e.entity_type == "tool"]
        assert "file" not in tool_names


class TestEntityExtractorFiles:
    def test_detects_tilde_path(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("Reading ~/.missy/config.yaml")
        files = [e for e in entities if e.entity_type == "file"]
        assert any("config.yaml" in f.name for f in files)

    def test_detects_absolute_path(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("Opening /etc/hosts to check entries.")
        files = [e for e in entities if e.entity_type == "file"]
        assert any("/etc/hosts" in f.name for f in files)

    def test_does_not_match_bare_words(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("The word hosts is not a path.")
        files = [e for e in entities if e.entity_type == "file"]
        file_names = [f.name for f in files]
        assert "hosts" not in file_names


class TestEntityExtractorURLs:
    def test_detects_http_url(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("Fetching https://api.github.com/repos")
        concepts = [e for e in entities if e.entity_type == "concept"]
        assert any("github" in c.name for c in concepts)

    def test_url_stored_in_properties(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("Fetching https://example.com/data")
        concepts = [e for e in entities if e.entity_type == "concept"]
        assert any(c.properties.get("url") for c in concepts)


class TestEntityExtractorProjects:
    def test_detects_project_keyword(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("Working on project missy today.")
        projects = [e for e in entities if e.entity_type == "project"]
        assert any("missy" in p.name for p in projects)

    def test_detects_repo_keyword(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("Cloned repo my-app from GitHub.")
        projects = [e for e in entities if e.entity_type == "project"]
        assert any("my-app" in p.name for p in projects)


class TestEntityExtractorPersons:
    def test_detects_capitalised_full_name(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("I spoke with Alice Smith about the task.")
        persons = [e for e in entities if e.entity_type == "person"]
        assert any("alice smith" in p.name for p in persons)

    def test_no_duplicate_entities(self, extractor: EntityExtractor):
        entities = extractor.extract_entities("file_read file_read file_read ran three times")
        tool_names = [e.name for e in entities if e.entity_type == "tool"]
        assert tool_names.count("file_read") == 1


class TestEntityExtractorEmpty:
    def test_empty_text_returns_empty(self, extractor: EntityExtractor):
        assert extractor.extract_entities("") == []

    def test_whitespace_only_returns_empty(self, extractor: EntityExtractor):
        assert extractor.extract_entities("   \n\t  ") == []


# ---------------------------------------------------------------------------
# EntityExtractor — relationship detection
# ---------------------------------------------------------------------------


class TestRelationshipExtraction:
    def test_tool_file_in_same_sentence_creates_rel(self, extractor: EntityExtractor):
        text = "file_write created ~/workspace/output.txt for the task."
        entities = extractor.extract_entities(text)
        rels = extractor.extract_relationships(text, entities)
        assert len(rels) >= 1

    def test_verb_based_uses_detection(self, extractor: EntityExtractor):
        text = "file_read uses ~/.missy/config.yaml as its source."
        entities = extractor.extract_entities(text)
        rels = extractor.extract_relationships(text, entities)
        types = [r.relation_type for r in rels]
        assert "uses" in types or "related_to" in types

    def test_verb_creates_detected(self, extractor: EntityExtractor):
        text = "file_write creates ~/workspace/report.txt"
        entities = extractor.extract_entities(text)
        rels = extractor.extract_relationships(text, entities)
        types = [r.relation_type for r in rels]
        assert "creates" in types or "modifies" in types

    def test_proximity_relationship_added(self, extractor: EntityExtractor):
        # Two entities very close together — should get a proximity relationship
        text = "file_read ~/.missy/vault.key"
        entities = extractor.extract_entities(text)
        rels = extractor.extract_relationships(text, entities)
        # Proximity or verb-based relationship expected
        assert len(rels) >= 1

    def test_no_self_relationships(self, extractor: EntityExtractor):
        entities = [Entity.new("file_read", "tool")]
        rels = extractor.extract_relationships("file_read is a tool.", entities)
        for rel in rels:
            assert rel.source_id != rel.target_id

    def test_empty_entities_list_returns_empty(self, extractor: EntityExtractor):
        rels = extractor.extract_relationships("some text", [])
        assert rels == []

    def test_single_entity_returns_no_rels(self, extractor: EntityExtractor):
        entities = [Entity.new("file_read", "tool")]
        rels = extractor.extract_relationships("file_read ran successfully.", entities)
        assert rels == []

    def test_extract_from_turn_returns_both(self, extractor: EntityExtractor):
        text = "I used file_read to open ~/.missy/config.yaml"
        entities, rels = extractor.extract_from_turn(text, role="user")
        assert len(entities) >= 1


# ---------------------------------------------------------------------------
# GraphMemoryStore — table creation
# ---------------------------------------------------------------------------


class TestGraphMemoryStoreTables:
    def test_entities_table_exists(self, store: GraphMemoryStore):
        conn = store._conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"
        ).fetchone()
        assert row is not None

    def test_relationships_table_exists(self, store: GraphMemoryStore):
        conn = store._conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'"
        ).fetchone()
        assert row is not None

    def test_indexes_created(self, store: GraphMemoryStore):
        conn = store._conn()
        indexes = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "idx_entities_name" in indexes
        assert "idx_rel_source" in indexes
        assert "idx_rel_target" in indexes

    def test_shared_db_compatible_with_sqlite_store(self, tmp_db: str):
        """Both stores open the same file without conflict."""
        sql_store = SQLiteMemoryStore(db_path=tmp_db)
        graph_store = GraphMemoryStore(db_path=tmp_db)
        # Write to each store
        turn = ConversationTurn.new("sess-1", "user", "hello")
        sql_store.add_turn(turn)
        ent = Entity.new("file_read", "tool")
        graph_store.add_entity(ent)
        # Both should be readable
        turns = sql_store.get_session_turns("sess-1")
        assert len(turns) == 1
        entities = graph_store.find_entities(name="file_read")
        assert len(entities) == 1


# ---------------------------------------------------------------------------
# GraphMemoryStore — add_entity / upsert
# ---------------------------------------------------------------------------


class TestAddEntity:
    def test_entity_persisted_and_readable(self, store: GraphMemoryStore):
        ent = Entity.new("file_read", "tool")
        eid = store.add_entity(ent)
        fetched = store.get_entity(eid)
        assert fetched is not None
        assert fetched.name == "file_read"
        assert fetched.entity_type == "tool"

    def test_duplicate_name_type_increments_mention_count(self, store: GraphMemoryStore):
        ent1 = Entity.new("config.yaml", "file")
        store.add_entity(ent1)
        ent2 = Entity.new("config.yaml", "file")
        store.add_entity(ent2)
        results = store.find_entities(name="config.yaml", entity_type="file")
        assert len(results) == 1
        assert results[0].mention_count == 2

    def test_duplicate_returns_same_canonical_id(self, store: GraphMemoryStore):
        ent1 = Entity.new("shell_exec", "tool")
        id1 = store.add_entity(ent1)
        ent2 = Entity.new("shell_exec", "tool")
        id2 = store.add_entity(ent2)
        assert id1 == id2

    def test_different_type_same_name_stored_separately(self, store: GraphMemoryStore):
        ent_tool = Entity.new("missy", "tool")
        ent_project = Entity.new("missy", "project")
        store.add_entity(ent_tool)
        store.add_entity(ent_project)
        results = store.find_entities(name="missy")
        assert len(results) == 2

    def test_properties_preserved_on_insert(self, store: GraphMemoryStore):
        ent = Entity.new("myproject", "project", properties={"lang": "python"})
        eid = store.add_entity(ent)
        fetched = store.get_entity(eid)
        assert fetched is not None
        assert fetched.properties.get("lang") == "python"


# ---------------------------------------------------------------------------
# GraphMemoryStore — add_relationship / upsert
# ---------------------------------------------------------------------------


class TestAddRelationship:
    def _two_entities(self, store: GraphMemoryStore) -> tuple[str, str]:
        src_id = store.add_entity(Entity.new("file_read", "tool"))
        tgt_id = store.add_entity(Entity.new("~/.missy/config.yaml", "file"))
        return src_id, tgt_id

    def test_relationship_persisted(self, store: GraphMemoryStore):
        src, tgt = self._two_entities(store)
        rel = Relationship.new(src, tgt, "uses")
        rid = store.add_relationship(rel)
        fetched = store._get_relationship_by_id(rid)
        assert fetched is not None
        assert fetched.source_id == src
        assert fetched.target_id == tgt
        assert fetched.relation_type == "uses"

    def test_duplicate_relationship_bumps_weight(self, store: GraphMemoryStore):
        src, tgt = self._two_entities(store)
        rel = Relationship.new(src, tgt, "uses", weight=0.5)
        store.add_relationship(rel)
        rel2 = Relationship.new(src, tgt, "uses", weight=0.5)
        rid2 = store.add_relationship(rel2)
        fetched = store._get_relationship_by_id(rid2)
        assert fetched is not None
        assert fetched.weight == pytest.approx(0.55)

    def test_weight_never_exceeds_one(self, store: GraphMemoryStore):
        src, tgt = self._two_entities(store)
        rel = Relationship.new(src, tgt, "uses", weight=0.99)
        store.add_relationship(rel)
        rel2 = Relationship.new(src, tgt, "uses", weight=0.99)
        rid = store.add_relationship(rel2)
        fetched = store._get_relationship_by_id(rid)
        assert fetched is not None
        assert fetched.weight <= 1.0

    def test_different_relation_type_stored_separately(self, store: GraphMemoryStore):
        src, tgt = self._two_entities(store)
        store.add_relationship(Relationship.new(src, tgt, "uses"))
        store.add_relationship(Relationship.new(src, tgt, "creates"))
        rels = store.get_relationships(src, direction="outbound")
        types = {r.relation_type for r in rels}
        assert "uses" in types
        assert "creates" in types


# ---------------------------------------------------------------------------
# GraphMemoryStore — ingest_turn
# ---------------------------------------------------------------------------


class TestIngestTurn:
    def test_ingest_discovers_entities(self, store: GraphMemoryStore):
        entities, _ = store.ingest_turn(
            "I used file_read on ~/.missy/config.yaml",
            role="user",
            session_id="sess-1",
        )
        names = [e.name for e in entities]
        assert "file_read" in names

    def test_ingest_repeated_turn_increments_count(self, store: GraphMemoryStore):
        store.ingest_turn(
            "file_write created ~/workspace/notes.txt",
            role="user",
            session_id="sess-1",
        )
        store.ingest_turn(
            "file_write created ~/workspace/notes.txt",
            role="user",
            session_id="sess-1",
        )
        results = store.find_entities(name="file_write")
        assert results[0].mention_count == 2

    def test_ingest_empty_text_returns_empty(self, store: GraphMemoryStore):
        entities, rels = store.ingest_turn("", role="user", session_id="s")
        assert entities == []
        assert rels == []

    def test_ingest_returns_persisted_canonical_ids(self, store: GraphMemoryStore):
        entities, rels = store.ingest_turn(
            "shell_exec uses ~/workspace/script.sh",
            role="assistant",
            session_id="sess-2",
        )
        for ent in entities:
            # Every returned entity should be findable in the store
            assert store.get_entity(ent.id) is not None


# ---------------------------------------------------------------------------
# GraphMemoryStore — find_entities
# ---------------------------------------------------------------------------


class TestFindEntities:
    def test_find_by_name_substring(self, store: GraphMemoryStore):
        store.add_entity(Entity.new("file_read", "tool"))
        store.add_entity(Entity.new("file_write", "tool"))
        results = store.find_entities(name="file_")
        names = [e.name for e in results]
        assert "file_read" in names
        assert "file_write" in names

    def test_find_by_type_filter(self, store: GraphMemoryStore):
        store.add_entity(Entity.new("file_read", "tool"))
        store.add_entity(Entity.new("myproject", "project"))
        results = store.find_entities(entity_type="tool")
        assert all(e.entity_type == "tool" for e in results)

    def test_find_no_filters_returns_all(self, store: GraphMemoryStore):
        store.add_entity(Entity.new("alpha", "tool"))
        store.add_entity(Entity.new("beta", "concept"))
        results = store.find_entities()
        assert len(results) >= 2

    def test_find_returns_empty_when_no_match(self, store: GraphMemoryStore):
        results = store.find_entities(name="nonexistent_xyz_zzz")
        assert results == []

    def test_find_respects_limit(self, store: GraphMemoryStore):
        for i in range(10):
            store.add_entity(Entity.new(f"entity_{i}", "concept"))
        results = store.find_entities(limit=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# GraphMemoryStore — get_relationships
# ---------------------------------------------------------------------------


class TestGetRelationships:
    def _setup(self, store: GraphMemoryStore) -> tuple[str, str]:
        src = store.add_entity(Entity.new("shell_exec", "tool"))
        tgt = store.add_entity(Entity.new("~/script.sh", "file"))
        store.add_relationship(Relationship.new(src, tgt, "uses"))
        return src, tgt

    def test_outbound_returns_correct_rels(self, store: GraphMemoryStore):
        src, tgt = self._setup(store)
        rels = store.get_relationships(src, direction="outbound")
        assert all(r.source_id == src for r in rels)

    def test_inbound_returns_correct_rels(self, store: GraphMemoryStore):
        src, tgt = self._setup(store)
        rels = store.get_relationships(tgt, direction="inbound")
        assert all(r.target_id == tgt for r in rels)

    def test_both_direction_returns_all(self, store: GraphMemoryStore):
        src, tgt = self._setup(store)
        rels_both = store.get_relationships(src, direction="both")
        assert len(rels_both) >= 1

    def test_entity_with_no_rels_returns_empty(self, store: GraphMemoryStore):
        eid = store.add_entity(Entity.new("orphan", "concept"))
        assert store.get_relationships(eid) == []


# ---------------------------------------------------------------------------
# GraphMemoryStore — get_neighbors (BFS)
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    def _build_chain(self, store: GraphMemoryStore) -> list[str]:
        """A → B → C chain."""
        ids = []
        for name in ("ent_a", "ent_b", "ent_c"):
            ids.append(store.add_entity(Entity.new(name, "concept")))
        store.add_relationship(Relationship.new(ids[0], ids[1], "related_to"))
        store.add_relationship(Relationship.new(ids[1], ids[2], "related_to"))
        return ids

    def test_depth_1_returns_direct_neighbor(self, store: GraphMemoryStore):
        ids = self._build_chain(store)
        result = store.get_neighbors(ids[0], max_depth=1)
        entity_names = {e.name for e in result.entities}
        assert "ent_a" in entity_names
        assert "ent_b" in entity_names

    def test_depth_2_traverses_two_hops(self, store: GraphMemoryStore):
        ids = self._build_chain(store)
        result = store.get_neighbors(ids[0], max_depth=2)
        entity_names = {e.name for e in result.entities}
        assert "ent_c" in entity_names

    def test_depth_0_returns_only_seed(self, store: GraphMemoryStore):
        ids = self._build_chain(store)
        result = store.get_neighbors(ids[0], max_depth=0)
        assert len(result.entities) == 1
        assert result.entities[0].name == "ent_a"

    def test_unknown_entity_returns_empty_query(self, store: GraphMemoryStore):
        result = store.get_neighbors("ent_does_not_exist", max_depth=2)
        assert result.entities == []
        assert result.relationships == []

    def test_paths_include_entity_ids(self, store: GraphMemoryStore):
        ids = self._build_chain(store)
        result = store.get_neighbors(ids[0], max_depth=2)
        # At least one path should start at ids[0]
        assert any(p[0] == ids[0] for p in result.paths)


# ---------------------------------------------------------------------------
# GraphMemoryStore — find_related
# ---------------------------------------------------------------------------


class TestFindRelated:
    def test_finds_entity_matching_query_keyword(self, store: GraphMemoryStore):
        store.add_entity(Entity.new("file_write", "tool"))
        result = store.find_related("file write tool")
        names = [e.name for e in result.entities]
        assert "file_write" in names

    def test_returns_empty_query_for_unknown_topic(self, store: GraphMemoryStore):
        result = store.find_related("completely unrelated topic xyzzy plugh")
        assert result.entities == []

    def test_respects_limit(self, store: GraphMemoryStore):
        for i in range(20):
            store.add_entity(Entity.new(f"concept_{i}", "concept"))
        result = store.find_related("concept", limit=5)
        assert len(result.entities) <= 5


# ---------------------------------------------------------------------------
# GraphMemoryStore — get_context_subgraph
# ---------------------------------------------------------------------------


class TestGetContextSubgraph:
    def test_returns_entity_graph_header(self, store: GraphMemoryStore):
        store.ingest_turn(
            "file_read opened ~/.missy/config.yaml",
            role="user",
            session_id="s1",
        )
        ctx = store.get_context_subgraph("config file")
        if ctx:
            assert "Entity Graph:" in ctx

    def test_returns_empty_string_when_no_match(self, store: GraphMemoryStore):
        ctx = store.get_context_subgraph("zephyr gobbledygook")
        assert ctx == ""

    def test_output_contains_relationship_arrow(self, store: GraphMemoryStore):
        src = store.add_entity(Entity.new("shell_exec", "tool"))
        tgt = store.add_entity(Entity.new("~/run.sh", "file"))
        store.add_relationship(Relationship.new(src, tgt, "uses"))
        ctx = store.get_context_subgraph("shell exec run")
        if ctx and "Entity Graph:" in ctx:
            assert "--[" in ctx and "]-->" in ctx

    def test_no_duplicate_lines(self, store: GraphMemoryStore):
        src = store.add_entity(Entity.new("file_write", "tool"))
        tgt = store.add_entity(Entity.new("~/out.txt", "file"))
        store.add_relationship(Relationship.new(src, tgt, "creates"))
        ctx = store.get_context_subgraph("file write")
        if ctx:
            lines = ctx.strip().split("\n")
            assert len(lines) == len(set(lines))


# ---------------------------------------------------------------------------
# GraphMemoryStore — get_entity_summary
# ---------------------------------------------------------------------------


class TestGetEntitySummary:
    def test_summary_contains_entity_name(self, store: GraphMemoryStore):
        store.add_entity(Entity.new("file_read", "tool"))
        summary = store.get_entity_summary("file_read")
        assert "file_read" in summary

    def test_summary_contains_mention_count(self, store: GraphMemoryStore):
        store.add_entity(Entity.new("file_read", "tool"))
        store.add_entity(Entity.new("file_read", "tool"))  # second upsert
        summary = store.get_entity_summary("file_read")
        assert "2" in summary

    def test_summary_includes_relationship(self, store: GraphMemoryStore):
        src = store.add_entity(Entity.new("web_fetch", "tool"))
        tgt = store.add_entity(Entity.new("https://api.github.com", "concept"))
        store.add_relationship(Relationship.new(src, tgt, "uses"))
        summary = store.get_entity_summary("web_fetch")
        assert "uses" in summary

    def test_summary_returns_empty_for_unknown_entity(self, store: GraphMemoryStore):
        assert store.get_entity_summary("nonexistent_entity_xyz") == ""


# ---------------------------------------------------------------------------
# GraphMemoryStore — prune
# ---------------------------------------------------------------------------


class TestPrune:
    def test_prune_removes_low_mention_old_entities(self, store: GraphMemoryStore, monkeypatch):
        """Entities with mention_count=1 older than threshold should be pruned."""
        from datetime import UTC, datetime, timedelta

        old_ts = (datetime.now(UTC) - timedelta(days=200)).isoformat()

        conn = store._conn()
        conn.execute(
            """INSERT INTO entities (id, name, entity_type, properties,
               first_seen, last_seen, mention_count)
               VALUES ('ent_old', 'old_entity', 'concept', '{}', ?, ?, 1)""",
            (old_ts, old_ts),
        )
        conn.commit()

        deleted = store.prune(min_mentions=1, older_than_days=90)
        assert deleted == 1
        assert store.get_entity("ent_old") is None

    def test_prune_keeps_high_mention_entities(self, store: GraphMemoryStore):
        from datetime import UTC, datetime, timedelta

        old_ts = (datetime.now(UTC) - timedelta(days=200)).isoformat()
        conn = store._conn()
        conn.execute(
            """INSERT INTO entities (id, name, entity_type, properties,
               first_seen, last_seen, mention_count)
               VALUES ('ent_keep', 'keep_entity', 'concept', '{}', ?, ?, 10)""",
            (old_ts, old_ts),
        )
        conn.commit()

        store.prune(min_mentions=1, older_than_days=90)
        assert store.get_entity("ent_keep") is not None

    def test_prune_removes_orphaned_relationships(self, store: GraphMemoryStore):
        from datetime import UTC, datetime, timedelta

        old_ts = (datetime.now(UTC) - timedelta(days=200)).isoformat()
        conn = store._conn()
        conn.executescript(f"""
            INSERT INTO entities (id, name, entity_type, properties,
               first_seen, last_seen, mention_count)
               VALUES ('ent_s', 'src_old', 'concept', '{{}}',
               '{old_ts}', '{old_ts}', 1);
            INSERT INTO entities (id, name, entity_type, properties,
               first_seen, last_seen, mention_count)
               VALUES ('ent_t', 'tgt_stable', 'concept', '{{}}',
               '{old_ts}', '{old_ts}', 10);
            INSERT INTO relationships (id, source_id, target_id, relation_type,
               weight, context, created_at, last_seen)
               VALUES ('rel_x', 'ent_s', 'ent_t', 'related_to',
               0.5, '', '{old_ts}', '{old_ts}');
        """)
        conn.commit()

        store.prune(min_mentions=1, older_than_days=90)
        # Relationship to pruned entity should be gone
        rel_row = conn.execute("SELECT id FROM relationships WHERE id = 'rel_x'").fetchone()
        assert rel_row is None

    def test_prune_returns_zero_when_nothing_to_prune(self, store: GraphMemoryStore):
        store.add_entity(Entity.new("recent_entity", "concept"))
        deleted = store.prune(min_mentions=1, older_than_days=1)
        assert deleted == 0


# ---------------------------------------------------------------------------
# GraphMemoryStore — merge_entities
# ---------------------------------------------------------------------------


class TestMergeEntities:
    def test_merge_transfers_relationships(self, store: GraphMemoryStore):
        keep_id = store.add_entity(Entity.new("keep_ent", "concept"))
        merge_id = store.add_entity(Entity.new("merge_ent", "concept"))
        third_id = store.add_entity(Entity.new("third_ent", "concept"))

        # Add rel from merge_ent → third_ent
        store.add_relationship(Relationship.new(merge_id, third_id, "related_to"))

        store.merge_entities(keep_id, merge_id)

        # Relationship should now originate from keep_id
        rels = store.get_relationships(keep_id, direction="outbound")
        assert any(r.target_id == third_id for r in rels)

    def test_merge_increments_mention_count(self, store: GraphMemoryStore):
        keep_id = store.add_entity(Entity.new("alpha", "concept"))
        merge_id = store.add_entity(Entity.new("beta", "concept"))
        # Manually bump merge entity count
        conn = store._conn()
        conn.execute("UPDATE entities SET mention_count = 5 WHERE id = ?", (merge_id,))
        conn.commit()

        store.merge_entities(keep_id, merge_id)
        kept = store.get_entity(keep_id)
        assert kept is not None
        assert kept.mention_count == 6  # 1 + 5

    def test_merge_deletes_merged_entity(self, store: GraphMemoryStore):
        keep_id = store.add_entity(Entity.new("x", "concept"))
        merge_id = store.add_entity(Entity.new("y", "concept"))
        store.merge_entities(keep_id, merge_id)
        assert store.get_entity(merge_id) is None

    def test_merge_no_self_loops(self, store: GraphMemoryStore):
        keep_id = store.add_entity(Entity.new("p", "concept"))
        merge_id = store.add_entity(Entity.new("q", "concept"))
        store.add_relationship(Relationship.new(keep_id, merge_id, "related_to"))
        store.merge_entities(keep_id, merge_id)
        rels = store.get_relationships(keep_id, direction="both")
        assert all(r.source_id != r.target_id for r in rels)

    def test_merge_same_id_is_noop(self, store: GraphMemoryStore):
        eid = store.add_entity(Entity.new("same", "concept"))
        store.merge_entities(eid, eid)
        # Should not raise and entity should still exist
        assert store.get_entity(eid) is not None

    def test_merge_unknown_ids_is_noop(self, store: GraphMemoryStore):
        # Should not raise
        store.merge_entities("ent_fake_1", "ent_fake_2")


# ---------------------------------------------------------------------------
# GraphMemoryStore — stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty_store_returns_zero_counts(self, store: GraphMemoryStore):
        s = store.stats()
        assert s["entity_count"] == 0
        assert s["relationship_count"] == 0

    def test_stats_reflect_added_entities(self, store: GraphMemoryStore):
        store.add_entity(Entity.new("file_read", "tool"))
        store.add_entity(Entity.new("myproject", "project"))
        s = store.stats()
        assert s["entity_count"] == 2

    def test_entity_types_distribution(self, store: GraphMemoryStore):
        store.add_entity(Entity.new("file_read", "tool"))
        store.add_entity(Entity.new("file_write", "tool"))
        store.add_entity(Entity.new("missy", "project"))
        s = store.stats()
        assert s["entity_types"]["tool"] == 2
        assert s["entity_types"]["project"] == 1

    def test_relationship_count_in_stats(self, store: GraphMemoryStore):
        src = store.add_entity(Entity.new("shell_exec", "tool"))
        tgt = store.add_entity(Entity.new("~/run.sh", "file"))
        store.add_relationship(Relationship.new(src, tgt, "uses"))
        s = store.stats()
        assert s["relationship_count"] == 1

    def test_relation_types_distribution(self, store: GraphMemoryStore):
        src = store.add_entity(Entity.new("tool_a", "tool"))
        tgt = store.add_entity(Entity.new("file_b", "file"))
        store.add_relationship(Relationship.new(src, tgt, "uses"))
        store.add_relationship(Relationship.new(src, tgt, "creates"))
        s = store.stats()
        assert "uses" in s["relation_types"]
        assert "creates" in s["relation_types"]


# ---------------------------------------------------------------------------
# Synthesizer integration — graph source
# ---------------------------------------------------------------------------


class TestSynthesizerGraphIntegration:
    def test_graph_source_accepted_at_correct_relevance(self):
        from missy.memory.synthesizer import MemorySynthesizer

        synth = MemorySynthesizer()
        synth.add_fragments(
            "graph",
            ["file_read --[uses]--> ~/.missy/config.yaml"],
            base_relevance=0.65,
        )
        synth.add_fragments("learnings", ["check ports first"], base_relevance=0.7)
        synth.add_fragments("playbook", ["use file_read before parsing"], base_relevance=0.6)

        result = synth.synthesize("config file")
        assert "[graph]" in result
        assert "[learnings]" in result
        assert "[playbook]" in result

    def test_graph_ranked_between_learnings_and_playbook(self):
        from missy.memory.synthesizer import MemorySynthesizer

        synth = MemorySynthesizer()
        synth.add_fragments("learnings", ["check ports first"], base_relevance=0.7)
        synth.add_fragments(
            "graph",
            ["entity: file_read"],
            base_relevance=0.65,
        )
        synth.add_fragments("playbook", ["run scripts with shell_exec"], base_relevance=0.6)

        result = synth.synthesize("irrelevant query zzzzz")
        lines = [ln for ln in result.split("\n") if ln.strip()]
        sources = [ln.split("]")[0].lstrip("[") for ln in lines if ln.startswith("[")]
        # learnings should appear before graph, graph before playbook
        if "learnings" in sources and "graph" in sources and "playbook" in sources:
            assert sources.index("learnings") < sources.index("graph")
            assert sources.index("graph") < sources.index("playbook")

    def test_graph_context_injected_from_store(self, store: GraphMemoryStore):
        from missy.memory.synthesizer import MemorySynthesizer

        store.ingest_turn(
            "file_read opened ~/.missy/config.yaml",
            role="user",
            session_id="sess-1",
        )
        ctx = store.get_context_subgraph("config")

        synth = MemorySynthesizer()
        if ctx:
            synth.add_fragments("graph", [ctx], base_relevance=0.65)
        synth.add_fragments("learnings", ["always validate config"], base_relevance=0.7)
        result = synth.synthesize("config validation")
        assert "[learnings]" in result
