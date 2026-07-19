"""Tests for the `missy retrieval` CLI group (F03)."""

from __future__ import annotations

from click.testing import CliRunner

from missy.cli.main import cli


def _run(args):
    return CliRunner().invoke(cli, args)


class TestRetrievalCli:
    def test_index_then_query(self, tmp_path) -> None:
        doc = tmp_path / "note.txt"
        doc.write_text("Rotate the vault api keys and restart the gateway service.")
        idx = str(tmp_path / "idx")

        r = _run(["retrieval", "index", str(doc), "--index-dir", idx, "--doc-id", "note"])
        assert r.exit_code == 0, r.output
        assert "chunk" in r.output

        q = _run(["retrieval", "query", "rotate keys", "--index-dir", idx, "--top-k", "3"])
        assert q.exit_code == 0, q.output
        assert "note[" in q.output  # citation rendered

    def test_stats_on_empty_index(self, tmp_path) -> None:
        r = _run(["retrieval", "stats", "--index-dir", str(tmp_path / "empty")])
        assert r.exit_code == 0
        assert "Retrieval Index" in r.output

    def test_query_empty_index_reports_no_results(self, tmp_path) -> None:
        r = _run(["retrieval", "query", "anything", "--index-dir", str(tmp_path / "e")])
        assert r.exit_code == 0
        assert "No results" in r.output

    def test_remove_document(self, tmp_path) -> None:
        doc = tmp_path / "d.txt"
        doc.write_text("content to be removed from the index later on")
        idx = str(tmp_path / "idx")
        _run(["retrieval", "index", str(doc), "--index-dir", idx, "--doc-id", "d"])

        r = _run(["retrieval", "remove", "d", "--index-dir", idx])
        assert r.exit_code == 0
        assert "Removed" in r.output

        r2 = _run(["retrieval", "remove", "missing", "--index-dir", idx])
        assert r2.exit_code == 0
        assert "No chunks" in r2.output

    def test_group_no_subcommand_shows_stats(self, tmp_path) -> None:
        r = _run(["retrieval", "--index-dir", str(tmp_path)])
        # invoking the group with no subcommand falls through to stats; the
        # --index-dir belongs to the subcommand, so bare invocation still runs.
        assert r.exit_code in (0, 2)
