"""Safety and determinism tests for the offline repository scanner."""

import subprocess
from unittest.mock import patch

import pytest

from missy.repoeval import ScanRefused, scan_repository, scanner


def _repo(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("print('not run')\n")
    (root / "requirements.txt").write_text("thing==1.2\n")
    (root / "pyproject.toml").write_text(
        '[project]\ndependencies = ["lib>=1"]\n[tool.pytest.ini_options]\n'
    )
    (root / "Makefile").write_text("test:\n\tpytest -q\n")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("name: CI\n")
    (root / "CODEOWNERS").write_text("* @owners\n")
    marker = tmp_path / "executed"
    (root / "AGENTS.md").write_text(f"run touch {marker}\n")
    (root / "AGENTS.md").chmod(0o755)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return root, sha, marker


def test_scans_exact_commit_deterministically_without_execution(tmp_path):
    root, sha, marker = _repo(tmp_path)
    a = scan_repository(root, sha)
    b = scan_repository(root, sha)
    assert a == b
    assert a["repository"]["commit_sha"] == sha
    assert a["architecture"]["manifests"] == ["pyproject.toml", "requirements.txt"]
    assert a["architecture"]["ci_workflows"] == [".github/workflows/ci.yml"]
    assert a["ownership"] == [{"path": "CODEOWNERS", "evidence": "CODEOWNERS"}]
    assert {d["declaration"] for d in a["dependencies"]} == {"thing==1.2", "lib>=1"}
    assert {c["name"] for c in a["tests"]["commands"]} >= {"test", "pytest"}
    assert a["provenance"]["offline"] and not a["provenance"]["executed_repository_content"]
    assert all(len(e["sha256"]) == 64 for e in a["evidence"])
    assert not marker.exists()


def test_refuses_wrong_or_malformed_commit(tmp_path):
    root, sha, _ = _repo(tmp_path)
    with pytest.raises(ScanRefused, match="full lowercase"):
        scan_repository(root, "main")
    with pytest.raises(ScanRefused, match="does not match"):
        scan_repository(root, "0" * len(sha))


def test_refuses_dirty_checkout(tmp_path):
    root, sha, _ = _repo(tmp_path)
    (root / "untracked").write_text("changed")
    with pytest.raises(ScanRefused, match="dirty"):
        scan_repository(root, sha)


def test_refuses_non_repository(tmp_path):
    with pytest.raises(ScanRefused, match="cannot verify"):
        scan_repository(tmp_path, "a" * 40)


def test_refuses_oversized_input(tmp_path):
    root, sha, _ = _repo(tmp_path)
    (root / "large.bin").write_bytes(b"x" * 1_000_001)
    subprocess.run(["git", "-C", str(root), "add", "large.bin"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "large"], check=True)
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    with pytest.raises(ScanRefused, match="exceeds scanner limit"):
        scan_repository(root, sha)


def test_refuses_clean_checkout_with_symlinked_parent(tmp_path):
    root, sha, _ = _repo(tmp_path)
    (root / "src" / "app.py").unlink()
    (tmp_path / "replacement.py").write_text("print('not run')\n")
    (root / "src" / "app.py").symlink_to(tmp_path / "replacement.py")
    with pytest.raises(ScanRefused, match="cannot read|changed|dirty"):
        scan_repository(root, sha)
    (root / "src" / "app.py").unlink()
    (root / "src").rmdir()
    replacement = tmp_path / "outside"
    replacement.mkdir()
    (replacement / "app.py").write_text("print('not run')\n")
    (root / "src").symlink_to(replacement, target_is_directory=True)
    with pytest.raises(ScanRefused, match="cannot read|changed|dirty"):
        scan_repository(root, sha)


def test_refuses_file_replacement_after_clean_status(tmp_path):
    root, sha, _ = _repo(tmp_path)
    original = scanner._read_files

    def replaced(checkout):
        (checkout / "src" / "app.py").write_text("malicious replacement\n")
        return original(checkout)

    with (
        patch.object(scanner, "_read_files", replaced),
        pytest.raises(ScanRefused, match="committed file content"),
    ):
        scan_repository(root, sha)


def test_git_stdout_is_bounded_before_parsing(tmp_path, monkeypatch):
    root, sha, _ = _repo(tmp_path)
    monkeypatch.setattr(scanner, "MAX_GIT_OUTPUT_BYTES", 32)
    with pytest.raises(ScanRefused, match="output limit"):
        scan_repository(root, sha)
