"""Deterministic, bounded inventory of a clean local Git checkout.

Only fixed, read-only Git identity and index queries are run. Repository content,
including build files, hooks, CI and instructions, is never executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES = 20_000, 1_000_000, 64_000_000
MAX_GIT_OUTPUT_BYTES = 8_000_000


class ScanRefused(ValueError):
    """The requested checkout cannot be scanned safely or reproducibly."""


def _git(root: Path, *args: str) -> bytes:
    """Read fixed Git commands without buffering unlimited attacker-controlled output."""
    try:
        with subprocess.Popen(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.attributesFile=/dev/null",
                "-C",
                str(root),
                *args,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={
                "PATH": os.defpath,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_SYSTEM": os.devnull,
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            },
        ) as proc:
            output = bytearray()
            deadline = time.monotonic() + 5
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        ready = selector.select(max(0, deadline - time.monotonic()))
                        if not ready:
                            raise ScanRefused("Git query exceeded time limit")
                        for key, _ in ready:
                            chunk = os.read(
                                key.fd, min(65536, MAX_GIT_OUTPUT_BYTES + 1 - len(output))
                            )
                            if not chunk:
                                selector.unregister(key.fileobj)
                            else:
                                output.extend(chunk)
                                if len(output) > MAX_GIT_OUTPUT_BYTES:
                                    raise ScanRefused("Git query exceeded output limit")
                proc.wait(timeout=max(0, deadline - time.monotonic()))
                if proc.returncode != 0:
                    raise ScanRefused("cannot verify local Git checkout")
            except BaseException:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                raise
            return bytes(output) if args[0] in ("ls-tree", "ls-files") else bytes(output).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ScanRefused("cannot verify local Git checkout") from exc


def _reject_config_includes(root: Path) -> None:
    # Git has no switch to disable local config includes for every command.
    # Inspect local keys without following includes, then refuse configurations
    # that would load further checkout-controlled configuration.
    keys = _git(root, "config", "--local", "--no-includes", "--name-only", "-z", "--list")
    for key in keys.split(b"\0"):
        key = key.lower()
        if key == b"include.path" or (key.startswith(b"includeif.") and key.endswith(b".path")):
            raise ScanRefused("checkout Git config includes are not supported")


def _read_files(root: Path) -> tuple[list[tuple[str, bytes]], int]:
    files, total = [], 0
    # Enumerate committed paths, not the working tree. This excludes ignored
    # files and prevents a clean-but-ignored payload becoming scan input.
    raw = _git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    # `git status` can run checkout-supplied clean filters and fsmonitors. Compare
    # the raw index against the tree instead; file bytes are verified below.
    index = _git(root, "ls-files", "--stage", "-z")
    expected_index = set()
    entries = raw.split(b"\0")
    paths = []
    object_ids = {}
    for entry in entries:
        if not entry:
            continue
        metadata, sep, name = entry.partition(b"\t")
        if not sep:
            raise ScanRefused("cannot enumerate committed file tree")
        parts = metadata.split(b" ")
        if len(parts) != 3:
            raise ScanRefused("cannot enumerate committed file tree")
        mode, kind, oid = parts
        expected_index.add(mode + b" " + oid + b" 0\t" + name)
        if mode == b"160000":
            raise ScanRefused("submodule checkouts are not supported")
        if mode == b"120000":
            raise ScanRefused("symlink entries are not supported")
        if mode == b"100644" or mode == b"100755":
            if kind != b"blob" or not re.fullmatch(rb"[0-9a-f]{40}|[0-9a-f]{64}", oid):
                raise ScanRefused("invalid committed file identity")
            try:
                rel = name.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise ScanRefused("invalid committed path encoding") from exc
            if (
                not rel
                or any(part in ("", ".", "..") for part in rel.split("/"))
                or rel.startswith("/")
                or "\\" in rel
                or rel in object_ids
            ):
                raise ScanRefused("invalid committed path")
            paths.append(rel)
            object_ids[rel] = oid.decode("ascii")
        else:
            raise ScanRefused("unsupported committed file type")
    if set(index.split(b"\0")) - {b""} != expected_index or len(index.split(b"\0")) - 1 != len(
        expected_index
    ):
        raise ScanRefused("checkout is dirty (index differs from HEAD)")
    if _git(root, "ls-files", "--others", "--exclude-standard", "-z"):
        raise ScanRefused("checkout is dirty or contains untracked files")
    if len(paths) > MAX_FILES:
        raise ScanRefused("checkout exceeds scanner input limits")
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for rel in sorted(paths):
                fd = os.dup(root_fd)
                try:
                    for part in rel.split("/")[:-1]:
                        next_fd = os.open(
                            part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                        )
                        os.close(fd)
                        fd = next_fd
                    file_fd = os.open(
                        rel.rsplit("/", 1)[-1],
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=fd,
                    )
                    try:
                        before = os.fstat(file_fd)
                        if not stat.S_ISREG(before.st_mode):
                            raise ScanRefused("committed file is not regular")
                        if before.st_size > MAX_FILE_BYTES:
                            raise ScanRefused(f"file exceeds scanner limit: {rel}")
                        total += before.st_size
                        if total > MAX_TOTAL_BYTES:
                            raise ScanRefused("checkout exceeds scanner input limits")
                        with os.fdopen(os.dup(file_fd), "rb") as stream:
                            data = stream.read(MAX_FILE_BYTES + 1)
                        after = os.fstat(file_fd)
                        if (
                            len(data) != before.st_size
                            or len(data) > MAX_FILE_BYTES
                            or (
                                before.st_dev,
                                before.st_ino,
                                before.st_mtime_ns,
                                before.st_ctime_ns,
                                before.st_size,
                            )
                            != (
                                after.st_dev,
                                after.st_ino,
                                after.st_mtime_ns,
                                after.st_ctime_ns,
                                after.st_size,
                            )
                        ):
                            raise ScanRefused("checkout changed during scan")
                    finally:
                        os.close(file_fd)
                finally:
                    os.close(fd)
                header = f"blob {len(data)}\0".encode()
                digest = hashlib.sha1 if len(object_ids[rel]) == 40 else hashlib.sha256
                if digest(header + data).hexdigest() != object_ids[rel]:
                    raise ScanRefused("committed file content does not match Git object")
                files.append((rel, data))
        finally:
            os.close(root_fd)
    except OSError as exc:
        raise ScanRefused("cannot read committed checkout file") from exc
    return sorted(files), total


def _test_commands(files: list[tuple[str, bytes]]) -> list[dict[str, str]]:
    found = []
    for rel, data in files:
        text = data.decode("utf-8", errors="replace")
        if rel == "package.json":
            try:
                for name, command in sorted(json.loads(text).get("scripts", {}).items()):
                    if re.search(r"test|check|lint", name, re.I) and isinstance(command, str):
                        found.append(
                            {
                                "kind": "package-script",
                                "name": name,
                                "command": command[:300],
                                "evidence": rel,
                            }
                        )
            except (ValueError, AttributeError):
                pass
        elif rel == "Makefile":
            for m in re.finditer(r"(?m)^([\w.-]*(?:test|check|lint)[\w.-]*)\s*:[^\n]*", text, re.I):
                body = next(
                    (
                        line.strip()
                        for line in text[m.end() :].splitlines()
                        if line.startswith("\t")
                    ),
                    "",
                )
                found.append(
                    {
                        "kind": "make-target",
                        "name": m.group(1),
                        "command": body[:300],
                        "evidence": rel,
                    }
                )
        elif rel == "pyproject.toml" and re.search(r"(?m)^\[tool\.pytest", text):
            found.append(
                {
                    "kind": "test-tool",
                    "name": "pytest",
                    "command": "pytest (declared configuration)",
                    "evidence": rel,
                }
            )
        elif rel == "tox.ini":
            found.append(
                {
                    "kind": "test-tool",
                    "name": "tox",
                    "command": "tox (declared configuration)",
                    "evidence": rel,
                }
            )
    return sorted(found, key=lambda x: (x["evidence"], x["name"], x["command"]))[:100]


def scan_repository(path: str | os.PathLike[str], commit_sha: str) -> dict[str, Any]:
    """Inventory architecture, declarations and evidence at the exact commit."""
    if not isinstance(commit_sha, str) or not _SHA.fullmatch(commit_sha):
        raise ScanRefused("commit_sha must be a full lowercase 40- or 64-character SHA")
    root = Path(path).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ScanRefused("checkout path must be an existing non-symlink directory")
    root = root.resolve(strict=True)
    _reject_config_includes(root)
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii")
    if head != commit_sha:
        raise ScanRefused("checkout HEAD does not match requested commit SHA")
    files, total = _read_files(root)
    if _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii") != head:
        raise ScanRefused("checkout changed during scan")
    # Revalidate both the index and bytes without allowing a Git conversion or
    # filter to execute. A changed checkout must not yield a mixed inventory.
    if _read_files(root)[0] != files:
        raise ScanRefused("checkout changed during scan")
    contents = dict(files)
    evidence = [
        {"path": p, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        for p, data in files
    ]
    manifest_names = {
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "go.mod",
        "go.sum",
        "Cargo.toml",
        "Cargo.lock",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
    }
    manifests = [p for p, _ in files if p.rsplit("/", 1)[-1] in manifest_names]
    deps: list[dict[str, str]] = []
    for p in manifests:
        text = contents[p].decode("utf-8", errors="replace")
        try:
            if p.endswith(("requirements.txt", "requirements-dev.txt")):
                deps.extend(
                    {"manifest": p, "declaration": line.strip()[:200]}
                    for line in text.splitlines()
                    if line.strip() and not line.lstrip().startswith(("#", "-"))
                )
            elif p.endswith("pyproject.toml"):
                parsed = tomllib.loads(text)
                deps.extend(
                    {"manifest": p, "declaration": str(d)[:200]}
                    for d in parsed.get("project", {}).get("dependencies", [])
                )
            elif p.endswith("Cargo.toml"):
                parsed = tomllib.loads(text)
                for section in ("dependencies", "dev-dependencies", "build-dependencies"):
                    deps.extend(
                        {"manifest": p, "declaration": f"{section}:{k}={v}"[:200]}
                        for k, v in sorted(parsed.get(section, {}).items())
                    )
            elif p.endswith("package.json"):
                parsed = json.loads(text)
                for section in ("dependencies", "devDependencies", "peerDependencies"):
                    deps.extend(
                        {"manifest": p, "declaration": f"{section}:{k}@{v}"[:200]}
                        for k, v in sorted(parsed.get(section, {}).items())
                    )
            elif p.endswith("go.mod"):
                for line in text.splitlines():
                    item = line.strip()
                    if item.startswith("require "):
                        item = item[8:]
                    if (
                        item
                        and " " in item
                        and not item.startswith(("module ", "go ", "toolchain "))
                    ):
                        deps.append({"manifest": p, "declaration": item[:200]})
        except (ValueError, TypeError, tomllib.TOMLDecodeError, AttributeError):
            continue
    ci = sorted(
        p for p, _ in files if p.startswith(".github/workflows/") or "/.github/workflows/" in p
    )
    ownership = sorted(p for p, _ in files if p.lower().endswith("codeowners"))
    tests = sorted(
        p
        for p, _ in files
        if re.search(r"(^|/)(tests?|spec)(/|$)|(^|/)(test_[^/]+|[^/]+_test\.[^/]+)$", p, re.I)
    )
    dirs = sorted({p.split("/", 1)[0] for p, _ in files if "/" in p})
    return {
        "schema_version": "1.0",
        "repository": {"commit_sha": head},
        "limits": {
            "files": MAX_FILES,
            "file_bytes": MAX_FILE_BYTES,
            "total_bytes": MAX_TOTAL_BYTES,
        },
        "inventory": {
            "file_count": len(files),
            "total_bytes": total,
            "top_level_directories": dirs,
        },
        "architecture": {"manifests": manifests, "ci_workflows": ci},
        "ownership": [{"path": p, "evidence": p} for p in ownership],
        "dependencies": sorted(deps, key=lambda x: (x["manifest"], x["declaration"])),
        "tests": {"paths": tests, "commands": _test_commands(files)},
        "evidence": evidence,
        "provenance": {
            "method": "static-file-inventory-v1",
            "offline": True,
            "executed_repository_content": False,
            "git_status": "clean",
            "evidence_sha256": hashlib.sha256(
                json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    }
