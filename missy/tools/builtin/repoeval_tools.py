"""Opt-in, project-scoped Foundry client with request-time file credentials."""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import os
import re
import stat
import threading
from typing import Any
from urllib.parse import quote, urlsplit

from missy.gateway.client import PolicyHTTPClient
from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_KEY = re.compile(r"[A-Za-z0-9._:-]{8,128}\Z")
_TOKEN = re.compile(rb"[A-Za-z0-9._~+/-]+=*\Z")
_TOKEN_MAX_BYTES = 4096
_URI = re.compile(r"(?i)\b(?:[a-z][a-z0-9+.-]*://|(?:file|data|javascript):)")
_ACTIONS = {
    "list",
    "plan",
    "snapshot",
    "start",
    "status",
    "compare",
    "artifacts",
    "cancel",
    "report",
}
_READ_ACTIONS = frozenset({"list", "plan", "status", "compare", "artifacts", "report"})
_MUTATION_ACTIONS = frozenset({"snapshot", "start", "cancel"})
_SECRET = re.compile(
    r"(?i)(?<![a-z0-9])(?:token|password|secret|api[_-]?key|authorization|cookie|credential|private[_-]?key|url|uri|href|location|stdout|stderr|raw[_-]?output|environment|env|content)(?![a-z0-9])"
)
_SECRET_TEXT = re.compile(
    r"(?i)\b(?:bearer\s+\S+|(?:token|password|secret|api[_-]?key)\s*[:=]\s*\S+|sk-[A-Za-z0-9_-]{3,})"
)
_SAFE_SETTING_KEYS = frozenset({"token_budget", "max_tokens"})
_FAIL = "Foundry request refused or unavailable; no execution is confirmed"
_RESPONSE_LIMIT = 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_ID = re.compile(r"artifact-[A-Za-z0-9_-]{8,128}\Z")
_KIND = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
# Client-side allowlist for Foundry's version 1.0 comparison reason paths.
# Keep this pinned at the wire boundary: Missy must not import the Foundry core.
_COMPARABILITY_PATHS_V1 = frozenset(
    {
        "repository.repository_id",
        "repository.commit_sha",
        "repository.snapshot_id",
        "repository.subdirectory",
        "task.class",
        "task.prompt_sha256",
        "task.fixture_digests",
        "task.tool_schema_uris",
        "workload_id",
        "workload_version",
        "definition_sha256",
        "sandbox.image_digest",
        "sandbox.network_policy",
        "sandbox.cpu_mhz",
        "sandbox.memory_mb",
        "sandbox.disk_mb",
        "sandbox.timeout_seconds",
        "sandbox.architecture",
        "evaluator_version",
        "validators",
        "provider_independent_settings_sha256",
        "tool_schemas_sha256",
    }
)
_REASONS = _COMPARABILITY_PATHS_V1 | {
    "comparability_key_invalid",
    "comparability_key_mismatch",
    "comparability_rules_version_mismatch",
    "claimed_key_mismatch",
    "missing_or_invalid_comparability_key",
}


def _identity_digest(value: Any) -> str:
    """Coordinator identity encoding, independent of the untrusted reply."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def _open_token_file(path: str) -> int:
    """Open and validate without following any symlink, including parent dirs.

    Directory-relative opens bind each checked component to an open descriptor;
    replacing a path component cannot redirect the final read through a symlink.
    No bytes are read here, including during registration.
    """
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or len(path) > 4096
        or "~" in path
        or "\\" in path
        or any(ord(c) < 33 or ord(c) == 127 for c in path)
        or any(part in ("", ".", "..") for part in path.split("/")[1:])
    ):
        raise ValueError("Invalid Foundry credential file")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    fd = None
    try:
        parts = path.split("/")[1:]
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=directory,
        )
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or not 1 <= info.st_size <= _TOKEN_MAX_BYTES
        ):
            raise ValueError("Invalid Foundry credential file")
        return fd
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise
    finally:
        os.close(directory)


def _token_file_valid(path: str) -> bool:
    try:
        os.close(_open_token_file(path))
        return True
    except (OSError, ValueError, TypeError):
        return False


def _read_token(path: str) -> str:
    fd = _open_token_file(path)
    try:
        before = os.fstat(fd)
        raw = os.read(fd, _TOKEN_MAX_BYTES + 1)
        after = os.fstat(fd)
        if (
            any(
                getattr(before, key) != getattr(after, key)
                for key in (
                    "st_mode",
                    "st_uid",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
            or len(raw) != before.st_size
            or len(raw) > _TOKEN_MAX_BYTES
        ):
            raise ValueError("Invalid Foundry credential file")
        # An optional single trailing LF supports normal operator-created files.
        token = raw.removesuffix(b"\n")
        if not _TOKEN.fullmatch(token):
            raise ValueError("Invalid Foundry credential file")
        return token.decode("ascii")
    finally:
        os.close(fd)


def _hostname_valid(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return len(host) <= 253 and all(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in host.split(".")
        )


def _secret_key(key: str) -> bool:
    return key.lower() not in _SAFE_SETTING_KEYS and bool(_SECRET.search(key))


def _safe(value: Any, depth: int = 0) -> Any:
    if depth >= 8:
        return "[omitted]"
    if isinstance(value, dict):
        return {
            k: "[redacted]" if _secret_key(k) else _safe(v, depth + 1)
            for k, v in list(value.items())[:100]
            if isinstance(k, str) and not _SECRET_TEXT.search(k) and not _URI.search(k)
        }
    if isinstance(value, list):
        return [_safe(item, depth + 1) for item in value[:100]]
    if isinstance(value, str):
        return "[redacted]" if _SECRET_TEXT.search(value) or _URI.search(value) else value[:2048]
    return value if value is None or type(value) in (bool, int, float) else "[omitted]"


def _no_secrets(value: Any, depth: int = 0) -> bool:
    """Reject credential-bearing request objects; never relay arbitrary secrets."""
    if depth > 10:
        return False
    if isinstance(value, dict):
        return len(value) <= 100 and all(
            isinstance(key, str) and not _secret_key(key) and _no_secrets(item, depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return len(value) <= 100 and all(_no_secrets(item, depth + 1) for item in value)
    if isinstance(value, str):
        return len(value) <= 2048 and not _SECRET_TEXT.search(value)
    return value is None or type(value) in (bool, int, float)


class RepoevalFoundryTool(BaseTool):
    name = "repoeval_foundry"
    description = "Authenticated project-scoped Foundry list/status, staging plan/snapshot/start/cancel, verified comparison/artifact metadata/draft reports."
    permissions = ToolPermissions(network=True)
    writes_state = True

    def __init__(
        self,
        *,
        base_url: str,
        project_id: str,
        allowed_hosts: list[str],
        http_client: PolicyHTTPClient | None = None,
        api_available: bool = False,
        token_file: str = "",
    ) -> None:
        self.permissions = ToolPermissions(network=True)
        self._auth_lock = threading.RLock()
        self._identity = (
            base_url,
            project_id,
            token_file,
            tuple(allowed_hosts) if isinstance(allowed_hosts, list) else None,
        )
        self._valid = False
        self._available = False
        self._token_file = token_file
        self._client = http_client
        self._plans: dict[str, dict[str, Any]] = {}
        self._repos: set[str] = set()
        self._listed = False
        try:
            if not isinstance(base_url, str) or any(
                ord(c) < 33 or ord(c) > 126 or c == "\\" for c in base_url
            ):
                return
            url = urlsplit(base_url)
            host = (url.hostname or "").lower()
            if (
                url.scheme not in ("http", "https")
                or not _hostname_valid(host)
                or url.port == 0
                or url.netloc.endswith(":")
                or url.username is not None
                or url.password is not None
                or url.query
                or url.fragment
                or url.path not in ("", "/", "/api", "/api/")
                or not isinstance(allowed_hosts, list)
                or not isinstance(project_id, str)
                or not _ID.fullmatch(project_id)
                or project_id in (".", "..")
                or host not in {h.lower() for h in allowed_hosts if isinstance(h, str)}
                or (url.scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"))
                or any(p in (".", "..") for p in url.path.split("/"))
            ):
                return
            self._base = (
                f"{url.scheme}://{url.netloc}{'/api' if url.path.startswith('/api') else ''}"
            )
            self._project = project_id
            self.permissions.allowed_hosts = [host]
            self._valid = True
            if api_available is True and _token_file_valid(token_file):
                if self._client is None:
                    self._client = PolicyHTTPClient(
                        category="tool", max_response_bytes=_RESPONSE_LIMIT
                    )
                self._available = getattr(self._client, "category", None) == "tool"
        except (TypeError, ValueError, AttributeError):
            pass

    @property
    def registration_ready(self) -> bool:
        """Configuration and file metadata only, not a server/auth health claim."""
        return self._valid and self._available

    def matches_config(self, config: Any) -> bool:
        """Only the original operator authorization may keep this client alive."""
        return (
            getattr(config, "enabled", False) is True
            and getattr(config, "api_available", False) is True
            and self._identity
            == (
                getattr(config, "base_url", None),
                getattr(config, "project_id", None),
                getattr(config, "token_file", None),
                tuple(config.allowed_hosts)
                if isinstance(getattr(config, "allowed_hosts", None), list)
                else None,
            )
        )

    def revoke(self) -> None:
        """Permanently retire this instance, including direct references to it.

        Serializing with execute ensures reload cannot complete while a stale
        authenticated request is still being initiated by this instance.
        """
        with self._auth_lock:
            self._available = False
            self._repos.clear()
            self._plans.clear()
            self._listed = False

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        del kwargs
        if not self._valid:
            raise ValueError("Foundry endpoint not authorized")
        url = urlsplit(self._base)
        host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
        return [f"{host}:{url.port or (443 if url.scheme == 'https' else 80)}"]

    @staticmethod
    def _id(value: Any) -> str:
        if not isinstance(value, str) or not _ID.fullmatch(value) or value in (".", ".."):
            raise ValueError("Invalid resource ID")
        return value

    @staticmethod
    def _repository_id(value: Any) -> str:
        """A repository is a legacy single ID or exactly two owner/repo segments.

        Never normalize or decode a remote identity: percent escapes, extra
        separators and dot segments are not repository names.
        """
        if not isinstance(value, str):
            raise ValueError("Invalid repository ID")
        parts = value.split("/")
        if not 1 <= len(parts) <= 2 or any(
            not _ID.fullmatch(part) or part in (".", "..") for part in parts
        ):
            raise ValueError("Invalid repository ID")
        return value

    @staticmethod
    def _key(value: Any) -> str:
        if not isinstance(value, str) or not _KEY.fullmatch(value):
            raise ValueError("Explicit idempotency key required")
        return value

    @staticmethod
    def _run_ids(value: Any) -> list[str]:
        if not isinstance(value, list) or not 1 <= len(value) <= 16:
            raise ValueError("Run IDs required")
        ids = [RepoevalFoundryTool._id(item) for item in value]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate run IDs")
        return ids

    @staticmethod
    def _approved(plan: dict[str, Any]) -> bool:
        checks = plan.get("policy_checks")
        placement = plan.get("placement")
        return (
            plan.get("state") == "planned"
            and isinstance(placement, dict)
            and placement.get("pool") == "staging"
            and isinstance(checks, dict)
            and all(
                checks.get(k) is True
                for k in ("repository", "image", "providers", "budget", "quota", "egress", "audit")
            )
        )

    def _reserved_start_ack(self, result: dict[str, Any], plan_id: str, key: str) -> bool:
        """Recognize only this plan/key's complete, inert coordinator reservation.

        HTTP 202 is not evidence that the scheduler saw a job. The expected
        parent and child identities are computed independently of the reply.
        """
        plan = self._plans[plan_id]
        execution = plan["workload"].get("execution")
        providers = plan["workload"].get("providers")
        if not isinstance(execution, dict) or not isinstance(providers, list):
            return False
        repetitions = execution.get("repetitions")
        if (
            type(repetitions) is not int
            or not 1 <= repetitions <= 32
            or not 1 <= len(providers) <= 8
        ):
            return False
        expected_run = "run-" + _identity_digest([self._project, plan_id, key])[:24]
        children = result.get("children")
        required = result.get("required_artifacts")
        artifact_policy = plan["workload"].get("artifacts")
        if (
            set(result)
            != {"id", "project_id", "plan_id", "state", "job_id", "children", "required_artifacts"}
            or result["id"] != expected_run
            or result["plan_id"] != plan_id
            or result["state"] != "reserved"
            or result["job_id"] is not None
            or not isinstance(children, list)
            or len(children) != len(providers) * repetitions
            or len(children) > 256
            or not isinstance(required, list)
            or len(required) > 32
            or any(not isinstance(kind, str) or not _KIND.fullmatch(kind) for kind in required)
            or len(set(required)) != len(required)
            or (
                isinstance(artifact_policy, dict)
                and required != artifact_policy.get("required_kinds")
            )
        ):
            return False
        expected = {(i, r) for i in range(len(providers)) for r in range(repetitions)}
        observed = set()
        for child in children:
            if not isinstance(child, dict) or set(child) != {
                "run_id",
                "provider_index",
                "repetition",
                "state",
                "job_id",
            }:
                return False
            i, r = child["provider_index"], child["repetition"]
            if (
                type(i) is not int
                or type(r) is not int
                or (i, r) not in expected
                or (i, r) in observed
                or child["run_id"] != "run-" + _identity_digest([expected_run, i, r])[:24]
                or child["state"] != "reserved"
                or child["job_id"] != f"foundry-{child['run_id']}"
            ):
                return False
            observed.add((i, r))
        return observed == expected

    def _read_contract(self, action: str, result: dict[str, Any], args: dict[str, Any]) -> bool:
        """Validate bounded read identities. Never infer server evidence from a 200."""
        try:
            if action == "artifacts":
                rows = result["artifacts"]
                if (
                    set(result) != {"project_id", "run_id", "artifacts"}
                    or result.get("run_id") != args["run_id"]
                    or not isinstance(rows, list)
                    or len(rows) > 100
                ):
                    return False
                seen = set()
                for row in rows:
                    if not isinstance(row, dict):
                        return False
                    if (
                        set(row)
                        != {"artifact_id", "run_id", "kind", "sha256", "size_bytes", "media_type"}
                        or not isinstance(row["artifact_id"], str)
                        or not _ARTIFACT_ID.fullmatch(row["artifact_id"])
                        or row["artifact_id"] in seen
                        or row["run_id"] != args["run_id"]
                        or not isinstance(row["kind"], str)
                        or not _KIND.fullmatch(row["kind"])
                        or not isinstance(row["sha256"], str)
                        or not _DIGEST.fullmatch(row["sha256"])
                        or type(row["size_bytes"]) is not int
                        or not 0 <= row["size_bytes"] <= 64_000_000
                        or not isinstance(row["media_type"], str)
                        or not 1 <= len(row["media_type"]) <= 255
                        or _SECRET_TEXT.search(row["media_type"])
                        or _URI.search(row["media_type"])
                    ):
                        return False
                    seen.add(row["artifact_id"])
                return True
            ids = args["run_ids"]
            if result.get("run_ids") != ids:
                return False
            if action == "compare":
                if (
                    set(result) != {"project_id", "comparison_id", "run_ids", "comparable", "pairs"}
                    or not isinstance(result.get("comparison_id"), str)
                    or not re.fullmatch(r"comparison-[0-9a-f]{24}", result["comparison_id"])
                ):
                    return False
                pairs = result.get("pairs")
                if not isinstance(pairs, list) or len(pairs) != len(ids) * (len(ids) - 1) // 2:
                    return False
                expected = {
                    frozenset((a, b)) for index, a in enumerate(ids) for b in ids[index + 1 :]
                }
                found = set()
                for row in pairs:
                    if not isinstance(row, dict):
                        return False
                    if set(row) != {
                        "left_run_id",
                        "right_run_id",
                        "comparable",
                        "key",
                        "reasons",
                    }:
                        return False
                    pair = frozenset((row["left_run_id"], row["right_run_id"]))
                    if pair not in expected or pair in found or type(row["comparable"]) is not bool:
                        return False
                    found.add(pair)
                    if (
                        not isinstance(row["reasons"], list)
                        or len(row["reasons"]) > 32
                        or any(not isinstance(x, str) or x not in _REASONS for x in row["reasons"])
                    ):
                        return False
                    if row["comparable"]:
                        if (
                            row["reasons"]
                            or not isinstance(row["key"], str)
                            or not _DIGEST.fullmatch(row["key"])
                        ):
                            return False
                    elif row["key"] is not None or not row["reasons"]:
                        return False
                return found == expected and result.get("comparable") is all(
                    p["comparable"] for p in pairs
                )
            if (
                set(result)
                != {
                    "project_id",
                    "report_id",
                    "status",
                    "published",
                    "run_ids",
                    "schema_version",
                    "groups",
                    "incomparable",
                }
                or not isinstance(result.get("report_id"), str)
                or not re.fullmatch(r"report-[0-9a-f]{24}", result["report_id"])
                or result.get("status") != "draft"
                or result.get("published") is not False
                or result.get("schema_version") != "1.0"
            ):
                return False
            groups, excluded = result.get("groups"), result.get("incomparable")
            if (
                not isinstance(groups, list)
                or not isinstance(excluded, list)
                or len(groups) + len(excluded) > len(ids)
            ):
                return False
            seen = set()
            for group in groups:
                if not isinstance(group, dict):
                    return False
                if set(group) != {"comparability_key", "run_ids"}:
                    return False
                if not isinstance(group["comparability_key"], str) or not _DIGEST.fullmatch(
                    group["comparability_key"]
                ):
                    return False
                if not isinstance(group["run_ids"], list) or not group["run_ids"]:
                    return False
                for run in group["run_ids"]:
                    if run not in ids or run in seen:
                        return False
                    seen.add(run)
            for row in excluded:
                if not isinstance(row, dict):
                    return False
                if (
                    set(row) != {"run_id", "reason"}
                    or row["run_id"] not in ids
                    or row["run_id"] in seen
                ):
                    return False
                if (
                    not isinstance(row["reason"], list)
                    or not row["reason"]
                    or len(row["reason"]) > 32
                    or any(
                        not isinstance(reason, str) or reason not in _REASONS
                        for reason in row["reason"]
                    )
                ):
                    return False
                seen.add(row["run_id"])
            return seen == set(ids)
        except (KeyError, TypeError, ValueError):
            return False

    def execute(self, *, action: str, **kwargs: Any) -> ToolResult:
        with self._auth_lock:
            if not self._available:
                return ToolResult(False, None, "Foundry HTTP API unavailable; no run was submitted")
            return self._execute_active(action=action, **kwargs)

    def _execute_active(self, *, action: str, **kwargs: Any) -> ToolResult:
        if not isinstance(action, str) or action not in _ACTIONS:
            return ToolResult(False, None, "Unsupported Foundry action")
        if not self._valid or not self._available:
            return ToolResult(False, None, "Foundry HTTP API unavailable; no run was submitted")
        try:
            method, path, body, headers = self._request(action, kwargs)
            if action == "list":
                # Failed refresh must not preserve a stale repository grant.
                self._repos, self._listed = set(), False
                self._plans.clear()
            if action == "plan":
                self._plans.clear()
            client = self._client
            if client is None or client.category != "tool":
                return ToolResult(False, None, "Authenticated Foundry transport unavailable")
            url = f"{self._base}/projects/{quote(self._project, safe='')}{path}"
            token = _read_token(self._token_file)
            headers["Authorization"] = f"Bearer {token}"
            response = (
                client.get_limited(url, _RESPONSE_LIMIT, headers=headers, follow_redirects=False)
                if method == "GET"
                else client.post_limited(
                    url, _RESPONSE_LIMIT, headers=headers, json=body, follow_redirects=False
                )
            )
            expected_status = 202 if action in ("snapshot", "start", "cancel") else 200
            if response.status_code != expected_status:
                return ToolResult(False, None, _FAIL)
            data = response.json()
            # The server is untrusted output. Refuse an exact credential echo
            # anywhere (including keys), even without a recognizable secret label.
            encoded = json.dumps(data, ensure_ascii=True, allow_nan=False)
            if len(encoded) > 1024 * 1024 or token in encoded:
                return ToolResult(False, None, _FAIL)
            if not isinstance(data, dict) or data.get("ok") is not True:
                return ToolResult(False, None, "Foundry response lacks project-bound evidence")
            result = data["data"]
            if action == "list":
                # PR1 returns a bare list of IDs. Its authenticated route checks
                # principal.project_id against the fixed project path. Do not
                # pretend the list itself contains a server project assertion.
                repos = result
                if not isinstance(repos, list) or len(repos) > 1000:
                    return ToolResult(False, None, "Foundry repository scope not verified")
                ids = [self._repository_id(repo) for repo in repos]
                if len(set(ids)) != len(ids):
                    return ToolResult(False, None, "Foundry repository scope not verified")
                self._repos, self._listed = set(ids), True
                return ToolResult(
                    True,
                    {
                        "repositories": ids,
                        "route_project_id": self._project,
                        "scope_source": "authenticated_project_route",
                    },
                )
            # Foundry's cancellation v1 payload has no project_id; scope there
            # comes from the authenticated, fixed project route and run ID.
            if not isinstance(result, dict) or (
                action != "cancel" and result.get("project_id") != self._project
            ):
                return ToolResult(False, None, "Foundry response project scope mismatch")
            if action in ("compare", "artifacts", "report") and not self._read_contract(
                action, result, kwargs
            ):
                return ToolResult(
                    False, None, "Foundry verified project-bound read contract missing"
                )
            if action == "plan":
                plan_id = self._id(result.get("id"))
                if not self._approved(result) or result.get("workload") != kwargs["workload"]:
                    self._plans.pop(plan_id, None)
                    return ToolResult(
                        False,
                        None,
                        "Foundry plan lacks reviewed staging placement, policy or matching workload evidence",
                    )
                self._plans[plan_id] = copy.deepcopy(result)
            if action == "status" and result.get("id") != kwargs["resource_id"]:
                return ToolResult(False, None, "Foundry resource identity mismatch")
            if action in ("snapshot", "start", "cancel") and (
                response.status_code != 202
                or not isinstance(result.get("id"), str)
                or not _ID.fullmatch(result["id"])
                or (
                    action == "snapshot"
                    and (
                        result.get("repository_id") != kwargs["repository_id"]
                        or result.get("commit_sha") != kwargs["commit_sha"]
                        or result.get("state") != "requested"
                    )
                )
                or (
                    action == "start"
                    and (
                        result.get("plan_id") != kwargs["plan_id"]
                        or not (
                            self._reserved_start_ack(
                                result, kwargs["plan_id"], kwargs["idempotency_key"]
                            )
                            if result.get("state") == "reserved"
                            else result.get("state") == "submitted"
                            and isinstance(result.get("job_id"), str)
                            and bool(_ID.fullmatch(result["job_id"]))
                            and result["job_id"] not in (".", "..")
                        )
                    )
                )
                or (
                    action == "cancel"
                    and (
                        result["id"] != kwargs["run_id"]
                        or result.get("state") not in ("cancelled", "cancel_pending", "failed")
                        or set(result) != {"id", "state", "cancel_requested"}
                        or result["cancel_requested"] is not True
                    )
                )
            ):
                return ToolResult(
                    False, None, "Mutation acknowledgement uncertain; check status before retrying"
                )
            if action in ("snapshot", "start", "cancel"):
                return ToolResult(
                    True,
                    {
                        "acknowledged": True,
                        "execution_complete": False,
                        "resource": _safe(result),
                        **(
                            {
                                "route_project_id": self._project,
                                "scope_source": "authenticated_project_route",
                            }
                            if action == "cancel"
                            else {}
                        ),
                    },
                )
            return ToolResult(True, _safe(result))
        except (KeyError, TypeError, ValueError):
            return ToolResult(False, None, "Invalid or unverified Foundry request/response")
        except Exception:
            # Exception messages and remote error bodies may contain credentials.
            return ToolResult(False, None, _FAIL)

    def _request(
        self, action: str, args: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any] | None, dict[str, str]]:
        fields = {
            "list": set(),
            "plan": {"workload"},
            "snapshot": {
                "repository_id",
                "commit_sha",
                "idempotency_key",
                "acknowledge_project_scope",
            },
            "start": {"plan_id", "idempotency_key", "acknowledge_project_scope"},
            "status": {"resource_type", "resource_id"},
            "compare": {"run_ids"},
            "artifacts": {"run_id"},
            "cancel": {"run_id"},
            "report": {"run_ids"},
        }
        if set(args) != fields[action]:
            raise ValueError("Unsupported or missing Foundry argument")
        headers = {"Accept": "application/json"}
        if action == "list":
            return "GET", "/repositories", None, headers
        if action == "plan":
            workload = args["workload"]
            if (
                not isinstance(workload, dict)
                or not _no_secrets(workload)
                or set(workload)
                - {
                    "schema_version",
                    "id",
                    "version",
                    "description",
                    "repository",
                    "task",
                    "providers",
                    "sandbox",
                    "execution",
                    "validation",
                    "artifacts",
                }
            ):
                raise ValueError("Invalid workload")
            repo, sandbox, execution = (
                workload.get(k) for k in ("repository", "sandbox", "execution")
            )
            if (
                not self._listed
                or not isinstance(repo, dict)
                or self._repository_id(repo.get("repository_id")) not in self._repos
                or not isinstance(repo.get("commit_sha"), str)
                or not _SHA.fullmatch(repo["commit_sha"])
                or not isinstance(repo.get("snapshot_id"), str)
                or not _ID.fullmatch(repo["snapshot_id"])
                or not isinstance(sandbox, dict)
                or not re.fullmatch(
                    r"[^@\s]+@sha256:[0-9a-f]{64}", str(sandbox.get("image_digest", ""))
                )
                or not isinstance(execution, dict)
                or any(
                    type(execution.get(k)) is not int or not 1 <= execution[k] <= bound
                    for k, bound in (
                        ("repetitions", 32),
                        ("parallelism", 8),
                        ("timeout_seconds", 86400),
                        ("max_attempts", 3),
                    )
                )
                or not isinstance(workload.get("providers"), list)
                or not 1 <= len(workload["providers"]) <= 8
            ):
                raise ValueError("Workload must be bounded and pinned")
            return "POST", "/benchmark/plan", {"workload": workload}, headers
        if action == "snapshot":
            repo = self._repository_id(args["repository_id"])
            if (
                not self._listed
                or repo not in self._repos
                or args["acknowledge_project_scope"] is not True
            ):
                raise ValueError(
                    "Registered repository and explicit project-scope acknowledgement required"
                )
            sha = args["commit_sha"]
            if not isinstance(sha, str) or not _SHA.fullmatch(sha):
                raise ValueError("Immutable commit required")
            headers["Idempotency-Key"] = self._key(args["idempotency_key"])
            return "POST", "/snapshots", {"repository_id": repo, "commit_sha": sha}, headers
        if action == "start":
            plan_id = self._id(args["plan_id"])
            if args["acknowledge_project_scope"] is not True or not self._approved(
                self._plans.get(plan_id, {})
            ):
                raise ValueError(
                    "Reviewed staging plan and explicit project-scope acknowledgement required"
                )
            headers["Idempotency-Key"] = self._key(args["idempotency_key"])
            return "POST", "/benchmark/start", {"plan_id": plan_id}, headers
        if action == "status":
            if args["resource_type"] not in ("run", "snapshot"):
                raise ValueError("Unsupported resource type")
            path = "runs" if args["resource_type"] == "run" else "snapshots"
            return "GET", f"/{path}/{quote(self._id(args['resource_id']), safe='')}", None, headers
        if action in ("compare", "report"):
            if action == "compare" and len(args["run_ids"]) < 2:
                raise ValueError("Comparison requires two runs")
            return "POST", f"/{action}", {"run_ids": self._run_ids(args["run_ids"])}, headers
        if action == "artifacts":
            return (
                "GET",
                f"/runs/{quote(self._id(args['run_id']), safe='')}/artifacts",
                None,
                headers,
            )
        if action == "cancel":
            return "POST", f"/runs/{quote(self._id(args['run_id']), safe='')}/cancel", {}, headers
        raise ValueError("Unsupported Foundry action")

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {"type": "string", "enum": sorted(_ACTIONS)},
                    "repository_id": {"type": "string"},
                    "commit_sha": {"type": "string"},
                    "workload": {"type": "object"},
                    "plan_id": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                    "acknowledge_project_scope": {
                        "type": "boolean",
                        "description": "Caller acknowledgement only; the authenticated Foundry permission is authoritative.",
                    },
                    "resource_type": {"type": "string", "enum": ["run", "snapshot"]},
                    "resource_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "run_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
                },
                "required": ["action"],
            },
        }


class _FoundrySurface(BaseTool):
    """Distinct policy identity, with shared project evidence and revocation.

    Deliberately do not inherit from RepoevalFoundryTool: doing so would expose
    its unrestricted execute method through a policy-granted read tool.
    """

    permissions = ToolPermissions(network=True)
    actions: frozenset[str] = frozenset()
    writes_state = False

    def __init__(self, client: RepoevalFoundryTool) -> None:
        self._foundry = client
        self.permissions = client.permissions

    def execute(self, *, action: str, **kwargs: Any) -> ToolResult:
        if not isinstance(action, str) or action not in self.actions:
            return ToolResult(False, None, "Unsupported Foundry action for this tool")
        if "self_approve" in kwargs:
            return ToolResult(
                False, None, "Use acknowledge_project_scope; the server authorizes mutations"
            )
        return self._foundry.execute(action=action, **kwargs)

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        if kwargs.get("action") not in self.actions:
            raise ValueError("Unsupported Foundry action for this tool")
        return self._foundry.resolve_network_hosts(kwargs)

    def matches_config(self, config: Any) -> bool:
        return self._foundry.matches_config(config)

    def revoke(self) -> None:
        self._foundry.revoke()

    def get_schema(self) -> dict[str, Any]:
        schema = self._foundry.get_schema()
        schema["name"] = self.name
        schema["description"] = self.description
        schema["parameters"]["properties"]["action"]["enum"] = sorted(self.actions)
        props = schema["parameters"]["properties"]
        if not self.writes_state:
            for key in ("idempotency_key", "acknowledge_project_scope", "commit_sha"):
                props.pop(key)
        else:
            for key in ("workload", "resource_type", "resource_id", "run_ids"):
                props.pop(key)
        return schema


class RepoevalFoundryReadTool(_FoundrySurface):
    name = "repoeval_foundry_read"
    description = "Read project repositories, stage a bounded plan without starting it, check run/snapshot status, compare runs and inspect artifact metadata or draft reports."
    actions = _READ_ACTIONS


class RepoevalFoundryMutateTool(_FoundrySurface):
    name = "repoeval_foundry_mutate"
    description = "Request project-scoped repository snapshots, start a reviewed staging plan, or cancel a run; requires explicit Foundry mutation permission."
    actions = _MUTATION_ACTIONS
    writes_state = True
