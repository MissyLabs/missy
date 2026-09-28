"""Opt-in, project-scoped Foundry client with request-time file credentials."""

from __future__ import annotations

import copy
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
    "capabilities",
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
_SECRET = re.compile(
    r"(?i)(?<![a-z0-9])(?:token|password|secret|api[_-]?key|authorization|cookie|credential|private[_-]?key|url|uri|href|location|stdout|stderr|raw[_-]?output|environment|env|content)(?![a-z0-9])"
)
_SECRET_TEXT = re.compile(
    r"(?i)\b(?:bearer\s+\S+|(?:token|password|secret|api[_-]?key)\s*[:=]\s*\S+|sk-[A-Za-z0-9_-]{3,})"
)
_SAFE_SETTING_KEYS = frozenset({"token_budget", "max_tokens"})
_FAIL = "Foundry request refused or unavailable; no execution is confirmed"
_RESPONSE_LIMIT = 1024 * 1024


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
    description = "Authenticated project-scoped Foundry list/status, bounded staging plan/snapshot/start/cancel. Other actions fail closed until their wire contracts exist."
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

    def execute(self, *, action: str, **kwargs: Any) -> ToolResult:
        with self._auth_lock:
            if not self._available:
                return ToolResult(False, None, "Foundry HTTP API unavailable; no run was submitted")
            return self._execute_active(action=action, **kwargs)

    def _execute_active(self, *, action: str, **kwargs: Any) -> ToolResult:
        if not isinstance(action, str) or action not in _ACTIONS:
            return ToolResult(False, None, "Unsupported Foundry action")
        if action == "capabilities":
            return ToolResult(False, None, "Foundry capabilities route unavailable")
        if action in ("compare", "artifacts", "report"):
            return ToolResult(
                False, None, "Foundry project-bound response contract unavailable for this action"
            )
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
            if not isinstance(result, dict) or result.get("project_id") != self._project:
                return ToolResult(False, None, "Foundry response project scope mismatch")
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
            if action == "artifacts" and result.get("resource_id") != kwargs["run_id"]:
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
                        or result.get("state") != "submitted"
                        or not isinstance(result.get("job_id"), str)
                        or not result["job_id"]
                    )
                )
                or (
                    action == "cancel"
                    and (result["id"] != kwargs["run_id"] or result.get("state") != "cancelled")
                )
            ):
                return ToolResult(
                    False, None, "Mutation acknowledgement uncertain; check status before retrying"
                )
            if action in ("snapshot", "start", "cancel"):
                return ToolResult(
                    True,
                    {"acknowledged": True, "execution_complete": False, "resource": _safe(result)},
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
            "capabilities": set(),
            "list": set(),
            "plan": {"workload"},
            "snapshot": {"repository_id", "commit_sha", "idempotency_key", "self_approve"},
            "start": {"plan_id", "idempotency_key", "self_approve"},
            "status": {"resource_type", "resource_id"},
            "compare": {"run_ids"},
            "artifacts": {"run_id"},
            "cancel": {"run_id", "idempotency_key"},
            "report": {"run_ids"},
        }
        if set(args) != fields[action]:
            raise ValueError("Unsupported or missing Foundry argument")
        headers = {"Accept": "application/json"}
        if action == "capabilities":
            raise ValueError("Foundry capabilities route unavailable")
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
            if not self._listed or repo not in self._repos or args["self_approve"] is not True:
                raise ValueError(
                    "Registered repository and explicit bounded self approval required"
                )
            sha = args["commit_sha"]
            if not isinstance(sha, str) or not _SHA.fullmatch(sha):
                raise ValueError("Immutable commit required")
            headers["Idempotency-Key"] = self._key(args["idempotency_key"])
            return "POST", "/snapshots", {"repository_id": repo, "commit_sha": sha}, headers
        if action == "start":
            plan_id = self._id(args["plan_id"])
            if args["self_approve"] is not True or not self._approved(self._plans.get(plan_id, {})):
                raise ValueError(
                    "Reviewed staging plan and explicit bounded self approval required"
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
            headers["Idempotency-Key"] = self._key(args["idempotency_key"])
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
                    "self_approve": {"type": "boolean"},
                    "resource_type": {"type": "string", "enum": ["run", "snapshot"]},
                    "resource_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "run_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
                },
                "required": ["action"],
            },
        }
