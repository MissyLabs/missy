"""Finite RepoEval worker launch contract, not a production transport.

Only an approved, immutable digest is accepted from the sealed input mount.
Store access, independent approval, and a credential-custody provider broker are
injected capabilities. No production implementations are wired into the CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .contracts import canonical_json
from .provider import ProviderBroker
from .worker import WorkerRequest, execute_worker

INPUT = Path("/run/repoeval/input/manifest.sha256")
RESULT = Path("/run/repoeval/result/result.json")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RESULT = 4096


class CustodyStore(Protocol):
    """Trusted store: resolve an exact digest to canonical manifest and payloads.

    Its implementation must independently enforce immutable object identity,
    project access and snapshot custody. It must never take an arbitrary URL.
    """

    def resolve(self, digest: str) -> SealedBundle: ...


@dataclass(frozen=True)
class SealedBundle:
    """Trusted custody output. Settings must derive from approved run metadata.

    Neither repository_id nor an arbitrary provider in a manifest authorizes
    project access. The custody backend binds these settings to the run.
    """

    manifest_bytes: bytes
    payloads: Mapping[str, bytes]
    project_id: str
    registry_key: str
    model: str
    budget_microusd: int
    timeout_seconds: int
    token_cap: int


Approval = Callable[[str, Mapping[str, Any]], bool]


class LaunchRefused(Exception):
    """Only constant reason codes may be exposed, never exception details."""


def _read_digest(path: Path) -> str:
    # The containing directory MUST be a trusted sealed mount. Opening a
    # regular file without following the final symlink prevents path switching.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise LaunchRefused("invalid_reference")
            raw = stream.read(66)
        digest = raw.decode("ascii")
    except (OSError, UnicodeError):
        raise LaunchRefused("invalid_reference") from None
    if not (len(raw) == 64 or len(raw) == 65 and raw.endswith(b"\n")) or not _SHA.fullmatch(
        digest.rstrip("\n")
    ):
        raise LaunchRefused("invalid_reference")
    return digest.rstrip("\n")


def _request(digest: str, store: CustodyStore, approve: Approval) -> WorkerRequest:
    try:
        bundle = store.resolve(digest)
        if not isinstance(bundle, SealedBundle):
            raise LaunchRefused("custody_unavailable")
        raw, payloads = bundle.manifest_bytes, bundle.payloads
        if not isinstance(raw, bytes) or len(raw) > 32768 or not isinstance(payloads, Mapping):
            raise LaunchRefused("custody_unavailable")
        manifest = json.loads(
            raw.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError())
        )
        if (
            not isinstance(manifest, dict)
            or canonical_json(manifest) != raw
            or hashlib.sha256(raw).hexdigest() != digest
        ):
            raise LaunchRefused("custody_unavailable")
        # Do not trust manifest-declared approval, provider or execution limits.
        # A separate verifier must attest this exact digest and current rights.
        if approve(digest, manifest) is not True:
            raise LaunchRefused("approval_required")
        selected = manifest["providers"]
        if (
            not isinstance(selected, list)
            or len(selected) != 1
            or not isinstance(selected[0], dict)
            or selected[0].get("registry_key") != bundle.registry_key
            or selected[0].get("model") != bundle.model
        ):
            raise LaunchRefused("invalid_manifest")
        # Worker enforces ceilings; custody owns the project and model binding.
        return WorkerRequest(
            manifest=manifest,
            manifest_sha256=digest,
            payloads=payloads,
            project_id=bundle.project_id,
            registry_key=bundle.registry_key,
            model=bundle.model,
            budget_microusd=bundle.budget_microusd,
            timeout_seconds=bundle.timeout_seconds,
            token_cap=bundle.token_cap,
        )
    except LaunchRefused:
        raise
    except Exception:
        # Neither store failures nor credential/approval errors are printable.
        raise LaunchRefused("custody_unavailable") from None


def _write_result(path: Path, report: dict[str, Any]) -> None:
    encoded = canonical_json(report) + b"\n"
    if len(encoded) > _MAX_RESULT:
        raise LaunchRefused("result_unavailable")
    # Output directory must be a dedicated bounded writable mount, not a
    # shared directory. O_EXCL prevents overwriting a prior attempt.
    fd = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if fd is not None:
            os.close(fd)
        raise LaunchRefused("result_unavailable") from None


def run_once(
    *,
    store: CustodyStore | None = None,
    approve: Approval | None = None,
    broker: ProviderBroker | None = None,
    reference: Path = INPUT,
    result: Path = RESULT,
) -> int:
    """Return 0 for completed worker, 2 for refusal/failure. Paths are test seams.

    Production must supply independent trusted capabilities, enforce wallclock
    externally, and pin the exact paths. This is not a network transport.
    """
    if store is None or approve is None or broker is None:
        reason, completed = "backend_unavailable", False
    else:
        try:
            digest = _read_digest(reference)
            req = _request(digest, store, approve)
            worker_result = execute_worker(req, approve=approve, broker=broker)
            report = worker_result.report()
            completed = worker_result.status == "passed"
            reason = ""
        except LaunchRefused as exc:
            reason, completed = str(exc), False
        except Exception:
            reason, completed = "worker_unavailable", False
    if reason:
        report = {"status": "refused", "reason": reason}
    try:
        _write_result(result, report)
    except LaunchRefused:
        return 2
    return 0 if completed else 2


def main() -> int:
    # Deliberately no env-driven import hooks, dynamic plugins, credentials,
    # network, shell, or default no-op authorizer. Do not deploy this as a job.
    return run_once()


if __name__ == "__main__":
    raise SystemExit(main())
