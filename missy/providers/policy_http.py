"""Policy-aware ``httpx`` client factory for provider SDKs.

The Anthropic and OpenAI SDKs create their own ``httpx`` clients and issue
requests directly, bypassing :class:`~missy.gateway.client.PolicyHTTPClient`.
This module builds an :class:`httpx.Client` with a ``request`` event hook that
runs the process-level network policy check *before* each request leaves the
process, keeping provider egress consistent with the rest of Missy's outbound
HTTP (network policy + audit events).

The SDKs accept a custom client via their ``http_client=`` constructor
argument; :func:`build_policy_http_client` returns exactly such a client.

Design notes:

* The hook is fail-closed when policy bootstrap is absent or denies the
  destination.
* The exact IP validated by policy is pinned into the transport used by the
  SDK, closing the check/connect DNS-rebinding window.
* REST method/path policy is applied to provider traffic as well as ordinary
  gateway HTTP traffic.
"""

from __future__ import annotations

import logging
import posixpath

import httpx

from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError

logger = logging.getLogger(__name__)


def _emit_network_event(method: str, url: str, result: str, detail: str) -> None:
    """Publish a ``network_request`` audit event for provider egress."""
    try:
        event = AuditEvent.now(
            session_id="",
            task_id="",
            event_type="network_request",
            category="network",
            result=result,  # type: ignore[arg-type]
            detail={"method": method, "url": url, "source": "provider_sdk", "message": detail},
        )
        event_bus.publish(event)
    except Exception:  # pragma: no cover - audit must never break requests
        logger.exception("Failed to emit provider network audit event")


def _safe_request_url(url: httpx.URL) -> str:
    """Return a bounded URL without credentials, query data, or fragments."""
    host = url.host or "<missing-host>"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{url.port}" if url.port is not None else ""
    path = url.path or "/"
    return f"{url.scheme}://{host}{port}{path}"[:2048]


def _normalized_path(raw_path: str) -> str:
    path = posixpath.normpath(raw_path or "/")
    if raw_path.endswith("/") and not path.endswith("/"):
        path += "/"
    return path


def _policy_request_hook(request: httpx.Request) -> None:
    """httpx ``request`` event hook enforcing network policy on provider calls.

    Args:
        request: The outbound :class:`httpx.Request` about to be sent.

    Raises:
        PolicyViolationError: When the destination host is denied by the
            active network policy.
    """
    host = request.url.host
    method = request.method
    url = _safe_request_url(request.url)
    if not host:
        raise PolicyViolationError(
            "Provider request has no policy-checkable destination host.",
            category="network",
            detail="Provider SDK request URL did not contain a host.",
        )

    from missy.policy.engine import get_policy_engine

    try:
        engine = get_policy_engine()
    except RuntimeError as exc:
        _emit_network_event(method, url, "deny", "provider policy is not initialized")
        raise PolicyViolationError(
            "Provider request denied because network policy is not initialized.",
            category="network",
            detail="Provider SDK egress requires an initialized policy engine.",
        ) from exc

    try:
        _allowed, resolved_ip = engine.check_network_resolved(host, category="provider")
        path = _normalized_path(request.url.path)
        rest_result = engine.rest_policy.check(host, method, path)
        if rest_result == "deny":
            raise PolicyViolationError(
                f"REST policy denied provider request {method} {host}{path}",
                category="network",
                detail=f"REST rule denied provider request {method} {path} on {host}",
            )
        from missy.gateway.pinned_transport import pin_host

        pin_host(host, resolved_ip)
    except PolicyViolationError:
        _emit_network_event(method, url, "deny", f"network policy denied host {host}")
        logger.warning("Network policy denied provider request to %s", host)
        raise

    _emit_network_event(method, url, "allow", f"provider request to {host}")


def build_policy_http_client(timeout: float = 30.0) -> httpx.Client:
    """Return an :class:`httpx.Client` that enforces network policy per request.

    Args:
        timeout: Per-request timeout in seconds.

    Returns:
        An :class:`httpx.Client` with a policy-enforcing ``request`` event
        hook, suitable for passing to a provider SDK via ``http_client=``.
    """
    from missy.gateway.pinned_transport import PinnedHTTPTransport

    return httpx.Client(
        timeout=timeout,
        transport=PinnedHTTPTransport(),
        event_hooks={"request": [_policy_request_hook]},
    )
