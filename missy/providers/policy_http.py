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

* The hook is **fail-closed** when the policy engine is initialised: a denied
  host raises :class:`~missy.core.exceptions.PolicyViolationError` before any
  bytes are sent.
* The hook is **defensive at startup**: if the policy engine has not yet been
  initialised, the request is allowed (with a debug log) so provider
  construction never crashes.  It does *not* silently skip the check once the
  engine exists.
"""

from __future__ import annotations

import logging

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
    url = str(request.url)
    if not host:
        return

    from missy.policy.engine import get_policy_engine

    try:
        engine = get_policy_engine()
    except RuntimeError:
        # Policy engine not initialised yet (e.g. provider used before
        # runtime bootstrap). Allow but log — do not crash provider startup.
        logger.debug("Policy engine not initialised; allowing provider request to %s", host)
        return

    try:
        engine.check_network(host, category="provider")
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
    return httpx.Client(
        timeout=timeout,
        event_hooks={"request": [_policy_request_hook]},
    )
