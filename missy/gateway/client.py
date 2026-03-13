"""Policy-enforcing HTTP client for the Missy framework.

:class:`PolicyHTTPClient` wraps :mod:`httpx` and enforces the active
:class:`~missy.policy.engine.PolicyEngine` network policy before issuing any
request.  If the destination host is not permitted, a
:class:`~missy.core.exceptions.PolicyViolationError` is raised before any
network I/O occurs.

Both synchronous and async request methods are provided.  The underlying
``httpx`` client instances are created lazily and reused across calls to
benefit from connection pooling.

Example::

    from missy.gateway.client import create_client

    client = create_client(session_id="s1", task_id="t1")
    response = client.get("https://api.github.com/zen")

Async example::

    import asyncio
    from missy.gateway.client import create_client

    async def main():
        client = create_client(session_id="s1", task_id="t1")
        response = await client.aget("https://api.github.com/zen")

    asyncio.run(main())
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from missy.core.events import AuditEvent, event_bus
from missy.policy.engine import get_policy_engine

logger = logging.getLogger(__name__)


class PolicyHTTPClient:
    """HTTP client that enforces network policy before making any request.

    Network policy is evaluated via the process-level
    :class:`~missy.policy.engine.PolicyEngine`; therefore
    :func:`~missy.policy.engine.init_policy_engine` must be called before
    any request method is used.

    The client owns both a synchronous :class:`httpx.Client` and an
    asynchronous :class:`httpx.AsyncClient`.  Both are created lazily on
    first use and closed when the instance is used as a context manager.

    Args:
        session_id: Session identifier forwarded to audit events and the
            policy engine.
        task_id: Task identifier forwarded to audit events and the policy
            engine.
        timeout: Default request timeout in seconds.  Applies to both
            synchronous and asynchronous requests.
    """

    def __init__(
        self,
        session_id: str = "",
        task_id: str = "",
        timeout: int = 30,
        category: str = "",
    ) -> None:
        self.session_id = session_id
        self.task_id = task_id
        self.timeout = timeout
        self.category = category
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Synchronous interface
    # ------------------------------------------------------------------

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform a synchronous HTTP GET after a policy check.

        Args:
            url: The target URL.
            **kwargs: Extra keyword arguments forwarded verbatim to
                :meth:`httpx.Client.get`.

        Returns:
            The HTTP response.

        Raises:
            PolicyViolationError: When the destination host is denied.
            httpx.HTTPError: On network or protocol errors.
        """
        self._check_url(url)
        response = self._get_sync_client().get(url, **kwargs)
        self._emit_request_event("GET", url, response.status_code)
        return response

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform a synchronous HTTP POST after a policy check.

        Args:
            url: The target URL.
            **kwargs: Extra keyword arguments forwarded verbatim to
                :meth:`httpx.Client.post`.

        Returns:
            The HTTP response.

        Raises:
            PolicyViolationError: When the destination host is denied.
            httpx.HTTPError: On network or protocol errors.
        """
        self._check_url(url)
        response = self._get_sync_client().post(url, **kwargs)
        self._emit_request_event("POST", url, response.status_code)
        return response

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform a synchronous HTTP PUT after a policy check."""
        self._check_url(url)
        response = self._get_sync_client().put(url, **kwargs)
        self._emit_request_event("PUT", url, response.status_code)
        return response

    # ------------------------------------------------------------------
    # Asynchronous interface
    # ------------------------------------------------------------------

    async def aget(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform an asynchronous HTTP GET after a policy check.

        Args:
            url: The target URL.
            **kwargs: Extra keyword arguments forwarded verbatim to
                :meth:`httpx.AsyncClient.get`.

        Returns:
            The HTTP response.

        Raises:
            PolicyViolationError: When the destination host is denied.
            httpx.HTTPError: On network or protocol errors.
        """
        self._check_url(url)
        response = await self._get_async_client().get(url, **kwargs)
        self._emit_request_event("GET", url, response.status_code)
        return response

    async def apost(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform an asynchronous HTTP POST after a policy check.

        Args:
            url: The target URL.
            **kwargs: Extra keyword arguments forwarded verbatim to
                :meth:`httpx.AsyncClient.post`.

        Returns:
            The HTTP response.

        Raises:
            PolicyViolationError: When the destination host is denied.
            httpx.HTTPError: On network or protocol errors.
        """
        self._check_url(url)
        response = await self._get_async_client().post(url, **kwargs)
        self._emit_request_event("POST", url, response.status_code)
        return response

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> PolicyHTTPClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> PolicyHTTPClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def close(self) -> None:
        """Close the underlying synchronous :class:`httpx.Client`.

        Safe to call even if the client was never created.
        """
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """Close the underlying asynchronous :class:`httpx.AsyncClient`.

        Safe to call even if the client was never created.
        """
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_url(self, url: str) -> None:
        """Extract the host from *url* and run a network policy check.

        Args:
            url: A fully-qualified URL string.

        Raises:
            PolicyViolationError: When the host is denied by the policy engine.
            ValueError: When the URL is malformed or contains no host component.
        """
        parsed = urlparse(url)
        host = parsed.hostname  # Returns None for malformed URLs; strips brackets from IPv6.
        if not host:
            raise ValueError(
                f"Cannot determine host from URL {url!r}. "
                "Ensure the URL includes a scheme (e.g. https://)."
            )
        get_policy_engine().check_network(
            host, self.session_id, self.task_id, category=self.category,
        )

    def _get_sync_client(self) -> httpx.Client:
        """Return the shared synchronous client, creating it on first call."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.timeout)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Return the shared async client, creating it on first call."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    def _emit_request_event(self, method: str, url: str, status_code: int) -> None:
        """Publish a successful HTTP request audit event.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, etc.).
            url: The request URL.
            status_code: The HTTP response status code.
        """
        event = AuditEvent.now(
            session_id=self.session_id,
            task_id=self.task_id,
            event_type="network_request",
            category="network",
            result="allow",
            detail={"method": method, "url": url, "status_code": status_code},
        )
        event_bus.publish(event)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_client(
    session_id: str = "",
    task_id: str = "",
    timeout: int = 30,
    category: str = "",
) -> PolicyHTTPClient:
    """Construct a :class:`PolicyHTTPClient` with the given parameters.

    This is the recommended way to create clients; it keeps calling code
    decoupled from the concrete constructor signature.

    Args:
        session_id: Session identifier forwarded to the policy engine and
            audit events.
        task_id: Task identifier forwarded to the policy engine and audit
            events.
        timeout: Default request timeout in seconds.
        category: Request category (``"provider"``, ``"tool"``,
            ``"discord"``) forwarded to the policy engine so per-category
            host allowlists are checked.

    Returns:
        A configured :class:`PolicyHTTPClient` instance.
    """
    return PolicyHTTPClient(
        session_id=session_id, task_id=task_id, timeout=timeout, category=category,
    )
