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

    #: Default maximum response body size (50 MB).  Responses larger than
    #: this are rejected to prevent memory exhaustion from malicious servers.
    DEFAULT_MAX_RESPONSE_BYTES: int = 50 * 1024 * 1024

    def __init__(
        self,
        session_id: str = "",
        task_id: str = "",
        timeout: int = 30,
        category: str = "",
        max_response_bytes: int = 0,
    ) -> None:
        self.session_id = session_id
        self.task_id = task_id
        self.timeout = timeout
        self.category = category
        self.max_response_bytes = max_response_bytes or self.DEFAULT_MAX_RESPONSE_BYTES
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
        response = self._get_sync_client().get(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
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
        response = self._get_sync_client().post(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("POST", url, response.status_code)
        return response

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform a synchronous HTTP PUT after a policy check."""
        self._check_url(url)
        response = self._get_sync_client().put(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("PUT", url, response.status_code)
        return response

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform a synchronous HTTP DELETE after a policy check."""
        self._check_url(url)
        response = self._get_sync_client().delete(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("DELETE", url, response.status_code)
        return response

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform a synchronous HTTP PATCH after a policy check."""
        self._check_url(url)
        response = self._get_sync_client().patch(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("PATCH", url, response.status_code)
        return response

    def head(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform a synchronous HTTP HEAD after a policy check."""
        self._check_url(url)
        response = self._get_sync_client().head(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("HEAD", url, response.status_code)
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
        response = await self._get_async_client().get(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
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
        response = await self._get_async_client().post(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("POST", url, response.status_code)
        return response

    async def adelete(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform an asynchronous HTTP DELETE after a policy check."""
        self._check_url(url)
        response = await self._get_async_client().delete(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("DELETE", url, response.status_code)
        return response

    async def apatch(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform an asynchronous HTTP PATCH after a policy check."""
        self._check_url(url)
        response = await self._get_async_client().patch(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("PATCH", url, response.status_code)
        return response

    async def aput(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform an asynchronous HTTP PUT after a policy check."""
        self._check_url(url)
        response = await self._get_async_client().put(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("PUT", url, response.status_code)
        return response

    async def ahead(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform an asynchronous HTTP HEAD after a policy check."""
        self._check_url(url)
        response = await self._get_async_client().head(url, **self._sanitize_kwargs(kwargs))
        self._check_response_size(response, url)
        self._emit_request_event("HEAD", url, response.status_code)
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

    _ALLOWED_SCHEMES = {"http", "https"}

    def _check_url(self, url: str) -> None:
        """Extract the host from *url* and run a network policy check.

        Args:
            url: A fully-qualified URL string.

        Raises:
            PolicyViolationError: When the host is denied by the policy engine.
            ValueError: When the URL is malformed, uses a disallowed scheme,
                or contains no host component.
        """
        parsed = urlparse(url)
        if parsed.scheme not in self._ALLOWED_SCHEMES:
            raise ValueError(
                f"Unsupported URL scheme {parsed.scheme!r}. "
                "Only http:// and https:// are permitted."
            )
        host = parsed.hostname  # Returns None for malformed URLs; strips brackets from IPv6.
        if not host:
            raise ValueError(
                f"Cannot determine host from URL {url!r}. "
                "Ensure the URL includes a scheme (e.g. https://)."
            )
        get_policy_engine().check_network(
            host,
            self.session_id,
            self.task_id,
            category=self.category,
        )

    #: Kwargs that are safe to pass through to httpx request methods.
    #: Everything else is stripped to prevent security bypass (e.g.
    #: ``verify=False`` to disable TLS, ``base_url`` to redirect traffic,
    #: ``transport`` to bypass the policy layer, ``auth`` to inject creds).
    _ALLOWED_KWARGS = frozenset({
        "headers",
        "params",
        "data",
        "json",
        "content",
        "cookies",
        "timeout",
        "files",
        "extensions",
    })

    @classmethod
    def _sanitize_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Allow only safe kwargs; strip everything else."""
        return {k: v for k, v in kwargs.items() if k in cls._ALLOWED_KWARGS}

    #: Explicit connection pool limits to prevent resource exhaustion.
    _POOL_LIMITS = httpx.Limits(
        max_connections=20,
        max_keepalive_connections=10,
        keepalive_expiry=30,  # seconds
    )

    def _get_sync_client(self) -> httpx.Client:
        """Return the shared synchronous client, creating it on first call."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=False,
                limits=self._POOL_LIMITS,
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Return the shared async client, creating it on first call."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=False,
                limits=self._POOL_LIMITS,
            )
        return self._async_client

    def _check_response_size(self, response: httpx.Response, url: str) -> None:
        """Reject responses exceeding the configured size limit.

        Checks the ``Content-Length`` header first (fast path).  When no
        ``Content-Length`` is present (e.g. chunked transfer-encoding), falls
        back to checking the actual body length after it has been buffered by
        httpx.

        Raises:
            ValueError: When the response body exceeds ``max_response_bytes``.
        """
        headers = getattr(response, "headers", None)
        if headers is None:
            return

        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except (ValueError, TypeError):
                size = 0
            if size > self.max_response_bytes:
                raise ValueError(
                    f"Response from {url} too large: "
                    f"{size} bytes > {self.max_response_bytes} limit"
                )
        else:
            # No Content-Length header (chunked encoding, HTTP/1.0, etc.) —
            # check the actual buffered body length as a fallback.
            try:
                body_len = len(response.content)
            except Exception:
                return
            if body_len > self.max_response_bytes:
                raise ValueError(
                    f"Response from {url} too large: "
                    f"{body_len} bytes > {self.max_response_bytes} limit"
                )

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
        session_id=session_id,
        task_id=task_id,
        timeout=timeout,
        category=category,
    )
