"""Pin policy-validated DNS resolutions to the actual TCP connection.

SR-1.9b: :meth:`~missy.policy.network.NetworkPolicyEngine.check_host` (and
its rebinding defense added for SR-1.9a) resolves a hostname, validates the
resulting IP is not private/loopback/link-local/reserved, and then discards
that IP. The caller that actually opens the connection --
:class:`~missy.gateway.client.PolicyHTTPClient`, via ``httpx``/``httpcore``
-- performs its *own*, independent DNS resolution when it connects. Between
the check and the connect, a low-TTL DNS record (attacker-controlled or
otherwise) can return a different address: a public IP at check time, then
``169.254.169.254`` (cloud metadata) or an internal address at connect time.
This is a classic check-then-use TOCTOU, not fixable by resolving more
carefully at check time alone -- the validated *resolution* has to be what
the connection actually uses.

The fix binds the connection to the specific IP the policy check validated,
via a custom ``httpcore`` network backend that substitutes the pinned IP for
the hostname at the TCP-connect layer only. TLS SNI and certificate
verification are unaffected: ``httpcore`` passes the original hostname as
``server_hostname`` for the TLS handshake independently of what address the
socket actually connects to, and the ``Host`` header is built from the
original request URL, never from the transport layer. Only the raw
``socket.create_connection()``/``anyio.connect_tcp()`` target address
changes -- exactly the address that already passed policy.
"""

from __future__ import annotations

import contextvars
import typing
from typing import Any

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend
from httpcore._backends.sync import SyncBackend

#: Per-request pinned IPs, keyed by hostname. A contextvar (not a plain
#: dict/thread-local) so it is correctly isolated per async task as well as
#: per thread -- concurrent requests to different hosts on the same shared
#: httpx.Client/AsyncClient instance must never see each other's pins.
_pinned_ips: contextvars.ContextVar[dict[str, str | None] | None] = contextvars.ContextVar(
    "missy_pinned_ips", default=None
)


def pin_host(host: str, ip: str | None) -> None:
    """Record the policy-validated IP for *host* for the next connection.

    Must be called immediately before the ``httpx`` request that will
    connect to *host*, in the same thread/async task -- the pin is
    consumed (and left in place, in case of retries within the same
    request) by :class:`_PinnedSyncBackend`/:class:`_PinnedAsyncBackend`'s
    ``connect_tcp``.

    Args:
        host: The hostname the upcoming request targets (as it will
            appear in the request URL -- case is not normalised here,
            callers should pass it exactly as ``httpx`` will see it).
        ip: The validated IP to pin the connection to, or ``None`` when
            no IP could be resolved (only valid in ``default_deny=False``
            policy mode, where there is no security boundary to enforce
            and the connection is allowed to fall back to normal,
            unpinned resolution).
    """
    current = dict(_pinned_ips.get() or {})
    current[host] = ip
    _pinned_ips.set(current)


def clear_pin(host: str) -> None:
    """Remove any pin for *host*, e.g. after the request completes."""
    current = dict(_pinned_ips.get() or {})
    current.pop(host, None)
    _pinned_ips.set(current)


class _PinnedSyncBackend(httpcore.NetworkBackend):
    """Substitutes the pinned IP for the hostname at TCP-connect time."""

    def __init__(self) -> None:
        self._inner = SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        pins = _pinned_ips.get() or {}
        if host not in pins:
            # Fail closed: every connection through this transport must
            # be preceded by PolicyHTTPClient._check_url() pinning an IP
            # (even a None "no boundary to enforce" pin in default-allow
            # mode). Reaching this with no pin at all means the policy
            # check was bypassed somewhere -- never fall back to a silent,
            # unvalidated DNS resolution.
            raise httpcore.ConnectError(
                f"Refusing to connect to {host!r}: no policy-validated IP "
                "pinned for this request. This connection did not go "
                "through PolicyHTTPClient._check_url()."
            )
        pinned_ip = pins[host]
        target = pinned_ip if pinned_ip is not None else host
        return self._inner.connect_tcp(
            target,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        return self._inner.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)


class _PinnedAsyncBackend(httpcore.AsyncNetworkBackend):
    """Async counterpart of :class:`_PinnedSyncBackend`."""

    def __init__(self) -> None:
        self._inner = AutoBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        pins = _pinned_ips.get() or {}
        if host not in pins:
            raise httpcore.ConnectError(
                f"Refusing to connect to {host!r}: no policy-validated IP "
                "pinned for this request. This connection did not go "
                "through PolicyHTTPClient._check_url()."
            )
        pinned_ip = pins[host]
        target = pinned_ip if pinned_ip is not None else host
        return await self._inner.connect_tcp(
            target,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return await self._inner.connect_unix_socket(
            path, timeout=timeout, socket_options=socket_options
        )


class PinnedHTTPTransport(httpx.HTTPTransport):
    """``httpx.HTTPTransport`` whose connections are pinned via :func:`pin_host`.

    Identical to the stock transport in every other respect (TLS, HTTP/1.1
    vs HTTP/2, connection pooling/limits, retries) -- only the underlying
    ``httpcore.ConnectionPool``'s ``network_backend`` differs, which
    ``httpx.HTTPTransport`` does not expose as a constructor parameter, so
    this subclass rebuilds the pool itself with the same arguments plus the
    pinned backend.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Replace the pool __init__ built with a pool using our pinned
        # backend, preserving every other setting it already computed.
        self._pool = httpcore.ConnectionPool(
            ssl_context=self._pool._ssl_context,
            max_connections=self._pool._max_connections,
            max_keepalive_connections=self._pool._max_keepalive_connections,
            keepalive_expiry=self._pool._keepalive_expiry,
            http1=self._pool._http1,
            http2=self._pool._http2,
            retries=self._pool._retries,
            network_backend=_PinnedSyncBackend(),
        )


class PinnedAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """Async counterpart of :class:`PinnedHTTPTransport`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=self._pool._ssl_context,
            max_connections=self._pool._max_connections,
            max_keepalive_connections=self._pool._max_keepalive_connections,
            keepalive_expiry=self._pool._keepalive_expiry,
            http1=self._pool._http1,
            http2=self._pool._http2,
            retries=self._pool._retries,
            network_backend=_PinnedAsyncBackend(),
        )
