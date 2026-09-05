"""Built-in tool: fetch a URL via HTTP GET.

Requires network policy approval before the request is made.  All requests
are routed through :class:`~missy.gateway.client.PolicyHTTPClient`, which
enforces the active policy engine's network restrictions.

Example::

    from missy.tools.builtin.web_fetch import WebFetchTool

    tool = WebFetchTool()
    result = tool.execute(url="https://example.com")
    assert result.success
"""

from __future__ import annotations

from typing import Any

from missy.security.sanitizer import sanitizer as input_sanitizer
from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_MAX_RESPONSE_BYTES = 65_536  # 64 KB
_DEFAULT_TIMEOUT = 30
_LOW_CONFIDENCE_HTML_COMMENT_PATTERN = r"<!--(?:(?!-->)[\s\S])*-->"
_SECURITY_BLOCK = (
    "[SECURITY BLOCK: The fetched response body was omitted because it "
    "contained prompt-injection-like instructions.]"
)
_SCAN_FAILURE_BLOCK = (
    "[SECURITY BLOCK: The fetched response body was omitted because its "
    "prompt-injection security scan failed.]"
)


class WebFetchTool(BaseTool):
    """Fetch a URL via HTTP GET through the policy-enforcing HTTP client.

    The destination host must be permitted by the active network policy.
    Response bodies larger than 64 KB are truncated.

    Attributes:
        name: ``"web_fetch"``
        description: One-line description for function-calling schemas.
        permissions: ``network=True``; all other flags ``False``.
    """

    name = "web_fetch"
    description = (
        "Fetch a URL via HTTP GET. The destination must be permitted by the network policy."
    )
    permissions = ToolPermissions(network=True)

    #: Headers that must be stripped from user-supplied request headers
    #: to prevent Host injection, credential forwarding, or proxy manipulation.
    _BLOCKED_HEADERS = frozenset(
        {
            "host",
            "authorization",
            "cookie",
            "x-forwarded-for",
            "x-forwarded-host",
            "x-forwarded-proto",
            "x-real-ip",
            "proxy-authorization",
        }
    )

    def execute(
        self,
        *,
        url: str,
        timeout: int = _DEFAULT_TIMEOUT,
        headers: dict[str, str] | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        """Perform an HTTP GET request against *url*.

        Args:
            url: The fully-qualified URL to fetch (must include a scheme,
                e.g. ``https://``).
            timeout: Request timeout in seconds (default: 30).
            headers: Optional mapping of additional HTTP request headers.

        Returns:
            :class:`~missy.tools.base.ToolResult` with:

            * ``success=True`` when the server returns a 2xx status, with
              ``output`` set to the response text (truncated to 64 KB).
            * ``success=False`` with ``error`` describing the HTTP status
              or any network/policy exception otherwise.
        """
        try:
            from missy.gateway.client import create_client

            http = create_client(session_id="web_fetch_tool", task_id="fetch", timeout=timeout)
            request_kwargs: dict[str, Any] = {}
            if headers:
                safe_headers = {
                    k: v for k, v in headers.items() if k.lower() not in self._BLOCKED_HEADERS
                }
                if safe_headers:
                    request_kwargs["headers"] = safe_headers

            response = http.get(url, **request_kwargs)
            content = response.text
            if len(content.encode("utf-8", errors="replace")) > _MAX_RESPONSE_BYTES:
                # Truncate by character count as a close approximation.
                content = content[:_MAX_RESPONSE_BYTES] + "\n[Response truncated]"

            security_flags: list[str] = []
            try:
                matches = input_sanitizer.check_for_injection(content)
                high_confidence_matches = [
                    match for match in matches if match != _LOW_CONFIDENCE_HTML_COMMENT_PATTERN
                ]
                if high_confidence_matches:
                    security_flags.append("prompt_injection")
                    content = _SECURITY_BLOCK
            except Exception:
                security_flags.append("prompt_injection_scan_failed")
                content = _SCAN_FAILURE_BLOCK

            status = response.status_code
            success = 200 <= status < 300
            error = f"HTTP {status}" if not success else None
            return ToolResult(
                success=success,
                output=content,
                error=error,
                security_flags=security_flags,
            )
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    def get_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for this tool's parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (must include a scheme, e.g. https://).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": f"Request timeout in seconds (default: {_DEFAULT_TIMEOUT}).",
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP request headers as a key/value mapping.",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["url"],
            },
        }
