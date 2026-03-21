"""Missy Agent-as-a-Service REST API.

Exposes :class:`ApiServer` and :class:`ApiConfig` as the primary public
interface.  Start the server with a minimal runtime::

    from missy.api import ApiConfig, ApiServer
    from missy.agent.runtime import AgentRuntime, AgentConfig

    runtime = AgentRuntime(AgentConfig())
    server = ApiServer(ApiConfig(api_key="changeme"), runtime=runtime)
    server.start()
    # Available at http://127.0.0.1:8080/api/v1/
    server.stop()
"""

from missy.api.server import ApiConfig, ApiServer

__all__ = ["ApiConfig", "ApiServer"]
