"""Standard topic constants for the Missy message bus.

Import these instead of using raw strings to avoid typos and enable
IDE-assisted discovery of available topics.

Topic naming convention: ``<subsystem>.<action>[.<detail>]``
"""

# ---------------------------------------------------------------------------
# Channel messages
# ---------------------------------------------------------------------------

#: A user message has been received by a channel.
CHANNEL_INBOUND = "channel.inbound"

#: A response is ready to send back through a channel.
CHANNEL_OUTBOUND = "channel.outbound"

# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------

#: An agent run has started.
AGENT_RUN_START = "agent.run.start"

#: An agent run completed successfully.
AGENT_RUN_COMPLETE = "agent.run.complete"

#: An agent run failed with an error.
AGENT_RUN_ERROR = "agent.run.error"

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

#: A tool call is about to be executed.
TOOL_REQUEST = "tool.request"

#: A tool call has completed (success or failure).
TOOL_RESULT = "tool.result"

# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

#: The system is starting up.
SYSTEM_STARTUP = "system.startup"

#: The system is shutting down.
SYSTEM_SHUTDOWN = "system.shutdown"

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

#: A security policy violation was detected.
SECURITY_VIOLATION = "security.violation"

#: A sensitive operation requires human approval.
SECURITY_APPROVAL_NEEDED = "security.approval.needed"

#: A human approval response was received.
SECURITY_APPROVAL_RESPONSE = "security.approval.response"

# ---------------------------------------------------------------------------
# Sleeptime computing
# ---------------------------------------------------------------------------

#: A background memory-processing cycle has started.
SLEEPTIME_CYCLE_START = "sleeptime.cycle.start"

#: A background memory-processing cycle completed successfully.
SLEEPTIME_CYCLE_COMPLETE = "sleeptime.cycle.complete"

#: An error occurred during a sleeptime processing cycle.
SLEEPTIME_ERROR = "sleeptime.error"
