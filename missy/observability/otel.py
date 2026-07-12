"""OpenTelemetry integration for Missy audit events."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OtelExporter:
    """Exports audit events to an OpenTelemetry collector.

    Wraps the event bus's ``publish()`` (mirroring
    :class:`~missy.observability.audit_logger.AuditLogger`'s established
    pattern -- see SR-4.6) so every published event, regardless of its
    ``event_type``, reaches the collector; the bus has no wildcard
    ``subscribe()`` and only ever notifies subscribers registered for the
    exact matching type.

    Requires: opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

    Args:
        endpoint: OTLP endpoint (e.g. "http://localhost:4317").
        protocol: "grpc" or "http/protobuf".
        service_name: Service name for span attributes.
        max_queue_size: Maximum spans the batch processor buffers before
            dropping the oldest (bounded so a stalled/unreachable
            collector cannot grow memory unboundedly).
        max_export_batch_size: Maximum spans sent per export call.
        schedule_delay_millis: Delay between batch export attempts.
        export_timeout_millis: Per-export network timeout.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        protocol: str = "grpc",
        service_name: str = "missy",
        max_queue_size: int = 2048,
        max_export_batch_size: int = 512,
        schedule_delay_millis: int = 5000,
        export_timeout_millis: int = 30000,
    ):
        self._endpoint = endpoint
        self._protocol = protocol
        self._service_name = service_name
        self._max_queue_size = max_queue_size
        self._max_export_batch_size = max_export_batch_size
        self._schedule_delay_millis = schedule_delay_millis
        self._export_timeout_millis = export_timeout_millis
        self._tracer = None
        self._enabled = False
        # SR-4.6: exporter failures were previously only logged at DEBUG
        # (invisible in normal operation). Tracked explicitly so callers
        # (e.g. `missy doctor`) can surface "OTLP export is silently
        # failing" rather than an operator only discovering it by noticing
        # the collector never received anything.
        self._export_failure_count = 0
        self._last_export_error: str | None = None
        self._bus: Any = None
        self._original_publish: Any = None
        self._setup()

    def _setup(self) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create({SERVICE_NAME: self._service_name})
            provider = TracerProvider(resource=resource)

            if self._protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=self._endpoint)

            provider.add_span_processor(
                BatchSpanProcessor(
                    exporter,
                    max_queue_size=self._max_queue_size,
                    max_export_batch_size=self._max_export_batch_size,
                    schedule_delay_millis=self._schedule_delay_millis,
                    export_timeout_millis=self._export_timeout_millis,
                )
            )
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self._service_name)
            self._enabled = True
            logger.info("OtelExporter: connected to %s", self._endpoint)
        except ImportError:
            logger.debug(
                "OtelExporter: opentelemetry packages not installed; OTLP export disabled. "
                "Install: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc"
            )
        except Exception as exc:
            logger.warning("OtelExporter: setup failed: %s", exc)

    def export_event(self, event: dict) -> None:
        """Export a single audit event as an OTLP span.

        SR-4.6: the event's ``detail`` is redacted (via the same
        :func:`~missy.observability.audit_logger._redact_detail` SR-1.10
        already applies before writing audit events to disk) before any
        value becomes a span attribute -- an unredacted export would
        otherwise leak the exact secrets SR-1.10 closed off for the
        on-disk audit log, just through the OTLP path instead.
        """
        if not self._enabled or not self._tracer:
            return
        try:
            from missy.observability.audit_logger import _redact_detail

            with self._tracer.start_as_current_span(event.get("event_type", "missy.event")) as span:
                for key, value in event.items():
                    if key == "detail" and isinstance(value, dict):
                        for k, v in _redact_detail(value).items():
                            span.set_attribute(f"missy.{k}", str(v))
                    elif isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"missy.{key}", value)
        except Exception as exc:
            self._export_failure_count += 1
            self._last_export_error = str(exc)
            logger.warning("OtelExporter: export failed (%d total failures): %s", self._export_failure_count, exc)

    def subscribe(self) -> None:
        """Attach to the event bus so every published event is exported.

        SR-4.6: previously called ``event_bus.subscribe(_handler)`` --
        but :meth:`~missy.core.events.EventBus.subscribe` requires an
        ``event_type`` string *and* a callback (``subscribe(event_type,
        callback)``), and has no wildcard/catch-all mode. That call
        always raised ``TypeError``, which was caught and merely logged
        as a warning -- meaning OTLP export silently received zero
        events in every configuration, regardless of ``otel_enabled``.
        Fixed by wrapping ``publish()`` itself (the same pattern
        :class:`~missy.observability.audit_logger.AuditLogger` already
        uses to capture every event without per-type registration).
        """
        try:
            from missy.core.events import event_bus

            self._bus = event_bus
            self._original_publish = event_bus.publish

            original_publish = self._original_publish

            def _patched_publish(evt: object) -> None:
                original_publish(evt)
                try:
                    self.export_event(evt.__dict__ if hasattr(evt, "__dict__") else {})
                except Exception:
                    logger.debug("OtelExporter: export_event raised during publish hook", exc_info=True)

            event_bus.publish = _patched_publish
            logger.debug("OtelExporter: subscribed to event bus (publish wrapped)")
        except Exception as exc:
            logger.warning("OtelExporter: subscribe failed: %s", exc)

    def unsubscribe(self) -> None:
        """Undo :meth:`subscribe`'s ``publish()`` wrap, restoring the prior function.

        Required before installing a *new* exporter's wrapper (e.g. on a
        config hot-reload) -- without this, each re-init stacks another
        layer of ``_patched_publish`` around the bus's real ``publish()``,
        so a single event would be exported once per historical reload
        rather than once.
        """
        if self._bus is not None and self._original_publish is not None:
            try:
                if self._bus.publish is not self._original_publish:
                    self._bus.publish = self._original_publish
            except Exception:
                logger.debug("OtelExporter: unsubscribe failed", exc_info=True)
            finally:
                self._bus = None
                self._original_publish = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def export_failure_count(self) -> int:
        """Total export attempts that raised an exception since construction."""
        return self._export_failure_count

    @property
    def last_export_error(self) -> str | None:
        """The most recent export failure message, or ``None`` if none occurred."""
        return self._last_export_error


def _disabled_stub() -> OtelExporter:
    """Return a fully-attributed disabled OtelExporter, skipping `_setup()`.

    Pre-existing bug fixed alongside SR-4.6: the previous
    ``OtelExporter.__new__(OtelExporter)`` skipped ``__init__`` entirely,
    leaving *no* instance attributes set at all -- touching
    ``.is_enabled`` (or any other attribute) on the result raised
    ``AttributeError`` immediately. This constructs the same "disabled,
    no real SDK/network setup performed" object, but with every
    attribute a real caller might read safely initialised.
    """
    stub = OtelExporter.__new__(OtelExporter)
    stub._endpoint = ""
    stub._protocol = ""
    stub._service_name = ""
    stub._max_queue_size = 0
    stub._max_export_batch_size = 0
    stub._schedule_delay_millis = 0
    stub._export_timeout_millis = 0
    stub._tracer = None
    stub._enabled = False
    stub._export_failure_count = 0
    stub._last_export_error = None
    stub._bus = None
    stub._original_publish = None
    return stub


_active_exporter: OtelExporter | None = None


def get_active_exporter() -> OtelExporter | None:
    """Return the most recently installed exporter, or ``None`` if
    :func:`init_otel` has never been called.

    Lets a long-lived caller (e.g. a future ``missy doctor``/Web TUI check
    running *inside* the same process as the live exporter) read
    ``is_enabled``/``export_failure_count``/``last_export_error`` off the
    actual exporter currently wired to the event bus, rather than a
    throwaway instance a separate CLI invocation would otherwise have to
    construct from scratch (which would misleadingly always report zero
    failures, having never handled a real event).
    """
    return _active_exporter


def init_otel(config) -> OtelExporter:
    """Initialise OpenTelemetry from config. Returns exporter (may be disabled).

    Safe to call more than once for the same process -- e.g. from
    :func:`missy.config.hotreload._apply_config` when the operator changes
    ``observability.otel_enabled``/``otel_endpoint``/``otel_protocol`` on a
    running ``missy gateway start`` daemon. Any previously active exporter
    has its ``publish()`` wrapper unwound first, so re-init never stacks a
    second layer of export around the same event.
    """
    global _active_exporter

    if _active_exporter is not None:
        _active_exporter.unsubscribe()
        _active_exporter = None

    obs = getattr(config, "observability", None)
    if obs is None or not getattr(obs, "otel_enabled", False):
        _active_exporter = _disabled_stub()
        return _active_exporter

    exporter = OtelExporter(
        endpoint=obs.otel_endpoint,
        protocol=obs.otel_protocol,
        service_name=obs.otel_service_name,
    )
    if exporter.is_enabled:
        exporter.subscribe()
    _active_exporter = exporter
    return exporter
