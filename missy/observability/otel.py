"""OpenTelemetry integration for Missy audit events."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class OtelExporter:
    """Exports audit events to an OpenTelemetry collector.

    Subscribes to the event bus and forwards events as OTLP spans.
    Requires: opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

    Args:
        endpoint: OTLP endpoint (e.g. "http://localhost:4317").
        protocol: "grpc" or "http/protobuf".
        service_name: Service name for span attributes.
    """

    def __init__(self, endpoint: str = "http://localhost:4317", protocol: str = "grpc", service_name: str = "missy"):
        self._endpoint = endpoint
        self._protocol = protocol
        self._service_name = service_name
        self._tracer = None
        self._enabled = False
        self._setup()

    def _setup(self) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME

            resource = Resource.create({SERVICE_NAME: self._service_name})
            provider = TracerProvider(resource=resource)

            if self._protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                exporter = OTLPSpanExporter(endpoint=self._endpoint)
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                exporter = OTLPSpanExporter(endpoint=self._endpoint)

            provider.add_span_processor(BatchSpanProcessor(exporter))
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
        """Export a single audit event as an OTLP span."""
        if not self._enabled or not self._tracer:
            return
        try:
            with self._tracer.start_as_current_span(event.get("event_type", "missy.event")) as span:
                for key, value in event.items():
                    if key == "detail" and isinstance(value, dict):
                        for k, v in value.items():
                            span.set_attribute(f"missy.{k}", str(v))
                    elif isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"missy.{key}", value)
        except Exception as exc:
            logger.debug("OtelExporter: export failed: %s", exc)

    def subscribe(self) -> None:
        """Subscribe to the event bus to receive all audit events."""
        try:
            from missy.core.events import event_bus

            def _handler(event):
                self.export_event(
                    event.__dict__ if hasattr(event, "__dict__") else {}
                )

            event_bus.subscribe(_handler)
            logger.debug("OtelExporter: subscribed to event bus")
        except Exception as exc:
            logger.warning("OtelExporter: subscribe failed: %s", exc)

    @property
    def is_enabled(self) -> bool:
        return self._enabled


def init_otel(config) -> OtelExporter:
    """Initialise OpenTelemetry from config. Returns exporter (may be disabled)."""
    obs = getattr(config, "observability", None)
    if obs is None or not getattr(obs, "otel_enabled", False):
        return OtelExporter.__new__(OtelExporter)  # disabled stub

    exporter = OtelExporter(
        endpoint=obs.otel_endpoint,
        protocol=obs.otel_protocol,
        service_name=obs.otel_service_name,
    )
    if exporter.is_enabled:
        exporter.subscribe()
    return exporter
