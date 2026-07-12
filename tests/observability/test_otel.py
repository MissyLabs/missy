"""Tests for OpenTelemetry integration."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from missy.observability.otel import OtelExporter, init_otel


def _make_exporter(enabled: bool = True, tracer=None) -> OtelExporter:
    """Manually construct an OtelExporter bypassing __init__/_setup().

    Sets every attribute __init__ would, so this stays accurate as the
    class gains new attributes (SR-4.6 added export_failure_count/
    last_export_error/_bus/_original_publish after this pattern was
    already established, and previously-missing attributes were exactly
    the reason test_export_event_exception_handled started raising
    AttributeError instead of exercising the intended code path).
    """
    exp = OtelExporter.__new__(OtelExporter)
    exp._endpoint = "http://localhost:4317"
    exp._protocol = "grpc"
    exp._service_name = "missy-test"
    exp._max_queue_size = 2048
    exp._max_export_batch_size = 512
    exp._schedule_delay_millis = 5000
    exp._export_timeout_millis = 30000
    exp._enabled = enabled
    exp._tracer = tracer
    exp._export_failure_count = 0
    exp._last_export_error = None
    exp._bus = None
    exp._original_publish = None
    return exp


def _mock_tracer_with_span():
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    return mock_tracer, mock_span


class TestOtelExporter:
    """Tests for OtelExporter."""

    def test_disabled_when_import_fails(self) -> None:
        with patch.dict("sys.modules", {"opentelemetry": None}):
            exp = _make_exporter(enabled=False)
            assert not exp.is_enabled

    def test_export_event_noop_when_disabled(self) -> None:
        exp = _make_exporter(enabled=False)
        # Should not raise
        exp.export_event({"event_type": "test", "session_id": "s1"})

    def test_export_event_with_mock_tracer(self) -> None:
        mock_tracer, _ = _mock_tracer_with_span()
        exp = _make_exporter(tracer=mock_tracer)

        exp.export_event(
            {
                "event_type": "test.event",
                "session_id": "s1",
                "result": "allow",
            }
        )
        mock_tracer.start_as_current_span.assert_called_once_with("test.event")

    def test_export_event_handles_detail_dict(self) -> None:
        mock_tracer, mock_span = _mock_tracer_with_span()
        exp = _make_exporter(tracer=mock_tracer)

        exp.export_event(
            {
                "event_type": "test.event",
                "detail": {"key1": "val1", "key2": 42},
            }
        )
        mock_span.set_attribute.assert_any_call("missy.key1", "val1")
        mock_span.set_attribute.assert_any_call("missy.key2", "42")

    def test_export_event_redacts_detail_before_setting_attributes(self) -> None:
        """SR-4.6: an unredacted export would leak the exact secrets
        SR-1.10 already closed off for the on-disk audit log, just
        through the OTLP path instead."""
        mock_tracer, mock_span = _mock_tracer_with_span()
        exp = _make_exporter(tracer=mock_tracer)

        exp.export_event(
            {
                "event_type": "network.request",
                "detail": {
                    "url": "https://api.example.com/x?api_key=sk-secret1234567890123456",
                    "note": "ok",
                },
            }
        )
        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "sk-secret1234567890123456" not in calls["missy.url"]
        assert "[REDACTED]" in calls["missy.url"]
        assert calls["missy.note"] == "ok"

    def test_export_event_skips_non_primitive(self) -> None:
        mock_tracer, mock_span = _mock_tracer_with_span()
        exp = _make_exporter(tracer=mock_tracer)

        exp.export_event(
            {
                "event_type": "test",
                "complex": [1, 2, 3],  # not str/int/float/bool
            }
        )
        # Only event_type should be set as attribute
        calls = list(mock_span.set_attribute.call_args_list)
        keys = [c[0][0] for c in calls]
        assert "missy.complex" not in keys

    def test_export_event_exception_handled(self) -> None:
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.side_effect = RuntimeError("otel fail")
        exp = _make_exporter(tracer=mock_tracer)
        # Should not raise
        exp.export_event({"event_type": "test"})

    def test_export_event_exception_increments_failure_count(self) -> None:
        """SR-4.6: failures must be surfaced (countable/inspectable), not
        just silently logged at DEBUG as before."""
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.side_effect = RuntimeError("otel fail")
        exp = _make_exporter(tracer=mock_tracer)

        assert exp.export_failure_count == 0
        exp.export_event({"event_type": "test"})
        assert exp.export_failure_count == 1
        assert exp.last_export_error == "otel fail"

        exp.export_event({"event_type": "test"})
        assert exp.export_failure_count == 2

    def test_export_event_missing_event_type(self) -> None:
        mock_tracer, mock_span = _mock_tracer_with_span()
        exp = _make_exporter(tracer=mock_tracer)

        exp.export_event({"session_id": "s1"})
        mock_tracer.start_as_current_span.assert_called_once_with("missy.event")

    def test_subscribe_wraps_publish_not_broken_subscribe_call(self) -> None:
        """SR-4.6: subscribe() previously called event_bus.subscribe(_handler)
        with only one argument, but EventBus.subscribe(event_type, callback)
        requires two -- that call always raised TypeError (silently caught),
        so OTLP export never received a single event in any configuration.
        The fix wraps publish() instead (mirroring AuditLogger's pattern)."""
        from missy.core.events import EventBus

        mock_tracer, _ = _mock_tracer_with_span()
        exp = _make_exporter(tracer=mock_tracer)

        bus = EventBus()
        with patch("missy.core.events.event_bus", bus):
            exp.subscribe()

        assert bus.publish is not EventBus.publish  # publish was wrapped
        assert exp._bus is bus

    def test_subscribe_actually_delivers_published_events(self) -> None:
        """End-to-end proof subscribe() results in export_event() being
        called for every published event, regardless of event_type --
        the exact capability the broken subscribe(_handler) call never
        provided."""
        from missy.core.events import AuditEvent, EventBus

        mock_tracer, mock_span = _mock_tracer_with_span()
        exp = _make_exporter(tracer=mock_tracer)

        bus = EventBus()
        with patch("missy.core.events.event_bus", bus):
            exp.subscribe()

        bus.publish(
            AuditEvent.now(
                session_id="s1",
                task_id="t1",
                event_type="some.arbitrary.type",
                category="plugin",
                result="allow",
                detail={},
            )
        )
        mock_tracer.start_as_current_span.assert_called_once_with("some.arbitrary.type")

    def test_subscribe_exception_handled(self) -> None:
        """subscribe() itself must not raise even if attaching to the bus
        fails for some reason (e.g. the bus module is unexpectedly
        unavailable)."""
        exp = _make_exporter()
        with patch("missy.core.events.event_bus", None):
            # None.publish raises AttributeError inside subscribe()'s try block.
            exp.subscribe()  # Should not raise

    def test_is_enabled_property(self) -> None:
        exp = _make_exporter(enabled=True)
        assert exp.is_enabled is True
        exp._enabled = False
        assert exp.is_enabled is False

    def test_export_failure_count_starts_at_zero(self) -> None:
        exp = _make_exporter()
        assert exp.export_failure_count == 0
        assert exp.last_export_error is None


class TestQueueBounds:
    """SR-4.6: BatchSpanProcessor bounds must be explicit, not implicit
    undocumented SDK defaults."""

    def test_default_queue_bounds_are_set(self) -> None:
        exp = _make_exporter()
        assert exp._max_queue_size == 2048
        assert exp._max_export_batch_size == 512
        assert exp._schedule_delay_millis == 5000
        assert exp._export_timeout_millis == 30000

    def test_custom_queue_bounds_accepted(self) -> None:
        exp = OtelExporter(
            endpoint="http://localhost:4317",
            protocol="grpc",
            service_name="missy",
            max_queue_size=100,
            max_export_batch_size=10,
        )
        assert exp._max_queue_size == 100
        assert exp._max_export_batch_size == 10


class TestDisabledStub:
    """SR-4.6: the pre-existing OtelExporter.__new__(OtelExporter) disabled
    stub had zero attributes set -- touching .is_enabled crashed with
    AttributeError. Fixed alongside the subscription bug since a real
    "surface exporter failures" story requires .is_enabled/
    .export_failure_count to be safely readable in the disabled case too."""

    def test_disabled_stub_is_enabled_does_not_raise(self) -> None:
        from missy.observability.otel import _disabled_stub

        stub = _disabled_stub()
        assert stub.is_enabled is False

    def test_disabled_stub_export_failure_count_does_not_raise(self) -> None:
        from missy.observability.otel import _disabled_stub

        stub = _disabled_stub()
        assert stub.export_failure_count == 0
        assert stub.last_export_error is None

    def test_disabled_stub_export_event_is_safe_noop(self) -> None:
        from missy.observability.otel import _disabled_stub

        stub = _disabled_stub()
        stub.export_event({"event_type": "test"})  # must not raise


class TestInitOtel:
    """Tests for init_otel factory."""

    def test_disabled_when_no_observability(self) -> None:
        config = MagicMock(spec=[])  # no observability attr
        result = init_otel(config)
        assert isinstance(result, OtelExporter)
        assert result.is_enabled is False

    def test_disabled_when_otel_not_enabled(self) -> None:
        @dataclass
        class ObsCfg:
            otel_enabled: bool = False
            otel_endpoint: str = "http://localhost:4317"
            otel_protocol: str = "grpc"
            otel_service_name: str = "missy"

        config = MagicMock()
        config.observability = ObsCfg()
        result = init_otel(config)
        assert isinstance(result, OtelExporter)
        assert result.is_enabled is False

    def test_creates_exporter_when_enabled(self) -> None:
        @dataclass
        class ObsCfg:
            otel_enabled: bool = True
            otel_endpoint: str = "http://localhost:4317"
            otel_protocol: str = "grpc"
            otel_service_name: str = "test-missy"

        config = MagicMock()
        config.observability = ObsCfg()

        result = init_otel(config)
        assert isinstance(result, OtelExporter)


class TestInitOtelReinitialization:
    """Regression: init_otel() was only ever called once, at process
    bootstrap -- `missy config` hot-reload (ConfigWatcher/_apply_config)
    had no way to make a change to observability.otel_enabled/otel_endpoint/
    otel_protocol take effect on a running `missy gateway start` daemon
    without a restart, despite existing specifically for that purpose.
    init_otel() must now (a) be safe to call more than once in the same
    process, (b) unwind any previously active exporter's publish() wrapper
    before installing a new one so events aren't exported once per
    historical re-init, and (c) track the currently active exporter so a
    later caller in the same process can read its live state.
    """

    @pytest.fixture(autouse=True)
    def _reset_otel_state(self):
        import missy.observability.otel as otel_module
        from missy.core.events import event_bus

        original_publish = event_bus.publish
        original_active = otel_module._active_exporter
        yield
        if otel_module._active_exporter is not None:
            otel_module._active_exporter.unsubscribe()
        event_bus.publish = original_publish
        otel_module._active_exporter = original_active

    def _disabled_config(self):
        @dataclass
        class ObsCfg:
            otel_enabled: bool = False
            otel_endpoint: str = "http://localhost:4317"
            otel_protocol: str = "grpc"
            otel_service_name: str = "missy-test"

        config = MagicMock()
        config.observability = ObsCfg()
        return config

    def _enabled_config(self):
        @dataclass
        class ObsCfg:
            otel_enabled: bool = True
            otel_endpoint: str = "http://localhost:4317"
            otel_protocol: str = "grpc"
            otel_service_name: str = "missy-test"

        config = MagicMock()
        config.observability = ObsCfg()
        return config

    def test_get_active_exporter_tracks_the_most_recent_init(self) -> None:
        from missy.observability.otel import get_active_exporter

        result = init_otel(self._disabled_config())
        assert get_active_exporter() is result

    def test_reinit_unsubscribes_previous_exporter_before_new_one(self) -> None:
        """Re-running init_otel() (as a config hot-reload would) must not
        stack a second publish() wrapper around the first -- otherwise a
        single published event would be exported once per historical
        reload rather than once.
        """
        from missy.core.events import event_bus

        first = init_otel(self._enabled_config())
        assert first.is_enabled is True
        publish_after_first = event_bus.publish

        second = init_otel(self._enabled_config())
        assert second.is_enabled is True
        assert second is not first
        # The first exporter's wrapper must have been unwound: publish()
        # changed identity (a fresh wrap was installed by the second
        # exporter), and the first exporter no longer holds a live
        # reference to the bus/original publish function.
        assert event_bus.publish is not publish_after_first
        assert first._bus is None
        assert first._original_publish is None

    def test_disabling_after_enabled_restores_original_publish(self) -> None:
        """Toggling otel_enabled: true -> false at runtime (a hot-reload)
        must actually stop exporting -- the previously enabled exporter's
        publish() wrapper must be unwound when the disabled stub is
        installed, not merely for the *next* enabled re-init.
        """
        from missy.core.events import event_bus

        enabled_exporter = init_otel(self._enabled_config())
        assert enabled_exporter.is_enabled is True
        wrapped_publish = event_bus.publish

        disabled_exporter = init_otel(self._disabled_config())
        assert disabled_exporter.is_enabled is False
        assert event_bus.publish is not wrapped_publish
        assert enabled_exporter._bus is None
        assert enabled_exporter._original_publish is None


@pytest.fixture
def _skip_without_opentelemetry():
    pytest.importorskip("opentelemetry")


@pytest.mark.usefixtures("_skip_without_opentelemetry")
class TestEndToEndRealSdk:
    """SR-4.6: prove enabled telemetry genuinely reaches the configured
    collector -- using the real opentelemetry SDK with an in-memory span
    exporter standing in for the network collector (so the test is real
    end-to-end through OtelExporter's actual code, not a re-implementation
    of it), rather than asserting only that internal methods were called.
    """

    def _build_real_exporter_with_memory_collector(self):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exp = _make_exporter(enabled=False, tracer=None)
        provider = TracerProvider(resource=Resource.create({"service.name": "missy-test"}))
        memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
        # Get the tracer directly from this provider instance rather than
        # via the global trace.set_tracer_provider()/get_tracer() API --
        # the global provider is process-wide and can only be set once
        # (OTel refuses later overrides), so relying on it here would make
        # this test's outcome depend on whether some earlier test already
        # installed a real (network-exporting) global provider.
        exp._tracer = provider.get_tracer("missy-test")
        exp._enabled = True
        return exp, memory_exporter

    def test_published_event_arrives_as_a_real_span(self):
        from missy.core.events import AuditEvent, EventBus

        exp, memory_exporter = self._build_real_exporter_with_memory_collector()
        bus = EventBus()
        with patch("missy.core.events.event_bus", bus):
            exp.subscribe()

        bus.publish(
            AuditEvent.now(
                session_id="s1",
                task_id="t1",
                event_type="tool_execute",
                category="plugin",
                result="allow",
                detail={"tool": "calculator"},
            )
        )

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "tool_execute"
        assert dict(spans[0].attributes)["missy.tool"] == "calculator"

    def test_secret_in_detail_never_reaches_the_collector_unredacted(self):
        from missy.core.events import AuditEvent, EventBus

        exp, memory_exporter = self._build_real_exporter_with_memory_collector()
        bus = EventBus()
        with patch("missy.core.events.event_bus", bus):
            exp.subscribe()

        secret = "sk-live-abcdef0123456789abcdef0123456789"
        bus.publish(
            AuditEvent.now(
                session_id="s1",
                task_id="t1",
                event_type="network.request",
                category="network",
                result="allow",
                detail={"url": f"https://api.example.com/x?api_key={secret}"},
            )
        )

        spans = memory_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert secret not in attrs["missy.url"]

    def test_multiple_arbitrary_event_types_all_arrive(self):
        """Confirms the fix isn't accidentally scoped to one event_type --
        the real bug affected every single event regardless of type."""
        from missy.core.events import AuditEvent, EventBus

        exp, memory_exporter = self._build_real_exporter_with_memory_collector()
        bus = EventBus()
        with patch("missy.core.events.event_bus", bus):
            exp.subscribe()

        for event_type in ["agent.run.start", "mcp.tool_execute", "scheduler.job.run.start"]:
            bus.publish(
                AuditEvent.now(
                    session_id="s1",
                    task_id="t1",
                    event_type=event_type,
                    category="plugin",
                    result="allow",
                    detail={},
                )
            )

        spans = memory_exporter.get_finished_spans()
        assert {s.name for s in spans} == {
            "agent.run.start",
            "mcp.tool_execute",
            "scheduler.job.run.start",
        }
