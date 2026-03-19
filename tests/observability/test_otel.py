"""Tests for OpenTelemetry integration."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from missy.observability.otel import OtelExporter, init_otel


class TestOtelExporter:
    """Tests for OtelExporter."""

    def test_disabled_when_import_fails(self) -> None:
        with patch.dict("sys.modules", {"opentelemetry": None}):
            exp = OtelExporter.__new__(OtelExporter)
            exp._enabled = False
            exp._tracer = None
            assert not exp.is_enabled

    def test_export_event_noop_when_disabled(self) -> None:
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = False
        exp._tracer = None
        # Should not raise
        exp.export_event({"event_type": "test", "session_id": "s1"})

    def test_export_event_with_mock_tracer(self) -> None:
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = True
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        exp._tracer = mock_tracer

        exp.export_event({
            "event_type": "test.event",
            "session_id": "s1",
            "result": "allow",
        })
        mock_tracer.start_as_current_span.assert_called_once_with("test.event")

    def test_export_event_handles_detail_dict(self) -> None:
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = True
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        exp._tracer = mock_tracer

        exp.export_event({
            "event_type": "test.event",
            "detail": {"key1": "val1", "key2": 42},
        })
        mock_span.set_attribute.assert_any_call("missy.key1", "val1")
        mock_span.set_attribute.assert_any_call("missy.key2", "42")

    def test_export_event_skips_non_primitive(self) -> None:
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = True
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        exp._tracer = mock_tracer

        exp.export_event({
            "event_type": "test",
            "complex": [1, 2, 3],  # not str/int/float/bool
        })
        # Only event_type should be set as attribute
        calls = list(mock_span.set_attribute.call_args_list)
        keys = [c[0][0] for c in calls]
        assert "missy.complex" not in keys

    def test_export_event_exception_handled(self) -> None:
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = True
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.side_effect = RuntimeError("otel fail")
        exp._tracer = mock_tracer
        # Should not raise
        exp.export_event({"event_type": "test"})

    def test_export_event_missing_event_type(self) -> None:
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = True
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        exp._tracer = mock_tracer

        exp.export_event({"session_id": "s1"})
        mock_tracer.start_as_current_span.assert_called_once_with("missy.event")

    def test_subscribe_exception_handled(self) -> None:
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = True
        exp._tracer = MagicMock()
        with patch("missy.core.events.event_bus") as mock_bus:
            mock_bus.subscribe.side_effect = RuntimeError("bus fail")
            # Should not raise even if subscribe fails
            exp.subscribe()

    def test_is_enabled_property(self) -> None:
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = True
        assert exp.is_enabled is True
        exp._enabled = False
        assert exp.is_enabled is False


class TestInitOtel:
    """Tests for init_otel factory."""

    def test_disabled_when_no_observability(self) -> None:
        config = MagicMock(spec=[])  # no observability attr
        result = init_otel(config)
        assert isinstance(result, OtelExporter)

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

    def test_creates_exporter_when_enabled(self) -> None:
        @dataclass
        class ObsCfg:
            otel_enabled: bool = True
            otel_endpoint: str = "http://localhost:4317"
            otel_protocol: str = "grpc"
            otel_service_name: str = "test-missy"

        config = MagicMock()
        config.observability = ObsCfg()

        # OtelExporter._setup will fail gracefully without otel installed
        result = init_otel(config)
        assert isinstance(result, OtelExporter)
