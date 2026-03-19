"""Capture performance benchmarking and latency profiling.

Measures and reports on capture latency, throughput, pipeline processing
time, and quality statistics.  Used by ``missy vision doctor`` for
performance diagnostics and by developers for regression testing.

Example::

    from missy.vision.benchmark import CaptureBenchmark

    bench = CaptureBenchmark()
    bench.record_capture(latency_ms=45.2, quality=0.87)
    bench.record_pipeline(processing_ms=12.3)
    report = bench.report()
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """A single benchmark measurement."""

    timestamp: float
    category: str  # "capture", "pipeline", "burst", "save"
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class CaptureBenchmark:
    """Collects and analyzes capture performance metrics.

    Thread-safe: all mutations are guarded by a lock.

    Parameters
    ----------
    max_samples:
        Maximum number of samples retained per category.
    """

    def __init__(self, max_samples: int = 500) -> None:
        self._max_samples = max(10, max_samples)
        self._samples: dict[str, deque[BenchmarkSample]] = {}
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

    def record(
        self,
        category: str,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Record a benchmark sample.

        Parameters
        ----------
        category:
            Measurement category (e.g. "capture", "pipeline", "burst").
        duration_ms:
            Duration in milliseconds.
        **metadata:
            Additional context (device, resolution, etc.).
        """
        sample = BenchmarkSample(
            timestamp=time.time(),
            category=category,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        with self._lock:
            if category not in self._samples:
                self._samples[category] = deque(maxlen=self._max_samples)
            self._samples[category].append(sample)

    def record_capture(
        self,
        latency_ms: float,
        quality: float = 0.0,
        device: str = "",
    ) -> None:
        """Convenience: record a capture measurement."""
        self.record("capture", latency_ms, quality=quality, device=device)

    def record_pipeline(
        self,
        processing_ms: float,
        operation: str = "",
    ) -> None:
        """Convenience: record a pipeline processing measurement."""
        self.record("pipeline", processing_ms, operation=operation)

    def record_burst(
        self,
        total_ms: float,
        frame_count: int = 0,
    ) -> None:
        """Convenience: record a burst capture measurement."""
        self.record("burst", total_ms, frame_count=frame_count)

    def record_save(self, save_ms: float, format: str = "") -> None:
        """Convenience: record a file save measurement."""
        self.record("save", save_ms, format=format)

    def get_stats(self, category: str) -> dict[str, Any]:
        """Compute statistics for a category.

        Returns
        -------
        dict
            Keys: count, min_ms, max_ms, mean_ms, median_ms, p95_ms, p99_ms,
            stddev_ms.  Returns empty dict if no samples.
        """
        with self._lock:
            samples = self._samples.get(category)
            if not samples:
                return {}
            durations = [s.duration_ms for s in samples]

        if not durations:
            return {}

        sorted_d = sorted(durations)
        n = len(sorted_d)

        return {
            "count": n,
            "min_ms": round(sorted_d[0], 2),
            "max_ms": round(sorted_d[-1], 2),
            "mean_ms": round(statistics.mean(sorted_d), 2),
            "median_ms": round(statistics.median(sorted_d), 2),
            "p95_ms": round(sorted_d[int(n * 0.95)] if n >= 20 else sorted_d[-1], 2),
            "p99_ms": round(sorted_d[int(n * 0.99)] if n >= 100 else sorted_d[-1], 2),
            "stddev_ms": round(statistics.stdev(sorted_d), 2) if n >= 2 else 0.0,
        }

    def report(self) -> dict[str, Any]:
        """Generate a full benchmark report across all categories.

        Returns
        -------
        dict
            Keys: uptime_seconds, categories (dict of category → stats).
        """
        with self._lock:
            categories = list(self._samples.keys())

        result: dict[str, Any] = {
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "categories": {},
        }
        for cat in categories:
            stats = self.get_stats(cat)
            if stats:
                result["categories"][cat] = stats

        return result

    def throughput(self, category: str, window_seconds: float = 60.0) -> float:
        """Compute recent throughput (operations/second) for a category.

        Parameters
        ----------
        category:
            Measurement category.
        window_seconds:
            Time window to consider.

        Returns
        -------
        float
            Operations per second, or 0.0 if insufficient data.
        """
        cutoff = time.time() - window_seconds
        with self._lock:
            samples = self._samples.get(category)
            if not samples:
                return 0.0
            recent = [s for s in samples if s.timestamp >= cutoff]

        if len(recent) < 2:
            return 0.0

        time_span = recent[-1].timestamp - recent[0].timestamp
        if time_span <= 0:
            return 0.0
        return len(recent) / time_span

    def reset(self, category: str | None = None) -> None:
        """Clear benchmark data.

        Parameters
        ----------
        category:
            If given, only clear that category.  Otherwise clear all.
        """
        with self._lock:
            if category:
                self._samples.pop(category, None)
            else:
                self._samples.clear()
                self._start_time = time.monotonic()


# ---------------------------------------------------------------------------
# Timer context manager for easy benchmarking
# ---------------------------------------------------------------------------


class BenchmarkTimer:
    """Context manager that records elapsed time to a CaptureBenchmark.

    Example::

        bench = CaptureBenchmark()
        with BenchmarkTimer(bench, "capture", device="/dev/video0"):
            result = camera.capture()
    """

    def __init__(
        self,
        benchmark: CaptureBenchmark,
        category: str,
        **metadata: Any,
    ) -> None:
        self._benchmark = benchmark
        self._category = category
        self._metadata = metadata
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> BenchmarkTimer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000
        self._benchmark.record(self._category, self.elapsed_ms, **self._metadata)


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_benchmark: CaptureBenchmark | None = None


def get_benchmark() -> CaptureBenchmark:
    """Return the process-level singleton CaptureBenchmark."""
    global _benchmark
    if _benchmark is None:
        _benchmark = CaptureBenchmark()
    return _benchmark
