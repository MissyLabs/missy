"""Tests for missy.vision.benchmark.

Covers CaptureBenchmark, BenchmarkTimer, BenchmarkSample, and the
get_benchmark() module singleton.  All tests are pure unit tests — no
hardware or filesystem access required.
"""

from __future__ import annotations

import threading
import time

import pytest

from missy.vision.benchmark import (
    BenchmarkSample,
    BenchmarkTimer,
    CaptureBenchmark,
    get_benchmark,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bench(max_samples: int = 500) -> CaptureBenchmark:
    return CaptureBenchmark(max_samples=max_samples)


def _record_n(bench: CaptureBenchmark, category: str, values: list[float]) -> None:
    for v in values:
        bench.record(category, v)


# ---------------------------------------------------------------------------
# BenchmarkSample dataclass
# ---------------------------------------------------------------------------


class TestBenchmarkSample:
    def test_fields_set(self):
        sample = BenchmarkSample(
            timestamp=1000.0,
            category="capture",
            duration_ms=42.5,
            metadata={"device": "/dev/video0"},
        )
        assert sample.timestamp == 1000.0
        assert sample.category == "capture"
        assert sample.duration_ms == 42.5
        assert sample.metadata == {"device": "/dev/video0"}

    def test_metadata_defaults_empty_dict(self):
        sample = BenchmarkSample(timestamp=0.0, category="save", duration_ms=1.0)
        assert sample.metadata == {}

    def test_metadata_is_per_instance(self):
        s1 = BenchmarkSample(timestamp=0.0, category="a", duration_ms=1.0)
        s2 = BenchmarkSample(timestamp=0.0, category="b", duration_ms=2.0)
        s1.metadata["key"] = "val"
        assert "key" not in s2.metadata


# ---------------------------------------------------------------------------
# CaptureBenchmark — construction
# ---------------------------------------------------------------------------


class TestCaptureBenchmarkInit:
    def test_default_max_samples(self):
        bench = CaptureBenchmark()
        assert bench._max_samples == 500

    def test_custom_max_samples(self):
        bench = CaptureBenchmark(max_samples=200)
        assert bench._max_samples == 200

    def test_max_samples_minimum_enforced(self):
        # Values below 10 are clamped to 10.
        bench = CaptureBenchmark(max_samples=3)
        assert bench._max_samples == 10

    def test_max_samples_zero_clamped(self):
        bench = CaptureBenchmark(max_samples=0)
        assert bench._max_samples == 10

    def test_initially_no_samples(self):
        bench = _make_bench()
        assert bench._samples == {}


# ---------------------------------------------------------------------------
# CaptureBenchmark — record()
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_creates_category(self):
        bench = _make_bench()
        bench.record("capture", 50.0)
        assert "capture" in bench._samples

    def test_record_stores_duration(self):
        bench = _make_bench()
        bench.record("pipeline", 12.3)
        sample = bench._samples["pipeline"][0]
        assert sample.duration_ms == 12.3

    def test_record_stores_category(self):
        bench = _make_bench()
        bench.record("burst", 99.9)
        assert bench._samples["burst"][0].category == "burst"

    def test_record_stores_metadata_kwargs(self):
        bench = _make_bench()
        bench.record("save", 5.0, format="jpeg", size=1024)
        meta = bench._samples["save"][0].metadata
        assert meta["format"] == "jpeg"
        assert meta["size"] == 1024

    def test_record_multiple_samples_same_category(self):
        bench = _make_bench()
        bench.record("capture", 10.0)
        bench.record("capture", 20.0)
        bench.record("capture", 30.0)
        assert len(bench._samples["capture"]) == 3

    def test_record_multiple_categories_independent(self):
        bench = _make_bench()
        bench.record("capture", 10.0)
        bench.record("pipeline", 5.0)
        assert len(bench._samples["capture"]) == 1
        assert len(bench._samples["pipeline"]) == 1

    def test_record_timestamp_is_recent(self):
        bench = _make_bench()
        before = time.time()
        bench.record("capture", 1.0)
        after = time.time()
        ts = bench._samples["capture"][0].timestamp
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# CaptureBenchmark — max_samples limit (deque eviction)
# ---------------------------------------------------------------------------


class TestMaxSamplesLimit:
    def test_deque_evicts_oldest_when_full(self):
        # max_samples=10 is the minimum the constructor allows; record 15 items
        # so the first 5 are evicted and only the newest 10 remain.
        bench = _make_bench(max_samples=10)
        for i in range(15):
            bench.record("capture", float(i))
        samples = list(bench._samples["capture"])
        assert len(samples) == 10
        # Oldest (0–4) evicted; newest (5–14) retained.
        durations = [s.duration_ms for s in samples]
        assert durations == [float(i) for i in range(5, 15)]

    def test_deque_evicts_across_many_inserts(self):
        bench = _make_bench(max_samples=10)
        for i in range(100):
            bench.record("x", float(i))
        assert len(bench._samples["x"]) == 10

    def test_max_samples_minimum_10_respected(self):
        bench = _make_bench(max_samples=3)
        for i in range(15):
            bench.record("y", float(i))
        assert len(bench._samples["y"]) == 10


# ---------------------------------------------------------------------------
# CaptureBenchmark — convenience helpers
# ---------------------------------------------------------------------------


class TestConvenienceRecorders:
    def test_record_capture_stores_in_capture_category(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=45.2, quality=0.87, device="/dev/video0")
        assert "capture" in bench._samples
        meta = bench._samples["capture"][0].metadata
        assert meta["quality"] == 0.87
        assert meta["device"] == "/dev/video0"

    def test_record_capture_defaults(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=10.0)
        meta = bench._samples["capture"][0].metadata
        assert meta["quality"] == 0.0
        assert meta["device"] == ""

    def test_record_capture_duration(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=33.3)
        assert bench._samples["capture"][0].duration_ms == 33.3

    def test_record_pipeline_stores_in_pipeline_category(self):
        bench = _make_bench()
        bench.record_pipeline(processing_ms=7.5, operation="resize")
        assert "pipeline" in bench._samples
        assert bench._samples["pipeline"][0].duration_ms == 7.5
        assert bench._samples["pipeline"][0].metadata["operation"] == "resize"

    def test_record_pipeline_defaults(self):
        bench = _make_bench()
        bench.record_pipeline(processing_ms=1.0)
        assert bench._samples["pipeline"][0].metadata["operation"] == ""

    def test_record_burst_stores_in_burst_category(self):
        bench = _make_bench()
        bench.record_burst(total_ms=150.0, frame_count=5)
        assert "burst" in bench._samples
        assert bench._samples["burst"][0].duration_ms == 150.0
        assert bench._samples["burst"][0].metadata["frame_count"] == 5

    def test_record_burst_defaults(self):
        bench = _make_bench()
        bench.record_burst(total_ms=50.0)
        assert bench._samples["burst"][0].metadata["frame_count"] == 0

    def test_record_save_stores_in_save_category(self):
        bench = _make_bench()
        bench.record_save(save_ms=2.3, format="png")
        assert "save" in bench._samples
        assert bench._samples["save"][0].duration_ms == 2.3
        assert bench._samples["save"][0].metadata["format"] == "png"

    def test_record_save_defaults(self):
        bench = _make_bench()
        bench.record_save(save_ms=1.0)
        assert bench._samples["save"][0].metadata["format"] == ""


# ---------------------------------------------------------------------------
# CaptureBenchmark — get_stats()
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_empty_category_returns_empty_dict(self):
        bench = _make_bench()
        assert bench.get_stats("capture") == {}

    def test_nonexistent_category_returns_empty_dict(self):
        bench = _make_bench()
        assert bench.get_stats("no_such_category") == {}

    def test_single_sample_stats(self):
        bench = _make_bench()
        bench.record("capture", 50.0)
        stats = bench.get_stats("capture")
        assert stats["count"] == 1
        assert stats["min_ms"] == 50.0
        assert stats["max_ms"] == 50.0
        assert stats["mean_ms"] == 50.0
        assert stats["median_ms"] == 50.0
        # p95, p99 fall back to last element when n < threshold.
        assert stats["p95_ms"] == 50.0
        assert stats["p99_ms"] == 50.0
        # stddev requires at least 2 samples.
        assert stats["stddev_ms"] == 0.0

    def test_two_samples_stddev_nonzero(self):
        bench = _make_bench()
        bench.record("capture", 10.0)
        bench.record("capture", 20.0)
        stats = bench.get_stats("capture")
        assert stats["stddev_ms"] > 0.0

    def test_min_max_correct(self):
        bench = _make_bench()
        _record_n(bench, "pipeline", [30.0, 10.0, 50.0, 20.0, 40.0])
        stats = bench.get_stats("pipeline")
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 50.0

    def test_mean_correct(self):
        bench = _make_bench()
        _record_n(bench, "capture", [10.0, 20.0, 30.0])
        stats = bench.get_stats("capture")
        assert stats["mean_ms"] == 20.0

    def test_median_odd_count(self):
        bench = _make_bench()
        _record_n(bench, "capture", [1.0, 3.0, 2.0])
        stats = bench.get_stats("capture")
        assert stats["median_ms"] == 2.0

    def test_median_even_count(self):
        bench = _make_bench()
        _record_n(bench, "capture", [1.0, 2.0, 3.0, 4.0])
        stats = bench.get_stats("capture")
        # statistics.median averages middle two: (2+3)/2 = 2.5
        assert stats["median_ms"] == 2.5

    def test_p95_uses_fallback_when_fewer_than_20_samples(self):
        bench = _make_bench()
        _record_n(bench, "capture", [float(i) for i in range(1, 19)])  # 18 samples
        stats = bench.get_stats("capture")
        # Falls back to sorted_d[-1] == 18.0
        assert stats["p95_ms"] == 18.0

    def test_p95_computed_when_20_or_more_samples(self):
        bench = _make_bench()
        # 20 sorted values 1..20; p95 index = int(20*0.95) = 19 → value 20.
        _record_n(bench, "capture", [float(i) for i in range(1, 21)])
        stats = bench.get_stats("capture")
        assert stats["count"] == 20
        assert stats["p95_ms"] == 20.0

    def test_p99_uses_fallback_when_fewer_than_100_samples(self):
        bench = _make_bench()
        _record_n(bench, "capture", [float(i) for i in range(1, 51)])  # 50 samples
        stats = bench.get_stats("capture")
        assert stats["p99_ms"] == 50.0

    def test_p99_computed_when_100_or_more_samples(self):
        bench = _make_bench(max_samples=200)
        # 100 sorted values 1..100; p99 index = int(100*0.99) = 99 → value 100.
        _record_n(bench, "capture", [float(i) for i in range(1, 101)])
        stats = bench.get_stats("capture")
        assert stats["count"] == 100
        assert stats["p99_ms"] == 100.0

    def test_stats_keys_present(self):
        bench = _make_bench()
        bench.record("capture", 1.0)
        stats = bench.get_stats("capture")
        expected_keys = {
            "count",
            "min_ms",
            "max_ms",
            "mean_ms",
            "median_ms",
            "p95_ms",
            "p99_ms",
            "stddev_ms",
        }
        assert set(stats.keys()) == expected_keys

    def test_count_matches_samples_recorded(self):
        bench = _make_bench()
        _record_n(bench, "burst", [1.0] * 7)
        assert bench.get_stats("burst")["count"] == 7

    def test_stats_values_are_rounded_to_two_decimal_places(self):
        bench = _make_bench()
        # Insert values that would produce fractional ms.
        _record_n(bench, "capture", [1.1, 2.2, 3.3])
        stats = bench.get_stats("capture")
        for key in ("min_ms", "max_ms", "mean_ms", "median_ms", "p95_ms", "p99_ms"):
            val = stats[key]
            assert round(val, 2) == val, f"{key}={val} not rounded to 2 dp"

    def test_independent_categories_do_not_interfere(self):
        bench = _make_bench()
        _record_n(bench, "capture", [100.0, 200.0])
        _record_n(bench, "pipeline", [1.0, 2.0])
        capture_stats = bench.get_stats("capture")
        pipeline_stats = bench.get_stats("pipeline")
        assert capture_stats["mean_ms"] == 150.0
        assert pipeline_stats["mean_ms"] == 1.5


# ---------------------------------------------------------------------------
# CaptureBenchmark — report()
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_structure(self):
        bench = _make_bench()
        report = bench.report()
        assert "uptime_seconds" in report
        assert "categories" in report

    def test_report_empty_bench_has_no_categories(self):
        bench = _make_bench()
        report = bench.report()
        assert report["categories"] == {}

    def test_report_uptime_is_nonnegative(self):
        bench = _make_bench()
        assert bench.report()["uptime_seconds"] >= 0.0

    def test_report_uptime_increases_over_time(self):
        bench = _make_bench()
        r1 = bench.report()["uptime_seconds"]
        time.sleep(0.05)
        r2 = bench.report()["uptime_seconds"]
        assert r2 >= r1

    def test_report_includes_recorded_category(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=10.0)
        report = bench.report()
        assert "capture" in report["categories"]

    def test_report_includes_all_categories(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=10.0)
        bench.record_pipeline(processing_ms=5.0)
        bench.record_burst(total_ms=80.0)
        bench.record_save(save_ms=2.0)
        cats = bench.report()["categories"]
        for cat in ("capture", "pipeline", "burst", "save"):
            assert cat in cats

    def test_report_category_stats_match_get_stats(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=42.0)
        bench.record_capture(latency_ms=58.0)
        report = bench.report()
        assert report["categories"]["capture"] == bench.get_stats("capture")

    def test_report_uptime_rounded_to_one_decimal(self):
        bench = _make_bench()
        uptime = bench.report()["uptime_seconds"]
        # Check it's a float with at most 1 decimal place as produced by round(..., 1).
        assert round(uptime, 1) == uptime


# ---------------------------------------------------------------------------
# CaptureBenchmark — throughput()
# ---------------------------------------------------------------------------


class TestThroughput:
    def test_empty_category_returns_zero(self):
        bench = _make_bench()
        assert bench.throughput("capture") == 0.0

    def test_nonexistent_category_returns_zero(self):
        bench = _make_bench()
        assert bench.throughput("ghost") == 0.0

    def test_single_sample_returns_zero(self):
        bench = _make_bench()
        bench.record("capture", 10.0)
        assert bench.throughput("capture") == 0.0

    def test_two_samples_with_controlled_timestamps(self):
        bench = _make_bench()
        now = time.time()
        # Inject two samples 1 second apart, both within the window.
        s1 = BenchmarkSample(timestamp=now - 1.0, category="capture", duration_ms=10.0)
        s2 = BenchmarkSample(timestamp=now, category="capture", duration_ms=10.0)
        with bench._lock:
            from collections import deque

            bench._samples["capture"] = deque([s1, s2], maxlen=bench._max_samples)
        tput = bench.throughput("capture", window_seconds=60.0)
        # 2 samples / 1 second = 2.0 ops/sec
        assert abs(tput - 2.0) < 0.01

    def test_samples_outside_window_excluded(self):
        bench = _make_bench()
        now = time.time()
        old = BenchmarkSample(timestamp=now - 120.0, category="x", duration_ms=1.0)
        recent = BenchmarkSample(timestamp=now, category="x", duration_ms=1.0)
        with bench._lock:
            from collections import deque

            bench._samples["x"] = deque([old, recent], maxlen=bench._max_samples)
        # Only 1 recent sample — not enough for throughput.
        assert bench.throughput("x", window_seconds=60.0) == 0.0

    def test_throughput_higher_for_faster_operations(self):
        bench = _make_bench()
        now = time.time()
        from collections import deque

        # 10 samples over 1 second → ~10 ops/sec
        fast = deque(
            [
                BenchmarkSample(timestamp=now - 1.0 + i * 0.1, category="fast", duration_ms=1.0)
                for i in range(11)
            ],
            maxlen=500,
        )
        # 10 samples over 9 seconds → ~1.1 ops/sec
        slow = deque(
            [
                BenchmarkSample(timestamp=now - 9.0 + i * 1.0, category="slow", duration_ms=1.0)
                for i in range(10)
            ],
            maxlen=500,
        )
        with bench._lock:
            bench._samples["fast"] = fast
            bench._samples["slow"] = slow
        assert bench.throughput("fast", window_seconds=60.0) > bench.throughput(
            "slow", window_seconds=60.0
        )

    def test_zero_time_span_returns_zero(self):
        bench = _make_bench()
        now = time.time()
        # Both samples at exact same timestamp.
        s1 = BenchmarkSample(timestamp=now, category="c", duration_ms=1.0)
        s2 = BenchmarkSample(timestamp=now, category="c", duration_ms=2.0)
        with bench._lock:
            from collections import deque

            bench._samples["c"] = deque([s1, s2], maxlen=bench._max_samples)
        assert bench.throughput("c") == 0.0

    def test_custom_window_seconds(self):
        bench = _make_bench()
        now = time.time()
        # Place one sample 90 seconds ago — outside a 60 s window but inside 120 s.
        s1 = BenchmarkSample(timestamp=now - 90.0, category="w", duration_ms=1.0)
        s2 = BenchmarkSample(timestamp=now, category="w", duration_ms=1.0)
        with bench._lock:
            from collections import deque

            bench._samples["w"] = deque([s1, s2], maxlen=bench._max_samples)
        assert bench.throughput("w", window_seconds=60.0) == 0.0
        assert bench.throughput("w", window_seconds=120.0) > 0.0


# ---------------------------------------------------------------------------
# CaptureBenchmark — reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_all_clears_all_categories(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=1.0)
        bench.record_pipeline(processing_ms=1.0)
        bench.reset()
        assert bench._samples == {}

    def test_reset_all_resets_start_time(self):
        bench = _make_bench()
        original_start = bench._start_time
        time.sleep(0.01)
        bench.reset()
        assert bench._start_time >= original_start

    def test_reset_category_removes_only_that_category(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=1.0)
        bench.record_pipeline(processing_ms=1.0)
        bench.reset(category="capture")
        assert "capture" not in bench._samples
        assert "pipeline" in bench._samples

    def test_reset_category_does_not_reset_start_time(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=1.0)
        start_before = bench._start_time
        time.sleep(0.01)
        bench.reset(category="capture")
        assert bench._start_time == start_before

    def test_reset_nonexistent_category_is_noop(self):
        bench = _make_bench()
        bench.record_capture(latency_ms=1.0)
        bench.reset(category="nonexistent")
        assert "capture" in bench._samples

    def test_reset_all_then_record_works(self):
        bench = _make_bench()
        bench.record("capture", 50.0)
        bench.reset()
        bench.record("capture", 99.0)
        stats = bench.get_stats("capture")
        assert stats["count"] == 1
        assert stats["min_ms"] == 99.0

    def test_reset_category_then_record_works(self):
        bench = _make_bench()
        bench.record("capture", 50.0)
        bench.reset(category="capture")
        bench.record("capture", 99.0)
        stats = bench.get_stats("capture")
        assert stats["count"] == 1

    def test_get_stats_empty_after_reset_all(self):
        bench = _make_bench()
        bench.record("capture", 10.0)
        bench.reset()
        assert bench.get_stats("capture") == {}

    def test_report_empty_after_reset_all(self):
        bench = _make_bench()
        bench.record("capture", 10.0)
        bench.reset()
        assert bench.report()["categories"] == {}


# ---------------------------------------------------------------------------
# CaptureBenchmark — thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_records_do_not_corrupt_state(self):
        bench = _make_bench(max_samples=500)
        errors: list[Exception] = []

        def worker(category: str, n: int) -> None:
            try:
                for i in range(n):
                    bench.record(category, float(i))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=("capture", 50)),
            threading.Thread(target=worker, args=("pipeline", 50)),
            threading.Thread(target=worker, args=("burst", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        total = sum(len(v) for v in bench._samples.values())
        assert total == 150

    def test_concurrent_reset_and_record(self):
        bench = _make_bench(max_samples=500)
        errors: list[Exception] = []

        def recorder():
            try:
                for i in range(100):
                    bench.record("capture", float(i))
            except Exception as exc:
                errors.append(exc)

        def resetter():
            try:
                for _ in range(10):
                    bench.reset()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=recorder)
        t2 = threading.Thread(target=resetter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []


# ---------------------------------------------------------------------------
# BenchmarkTimer — context manager
# ---------------------------------------------------------------------------


class TestBenchmarkTimer:
    def test_records_elapsed_ms_to_benchmark(self):
        bench = _make_bench()
        with BenchmarkTimer(bench, "capture"):
            time.sleep(0.01)
        stats = bench.get_stats("capture")
        assert stats["count"] == 1
        # Should have recorded at least ~10 ms.
        assert stats["min_ms"] >= 1.0

    def test_elapsed_ms_attribute_set_after_exit(self):
        bench = _make_bench()
        with BenchmarkTimer(bench, "pipeline") as timer:
            time.sleep(0.005)
        assert timer.elapsed_ms >= 1.0

    def test_elapsed_ms_zero_before_exit(self):
        bench = _make_bench()
        timer = BenchmarkTimer(bench, "x")
        timer.__enter__()
        assert timer.elapsed_ms == 0.0
        timer.__exit__(None, None, None)

    def test_metadata_forwarded_to_benchmark(self):
        bench = _make_bench()
        with BenchmarkTimer(bench, "capture", device="/dev/video0", quality=0.9):
            pass
        meta = bench._samples["capture"][0].metadata
        assert meta["device"] == "/dev/video0"
        assert meta["quality"] == 0.9

    def test_records_even_when_body_raises(self):
        bench = _make_bench()
        with pytest.raises(ValueError), BenchmarkTimer(bench, "capture"):
            raise ValueError("boom")
        # The record should still have been made.
        assert bench.get_stats("capture")["count"] == 1

    def test_returns_self_on_enter(self):
        bench = _make_bench()
        timer = BenchmarkTimer(bench, "save")
        result = timer.__enter__()
        timer.__exit__(None, None, None)
        assert result is timer

    def test_nested_timers_record_independently(self):
        bench = _make_bench()
        with BenchmarkTimer(bench, "capture"), BenchmarkTimer(bench, "pipeline"):
            time.sleep(0.005)
        assert bench.get_stats("capture")["count"] == 1
        assert bench.get_stats("pipeline")["count"] == 1

    def test_elapsed_ms_reflects_actual_duration(self):
        bench = _make_bench()
        with BenchmarkTimer(bench, "save") as timer:
            time.sleep(0.02)
        # elapsed_ms should be at least 15 ms (generous lower bound for slow CI).
        assert timer.elapsed_ms >= 5.0

    def test_multiple_uses_accumulate_samples(self):
        bench = _make_bench()
        for _ in range(5):
            with BenchmarkTimer(bench, "capture"):
                pass
        assert bench.get_stats("capture")["count"] == 5

    def test_category_stored_on_sample(self):
        bench = _make_bench()
        with BenchmarkTimer(bench, "burst"):
            pass
        assert bench._samples["burst"][0].category == "burst"


# ---------------------------------------------------------------------------
# get_benchmark() — module singleton
# ---------------------------------------------------------------------------


class TestGetBenchmark:
    def setup_method(self):
        """Reset module-level singleton before each test."""
        import missy.vision.benchmark as _mod

        _mod._benchmark = None

    def test_returns_capture_benchmark_instance(self):
        bench = get_benchmark()
        assert isinstance(bench, CaptureBenchmark)

    def test_returns_same_instance_on_repeated_calls(self):
        b1 = get_benchmark()
        b2 = get_benchmark()
        assert b1 is b2

    def test_singleton_is_functional(self):
        bench = get_benchmark()
        bench.record_capture(latency_ms=5.0)
        assert bench.get_stats("capture")["count"] == 1

    def test_singleton_shared_across_calls(self):
        b1 = get_benchmark()
        b1.record_capture(latency_ms=7.0)
        b2 = get_benchmark()
        assert b2.get_stats("capture")["count"] == 1

    def test_singleton_created_fresh_after_none_reset(self):
        import missy.vision.benchmark as _mod

        _mod._benchmark = None
        b1 = get_benchmark()
        _mod._benchmark = None
        b2 = get_benchmark()
        assert b1 is not b2


# ---------------------------------------------------------------------------
# Edge cases — statistical correctness
# ---------------------------------------------------------------------------


class TestStatisticalEdgeCases:
    def test_all_identical_values_stddev_zero(self):
        bench = _make_bench()
        _record_n(bench, "capture", [42.0] * 10)
        stats = bench.get_stats("capture")
        assert stats["stddev_ms"] == 0.0

    def test_large_spread_stddev_nonzero(self):
        bench = _make_bench()
        _record_n(bench, "capture", [1.0, 1000.0])
        stats = bench.get_stats("capture")
        assert stats["stddev_ms"] > 100.0

    def test_fractional_durations_handled(self):
        # Use values that differ after rounding to 2 decimal places.
        bench = _make_bench()
        bench.record("capture", 0.01)
        bench.record("capture", 0.02)
        stats = bench.get_stats("capture")
        assert stats["min_ms"] < stats["max_ms"]

    def test_very_large_durations_handled(self):
        bench = _make_bench()
        bench.record("capture", 1_000_000.0)
        stats = bench.get_stats("capture")
        assert stats["max_ms"] == 1_000_000.0

    def test_zero_duration_recorded(self):
        bench = _make_bench()
        bench.record("save", 0.0)
        stats = bench.get_stats("save")
        assert stats["min_ms"] == 0.0
        assert stats["max_ms"] == 0.0

    def test_p95_boundary_exactly_20_samples(self):
        bench = _make_bench()
        # Exactly 20 samples: p95 index = int(20 * 0.95) = 19, value = 20.
        _record_n(bench, "capture", [float(i) for i in range(1, 21)])
        stats = bench.get_stats("capture")
        assert stats["count"] == 20
        # Index 19 of sorted [1..20] is 20.
        assert stats["p95_ms"] == 20.0

    def test_p99_boundary_exactly_100_samples(self):
        bench = _make_bench(max_samples=200)
        # Exactly 100 samples: p99 index = int(100 * 0.99) = 99, value = 100.
        _record_n(bench, "capture", [float(i) for i in range(1, 101)])
        stats = bench.get_stats("capture")
        assert stats["count"] == 100
        assert stats["p99_ms"] == 100.0

    def test_report_uptime_nonnegative_after_reset(self):
        bench = _make_bench()
        time.sleep(0.01)
        bench.reset()
        report = bench.report()
        assert report["uptime_seconds"] >= 0.0

    def test_get_stats_not_mutated_by_subsequent_records(self):
        bench = _make_bench()
        bench.record("capture", 10.0)
        stats_before = bench.get_stats("capture")
        bench.record("capture", 90.0)
        stats_after = bench.get_stats("capture")
        # First call should be a snapshot, not affected by later insert.
        assert stats_before["count"] == 1
        assert stats_after["count"] == 2
