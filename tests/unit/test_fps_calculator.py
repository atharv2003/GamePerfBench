"""Unit tests for FPSCalculator."""

import numpy as np

from src.capture.fps_capture import FPSCalculator


class TestComputeInstantaneousFPS:
    """Tests for compute_instantaneous_fps method."""

    def test_basic_conversion(self):
        """Test FPS = 1000 / frame_time_ms."""
        calc = FPSCalculator()
        frame_times = [16.6667, 10.0, 20.0]
        fps = calc.compute_instantaneous_fps(frame_times)

        expected = [1000.0 / 16.6667, 100.0, 50.0]
        np.testing.assert_array_almost_equal(fps, expected, decimal=2)

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        calc = FPSCalculator()
        frame_times = np.array([8.333, 16.667, 33.333])
        fps = calc.compute_instantaneous_fps(frame_times)

        assert len(fps) == 3
        assert abs(fps[0] - 120.0) < 1.0  # ~120 FPS
        assert abs(fps[1] - 60.0) < 1.0  # ~60 FPS
        assert abs(fps[2] - 30.0) < 1.0  # ~30 FPS

    def test_deterministic_output(self):
        """Test that output is deterministic."""
        calc = FPSCalculator()
        frame_times = [16.6667, 20.0, 25.0, 33.333]

        fps1 = calc.compute_instantaneous_fps(frame_times)
        fps2 = calc.compute_instantaneous_fps(frame_times)

        np.testing.assert_array_equal(fps1, fps2)

    def test_avoids_division_by_zero(self):
        """Test that zero frame times don't cause errors."""
        calc = FPSCalculator()
        frame_times = [0.0, 16.6667, 0.0]
        fps = calc.compute_instantaneous_fps(frame_times)

        # Zero frame times should become very high FPS (1000 / 0.001)
        assert fps[0] == 1000000.0  # 1000 / 0.001
        assert fps[2] == 1000000.0

    def test_empty_input(self):
        """Test with empty input."""
        calc = FPSCalculator()
        fps = calc.compute_instantaneous_fps([])
        assert len(fps) == 0


class TestCalculateSummaryMetrics:
    """Tests for calculate_summary_metrics method."""

    def test_basic_metrics(self):
        """Test basic metric calculations."""
        calc = FPSCalculator()
        # 100 frames at exactly 16.6667ms = 60 FPS
        frame_times = [16.6667] * 100
        metrics = calc.calculate_summary_metrics(frame_times)

        assert abs(metrics["avg_fps"] - 60.0) < 0.1
        assert abs(metrics["min_fps"] - 60.0) < 0.1
        assert abs(metrics["max_fps"] - 60.0) < 0.1
        assert abs(metrics["avg_frame_time_ms"] - 16.6667) < 0.01
        assert metrics["total_frames"] == 100

    def test_low_fps_calculation(self):
        """Test 1% low and 0.1% low FPS calculations.

        Definition used: 1% low = 1000 / 99th percentile frame time
        """
        calc = FPSCalculator()
        # 99 frames at 16.6667ms, 1 frame at 100ms
        frame_times = [16.6667] * 99 + [100.0]
        metrics = calc.calculate_summary_metrics(frame_times)

        # 99th percentile should be 100ms (the slow frame)
        # So 1% low FPS should be 1000/100 = 10 FPS
        assert metrics["p99_frame_time_ms"] == 100.0
        assert metrics["one_percent_low_fps"] == 10.0

    def test_percentile_method_determinism(self):
        """Test that percentile calculation is deterministic."""
        calc = FPSCalculator()
        frame_times = list(range(1, 101))  # 1ms to 100ms

        metrics1 = calc.calculate_summary_metrics(frame_times)
        metrics2 = calc.calculate_summary_metrics(frame_times)

        assert metrics1["p99_frame_time_ms"] == metrics2["p99_frame_time_ms"]
        assert metrics1["p99_9_frame_time_ms"] == metrics2["p99_9_frame_time_ms"]

    def test_empty_input(self):
        """Test with empty input returns zeros."""
        calc = FPSCalculator()
        metrics = calc.calculate_summary_metrics([])

        assert metrics["avg_fps"] == 0.0
        assert metrics["total_frames"] == 0
        assert metrics["duration_seconds"] == 0.0

    def test_all_keys_present(self):
        """Test all expected keys are present."""
        calc = FPSCalculator()
        metrics = calc.calculate_summary_metrics([16.6667] * 10)

        expected_keys = [
            "avg_fps",
            "min_fps",
            "max_fps",
            "avg_frame_time_ms",
            "min_frame_time_ms",
            "max_frame_time_ms",
            "one_percent_low_fps",
            "point_one_percent_low_fps",
            "p99_frame_time_ms",
            "p99_9_frame_time_ms",
            "total_frames",
            "duration_seconds",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_duration_calculation(self):
        """Test duration is sum of frame times."""
        calc = FPSCalculator()
        frame_times = [10.0, 20.0, 30.0]  # Total = 60ms
        metrics = calc.calculate_summary_metrics(frame_times)

        assert metrics["duration_seconds"] == 0.06  # 60ms = 0.06s

    def test_min_max_frame_time(self):
        """Test min/max frame time detection."""
        calc = FPSCalculator()
        frame_times = [10.0, 50.0, 20.0, 100.0, 5.0]
        metrics = calc.calculate_summary_metrics(frame_times)

        assert metrics["min_frame_time_ms"] == 5.0
        assert metrics["max_frame_time_ms"] == 100.0
