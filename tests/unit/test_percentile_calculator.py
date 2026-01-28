"""Unit tests for PercentileCalculator."""

from src.analysis.percentile_calculator import PercentileCalculator


class TestCalculateFrameTimePercentiles:
    """Tests for calculate_frame_time_percentiles method."""

    def test_known_values_simple(self):
        """Test percentiles on simple known data."""
        calc = PercentileCalculator()
        # 10 values from 1 to 10
        frame_times = list(range(1, 11))

        result = calc.calculate_frame_time_percentiles(frame_times, [50, 90])

        # Median of 1-10 with "higher" method should be 6
        # (values at index 4 and 5 are 5 and 6, higher = 6)
        assert result["p50_ms"] == 6.0
        # 90th percentile should be 10 (higher method)
        assert result["p90_ms"] == 10.0

    def test_default_percentiles(self):
        """Test default percentiles [50, 90, 95, 99, 99.9]."""
        calc = PercentileCalculator()
        frame_times = list(range(1, 1001))  # 1 to 1000

        result = calc.calculate_frame_time_percentiles(frame_times)

        expected_keys = ["p50_ms", "p90_ms", "p95_ms", "p99_ms", "p99_9_ms"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_custom_percentiles(self):
        """Test custom percentiles list."""
        calc = PercentileCalculator(percentiles=[25, 75])
        frame_times = list(range(1, 101))

        result = calc.calculate_frame_time_percentiles(frame_times)

        assert "p25_ms" in result
        assert "p75_ms" in result
        assert len(result) == 2

    def test_override_percentiles_in_call(self):
        """Test overriding percentiles in method call."""
        calc = PercentileCalculator(percentiles=[50])
        frame_times = list(range(1, 11))

        result = calc.calculate_frame_time_percentiles(frame_times, percentiles=[10])

        assert "p10_ms" in result
        assert "p50_ms" not in result

    def test_deterministic_results(self):
        """Test results are deterministic."""
        calc = PercentileCalculator()
        frame_times = [10.0, 20.0, 30.0, 40.0, 50.0]

        result1 = calc.calculate_frame_time_percentiles(frame_times)
        result2 = calc.calculate_frame_time_percentiles(frame_times)

        assert result1 == result2

    def test_empty_input(self):
        """Test with empty input returns zeros."""
        calc = PercentileCalculator()
        result = calc.calculate_frame_time_percentiles([])

        for value in result.values():
            assert value == 0.0

    def test_single_value(self):
        """Test with single value."""
        calc = PercentileCalculator()
        result = calc.calculate_frame_time_percentiles([42.0])

        # All percentiles should be the same value
        for value in result.values():
            assert value == 42.0

    def test_key_naming_decimal(self):
        """Test key naming for decimal percentiles like 99.9."""
        calc = PercentileCalculator(percentiles=[99.9])
        frame_times = list(range(1, 1001))

        result = calc.calculate_frame_time_percentiles(frame_times)

        assert "p99_9_ms" in result


class TestCalculateFPSPercentiles:
    """Tests for calculate_fps_percentiles method."""

    def test_fps_conversion(self):
        """Test FPS percentiles are 1000 / frame_time."""
        calc = PercentileCalculator()
        # All 10ms frame times = 100 FPS
        frame_times = [10.0] * 100

        result = calc.calculate_fps_percentiles(frame_times)

        assert abs(result["p50_fps"] - 100.0) < 0.01
        assert abs(result["p99_fps"] - 100.0) < 0.01

    def test_key_naming(self):
        """Test FPS keys use _fps suffix."""
        calc = PercentileCalculator(percentiles=[50, 99])
        frame_times = [10.0] * 10

        result = calc.calculate_fps_percentiles(frame_times)

        assert "p50_fps" in result
        assert "p99_fps" in result
        assert "p50_ms" not in result


class TestFormatKey:
    """Tests for _format_key static method."""

    def test_integer_percentile(self):
        """Test integer percentile key."""
        key = PercentileCalculator._format_key(50)
        assert key == "p50_ms"

    def test_decimal_percentile(self):
        """Test decimal percentile key."""
        key = PercentileCalculator._format_key(99.9)
        assert key == "p99_9_ms"

    def test_small_decimal(self):
        """Test small decimal percentile."""
        key = PercentileCalculator._format_key(0.1)
        assert key == "p0_1_ms"
