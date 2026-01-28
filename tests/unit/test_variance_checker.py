"""Unit tests for VarianceChecker."""

from datetime import datetime

from src.core.models import BenchmarkRun
from src.core.variance_checker import VarianceChecker


class TestVarianceCheckerBasic:
    """Basic tests for VarianceChecker."""

    def test_pass_with_low_variance(self):
        """Test pass with low variance runs."""
        checker = VarianceChecker()
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 60.5, "one_percent_low_fps": 54.5},
            {"avg_fps": 59.5, "one_percent_low_fps": 55.5},
        ]

        result = checker.check_variance(runs)

        assert result.passed is True
        assert "PASSED" in result.message

    def test_fail_with_high_variance(self):
        """Test fail with high variance runs."""
        checker = VarianceChecker()
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 80.0, "one_percent_low_fps": 70.0},
            {"avg_fps": 40.0, "one_percent_low_fps": 35.0},
        ]

        result = checker.check_variance(runs)

        assert result.passed is False
        assert "FAILED" in result.message

    def test_fail_insufficient_runs(self):
        """Test fail with only one run."""
        checker = VarianceChecker()
        runs = [{"avg_fps": 60.0, "one_percent_low_fps": 55.0}]

        result = checker.check_variance(runs)

        assert result.passed is False
        assert "Insufficient runs" in result.message

    def test_fail_empty_runs(self):
        """Test fail with empty runs list."""
        checker = VarianceChecker()
        result = checker.check_variance([])

        assert result.passed is False


class TestVarianceCheckerThresholds:
    """Tests for custom thresholds."""

    def test_custom_avg_fps_threshold(self):
        """Test with custom avg_fps CV threshold."""
        # Strict threshold
        checker = VarianceChecker(max_cv_avg_fps=0.01)  # 1%
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 61.0, "one_percent_low_fps": 56.0},
            {"avg_fps": 59.0, "one_percent_low_fps": 54.0},
        ]

        result = checker.check_variance(runs)
        # CV of [60, 61, 59] is ~1.67%, which exceeds 1%
        assert result.passed is False

    def test_custom_one_percent_threshold(self):
        """Test with custom one_percent_low CV threshold."""
        checker = VarianceChecker(max_cv_one_percent_low=0.05)  # 5%
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 50.0},
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 60.0, "one_percent_low_fps": 60.0},
        ]

        result = checker.check_variance(runs)
        # CV of [50, 55, 60] is ~9%, which exceeds 5%
        assert result.passed is False

    def test_custom_min_runs(self):
        """Test with custom minimum runs."""
        checker = VarianceChecker(min_runs=5)
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
        ]

        result = checker.check_variance(runs)
        assert result.passed is False
        assert "3 < 5 required" in result.message


class TestVarianceCheckerBenchmarkRun:
    """Tests with BenchmarkRun objects."""

    def test_with_benchmark_run_objects(self):
        """Test with BenchmarkRun objects instead of dicts."""
        checker = VarianceChecker()

        run1 = BenchmarkRun(run_id="run1", timestamp=datetime.now())
        run1.fps_metrics = {"avg_fps": 60.0, "one_percent_low_fps": 55.0}

        run2 = BenchmarkRun(run_id="run2", timestamp=datetime.now())
        run2.fps_metrics = {"avg_fps": 60.5, "one_percent_low_fps": 54.5}

        run3 = BenchmarkRun(run_id="run3", timestamp=datetime.now())
        run3.fps_metrics = {"avg_fps": 59.5, "one_percent_low_fps": 55.5}

        result = checker.check_variance([run1, run2, run3])

        assert result.passed is True

    def test_missing_metrics_in_run(self):
        """Test failure when metrics are missing."""
        checker = VarianceChecker()

        run1 = BenchmarkRun(run_id="run1", timestamp=datetime.now())
        run1.fps_metrics = {"avg_fps": 60.0}  # Missing one_percent_low_fps

        run2 = BenchmarkRun(run_id="run2", timestamp=datetime.now())
        run2.fps_metrics = {"avg_fps": 60.0, "one_percent_low_fps": 55.0}

        result = checker.check_variance([run1, run2])

        assert result.passed is False
        assert "could not extract" in result.message.lower()


class TestVarianceCheckerOutput:
    """Tests for output values."""

    def test_overall_mean(self):
        """Test overall_mean is calculated correctly."""
        checker = VarianceChecker()
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 70.0, "one_percent_low_fps": 65.0},
            {"avg_fps": 80.0, "one_percent_low_fps": 75.0},
        ]

        result = checker.check_variance(runs)

        assert result.overall_mean == 70.0  # (60 + 70 + 80) / 3

    def test_run_means_preserved(self):
        """Test run_means contains all input values."""
        checker = VarianceChecker()
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 61.0, "one_percent_low_fps": 56.0},
        ]

        result = checker.check_variance(runs)

        assert result.run_means == [60.0, 61.0]

    def test_max_deviation(self):
        """Test max_deviation calculation."""
        checker = VarianceChecker()
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 65.0, "one_percent_low_fps": 60.0},
            {"avg_fps": 55.0, "one_percent_low_fps": 50.0},
        ]
        # Mean = 60, max deviation = 5 (65-60 or 60-55)

        result = checker.check_variance(runs)

        assert result.max_deviation == 5.0

    def test_cv_percent_in_result(self):
        """Test cv_percent is included in result."""
        checker = VarianceChecker()
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
        ]

        result = checker.check_variance(runs)

        assert result.cv_percent == 0.0  # Identical values

    def test_message_includes_cv_values(self):
        """Test message includes CV values for debugging."""
        checker = VarianceChecker()
        runs = [
            {"avg_fps": 60.0, "one_percent_low_fps": 55.0},
            {"avg_fps": 60.5, "one_percent_low_fps": 54.5},
            {"avg_fps": 59.5, "one_percent_low_fps": 55.5},
        ]

        result = checker.check_variance(runs)

        assert "CV=" in result.message or "cv" in result.message.lower()
