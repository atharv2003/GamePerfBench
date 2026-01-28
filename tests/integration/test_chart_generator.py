"""
Integration tests for ChartGenerator.

Tests chart generation from BenchmarkSession objects.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from src.core.models import (
    BenchmarkRun,
    BenchmarkSession,
    StutterEvent,
    StutterSeverity,
    VarianceCheckResult,
)
from src.reporting.chart_generator import ChartGenerator


def create_test_session(num_runs: int = 2) -> BenchmarkSession:
    """Create a minimal test session for chart generation.

    Args:
        num_runs: Number of runs to create.

    Returns:
        BenchmarkSession with test data.
    """
    session = BenchmarkSession(
        session_id="test_session_123",
        game_name="TestGame",
        preset_name="high",
        timestamp=datetime.now(),
        config={"num_trials": num_runs},
    )

    for i in range(num_runs):
        # Create realistic frame times
        np.random.seed(42 + i)
        frame_times = np.random.lognormal(mean=np.log(16.67), sigma=0.05, size=100)
        timestamps = np.cumsum(frame_times)
        timestamps = np.insert(timestamps[:-1], 0, 0.0)

        # Create some stutter events
        stutter_events = []
        if i == 0:
            stutter_events = [
                StutterEvent(
                    start_index=10,
                    start_time_ms=100.0,
                    end_time_ms=150.0,
                    duration_ms=50.0,
                    max_frame_time_ms=55.0,
                    severity=StutterSeverity.MINOR,
                    frame_count=1,
                ),
                StutterEvent(
                    start_index=50,
                    start_time_ms=500.0,
                    end_time_ms=530.0,
                    duration_ms=30.0,
                    max_frame_time_ms=30.0,
                    severity=StutterSeverity.MICRO,
                    frame_count=1,
                ),
            ]

        run = BenchmarkRun(
            run_id=f"test_run_{i}",
            timestamp=datetime.now(),
            trial_index=i,
            config={"preset": "high"},
            frame_times_ms=frame_times,
            timestamps_ms=timestamps,
            fps_metrics={
                "avg_fps": 60.0 + i * 0.5,
                "one_percent_low_fps": 55.0 + i * 0.3,
                "duration_seconds": 1.67,
            },
            percentiles={
                "p50_ms": 16.5,
                "p90_ms": 17.5,
                "p95_ms": 18.0,
                "p99_ms": 20.0 + i,
                "p99_9_ms": 25.0,
            },
            stutter_events=stutter_events,
            stutter_summary={
                "total_events": len(stutter_events),
                "counts_by_severity": {
                    "micro": 1 if i == 0 else 0,
                    "minor": 1 if i == 0 else 0,
                    "major": 0,
                    "severe": 0,
                    "freeze": 0,
                },
                "worst_severity": "minor" if i == 0 else None,
            },
            duration_seconds=1.67,
            total_frames=100,
        )

        session.runs.append(run)

    # Add aggregate metrics
    session.aggregate_metrics = {
        "num_runs": num_runs,
        "avg_of_avg_fps": 60.25,
        "avg_of_one_percent_low_fps": 55.15,
        "worst_stutter_severity_over_session": "minor",
    }

    # Add variance result
    session.variance_result = VarianceCheckResult(
        passed=True,
        cv_percent=0.83,
        max_deviation=0.25,
        run_means=[60.0, 60.5],
        overall_mean=60.25,
        message="Variance check PASSED",
    )

    return session


class TestChartGeneratorBasic:
    """Basic chart generator tests."""

    def test_generate_session_charts_creates_files(self):
        """Test that generate_session_charts creates expected PNG files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            generator = ChartGenerator()

            charts = generator.generate_session_charts(session, session_dir)

            # Check that expected files exist
            expected_files = [
                "avg_fps_by_trial.png",
                "one_percent_low_by_trial.png",
                "frametime_p99_by_trial.png",
                "stutter_events_by_severity.png",
                "session_summary.png",
            ]

            assert len(charts) == len(expected_files)

            for filename in expected_files:
                filepath = session_dir / filename
                assert filepath.exists(), f"{filename} should exist"
                assert filepath.stat().st_size > 0, f"{filename} should not be empty"
                assert filename in charts

    def test_chart_files_are_valid_png(self):
        """Test that generated chart files have PNG headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            generator = ChartGenerator()

            charts = generator.generate_session_charts(session, session_dir)

            for filepath in charts.values():
                with open(filepath, "rb") as f:
                    header = f.read(8)
                    # PNG magic bytes
                    assert header[:4] == b"\x89PNG", f"{filepath} should be a valid PNG"


class TestAvgFPSChart:
    """Tests for average FPS chart."""

    def test_avg_fps_chart_content(self):
        """Test that avg_fps chart is generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=3)
            generator = ChartGenerator()

            generator.generate_session_charts(session, session_dir)

            filepath = session_dir / "avg_fps_by_trial.png"
            assert filepath.exists()
            assert filepath.stat().st_size > 1000  # Should be a reasonable size


class TestOnePercentLowChart:
    """Tests for 1% low FPS chart."""

    def test_one_percent_low_chart_exists(self):
        """Test that 1% low chart is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            generator = ChartGenerator()

            charts = generator.generate_session_charts(session, session_dir)

            filepath = session_dir / "one_percent_low_by_trial.png"
            assert filepath.exists()
            assert "one_percent_low_by_trial.png" in charts


class TestP99Chart:
    """Tests for P99 frame time chart."""

    def test_p99_chart_exists(self):
        """Test that P99 chart is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            generator = ChartGenerator()

            generator.generate_session_charts(session, session_dir)

            filepath = session_dir / "frametime_p99_by_trial.png"
            assert filepath.exists()


class TestStutterChart:
    """Tests for stutter events chart."""

    def test_stutter_chart_with_events(self):
        """Test stutter chart when events exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            generator = ChartGenerator()

            generator.generate_session_charts(session, session_dir)

            filepath = session_dir / "stutter_events_by_severity.png"
            assert filepath.exists()
            assert filepath.stat().st_size > 1000

    def test_stutter_chart_no_events(self):
        """Test stutter chart when no events exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)

            # Clear stutter data
            for run in session.runs:
                run.stutter_events = []
                run.stutter_summary = {
                    "total_events": 0,
                    "counts_by_severity": {
                        "micro": 0,
                        "minor": 0,
                        "major": 0,
                        "severe": 0,
                        "freeze": 0,
                    },
                }

            generator = ChartGenerator()
            generator.generate_session_charts(session, session_dir)

            filepath = session_dir / "stutter_events_by_severity.png"
            assert filepath.exists()


class TestSummaryChart:
    """Tests for session summary chart."""

    def test_summary_chart_exists(self):
        """Test that summary chart is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            generator = ChartGenerator()

            generator.generate_session_charts(session, session_dir)

            filepath = session_dir / "session_summary.png"
            assert filepath.exists()


class TestPerRunCharts:
    """Tests for per-run chart generation."""

    def test_generate_run_charts(self):
        """Test generating charts for a single run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            session = create_test_session(num_runs=1)
            run = session.runs[0]

            generator = ChartGenerator()
            charts = generator.generate_run_charts(run, run_dir, session.session_id)

            assert len(charts) > 0
            assert "trial_0_frametime_series.png" in charts

            filepath = run_dir / "trial_0_frametime_series.png"
            assert filepath.exists()
            assert filepath.stat().st_size > 1000


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_session(self):
        """Test chart generation with empty session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = BenchmarkSession(
                session_id="empty_session",
                game_name="EmptyGame",
                preset_name="high",
                timestamp=datetime.now(),
            )

            generator = ChartGenerator()
            charts = generator.generate_session_charts(session, session_dir)

            # Should still create charts (with "no data" messages)
            assert len(charts) == 5

    def test_single_run_session(self):
        """Test chart generation with single run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=1)

            generator = ChartGenerator()
            charts = generator.generate_session_charts(session, session_dir)

            assert len(charts) == 5
            for filepath in charts.values():
                assert Path(filepath).exists()

    def test_custom_dpi(self):
        """Test chart generation with custom DPI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)

            generator = ChartGenerator(dpi=150)
            charts = generator.generate_session_charts(session, session_dir)

            # Higher DPI should result in larger file sizes
            for filepath in charts.values():
                assert Path(filepath).exists()
