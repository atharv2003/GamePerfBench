"""
Integration tests for HTMLReporter.

Tests HTML report generation from BenchmarkSession objects.
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
from src.reporting.html_reporter import HTMLReporter


def create_test_session(num_runs: int = 2) -> BenchmarkSession:
    """Create a minimal test session for report generation.

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


class TestHTMLReporterBasic:
    """Basic HTML reporter tests."""

    def test_generate_report_creates_file(self):
        """Test that generate_report creates report.html."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            report_path = reporter.generate_report(session, session_dir)

            assert report_path.exists()
            assert report_path.name == "report.html"
            assert report_path.stat().st_size > 0

    def test_report_is_valid_html(self):
        """Test that generated report is valid HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert html_content.startswith("<!DOCTYPE html>")
            assert "<html" in html_content
            assert "</html>" in html_content
            assert "<head>" in html_content
            assert "<body>" in html_content

    def test_report_contains_session_info(self):
        """Test that report contains session information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert session.session_id in html_content
            assert session.game_name in html_content
            assert session.preset_name in html_content


class TestHTMLReporterSections:
    """Tests for specific HTML report sections."""

    def test_report_contains_summary_metrics(self):
        """Test that report contains summary metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert "60.25" in html_content  # avg FPS
            assert "55.15" in html_content  # 1% low FPS
            assert "Number of Runs" in html_content
            assert "Average FPS" in html_content

    def test_report_contains_variance_check(self):
        """Test that report contains variance check results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert "Variance Check" in html_content
            assert "PASSED" in html_content
            assert "0.83" in html_content  # CV percent
            assert "0.25" in html_content  # Max deviation

    def test_report_contains_run_details_table(self):
        """Test that report contains per-run details table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert "Run Details" in html_content
            assert "test_run_0" in html_content
            assert "test_run_1" in html_content
            assert "<th>Avg FPS</th>" in html_content
            assert "<th>1% Low FPS</th>" in html_content


class TestHTMLReporterChartLinks:
    """Tests for chart embedding in HTML reports."""

    def test_report_links_existing_charts(self):
        """Test that report includes links to existing chart files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)

            # Create dummy chart files
            chart_files = [
                "avg_fps_by_trial.png",
                "one_percent_low_by_trial.png",
                "session_summary.png",
            ]
            for chart in chart_files:
                (session_dir / chart).write_bytes(b"fake png data")

            reporter = HTMLReporter()
            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            for chart in chart_files:
                assert f'src="{chart}"' in html_content

    def test_report_handles_missing_charts(self):
        """Test that report handles missing chart files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            # No chart files created
            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert "No charts available" in html_content


class TestHTMLReporterCSVLinks:
    """Tests for CSV download links in HTML reports."""

    def test_report_links_existing_csvs(self):
        """Test that report includes download links for existing CSVs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)

            # Create dummy CSV files
            csv_files = [
                "run_summaries.csv",
                "raw_frametimes.csv",
                "stutter_events.csv",
            ]
            for csv in csv_files:
                (session_dir / csv).write_text("col1,col2\n1,2\n")

            reporter = HTMLReporter()
            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            for csv in csv_files:
                assert f'href="{csv}"' in html_content

    def test_report_handles_missing_csvs(self):
        """Test that report handles missing CSV files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            # No CSV files created
            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert "No data files available" in html_content


class TestHTMLReporterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_session(self):
        """Test report generation with empty session (no runs)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = BenchmarkSession(
                session_id="empty_session",
                game_name="EmptyGame",
                preset_name="high",
                timestamp=datetime.now(),
            )

            reporter = HTMLReporter()
            report_path = reporter.generate_report(session, session_dir)

            assert report_path.exists()
            html_content = report_path.read_text(encoding="utf-8")
            assert "No run data available" in html_content

    def test_single_run_session(self):
        """Test report generation with single run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=1)

            reporter = HTMLReporter()
            report_path = reporter.generate_report(session, session_dir)

            assert report_path.exists()
            html_content = report_path.read_text(encoding="utf-8")
            assert "test_run_0" in html_content

    def test_session_without_variance_result(self):
        """Test report generation when variance result is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            session.variance_result = None

            reporter = HTMLReporter()
            report_path = reporter.generate_report(session, session_dir)

            assert report_path.exists()
            html_content = report_path.read_text(encoding="utf-8")
            assert "Variance Check" in html_content
            assert "N/A" in html_content

    def test_custom_title(self):
        """Test report generation with custom title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)

            reporter = HTMLReporter(title="Custom Report Title")
            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert "Custom Report Title" in html_content

    def test_variance_check_failed(self):
        """Test report with failed variance check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            session.variance_result = VarianceCheckResult(
                passed=False,
                cv_percent=10.5,
                max_deviation=5.0,
                run_means=[55.0, 65.0],
                overall_mean=60.0,
                message="Variance check FAILED",
            )

            reporter = HTMLReporter()
            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert "FAILED" in html_content
            assert "status-failed" in html_content


class TestHTMLReporterCSS:
    """Tests for CSS styling in HTML reports."""

    def test_report_contains_inline_css(self):
        """Test that report contains inline CSS styles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            assert "<style>" in html_content
            assert "</style>" in html_content
            assert "font-family" in html_content
            assert ".container" in html_content

    def test_report_has_no_external_dependencies(self):
        """Test that report has no external CSS/JS dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(num_runs=2)
            reporter = HTMLReporter()

            report_path = reporter.generate_report(session, session_dir)

            html_content = report_path.read_text(encoding="utf-8")
            # Should not have external stylesheet links
            assert 'rel="stylesheet"' not in html_content
            # Should not have external script tags
            assert "<script src=" not in html_content
