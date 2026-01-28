"""Integration tests for CSVExporter."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.percentile_calculator import PercentileCalculator
from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.analysis.stutter_detector import StutterDetector
from src.capture.fps_capture import FPSCalculator
from src.core.models import BenchmarkRun, BenchmarkSession
from src.reporting.csv_exporter import CSVExporter


def create_mock_run(
    run_id: str,
    trial_index: int,
    frame_times: np.ndarray,
    timestamp: datetime,
) -> BenchmarkRun:
    """Create a mock BenchmarkRun with computed metrics.

    Args:
        run_id: Unique run identifier.
        trial_index: Index in session (0-based).
        frame_times: Array of frame times in ms.
        timestamp: Run start timestamp.

    Returns:
        BenchmarkRun with all computed metrics.
    """
    # Compute timestamps from frame times
    timestamps = np.cumsum(frame_times)
    timestamps = np.insert(timestamps[:-1], 0, 0.0)

    # Compute FPS metrics
    fps_calc = FPSCalculator()
    fps_metrics = fps_calc.calculate_summary_metrics(frame_times)

    # Compute percentiles
    perc_calc = PercentileCalculator()
    percentiles = perc_calc.calculate_frame_time_percentiles(frame_times)

    # Detect stutters
    stutter_det = StutterDetector()
    stutter_events = stutter_det.detect_stutters(timestamps, frame_times)
    stutter_summary = stutter_det.calculate_summary(
        stutter_events, total_duration_ms=float(np.sum(frame_times))
    )

    # Compute distribution stats (not stored in BenchmarkRun, but available)
    stat_analyzer = StatisticalAnalyzer()
    stat_analyzer.analyze_distribution(frame_times)

    run = BenchmarkRun(
        run_id=run_id,
        timestamp=timestamp,
        trial_index=trial_index,
        frame_times_ms=frame_times,
        timestamps_ms=timestamps,
        fps_metrics=fps_metrics,
        percentiles=percentiles,
        stutter_events=stutter_events,
        stutter_summary=stutter_summary,
        duration_seconds=fps_metrics.get("duration_seconds", 0.0),
        total_frames=len(frame_times),
    )

    return run


def create_mock_session(session_id: str = "test_session_001") -> BenchmarkSession:
    """Create a mock BenchmarkSession with 2 runs.

    Returns:
        BenchmarkSession with deterministic test data.
    """
    # Run 1: Mostly smooth with one minor stutter
    frame_times_1 = np.array(
        [16.6667] * 50  # 50 normal frames at 60 FPS
        + [60.0]  # 1 minor stutter (>50ms)
        + [16.6667] * 49  # 49 more normal frames
    )

    # Run 2: Smooth with one major stutter
    frame_times_2 = np.array(
        [16.6667] * 40  # 40 normal frames
        + [120.0]  # 1 major stutter (>100ms)
        + [16.6667] * 59  # 59 more normal frames
    )

    base_time = datetime(2024, 1, 15, 10, 0, 0)

    run1 = create_mock_run(
        run_id="run_001",
        trial_index=0,
        frame_times=frame_times_1,
        timestamp=base_time,
    )

    run2 = create_mock_run(
        run_id="run_002",
        trial_index=1,
        frame_times=frame_times_2,
        timestamp=datetime(2024, 1, 15, 10, 5, 0),
    )

    session = BenchmarkSession(
        session_id=session_id,
        game_name="test_game",
        preset_name="high",
        timestamp=base_time,
        runs=[run1, run2],
    )

    return session


class TestCSVExporterBasic:
    """Basic tests for CSVExporter."""

    def test_export_creates_directory(self, tmp_path: Path):
        """Test export creates output directory."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        result = exporter.export_session(session, output_dir=output_dir)

        assert result == output_dir
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_export_creates_all_csv_files(self, tmp_path: Path):
        """Test export creates all required CSV files."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        assert (output_dir / "raw_frametimes.csv").exists()
        assert (output_dir / "run_summaries.csv").exists()
        assert (output_dir / "stutter_events.csv").exists()

    def test_export_uses_default_directory(self, tmp_path: Path):
        """Test export uses default directory structure."""
        session = create_mock_session("my_session")
        exporter = CSVExporter(output_root=tmp_path)

        result = exporter.export_session(session)

        expected_dir = tmp_path / "runs" / "my_session"
        assert result == expected_dir
        assert expected_dir.exists()


class TestRawFrametimesCSV:
    """Tests for raw_frametimes.csv output."""

    def test_has_required_columns(self, tmp_path: Path):
        """Test raw_frametimes.csv has all required columns."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "raw_frametimes.csv")

        required_columns = [
            "session_id",
            "run_id",
            "trial_index",
            "timestamp_ms",
            "frame_time_ms",
            "present_time_ms",
            "fps_instantaneous",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_correct_row_count(self, tmp_path: Path):
        """Test raw_frametimes.csv has correct number of rows."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "raw_frametimes.csv")

        # Run 1: 100 frames, Run 2: 100 frames
        expected_rows = 100 + 100
        assert len(df) == expected_rows

    def test_frame_times_match_input(self, tmp_path: Path):
        """Test frame times in CSV match input data."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "raw_frametimes.csv")

        # Check run 1 has 60ms stutter frame
        run1_data = df[df["run_id"] == "run_001"]
        assert 60.0 in run1_data["frame_time_ms"].values

        # Check run 2 has 120ms stutter frame
        run2_data = df[df["run_id"] == "run_002"]
        assert 120.0 in run2_data["frame_time_ms"].values

    def test_sorted_by_trial_and_timestamp(self, tmp_path: Path):
        """Test data is sorted by trial_index then timestamp_ms."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "raw_frametimes.csv")

        # First 100 rows should be trial 0
        assert all(df.iloc[:100]["trial_index"] == 0)
        # Next 100 rows should be trial 1
        assert all(df.iloc[100:]["trial_index"] == 1)

        # Timestamps within each trial should be increasing
        trial0 = df[df["trial_index"] == 0]["timestamp_ms"]
        assert trial0.is_monotonic_increasing


class TestRunSummariesCSV:
    """Tests for run_summaries.csv output."""

    def test_has_required_columns(self, tmp_path: Path):
        """Test run_summaries.csv has all required columns."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "run_summaries.csv")

        required_columns = [
            "session_id",
            "run_id",
            "trial_index",
            "avg_fps",
            "one_percent_low_fps",
            "p99_ms",
            "stutter_count_total",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_two_rows(self, tmp_path: Path):
        """Test run_summaries.csv has one row per run."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "run_summaries.csv")

        assert len(df) == 2

    def test_avg_fps_values(self, tmp_path: Path):
        """Test avg_fps values are reasonable."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "run_summaries.csv")

        # Both runs are mostly 60 FPS with one stutter
        # Avg FPS should be close to 60 (within 10%)
        for avg_fps in df["avg_fps"]:
            assert 50 < avg_fps < 70

    def test_stutter_count_matches(self, tmp_path: Path):
        """Test stutter count matches detected stutters."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "run_summaries.csv")

        # Each run has 1 stutter
        assert df.iloc[0]["stutter_count_total"] == 1
        assert df.iloc[1]["stutter_count_total"] == 1

    def test_p99_ms_values(self, tmp_path: Path):
        """Test p99_ms percentile values are computed correctly."""
        session = create_mock_session()

        # Independently compute expected p99 for run 1
        perc_calc = PercentileCalculator()
        expected_p99 = perc_calc.calculate_frame_time_percentiles(
            session.runs[0].frame_times_ms
        )["p99_ms"]

        exporter = CSVExporter(output_root=tmp_path)
        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "run_summaries.csv")

        # p99_ms should match our independent calculation
        assert abs(df.iloc[0]["p99_ms"] - expected_p99) < 0.01


class TestStutterEventsCSV:
    """Tests for stutter_events.csv output."""

    def test_has_required_columns(self, tmp_path: Path):
        """Test stutter_events.csv has all required columns."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "stutter_events.csv")

        required_columns = [
            "session_id",
            "run_id",
            "trial_index",
            "start_time_ms",
            "end_time_ms",
            "duration_ms",
            "max_frame_time_ms",
            "severity",
            "frame_count",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_correct_event_count(self, tmp_path: Path):
        """Test stutter_events.csv has correct number of events."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "stutter_events.csv")

        # Each run has 1 stutter event
        assert len(df) == 2

    def test_severity_values(self, tmp_path: Path):
        """Test severity values are correct."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "stutter_events.csv")

        # Run 1 has 60ms stutter (minor)
        run1_event = df[df["run_id"] == "run_001"].iloc[0]
        assert run1_event["severity"] == "minor"

        # Run 2 has 120ms stutter (major)
        run2_event = df[df["run_id"] == "run_002"].iloc[0]
        assert run2_event["severity"] == "major"

    def test_max_frame_time_values(self, tmp_path: Path):
        """Test max_frame_time_ms values match stutter frames."""
        session = create_mock_session()
        exporter = CSVExporter(output_root=tmp_path)

        output_dir = tmp_path / "session_out"
        exporter.export_session(session, output_dir=output_dir)

        df = pd.read_csv(output_dir / "stutter_events.csv")

        # Run 1 stutter: 60ms
        run1_event = df[df["run_id"] == "run_001"].iloc[0]
        assert run1_event["max_frame_time_ms"] == 60.0

        # Run 2 stutter: 120ms
        run2_event = df[df["run_id"] == "run_002"].iloc[0]
        assert run2_event["max_frame_time_ms"] == 120.0


class TestCSVExporterEdgeCases:
    """Tests for edge cases."""

    def test_empty_session(self, tmp_path: Path):
        """Test export handles session with no runs."""
        session = BenchmarkSession(
            session_id="empty_session",
            game_name="test",
            preset_name="high",
            timestamp=datetime.now(),
            runs=[],
        )

        exporter = CSVExporter(output_root=tmp_path)
        output_dir = tmp_path / "empty_out"
        exporter.export_session(session, output_dir=output_dir)

        # All files should exist but be empty (just headers)
        assert (output_dir / "raw_frametimes.csv").exists()
        assert (output_dir / "run_summaries.csv").exists()
        assert (output_dir / "stutter_events.csv").exists()

    def test_run_with_no_stutters(self, tmp_path: Path):
        """Test export handles run with no stutters."""
        # All smooth frames, no stutters
        frame_times = np.array([16.6667] * 100)

        run = create_mock_run(
            run_id="smooth_run",
            trial_index=0,
            frame_times=frame_times,
            timestamp=datetime.now(),
        )

        session = BenchmarkSession(
            session_id="smooth_session",
            game_name="test",
            preset_name="high",
            timestamp=datetime.now(),
            runs=[run],
        )

        exporter = CSVExporter(output_root=tmp_path)
        output_dir = tmp_path / "smooth_out"
        exporter.export_session(session, output_dir=output_dir)

        # stutter_events.csv should exist but have no data rows
        df = pd.read_csv(output_dir / "stutter_events.csv")
        assert len(df) == 0
