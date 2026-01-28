"""Unit tests for core data models."""

from datetime import datetime

import numpy as np

from src.core.models import (
    BenchmarkRun,
    BenchmarkSession,
    DistributionStats,
    FrameData,
    HardwareSnapshot,
    StutterEvent,
    StutterSeverity,
    VarianceCheckResult,
)


class TestFrameData:
    """Tests for FrameData dataclass."""

    def test_instantiation_required_fields(self):
        """Test FrameData can be instantiated with required fields."""
        frame = FrameData(
            timestamp_ms=100.0,
            frame_time_ms=16.67,
            present_time_ms=16.67,
        )
        assert frame.timestamp_ms == 100.0
        assert frame.frame_time_ms == 16.67
        assert frame.present_time_ms == 16.67

    def test_optional_fields_default_none(self):
        """Test optional fields default to None."""
        frame = FrameData(
            timestamp_ms=0.0,
            frame_time_ms=8.33,
            present_time_ms=8.33,
        )
        assert frame.gpu_busy_ms is None
        assert frame.display_latency_ms is None

    def test_optional_fields_can_be_set(self):
        """Test optional fields can be provided."""
        frame = FrameData(
            timestamp_ms=0.0,
            frame_time_ms=8.33,
            present_time_ms=8.33,
            gpu_busy_ms=7.5,
            display_latency_ms=12.0,
        )
        assert frame.gpu_busy_ms == 7.5
        assert frame.display_latency_ms == 12.0


class TestHardwareSnapshot:
    """Tests for HardwareSnapshot dataclass."""

    def test_instantiation(self):
        """Test HardwareSnapshot can be instantiated."""
        snapshot = HardwareSnapshot(timestamp_ms=1000.0)
        assert snapshot.timestamp_ms == 1000.0
        assert snapshot.payload == {}

    def test_payload_default_not_shared(self):
        """Test payload dict is not shared between instances."""
        snap1 = HardwareSnapshot(timestamp_ms=0.0)
        snap2 = HardwareSnapshot(timestamp_ms=100.0)

        snap1.payload["cpu_usage"] = 50.0

        assert "cpu_usage" in snap1.payload
        assert "cpu_usage" not in snap2.payload

    def test_payload_with_data(self):
        """Test HardwareSnapshot with payload data."""
        snapshot = HardwareSnapshot(
            timestamp_ms=500.0,
            payload={
                "cpu_usage_percent": 45.0,
                "gpu_usage_percent": 95.0,
                "gpu_temp_celsius": 72.0,
            },
        )
        assert snapshot.payload["cpu_usage_percent"] == 45.0
        assert snapshot.payload["gpu_temp_celsius"] == 72.0


class TestStutterEvent:
    """Tests for StutterEvent dataclass."""

    def test_instantiation(self):
        """Test StutterEvent can be instantiated."""
        event = StutterEvent(
            start_index=100,
            start_time_ms=5000.0,
            end_time_ms=5050.0,
            duration_ms=50.0,
            max_frame_time_ms=50.0,
            severity=StutterSeverity.MINOR,
        )
        assert event.start_index == 100
        assert event.start_time_ms == 5000.0
        assert event.end_time_ms == 5050.0
        assert event.duration_ms == 50.0
        assert event.max_frame_time_ms == 50.0
        assert event.severity == StutterSeverity.MINOR
        assert event.frame_count == 1  # default

    def test_backward_compatibility_aliases(self):
        """Test backward compatibility property aliases."""
        event = StutterEvent(
            start_index=100,
            start_time_ms=5000.0,
            end_time_ms=5050.0,
            duration_ms=50.0,
            max_frame_time_ms=50.0,
            severity=StutterSeverity.MINOR,
        )
        # Legacy aliases
        assert event.index == event.start_index
        assert event.timestamp_ms == event.start_time_ms
        assert event.frame_time_ms == event.max_frame_time_ms
        assert event.duration_frames == event.frame_count

    def test_multiplier_calculation(self):
        """Test multiplier method."""
        event = StutterEvent(
            start_index=0,
            start_time_ms=0.0,
            end_time_ms=100.0,
            duration_ms=100.0,
            max_frame_time_ms=100.0,
            severity=StutterSeverity.MAJOR,
        )
        # 100ms frame time with 16.67ms average = ~6x multiplier
        multiplier = event.multiplier(avg_frame_time=16.67)
        assert abs(multiplier - 6.0) < 0.1

    def test_multiplier_zero_avg(self):
        """Test multiplier with zero average returns 0."""
        event = StutterEvent(
            start_index=0,
            start_time_ms=0.0,
            end_time_ms=50.0,
            duration_ms=50.0,
            max_frame_time_ms=50.0,
            severity=StutterSeverity.MINOR,
        )
        assert event.multiplier(avg_frame_time=0.0) == 0.0

    def test_all_severities(self):
        """Test all severity levels can be used."""
        for severity in StutterSeverity:
            event = StutterEvent(
                start_index=0,
                start_time_ms=0.0,
                end_time_ms=100.0,
                duration_ms=100.0,
                max_frame_time_ms=100.0,
                severity=severity,
            )
            assert event.severity == severity


class TestDistributionStats:
    """Tests for DistributionStats dataclass."""

    def test_instantiation(self):
        """Test DistributionStats can be instantiated."""
        stats = DistributionStats(
            mean=16.67,
            median=16.5,
            std=2.0,
            variance=4.0,
            skewness=0.5,
            kurtosis=3.0,
            iqr=2.5,
            cv=12.0,
        )
        assert stats.mean == 16.67
        assert stats.median == 16.5
        assert stats.cv == 12.0


class TestVarianceCheckResult:
    """Tests for VarianceCheckResult dataclass."""

    def test_instantiation_minimal(self):
        """Test VarianceCheckResult with minimal fields."""
        result = VarianceCheckResult(
            passed=True,
            cv_percent=2.5,
            max_deviation=1.5,
        )
        assert result.passed is True
        assert result.cv_percent == 2.5
        assert result.run_means == []
        assert result.message == ""

    def test_instantiation_full(self):
        """Test VarianceCheckResult with all fields."""
        result = VarianceCheckResult(
            passed=False,
            cv_percent=8.0,
            max_deviation=5.0,
            run_means=[60.0, 58.0, 65.0],
            overall_mean=61.0,
            message="CV 8.0% exceeds 5.0% threshold",
        )
        assert result.passed is False
        assert len(result.run_means) == 3
        assert result.message != ""

    def test_run_means_not_shared(self):
        """Test run_means list is not shared between instances."""
        result1 = VarianceCheckResult(passed=True, cv_percent=1.0, max_deviation=0.5)
        result2 = VarianceCheckResult(passed=True, cv_percent=1.0, max_deviation=0.5)

        result1.run_means.append(60.0)

        assert 60.0 in result1.run_means
        assert 60.0 not in result2.run_means


class TestBenchmarkRun:
    """Tests for BenchmarkRun dataclass."""

    def test_instantiation_minimal(self):
        """Test BenchmarkRun with minimal fields."""
        run = BenchmarkRun(
            run_id="run_001",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
        )
        assert run.run_id == "run_001"
        assert run.config == {}
        assert len(run.frame_times_ms) == 0
        assert run.hardware_snapshots == []
        assert run.fps_metrics == {}
        assert run.warnings == []

    def test_frame_times_array_not_shared(self):
        """Test frame_times_ms array is not shared between instances."""
        run1 = BenchmarkRun(run_id="run_1", timestamp=datetime.now())
        run2 = BenchmarkRun(run_id="run_2", timestamp=datetime.now())

        # Modify run1's array
        run1.frame_times_ms = np.array([16.67, 16.70, 16.65])

        assert len(run1.frame_times_ms) == 3
        assert len(run2.frame_times_ms) == 0

    def test_lists_not_shared(self):
        """Test list fields are not shared between instances."""
        run1 = BenchmarkRun(run_id="run_1", timestamp=datetime.now())
        run2 = BenchmarkRun(run_id="run_2", timestamp=datetime.now())

        run1.hardware_snapshots.append(HardwareSnapshot(timestamp_ms=0.0))
        run1.warnings.append("Test warning")

        assert len(run1.hardware_snapshots) == 1
        assert len(run2.hardware_snapshots) == 0
        assert len(run1.warnings) == 1
        assert len(run2.warnings) == 0

    def test_dicts_not_shared(self):
        """Test dict fields are not shared between instances."""
        run1 = BenchmarkRun(run_id="run_1", timestamp=datetime.now())
        run2 = BenchmarkRun(run_id="run_2", timestamp=datetime.now())

        run1.config["preset"] = "high"
        run1.fps_metrics["avg_fps"] = 60.0

        assert "preset" in run1.config
        assert "preset" not in run2.config
        assert "avg_fps" in run1.fps_metrics
        assert "avg_fps" not in run2.fps_metrics


class TestBenchmarkSession:
    """Tests for BenchmarkSession dataclass."""

    def test_instantiation_minimal(self):
        """Test BenchmarkSession with minimal fields."""
        session = BenchmarkSession(
            session_id="session_001",
            game_name="test_game",
            preset_name="high",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
        )
        assert session.session_id == "session_001"
        assert session.game_name == "test_game"
        assert session.preset_name == "high"
        assert session.config == {}
        assert session.runs == []
        assert session.variance_result is None
        assert session.aggregate_metrics == {}

    def test_runs_not_shared(self):
        """Test runs list is not shared between instances."""
        session1 = BenchmarkSession(
            session_id="s1",
            game_name="game",
            preset_name="high",
            timestamp=datetime.now(),
        )
        session2 = BenchmarkSession(
            session_id="s2",
            game_name="game",
            preset_name="high",
            timestamp=datetime.now(),
        )

        session1.runs.append(BenchmarkRun(run_id="run_1", timestamp=datetime.now()))

        assert len(session1.runs) == 1
        assert len(session2.runs) == 0

    def test_with_variance_result(self):
        """Test BenchmarkSession with variance result."""
        session = BenchmarkSession(
            session_id="session_001",
            game_name="test_game",
            preset_name="high",
            timestamp=datetime.now(),
            variance_result=VarianceCheckResult(
                passed=True, cv_percent=2.0, max_deviation=1.0
            ),
        )
        assert session.variance_result is not None
        assert session.variance_result.passed is True
