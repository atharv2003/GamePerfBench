"""Unit tests for SimulatedCaptureBackend."""

import numpy as np

from src.capture.frametime_capture import CaptureBackend, SimulatedCaptureBackend


class TestCaptureBackendInterface:
    """Tests for the CaptureBackend abstract interface."""

    def test_simulated_backend_is_capture_backend(self):
        """Test SimulatedCaptureBackend is a CaptureBackend."""
        backend = SimulatedCaptureBackend()
        assert isinstance(backend, CaptureBackend)


class TestSimulatedCaptureBackendAvailability:
    """Tests for SimulatedCaptureBackend availability."""

    def test_is_available_returns_true(self):
        """Test is_available always returns True."""
        backend = SimulatedCaptureBackend()
        assert backend.is_available() is True


class TestSimulatedCaptureBackendStartStop:
    """Tests for start/stop capture methods."""

    def test_start_capture_returns_true(self):
        """Test start_capture returns True."""
        backend = SimulatedCaptureBackend()
        result = backend.start_capture("game.exe")
        assert result is True

    def test_stop_capture_returns_empty_without_session(self):
        """Test stop_capture returns empty list without capture_session."""
        backend = SimulatedCaptureBackend()
        backend.start_capture("game.exe")
        frames = backend.stop_capture()
        assert frames == []

    def test_stop_capture_returns_frames_after_session(self):
        """Test stop_capture returns frames after capture_session."""
        backend = SimulatedCaptureBackend()
        backend.start_capture("game.exe")
        backend.capture_session(duration_seconds=1, target_fps=60.0, seed=42)
        frames = backend.stop_capture()
        assert len(frames) > 0


class TestSimulatedCaptureSessionDeterminism:
    """Tests for deterministic behavior with seed."""

    def test_same_seed_produces_identical_results(self):
        """Test same seed produces identical frame_time_ms arrays."""
        backend1 = SimulatedCaptureBackend()
        backend2 = SimulatedCaptureBackend()

        df1 = backend1.capture_session(duration_seconds=5, target_fps=60.0, seed=12345)
        df2 = backend2.capture_session(duration_seconds=5, target_fps=60.0, seed=12345)

        np.testing.assert_array_equal(
            df1["frame_time_ms"].values, df2["frame_time_ms"].values
        )

    def test_different_seeds_produce_different_results(self):
        """Test different seeds produce different results."""
        backend1 = SimulatedCaptureBackend()
        backend2 = SimulatedCaptureBackend()

        df1 = backend1.capture_session(duration_seconds=5, target_fps=60.0, seed=111)
        df2 = backend2.capture_session(duration_seconds=5, target_fps=60.0, seed=222)

        # Arrays should not be equal
        assert not np.array_equal(
            df1["frame_time_ms"].values, df2["frame_time_ms"].values
        )

    def test_none_seed_produces_random_results(self):
        """Test None seed produces different results each call."""
        backend = SimulatedCaptureBackend()

        df1 = backend.capture_session(duration_seconds=2, target_fps=60.0, seed=None)
        df2 = backend.capture_session(duration_seconds=2, target_fps=60.0, seed=None)

        # Very unlikely to be equal with random seeds
        # (not guaranteed, but extremely improbable)
        assert not np.array_equal(
            df1["frame_time_ms"].values, df2["frame_time_ms"].values
        )


class TestSimulatedCaptureSessionColumns:
    """Tests for DataFrame column structure."""

    def test_required_columns_exist(self):
        """Test DataFrame has all required columns."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(duration_seconds=1, target_fps=60.0, seed=42)

        required_columns = [
            "timestamp_ms",
            "frame_time_ms",
            "present_time_ms",
            "fps_instantaneous",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_column_count(self):
        """Test DataFrame has exactly 4 columns."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(duration_seconds=1, target_fps=60.0, seed=42)
        assert len(df.columns) == 4


class TestSimulatedCaptureSessionLength:
    """Tests for frame count and duration."""

    def test_frame_count_reasonable_for_duration(self):
        """Test frame count is reasonable for duration and target FPS."""
        backend = SimulatedCaptureBackend()
        duration = 5
        target_fps = 60.0

        df = backend.capture_session(
            duration_seconds=duration, target_fps=target_fps, seed=42
        )

        # Expected frames: duration * target_fps
        expected_frames = duration * target_fps

        # Allow 20% tolerance due to variance and trimming
        assert len(df) >= expected_frames * 0.8
        assert len(df) <= expected_frames * 1.2

    def test_total_duration_within_bounds(self):
        """Test total captured duration is close to requested."""
        backend = SimulatedCaptureBackend()
        duration = 10

        df = backend.capture_session(
            duration_seconds=duration, target_fps=60.0, seed=42
        )

        # Sum of frame times should be close to duration
        total_time_ms = df["frame_time_ms"].sum()
        total_time_s = total_time_ms / 1000.0

        # Should not exceed requested duration
        assert total_time_s <= duration
        # Should be reasonably close (at least 90%)
        assert total_time_s >= duration * 0.9


class TestSimulatedCaptureSessionFPS:
    """Tests for FPS calculations."""

    def test_mean_fps_close_to_target(self):
        """Test mean FPS is within tolerance of target."""
        backend = SimulatedCaptureBackend()
        target_fps = 60.0

        df = backend.capture_session(
            duration_seconds=10,
            target_fps=target_fps,
            seed=42,
            variance=0.1,  # Low variance for tighter tolerance
        )

        mean_fps = df["fps_instantaneous"].mean()

        # Allow 15% tolerance
        assert abs(mean_fps - target_fps) / target_fps < 0.15

    def test_fps_instantaneous_calculated_correctly(self):
        """Test fps_instantaneous = 1000 / frame_time_ms."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(duration_seconds=1, target_fps=60.0, seed=42)

        expected_fps = 1000.0 / df["frame_time_ms"].values
        np.testing.assert_array_almost_equal(
            df["fps_instantaneous"].values, expected_fps, decimal=6
        )

    def test_different_target_fps(self):
        """Test different target FPS values produce appropriate means."""
        backend = SimulatedCaptureBackend()

        for target_fps in [30.0, 60.0, 120.0, 144.0]:
            df = backend.capture_session(
                duration_seconds=5,
                target_fps=target_fps,
                seed=42,
                variance=0.1,
            )
            mean_fps = df["fps_instantaneous"].mean()
            # 20% tolerance
            assert abs(mean_fps - target_fps) / target_fps < 0.20


class TestSimulatedCaptureSessionFrameTimes:
    """Tests for frame time properties."""

    def test_all_frame_times_positive(self):
        """Test all frame times are strictly positive."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=5,
            target_fps=60.0,
            seed=42,
            variance=0.3,  # Higher variance
        )

        assert (df["frame_time_ms"] > 0).all()

    def test_present_time_equals_frame_time(self):
        """Test present_time_ms equals frame_time_ms."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(duration_seconds=1, target_fps=60.0, seed=42)

        np.testing.assert_array_equal(
            df["present_time_ms"].values, df["frame_time_ms"].values
        )

    def test_timestamps_monotonically_increasing(self):
        """Test timestamps are monotonically increasing."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(duration_seconds=5, target_fps=60.0, seed=42)

        timestamps = df["timestamp_ms"].values
        assert (np.diff(timestamps) >= 0).all()


class TestSimulatedCaptureSessionStutterInjection:
    """Tests for stutter injection."""

    def test_no_stutters_by_default(self):
        """Test no extreme stutters without stutter_injection."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=5,
            target_fps=60.0,
            seed=42,
            variance=0.1,  # Low variance
            stutter_injection=None,
        )

        # With low variance, shouldn't have frames >50ms (minor threshold)
        # at 60 FPS (base ~16.67ms)
        assert (df["frame_time_ms"] < 50.0).all()

    def test_micro_stutter_injection(self):
        """Test micro stutter injection (>25ms)."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=5,
            target_fps=60.0,
            seed=42,
            variance=0.05,  # Very low variance
            stutter_injection={"micro": 5},
        )

        # At least 5 frames should exceed 25ms threshold
        stutters_above_25ms = (df["frame_time_ms"] > 25.0).sum()
        assert stutters_above_25ms >= 5

    def test_minor_stutter_injection(self):
        """Test minor stutter injection (>50ms)."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=5,
            target_fps=60.0,
            seed=42,
            variance=0.05,
            stutter_injection={"minor": 3},
        )

        stutters_above_50ms = (df["frame_time_ms"] > 50.0).sum()
        assert stutters_above_50ms >= 3

    def test_major_stutter_injection(self):
        """Test major stutter injection (>100ms)."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=5,
            target_fps=60.0,
            seed=42,
            variance=0.05,
            stutter_injection={"major": 2},
        )

        stutters_above_100ms = (df["frame_time_ms"] > 100.0).sum()
        assert stutters_above_100ms >= 2

    def test_severe_stutter_injection(self):
        """Test severe stutter injection (>200ms)."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=5,
            target_fps=60.0,
            seed=42,
            variance=0.05,
            stutter_injection={"severe": 2},
        )

        stutters_above_200ms = (df["frame_time_ms"] > 200.0).sum()
        assert stutters_above_200ms >= 2

    def test_freeze_stutter_injection(self):
        """Test freeze stutter injection (>500ms)."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=10,  # Longer duration for freeze injection
            target_fps=60.0,
            seed=42,
            variance=0.05,
            stutter_injection={"freeze": 1},
        )

        stutters_above_500ms = (df["frame_time_ms"] > 500.0).sum()
        assert stutters_above_500ms >= 1

    def test_multiple_stutter_types(self):
        """Test multiple stutter types can be injected together."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=10,
            target_fps=60.0,
            seed=42,
            variance=0.05,
            stutter_injection={"micro": 5, "minor": 3, "major": 2},
        )

        # Check each threshold
        assert (df["frame_time_ms"] > 25.0).sum() >= 5
        assert (df["frame_time_ms"] > 50.0).sum() >= 3
        assert (df["frame_time_ms"] > 100.0).sum() >= 2

    def test_invalid_stutter_type_ignored(self):
        """Test invalid stutter types are ignored without error."""
        backend = SimulatedCaptureBackend()

        # Should not raise
        df = backend.capture_session(
            duration_seconds=1,
            target_fps=60.0,
            seed=42,
            stutter_injection={"invalid_type": 5},
        )

        assert len(df) > 0


class TestSimulatedCaptureSessionVariance:
    """Tests for variance parameter."""

    def test_low_variance_tight_distribution(self):
        """Test low variance produces tight distribution."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=5,
            target_fps=60.0,
            seed=42,
            variance=0.05,
        )

        cv = df["frame_time_ms"].std() / df["frame_time_ms"].mean()
        assert cv < 0.15  # CV should be relatively low

    def test_high_variance_wide_distribution(self):
        """Test high variance produces wider distribution."""
        backend = SimulatedCaptureBackend()
        df = backend.capture_session(
            duration_seconds=5,
            target_fps=60.0,
            seed=42,
            variance=0.3,
        )

        cv = df["frame_time_ms"].std() / df["frame_time_ms"].mean()
        assert cv > 0.1  # CV should be higher
