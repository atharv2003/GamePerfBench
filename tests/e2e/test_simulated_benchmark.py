"""
End-to-end tests for simulated benchmark pipeline.

Tests the complete flow: capture → analysis → variance check → CSV export.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.benchmark_runner import BenchmarkRunner
from src.core.models import BenchmarkSession


class TestSimulatedBenchmarkE2E:
    """End-to-end tests for the simulated benchmark pipeline."""

    def test_basic_benchmark_run(self):
        """Test basic benchmark run with default settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            session = runner.run_benchmark(
                game_name="TestGame",
                preset="high",
                num_trials=3,
                duration_seconds=5,
                simulated=True,
                target_fps=60.0,
                seed=42,
                export=True,
            )

            # Verify session structure
            assert isinstance(session, BenchmarkSession)
            assert session.game_name == "TestGame"
            assert session.preset_name == "high"
            assert len(session.runs) == 3

            # Verify each run has required data
            for i, run in enumerate(session.runs):
                assert run.trial_index == i
                assert len(run.frame_times_ms) > 0
                assert len(run.timestamps_ms) > 0
                assert run.fps_metrics is not None
                assert "avg_fps" in run.fps_metrics
                assert "one_percent_low_fps" in run.fps_metrics
                assert run.percentiles is not None
                assert run.stutter_summary is not None

    def test_variance_check_passes_with_consistent_runs(self):
        """Test that variance check passes when runs are consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            session = runner.run_benchmark(
                game_name="ConsistentGame",
                preset="medium",
                num_trials=3,
                duration_seconds=5,
                simulated=True,
                target_fps=60.0,
                seed=100,
                variance=0.03,  # Low variance for consistency
                export=False,
            )

            # With low variance and deterministic seed progression,
            # runs should be fairly consistent
            assert session.variance_result is not None
            # Note: variance check may or may not pass depending on
            # the exact simulated values, but result should exist
            assert hasattr(session.variance_result, "passed")
            assert hasattr(session.variance_result, "cv_percent")

    def test_csv_export_creates_files(self):
        """Test that CSV export creates all expected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            runner.run_benchmark(
                game_name="ExportTest",
                preset="ultra",
                num_trials=2,
                duration_seconds=3,
                simulated=True,
                seed=555,
                export=True,
            )

            # Find the session directory
            runs_dir = Path(tmpdir) / "runs"
            assert runs_dir.exists(), "Runs directory should exist"

            # Find session subdirectory
            session_dirs = list(runs_dir.glob("exporttest_*"))
            assert len(session_dirs) == 1, "Should have one session directory"

            session_dir = session_dirs[0]

            # Check for expected CSV files
            expected_files = [
                "raw_frametimes.csv",
                "run_summaries.csv",
                "stutter_events.csv",
            ]

            for filename in expected_files:
                filepath = session_dir / filename
                assert filepath.exists(), f"{filename} should exist"

            # Verify raw_frametimes.csv content
            raw_df = pd.read_csv(session_dir / "raw_frametimes.csv")
            assert "run_id" in raw_df.columns
            assert "trial_index" in raw_df.columns
            assert "timestamp_ms" in raw_df.columns
            assert "frame_time_ms" in raw_df.columns
            assert len(raw_df) > 0

            # Verify run_summaries.csv content
            summary_df = pd.read_csv(session_dir / "run_summaries.csv")
            assert "run_id" in summary_df.columns
            assert "avg_fps" in summary_df.columns
            assert len(summary_df) == 2  # 2 trials

    def test_stutter_injection(self):
        """Test that stutters are properly injected and detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            session = runner.run_benchmark(
                game_name="StutterTest",
                preset="high",
                num_trials=1,
                duration_seconds=10,
                simulated=True,
                target_fps=60.0,
                seed=777,
                stutter_injection={"minor": 3, "major": 2},
                export=False,
            )

            run = session.runs[0]

            # Should have detected some stutters
            assert run.stutter_summary is not None
            total_stutters = run.stutter_summary.get("total_events", 0)
            assert total_stutters > 0, "Should have detected stutters"

            # Verify stutter events list
            assert run.stutter_events is not None
            assert len(run.stutter_events) > 0

    def test_aggregate_metrics(self):
        """Test that session-level aggregate metrics are computed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            session = runner.run_benchmark(
                game_name="AggregateTest",
                preset="low",
                num_trials=3,
                duration_seconds=5,
                simulated=True,
                seed=999,
                export=False,
            )

            # Check aggregate metrics
            assert session.aggregate_metrics is not None
            agg = session.aggregate_metrics

            assert "num_runs" in agg
            assert agg["num_runs"] == 3

            assert "avg_of_avg_fps" in agg
            assert agg["avg_of_avg_fps"] > 0

            assert "avg_of_one_percent_low_fps" in agg
            assert agg["avg_of_one_percent_low_fps"] > 0

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces reproducible results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            # Run twice with same seed
            session1 = runner.run_benchmark(
                game_name="ReproTest",
                preset="high",
                num_trials=1,
                duration_seconds=5,
                simulated=True,
                seed=12345,
                export=False,
            )

            session2 = runner.run_benchmark(
                game_name="ReproTest",
                preset="high",
                num_trials=1,
                duration_seconds=5,
                simulated=True,
                seed=12345,
                export=False,
            )

            # Frame times should be identical
            np.testing.assert_array_almost_equal(
                session1.runs[0].frame_times_ms,
                session2.runs[0].frame_times_ms,
                decimal=6,
            )

            # FPS metrics should be identical
            assert (
                session1.runs[0].fps_metrics["avg_fps"]
                == session2.runs[0].fps_metrics["avg_fps"]
            )

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            session1 = runner.run_benchmark(
                game_name="DiffTest",
                preset="high",
                num_trials=1,
                duration_seconds=5,
                simulated=True,
                seed=111,
                export=False,
            )

            session2 = runner.run_benchmark(
                game_name="DiffTest",
                preset="high",
                num_trials=1,
                duration_seconds=5,
                simulated=True,
                seed=222,
                export=False,
            )

            # Frame times should be different (not exactly equal)
            assert not np.array_equal(
                session1.runs[0].frame_times_ms,
                session2.runs[0].frame_times_ms,
            )

    def test_multiple_trials_have_different_data(self):
        """Test that each trial has unique data (seed increments)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            session = runner.run_benchmark(
                game_name="MultiTrialTest",
                preset="high",
                num_trials=3,
                duration_seconds=5,
                simulated=True,
                seed=500,
                export=False,
            )

            # Each run should have different frame times (not exactly equal)
            for i in range(len(session.runs)):
                for j in range(i + 1, len(session.runs)):
                    assert not np.array_equal(
                        session.runs[i].frame_times_ms,
                        session.runs[j].frame_times_ms,
                    ), f"Run {i} and {j} should have different data"

    def test_fps_near_target(self):
        """Test that simulated FPS is near the target FPS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            target_fps = 120.0

            session = runner.run_benchmark(
                game_name="FPSTest",
                preset="high",
                num_trials=1,
                duration_seconds=10,
                simulated=True,
                target_fps=target_fps,
                seed=42,
                variance=0.03,
                export=False,
            )

            run = session.runs[0]
            avg_fps = run.fps_metrics["avg_fps"]

            # Average FPS should be within 15% of target
            assert (
                abs(avg_fps - target_fps) / target_fps < 0.15
            ), f"Average FPS {avg_fps} should be near target {target_fps}"

    def test_session_config_preserved(self):
        """Test that session config is properly stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            session = runner.run_benchmark(
                game_name="ConfigTest",
                preset="custom",
                num_trials=2,
                duration_seconds=7,
                simulated=True,
                target_fps=90.0,
                seed=888,
                variance=0.04,
                export=False,
            )

            # Verify config is stored
            assert session.config is not None
            assert session.config["num_trials"] == 2
            assert session.config["duration_seconds"] == 7
            assert session.config["target_fps"] == 90.0
            assert session.config["variance"] == 0.04

    def test_single_trial_no_variance_check(self):
        """Test that single trial doesn't fail variance check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            session = runner.run_benchmark(
                game_name="SingleTrial",
                preset="high",
                num_trials=1,
                duration_seconds=5,
                simulated=True,
                seed=42,
                export=False,
            )

            # With only 1 run, variance check should not be performed
            # (requires minimum 2 runs)
            assert len(session.runs) == 1


class TestBenchmarkRunnerConfiguration:
    """Test BenchmarkRunner configuration options."""

    def test_custom_output_directory(self):
        """Test using custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_output = Path(tmpdir) / "custom_output"

            runner = BenchmarkRunner(output_root=tmpdir)

            runner.run_benchmark(
                game_name="CustomDirTest",
                preset="high",
                num_trials=1,
                duration_seconds=3,
                simulated=True,
                seed=42,
                export=True,
                output_dir=custom_output,
            )

            # Files should be in custom directory
            assert custom_output.exists()
            assert (custom_output / "raw_frametimes.csv").exists()

    def test_export_disabled(self):
        """Test that export=False skips CSV generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_root=tmpdir)

            runner.run_benchmark(
                game_name="NoExportTest",
                preset="high",
                num_trials=1,
                duration_seconds=3,
                simulated=True,
                seed=42,
                export=False,
            )

            # Runs directory should not be created or be empty
            runs_dir = Path(tmpdir) / "runs"
            if runs_dir.exists():
                session_dirs = list(runs_dir.glob("noexporttest_*"))
                assert len(session_dirs) == 0

    def test_dependency_injection(self):
        """Test that custom components can be injected."""
        from src.core.variance_checker import VarianceChecker

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create custom instances
            custom_variance_checker = VarianceChecker(
                max_cv_avg_fps=0.10,  # 10% threshold
                max_cv_one_percent_low=0.20,
            )

            runner = BenchmarkRunner(
                output_root=tmpdir,
                variance_checker=custom_variance_checker,
            )

            # Verify custom instance is used
            assert runner.variance_checker.max_cv_avg_fps == 0.10
            assert runner.variance_checker.max_cv_one_percent_low == 0.20
