"""
End-to-end tests for CLI interface.

Tests the CLI commands via subprocess to ensure proper integration.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


class TestBenchmarkCommand:
    """Tests for the benchmark CLI command."""

    def test_benchmark_creates_csv_files(self):
        """Test that benchmark command creates expected CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that CSV files were created
            assert output_path.exists(), "Output directory not created"
            assert (output_path / "raw_frametimes.csv").exists()
            assert (output_path / "run_summaries.csv").exists()
            assert (output_path / "stutter_events.csv").exists()

    def test_benchmark_output_content(self):
        """Test that benchmark CSV files have valid content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0

            # Verify run_summaries.csv content
            summary_df = pd.read_csv(output_path / "run_summaries.csv")
            assert len(summary_df) == 2  # 2 trials
            assert "avg_fps" in summary_df.columns
            assert "one_percent_low_fps" in summary_df.columns

            # Verify raw_frametimes.csv content
            raw_df = pd.read_csv(output_path / "raw_frametimes.csv")
            assert len(raw_df) > 0
            assert "frame_time_ms" in raw_df.columns

    def test_benchmark_prints_summary(self):
        """Test that benchmark command prints a summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0

            # Check stdout contains expected summary info
            assert "BENCHMARK COMPLETE" in result.stdout
            assert "Session ID:" in result.stdout
            assert "Avg FPS:" in result.stdout
            assert "1% Low FPS:" in result.stdout

    def test_benchmark_no_export_flag(self):
        """Test that --no-export skips CSV generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "1",
                    "--simulated",
                    "--duration",
                    "1",
                    "--output",
                    str(output_path),
                    "--no-export",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0
            # Output directory should not be created when --no-export is used
            assert not output_path.exists() or not list(output_path.glob("*.csv"))

    def test_benchmark_with_stutter_injection(self):
        """Test benchmark with stutter injection parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "1",
                    "--simulated",
                    "--duration",
                    "5",
                    "--stutter",
                    "minor=2,major=1",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_benchmark_invalid_preset_error(self):
        """Test that invalid preset returns error."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "benchmark",
                "--game",
                "test",
                "--preset",
                "nonexistent_preset",
                "--trials",
                "1",
                "--simulated",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode != 0
        assert "Unknown preset" in result.stderr or "not found" in result.stderr.lower()

    def test_benchmark_different_presets(self):
        """Test that different presets can be used."""
        presets = ["low", "medium", "high", "ultra"]

        for preset in presets:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "test_session"

                result = subprocess.run(
                    [
                        sys.executable,
                        "main.py",
                        "benchmark",
                        "--game",
                        "test",
                        "--preset",
                        preset,
                        "--trials",
                        "1",
                        "--simulated",
                        "--duration",
                        "1",
                        "--output",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent.parent,
                )

                assert (
                    result.returncode == 0
                ), f"Preset {preset} failed: {result.stderr}"

    def test_benchmark_with_charts_creates_png_files(self):
        """Test that --charts generates PNG chart files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--charts",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that CSV files exist
            assert (output_path / "raw_frametimes.csv").exists()
            assert (output_path / "run_summaries.csv").exists()

            # Check that chart PNG files exist
            expected_charts = [
                "avg_fps_by_trial.png",
                "one_percent_low_by_trial.png",
                "frametime_p99_by_trial.png",
                "stutter_events_by_severity.png",
                "session_summary.png",
            ]

            for chart_name in expected_charts:
                chart_path = output_path / chart_name
                assert chart_path.exists(), f"Chart {chart_name} should exist"
                assert (
                    chart_path.stat().st_size > 0
                ), f"Chart {chart_name} should not be empty"

            # Check output mentions charts
            assert "Charts Generated" in result.stdout

    def test_benchmark_no_charts_skips_png_generation(self):
        """Test that --no-charts skips PNG generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--no-charts",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that CSV files exist
            assert (output_path / "raw_frametimes.csv").exists()

            # Check that chart PNG files do NOT exist
            chart_files = list(output_path.glob("*.png"))
            assert (
                len(chart_files) == 0
            ), "No PNG files should be created with --no-charts"

            # Output should not mention charts
            assert "Charts Generated" not in result.stdout

    def test_benchmark_creates_html_report(self):
        """Test that benchmark command creates HTML report by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that report.html exists
            report_path = output_path / "report.html"
            assert report_path.exists(), "report.html should be created by default"
            assert report_path.stat().st_size > 0, "report.html should not be empty"

            # Check output mentions HTML report
            assert "HTML Report:" in result.stdout

    def test_benchmark_report_is_valid_html(self):
        """Test that generated HTML report is valid HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--report",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            report_path = output_path / "report.html"
            assert report_path.exists()

            html_content = report_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in html_content
            assert "<html" in html_content
            assert "</html>" in html_content
            assert "test" in html_content.lower()  # Game name
            assert "high" in html_content.lower()  # Preset name

    def test_benchmark_no_report_skips_html_generation(self):
        """Test that --no-report skips HTML report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--no-report",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that CSV files exist
            assert (output_path / "raw_frametimes.csv").exists()

            # Check that report.html does NOT exist
            report_path = output_path / "report.html"
            assert (
                not report_path.exists()
            ), "report.html should not exist with --no-report"

            # Output should not mention HTML report
            assert "HTML Report:" not in result.stdout

    def test_benchmark_report_contains_session_data(self):
        """Test that HTML report contains expected session data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "reporttest",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            report_path = output_path / "report.html"
            html_content = report_path.read_text(encoding="utf-8")

            # Check for expected content
            assert "reporttest" in html_content.lower()  # Game name
            assert "Summary" in html_content
            assert "Average FPS" in html_content
            assert "1% Low FPS" in html_content
            assert "Run Details" in html_content


class TestBundleCommand:
    """Tests for ZIP bundle CLI functionality."""

    def test_benchmark_creates_bundle_by_default(self):
        """Test that benchmark creates ZIP bundle by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that ZIP bundle exists
            zip_files = list(output_path.glob("*.zip"))
            assert len(zip_files) == 1, "Should have exactly one ZIP bundle"
            assert zip_files[0].name.startswith("GamePerfBench_")

            # Check output mentions bundle
            assert "ZIP Bundle:" in result.stdout

    def test_benchmark_no_bundle_skips_zip(self):
        """Test that --no-bundle skips ZIP generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--no-bundle",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that no ZIP bundle exists
            zip_files = list(output_path.glob("*.zip"))
            assert len(zip_files) == 0, "No ZIP should be created with --no-bundle"

            # Check output does not mention bundle
            assert "ZIP Bundle:" not in result.stdout

    def test_bundle_contains_report_when_enabled(self):
        """Test that ZIP contains report.html when report is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--report",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            zip_files = list(output_path.glob("*.zip"))
            assert len(zip_files) == 1

            import zipfile

            with zipfile.ZipFile(zip_files[0], "r") as zf:
                names = zf.namelist()
                assert "report.html" in names

    def test_bundle_contains_pngs_only_when_charts_enabled(self):
        """Test that ZIP contains PNGs only when charts are enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with charts enabled
            output_path1 = Path(tmpdir) / "with_charts"
            subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--charts",
                    "--output",
                    str(output_path1),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            # Test without charts
            output_path2 = Path(tmpdir) / "no_charts"
            subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--no-charts",
                    "--output",
                    str(output_path2),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            import zipfile

            # Check with charts
            zip_files1 = list(output_path1.glob("*.zip"))
            assert len(zip_files1) == 1
            with zipfile.ZipFile(zip_files1[0], "r") as zf:
                png_files = [n for n in zf.namelist() if n.endswith(".png")]
                assert len(png_files) > 0, "Should have PNG files with --charts"

            # Check without charts
            zip_files2 = list(output_path2.glob("*.zip"))
            assert len(zip_files2) == 1
            with zipfile.ZipFile(zip_files2[0], "r") as zf:
                png_files = [n for n in zf.namelist() if n.endswith(".png")]
                assert len(png_files) == 0, "Should have no PNG files with --no-charts"

    def test_bundle_contains_metadata_json(self):
        """Test that ZIP bundle contains metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "metadatatest",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            import json
            import zipfile

            zip_files = list(output_path.glob("*.zip"))
            assert len(zip_files) == 1

            with zipfile.ZipFile(zip_files[0], "r") as zf:
                assert "metadata.json" in zf.namelist()
                metadata = json.loads(zf.read("metadata.json"))
                assert metadata["game"] == "metadatatest"
                assert metadata["preset"] == "high"
                assert metadata["num_runs"] == 2

    def test_bundle_contains_manifest_json(self):
        """Test that ZIP bundle contains manifest.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_session"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            import json
            import zipfile

            zip_files = list(output_path.glob("*.zip"))
            assert len(zip_files) == 1

            with zipfile.ZipFile(zip_files[0], "r") as zf:
                assert "manifest.json" in zf.namelist()
                manifest = json.loads(zf.read("manifest.json"))
                assert "files" in manifest
                assert len(manifest["files"]) > 0


class TestListCommand:
    """Tests for the list CLI command."""

    def test_list_empty_output(self):
        """Test list command when no sessions exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "list",
                    "--output-root",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0
            assert "No sessions found" in result.stdout

    def test_list_shows_sessions(self):
        """Test list command shows created sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)

            # First create a benchmark session
            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "benchmark",
                    "--game",
                    "test",
                    "--preset",
                    "high",
                    "--trials",
                    "2",
                    "--simulated",
                    "--duration",
                    "1",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
                env={
                    **dict(__import__("os").environ),
                },
            )

            # Create session in custom output root
            runs_dir = output_root / "runs" / "test_session"
            runs_dir.mkdir(parents=True, exist_ok=True)

            # Create a minimal run_summaries.csv
            summary_path = runs_dir / "run_summaries.csv"
            summary_path.write_text(
                "session_id,run_id,trial_index,avg_fps,one_percent_low_fps\n"
                "test_session,run_0,0,60.5,45.2\n"
                "test_session,run_1,1,61.2,46.1\n"
            )

            # Now run list command
            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "list",
                    "--output-root",
                    str(output_root),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0
            assert "BENCHMARK SESSIONS" in result.stdout
            assert "test_session" in result.stdout

    def test_list_with_limit(self):
        """Test list command respects limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)

            # Create multiple fake sessions
            for i in range(5):
                session_dir = output_root / "runs" / f"test_session_{i}"
                session_dir.mkdir(parents=True, exist_ok=True)

                summary_path = session_dir / "run_summaries.csv"
                summary_path.write_text(
                    "session_id,run_id,trial_index,avg_fps,one_percent_low_fps\n"
                    f"test_session_{i},run_0,0,60.0,45.0\n"
                )

            # List with limit
            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "list",
                    "--output-root",
                    str(output_root),
                    "--limit",
                    "3",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0
            assert "Showing 3 session(s)" in result.stdout

    def test_list_filter_by_game(self):
        """Test list command filters by game name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)

            # Create sessions for different games
            for game in ["game_a", "game_b"]:
                session_dir = output_root / "runs" / f"{game}_high_session"
                session_dir.mkdir(parents=True, exist_ok=True)

                summary_path = session_dir / "run_summaries.csv"
                summary_path.write_text(
                    "session_id,run_id,trial_index,avg_fps,one_percent_low_fps\n"
                    f"{game}_high_session,run_0,0,60.0,45.0\n"
                )

            # Filter by game_a
            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "list",
                    "--output-root",
                    str(output_root),
                    "--game",
                    "game_a",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode == 0
            assert "game_a" in result.stdout
            # game_b should be filtered out
            assert "game_b" not in result.stdout or "Showing 1 session" in result.stdout


class TestCLIHelp:
    """Tests for CLI help functionality."""

    def test_main_help(self):
        """Test main help message."""
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "benchmark" in result.stdout
        assert "list" in result.stdout

    def test_benchmark_help(self):
        """Test benchmark subcommand help."""
        result = subprocess.run(
            [sys.executable, "main.py", "benchmark", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "--game" in result.stdout
        assert "--preset" in result.stdout
        assert "--trials" in result.stdout

    def test_list_help(self):
        """Test list subcommand help."""
        result = subprocess.run(
            [sys.executable, "main.py", "list", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "--limit" in result.stdout
        assert "--output-root" in result.stdout

    def test_no_command_shows_help(self):
        """Test that running without command shows help."""
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "benchmark" in result.stdout
