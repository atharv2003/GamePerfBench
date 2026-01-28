"""
Integration tests for ReportPackager.

Tests ZIP bundle creation with various configurations.
"""

import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.core.models import (
    BenchmarkRun,
    BenchmarkSession,
    StutterEvent,
    StutterSeverity,
    VarianceCheckResult,
)
from src.reporting.report_packager import ReportPackager


def create_test_session(
    session_id: str = "test_session_123",
    game_name: str = "testgame",
    preset_name: str = "high",
    num_runs: int = 2,
    include_stutters: bool = True,
) -> BenchmarkSession:
    """Create a test session with sample data."""
    runs = []
    for i in range(num_runs):
        run = BenchmarkRun(
            run_id=f"{session_id}_run_{i:02d}",
            timestamp=datetime.now(),
            trial_index=i,
            config={"game_name": game_name, "preset": preset_name},
            frame_times_ms=np.array([16.5, 16.7, 16.6, 16.8, 16.5]),
            timestamps_ms=np.array([0.0, 16.5, 33.2, 49.8, 66.6]),
            fps_metrics={
                "avg_fps": 60.0 + i,
                "one_percent_low_fps": 55.0 + i,
                "duration_seconds": 5.0,
            },
            percentiles={"p99_ms": 16.8},
            stutter_events=(
                [
                    StutterEvent(
                        start_index=2,
                        start_time_ms=33.2,
                        end_time_ms=83.2,
                        duration_ms=50.0,
                        max_frame_time_ms=50.0,
                        severity=StutterSeverity.MINOR,
                        frame_count=1,
                    )
                ]
                if include_stutters
                else []
            ),
            stutter_summary=(
                {"total_stutters": 1, "minor": 1} if include_stutters else {}
            ),
            duration_seconds=5.0,
            total_frames=5,
        )
        runs.append(run)

    session = BenchmarkSession(
        session_id=session_id,
        game_name=game_name,
        preset_name=preset_name,
        timestamp=datetime.now(),
        config={"num_trials": num_runs},
        runs=runs,
        variance_result=VarianceCheckResult(
            passed=True,
            cv_percent=2.5,
            max_deviation=1.0,
            run_means=[60.0, 61.0],
            overall_mean=60.5,
            message="Variance check PASSED",
        ),
        aggregate_metrics={
            "avg_of_avg_fps": 60.5,
            "avg_of_one_percent_low_fps": 55.5,
            "avg_of_p99_ms": 16.8,
        },
    )

    return session


def create_test_files(session_dir: Path, include_charts: bool = True) -> None:
    """Create test artifact files in session directory."""
    # Create CSV files
    (session_dir / "raw_frametimes.csv").write_text(
        "session_id,run_id,frame_index,timestamp_ms,frame_time_ms\n"
        "test,run_0,0,0.0,16.5\n"
    )
    (session_dir / "run_summaries.csv").write_text(
        "session_id,run_id,avg_fps,one_percent_low_fps\n" "test,run_0,60.0,55.0\n"
    )
    (session_dir / "stutter_events.csv").write_text(
        "session_id,run_id,start_index,severity\n" "test,run_0,2,minor\n"
    )

    if include_charts:
        # Create fake PNG files (minimal valid PNG header)
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        (session_dir / "avg_fps_by_trial.png").write_bytes(png_header)
        (session_dir / "one_percent_low_by_trial.png").write_bytes(png_header)
        (session_dir / "frametime_p99_by_trial.png").write_bytes(png_header)
        (session_dir / "stutter_events_by_severity.png").write_bytes(png_header)
        (session_dir / "session_summary.png").write_bytes(png_header)

    # Create report.html
    (session_dir / "report.html").write_text(
        "<!DOCTYPE html><html><body>Test Report</body></html>"
    )


class TestReportPackager:
    """Tests for ReportPackager."""

    def test_creates_zip_successfully(self):
        """Test that ZIP bundle is created successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            assert bundle_path.exists()
            assert bundle_path.suffix == ".zip"
            assert bundle_path.stat().st_size > 0

    def test_zip_contains_expected_csv_entries(self):
        """Test that ZIP contains CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            with zipfile.ZipFile(bundle_path, "r") as zf:
                names = zf.namelist()
                assert "raw_frametimes.csv" in names
                assert "run_summaries.csv" in names
                assert "stutter_events.csv" in names

    def test_zip_contains_png_when_charts_enabled(self):
        """Test that ZIP contains PNG files when charts are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir, include_charts=True)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            with zipfile.ZipFile(bundle_path, "r") as zf:
                names = zf.namelist()
                assert "avg_fps_by_trial.png" in names
                assert "session_summary.png" in names

    def test_zip_works_without_charts(self):
        """Test that ZIP works when charts are not generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir, include_charts=False)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            assert bundle_path.exists()

            with zipfile.ZipFile(bundle_path, "r") as zf:
                names = zf.namelist()
                # Should still have CSVs and report
                assert "raw_frametimes.csv" in names
                assert "report.html" in names
                # Should not have PNGs
                png_files = [n for n in names if n.endswith(".png")]
                assert len(png_files) == 0

    def test_zip_works_without_report(self):
        """Test that ZIP works when report is not generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)
            # Remove report
            (session_dir / "report.html").unlink()

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            assert bundle_path.exists()

            with zipfile.ZipFile(bundle_path, "r") as zf:
                names = zf.namelist()
                assert "report.html" not in names
                # CSVs should still be present
                assert "raw_frametimes.csv" in names

    def test_all_arcnames_are_relative(self):
        """Test that all file paths in ZIP are relative (no absolute paths)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            with zipfile.ZipFile(bundle_path, "r") as zf:
                for name in zf.namelist():
                    # No drive letters (Windows)
                    assert ":" not in name, f"Absolute path found: {name}"
                    # No leading slashes
                    assert not name.startswith("/"), f"Leading slash: {name}"
                    assert not name.startswith("\\"), f"Leading backslash: {name}"
                    # Should be just filename, no directory components
                    assert "/" not in name, f"Directory path: {name}"
                    assert "\\" not in name, f"Directory path: {name}"

    def test_zip_contains_metadata_json(self):
        """Test that ZIP contains metadata.json with expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            with zipfile.ZipFile(bundle_path, "r") as zf:
                assert "metadata.json" in zf.namelist()

                metadata = json.loads(zf.read("metadata.json"))
                assert metadata["session_id"] == "test_session_123"
                assert metadata["game"] == "testgame"
                assert metadata["preset"] == "high"
                assert metadata["num_runs"] == 2
                assert "aggregate_metrics" in metadata
                assert metadata["aggregate_metrics"]["avg_fps"] == 60.5

    def test_zip_contains_manifest_json(self):
        """Test that ZIP contains manifest.json with file list and hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            with zipfile.ZipFile(bundle_path, "r") as zf:
                assert "manifest.json" in zf.namelist()

                manifest = json.loads(zf.read("manifest.json"))
                assert "version" in manifest
                assert "created_at" in manifest
                assert "files" in manifest
                assert len(manifest["files"]) > 0

                # Check file entries have expected fields
                for file_entry in manifest["files"]:
                    assert "name" in file_entry
                    assert "size_bytes" in file_entry
                    assert "sha256" in file_entry
                    # SHA256 should be 64 hex characters
                    assert len(file_entry["sha256"]) == 64

    def test_custom_bundle_name(self):
        """Test that custom bundle name is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(
                session, session_dir, bundle_name="custom_bundle.zip"
            )

            assert bundle_path.name == "custom_bundle.zip"
            assert bundle_path.exists()

    def test_custom_bundle_name_adds_zip_extension(self):
        """Test that .zip is added if missing from custom name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(
                session, session_dir, bundle_name="my_bundle"
            )

            assert bundle_path.name == "my_bundle.zip"
            assert bundle_path.exists()

    def test_default_bundle_name_format(self):
        """Test that default bundle name uses correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(session_id="game_high_20240101_abc123")
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            assert bundle_path.name == "GamePerfBench_game_high_20240101_abc123.zip"

    def test_works_with_empty_stutter_events(self):
        """Test that bundling works with no stutter events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session(include_stutters=False)
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            assert bundle_path.exists()

            with zipfile.ZipFile(bundle_path, "r") as zf:
                metadata = json.loads(zf.read("metadata.json"))
                assert metadata["aggregate_metrics"]["total_stutters"] == 0
                assert metadata["aggregate_metrics"]["worst_stutter_ms"] == 0.0

    def test_variance_result_in_metadata(self):
        """Test that variance result is included in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            with zipfile.ZipFile(bundle_path, "r") as zf:
                metadata = json.loads(zf.read("metadata.json"))
                assert "variance_result" in metadata
                assert metadata["variance_result"]["passed"] is True
                assert metadata["variance_result"]["cv_percent"] == 2.5

    def test_does_not_include_existing_zip_files(self):
        """Test that existing ZIP files in directory are not included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session = create_test_session()
            create_test_files(session_dir)

            # Create an existing ZIP file
            (session_dir / "old_bundle.zip").write_bytes(b"fake zip content")

            packager = ReportPackager()
            bundle_path = packager.package_session(session, session_dir)

            with zipfile.ZipFile(bundle_path, "r") as zf:
                names = zf.namelist()
                assert "old_bundle.zip" not in names
