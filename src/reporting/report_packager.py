"""
Report Packager Module.

Creates portable ZIP bundles containing all benchmark session artifacts.
Includes metadata.json and manifest.json for easy parsing.
"""

import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.models import BenchmarkSession


class ReportPackager:
    """Create portable ZIP bundles for benchmark sessions.

    Bundles include:
        - CSV data files (raw_frametimes, run_summaries, stutter_events)
        - PNG chart files (if generated)
        - HTML report (if generated)
        - metadata.json (session info and metrics)
        - manifest.json (file list with sizes and hashes)
    """

    # Whitelist of file extensions to include in bundle
    ALLOWED_EXTENSIONS = {".csv", ".png", ".html", ".json"}

    # Files to include by pattern
    EXPECTED_FILES = [
        "raw_frametimes.csv",
        "run_summaries.csv",
        "stutter_events.csv",
        "report.html",
        "avg_fps_by_trial.png",
        "one_percent_low_by_trial.png",
        "frametime_p99_by_trial.png",
        "stutter_events_by_severity.png",
        "session_summary.png",
    ]

    def __init__(self) -> None:
        """Initialize the report packager."""
        pass

    def package_session(
        self,
        session: BenchmarkSession,
        session_dir: Union[str, Path],
        bundle_name: Optional[str] = None,
    ) -> Path:
        """Create a ZIP bundle for a benchmark session.

        Args:
            session: BenchmarkSession with metadata.
            session_dir: Directory containing session artifacts.
            bundle_name: Optional custom bundle filename.
                Must end with .zip if provided.

        Returns:
            Path to the created ZIP file.
        """
        session_dir = Path(session_dir)

        # Determine bundle filename
        if bundle_name:
            if not bundle_name.endswith(".zip"):
                bundle_name = f"{bundle_name}.zip"
            bundle_path = session_dir / bundle_name
        else:
            bundle_path = session_dir / f"GamePerfBench_{session.session_id}.zip"

        # Collect files to include
        files_to_bundle = self._collect_files(session_dir)

        # Create metadata.json content
        metadata = self._create_metadata(session)

        # Create manifest.json content (with file list and hashes)
        manifest = self._create_manifest(session_dir, files_to_bundle)

        # Write metadata.json to session_dir temporarily
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Write manifest.json to session_dir temporarily
        manifest_path = session_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Add JSON files to the bundle list
        files_to_bundle.append(metadata_path)
        files_to_bundle.append(manifest_path)

        # Create the ZIP bundle
        self._create_zip(bundle_path, session_dir, files_to_bundle)

        return bundle_path

    def _collect_files(self, session_dir: Path) -> List[Path]:
        """Collect files to include in the bundle.

        Args:
            session_dir: Directory containing session artifacts.

        Returns:
            List of file paths to include.
        """
        files = []

        for filename in self.EXPECTED_FILES:
            filepath = session_dir / filename
            if filepath.exists():
                files.append(filepath)

        # Also include any other allowed files not in expected list
        for filepath in session_dir.iterdir():
            if filepath.is_file():
                ext = filepath.suffix.lower()
                if ext in self.ALLOWED_EXTENSIONS:
                    # Skip zip files and files already in list
                    if ext != ".zip" and filepath not in files:
                        # Skip metadata/manifest as we'll create fresh ones
                        if filepath.name not in ("metadata.json", "manifest.json"):
                            files.append(filepath)

        return files

    def _create_metadata(self, session: BenchmarkSession) -> Dict[str, Any]:
        """Create metadata dictionary for the session.

        Args:
            session: BenchmarkSession with data.

        Returns:
            Dict with session metadata.
        """
        agg = session.aggregate_metrics or {}
        variance = session.variance_result

        metadata: Dict[str, Any] = {
            "version": "1.0",
            "session_id": session.session_id,
            "game": session.game_name,
            "preset": session.preset_name,
            "timestamp": session.timestamp.isoformat(),
            "num_runs": len(session.runs),
            "aggregate_metrics": {
                "avg_fps": agg.get("avg_of_avg_fps", 0),
                "one_percent_low_fps": agg.get("avg_of_one_percent_low_fps", 0),
                "p99_frametime_ms": agg.get("avg_of_p99_ms", 0),
            },
        }

        # Add stutter summary if available
        if session.runs:
            total_stutters = 0
            worst_stutter_ms = 0.0
            for run in session.runs:
                stutter_summary = run.stutter_summary or {}
                total_stutters += stutter_summary.get("total_stutters", 0)
                for event in run.stutter_events:
                    if event.max_frame_time_ms > worst_stutter_ms:
                        worst_stutter_ms = event.max_frame_time_ms

            metadata["aggregate_metrics"]["total_stutters"] = total_stutters
            metadata["aggregate_metrics"]["worst_stutter_ms"] = worst_stutter_ms

        # Add variance result if available
        if variance:
            metadata["variance_result"] = {
                "passed": variance.passed,
                "cv_percent": variance.cv_percent,
                "max_deviation": variance.max_deviation,
                "message": variance.message,
            }

        return metadata

    def _create_manifest(
        self,
        session_dir: Path,
        files: List[Path],
    ) -> Dict[str, Any]:
        """Create manifest dictionary with file list and hashes.

        Args:
            session_dir: Base directory for relative paths.
            files: List of files to include.

        Returns:
            Dict with file manifest.
        """
        manifest: Dict[str, Any] = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "files": [],
        }

        for filepath in files:
            # Skip metadata.json and manifest.json as they don't exist yet
            if filepath.name in ("metadata.json", "manifest.json"):
                continue

            if filepath.exists():
                file_info = {
                    "name": filepath.name,
                    "size_bytes": filepath.stat().st_size,
                    "sha256": self._calculate_hash(filepath),
                }
                manifest["files"].append(file_info)

        return manifest

    def _calculate_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            filepath: Path to the file.

        Returns:
            Hex string of SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _create_zip(
        self,
        bundle_path: Path,
        session_dir: Path,
        files: List[Path],
    ) -> None:
        """Create the ZIP bundle.

        Args:
            bundle_path: Output ZIP file path.
            session_dir: Base directory for relative paths.
            files: List of files to include.
        """
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for filepath in files:
                if filepath.exists():
                    # Use relative path as arcname (no absolute paths)
                    arcname = filepath.name
                    zf.write(filepath, arcname)
