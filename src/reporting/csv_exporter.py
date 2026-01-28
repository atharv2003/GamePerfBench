"""
CSV Export Module.

Exports benchmark session data to CSV files:
- raw_frametimes.csv: Frame-by-frame timing data
- run_summaries.csv: Per-run aggregate metrics
- stutter_events.csv: Detected stutter events
"""

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.core.models import BenchmarkRun, BenchmarkSession


class CSVExporter:
    """Export benchmark data to CSV files.

    Creates a session directory containing:
        - raw_frametimes.csv: One row per frame per run
        - run_summaries.csv: One row per run with aggregate metrics
        - stutter_events.csv: One row per detected stutter event
    """

    def __init__(self, output_root: Union[str, Path] = "output"):
        """Initialize the CSV exporter.

        Args:
            output_root: Root directory for output files.
        """
        self.output_root = Path(output_root)

    def export_session(
        self,
        session: BenchmarkSession,
        output_dir: Union[str, Path, None] = None,
    ) -> Path:
        """Export a benchmark session to CSV files.

        Args:
            session: BenchmarkSession to export.
            output_dir: Optional specific output directory.
                If None, creates output_root/runs/<session_id>.

        Returns:
            Path to the session output directory.
        """
        # Determine output directory
        if output_dir is not None:
            session_dir = Path(output_dir)
        else:
            session_dir = self.output_root / "runs" / session.session_id

        # Create directory if missing
        session_dir.mkdir(parents=True, exist_ok=True)

        # Export each CSV file
        self._export_raw_frametimes(session, session_dir)
        self._export_run_summaries(session, session_dir)
        self._export_stutter_events(session, session_dir)

        return session_dir

    def _export_raw_frametimes(
        self,
        session: BenchmarkSession,
        session_dir: Path,
    ) -> None:
        """Export raw frame time data to CSV.

        Args:
            session: BenchmarkSession containing runs.
            session_dir: Directory to write CSV file.
        """
        rows: List[Dict[str, Any]] = []

        for run in session.runs:
            frame_data = self._extract_frame_data(session.session_id, run)
            rows.extend(frame_data)

        # Create DataFrame and sort
        df = pd.DataFrame(rows)

        if not df.empty:
            # Ensure column order
            columns = [
                "session_id",
                "run_id",
                "trial_index",
                "timestamp_ms",
                "frame_time_ms",
                "present_time_ms",
                "fps_instantaneous",
            ]
            # Only include columns that exist
            columns = [c for c in columns if c in df.columns]
            df = df[columns]

            # Sort by trial_index then timestamp_ms
            df = df.sort_values(["trial_index", "timestamp_ms"]).reset_index(drop=True)

        # Write CSV
        output_path = session_dir / "raw_frametimes.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")

    def _extract_frame_data(
        self,
        session_id: str,
        run: BenchmarkRun,
    ) -> List[Dict[str, Any]]:
        """Extract frame data from a benchmark run.

        Args:
            session_id: Session identifier.
            run: BenchmarkRun containing frame data.

        Returns:
            List of dicts, one per frame.
        """
        rows = []

        # Get frame times array
        frame_times = run.frame_times_ms
        if len(frame_times) == 0:
            return rows

        # Get or compute timestamps
        if len(run.timestamps_ms) > 0:
            timestamps = run.timestamps_ms
        else:
            # Compute cumulative timestamps from frame times
            timestamps = np.cumsum(frame_times)
            timestamps = np.insert(timestamps[:-1], 0, 0.0)

        # Compute instantaneous FPS
        fps_instant = 1000.0 / np.maximum(frame_times, 0.001)

        for i in range(len(frame_times)):
            rows.append(
                {
                    "session_id": session_id,
                    "run_id": run.run_id,
                    "trial_index": run.trial_index,
                    "timestamp_ms": round(timestamps[i], 3),
                    "frame_time_ms": round(frame_times[i], 3),
                    "present_time_ms": round(frame_times[i], 3),  # Same as frame_time
                    "fps_instantaneous": round(fps_instant[i], 2),
                }
            )

        return rows

    def _export_run_summaries(
        self,
        session: BenchmarkSession,
        session_dir: Path,
    ) -> None:
        """Export run summary metrics to CSV.

        Args:
            session: BenchmarkSession containing runs.
            session_dir: Directory to write CSV file.
        """
        rows: List[Dict[str, Any]] = []

        for run in session.runs:
            row = self._build_summary_row(session.session_id, run)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure column order
        columns = [
            "session_id",
            "run_id",
            "trial_index",
            "start_time_iso",
            "end_time_iso",
            "duration_seconds",
            "avg_fps",
            "one_percent_low_fps",
            "point_one_percent_low_fps",
            "p50_ms",
            "p90_ms",
            "p95_ms",
            "p99_ms",
            "p99_9_ms",
            "avg_frame_time_ms",
            "min_frame_time_ms",
            "max_frame_time_ms",
            "stutter_count_total",
            "stutter_worst_severity",
            "notes",
        ]
        # Only include columns that exist in the data
        existing_columns = [c for c in columns if c in df.columns]
        if not df.empty:
            df = df[existing_columns]

        # Write CSV
        output_path = session_dir / "run_summaries.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")

    def _build_summary_row(
        self,
        session_id: str,
        run: BenchmarkRun,
    ) -> Dict[str, Any]:
        """Build a summary row for a benchmark run.

        Args:
            session_id: Session identifier.
            run: BenchmarkRun to summarize.

        Returns:
            Dict of summary metrics.
        """
        fps = run.fps_metrics
        percs = run.percentiles
        stutter = run.stutter_summary

        # Compute end time if we have start time and duration
        start_time_iso = run.timestamp.isoformat() if run.timestamp else ""
        end_time_iso = ""
        if run.timestamp and run.duration_seconds > 0:
            end_time = run.timestamp + timedelta(seconds=run.duration_seconds)
            end_time_iso = end_time.isoformat()

        # Get worst stutter severity
        worst_severity = stutter.get("worst_severity", "")

        # Build notes from warnings
        notes = "; ".join(run.warnings) if run.warnings else ""

        return {
            "session_id": session_id,
            "run_id": run.run_id,
            "trial_index": run.trial_index,
            "start_time_iso": start_time_iso,
            "end_time_iso": end_time_iso,
            "duration_seconds": run.duration_seconds or fps.get("duration_seconds", ""),
            "avg_fps": fps.get("avg_fps", ""),
            "one_percent_low_fps": fps.get("one_percent_low_fps", ""),
            "point_one_percent_low_fps": fps.get("point_one_percent_low_fps", ""),
            "p50_ms": percs.get("p50_ms", ""),
            "p90_ms": percs.get("p90_ms", ""),
            "p95_ms": percs.get("p95_ms", ""),
            "p99_ms": percs.get("p99_ms", ""),
            "p99_9_ms": percs.get("p99_9_ms", ""),
            "avg_frame_time_ms": fps.get("avg_frame_time_ms", ""),
            "min_frame_time_ms": fps.get("min_frame_time_ms", ""),
            "max_frame_time_ms": fps.get("max_frame_time_ms", ""),
            "stutter_count_total": stutter.get("total_events", 0),
            "stutter_worst_severity": worst_severity,
            "notes": notes,
        }

    def _export_stutter_events(
        self,
        session: BenchmarkSession,
        session_dir: Path,
    ) -> None:
        """Export stutter events to CSV.

        Args:
            session: BenchmarkSession containing runs.
            session_dir: Directory to write CSV file.
        """
        rows: List[Dict[str, Any]] = []

        for run in session.runs:
            for event in run.stutter_events:
                rows.append(
                    {
                        "session_id": session.session_id,
                        "run_id": run.run_id,
                        "trial_index": run.trial_index,
                        "start_time_ms": round(event.start_time_ms, 3),
                        "end_time_ms": round(event.end_time_ms, 3),
                        "duration_ms": round(event.duration_ms, 3),
                        "max_frame_time_ms": round(event.max_frame_time_ms, 3),
                        "severity": event.severity.value,
                        "frame_count": event.frame_count,
                    }
                )

        # Define column order
        columns = [
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

        # Create DataFrame with explicit columns to ensure headers even when empty
        df = pd.DataFrame(rows, columns=columns)

        # Write CSV
        output_path = session_dir / "stutter_events.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
