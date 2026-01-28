"""
Benchmark Runner Module.

Main orchestration engine that coordinates capture, analysis, variance
checking, metrics collection, and CSV export for benchmark sessions.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from src.capture.frametime_capture import CaptureBackend, SimulatedCaptureBackend
from src.core.frame_analyzer import FrameAnalyzer
from src.core.metrics_collector import MetricsCollector
from src.core.models import BenchmarkRun, BenchmarkSession
from src.core.variance_checker import VarianceChecker
from src.reporting.chart_generator import ChartGenerator
from src.reporting.csv_exporter import CSVExporter
from src.reporting.html_reporter import HTMLReporter
from src.reporting.report_packager import ReportPackager


class BenchmarkRunner:
    """Main execution engine for benchmark sessions.

    Orchestrates the complete benchmarking workflow:
        1. Frame capture (simulated or real backend)
        2. Frame-level analysis (FPS, percentiles, stutters)
        3. Run creation with all computed metrics
        4. Variance checking across runs
        5. Session-level metric aggregation
        6. CSV export

    Supports dependency injection for all components, defaulting to
    sensible implementations when not provided.
    """

    def __init__(
        self,
        output_root: Union[str, Path] = "output",
        exporter: Optional[CSVExporter] = None,
        capture_backend: Optional[CaptureBackend] = None,
        frame_analyzer: Optional[FrameAnalyzer] = None,
        variance_checker: Optional[VarianceChecker] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        chart_generator: Optional[ChartGenerator] = None,
        html_reporter: Optional[HTMLReporter] = None,
        report_packager: Optional[ReportPackager] = None,
    ):
        """Initialize the benchmark runner.

        Args:
            output_root: Root directory for output files.
            exporter: Optional CSVExporter instance.
            capture_backend: Optional CaptureBackend instance.
            frame_analyzer: Optional FrameAnalyzer instance.
            variance_checker: Optional VarianceChecker instance.
            metrics_collector: Optional MetricsCollector instance.
            chart_generator: Optional ChartGenerator instance.
            html_reporter: Optional HTMLReporter instance.
            report_packager: Optional ReportPackager instance.
        """
        self.output_root = Path(output_root)
        self.exporter = exporter or CSVExporter(output_root=self.output_root)
        self.capture_backend = capture_backend
        self.frame_analyzer = frame_analyzer or FrameAnalyzer()
        self.variance_checker = variance_checker or VarianceChecker()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.chart_generator = chart_generator or ChartGenerator()
        self.html_reporter = html_reporter or HTMLReporter()
        self.report_packager = report_packager or ReportPackager()

    def run_benchmark(
        self,
        game_name: str,
        preset: str = "high",
        num_trials: int = 3,
        duration_seconds: int = 10,
        simulated: bool = True,
        target_fps: float = 60.0,
        seed: int = 123,
        variance: float = 0.05,
        stutter_injection: Optional[Dict[str, int]] = None,
        export: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        charts: bool = True,
        report: bool = True,
        bundle: bool = True,
        bundle_name: Optional[str] = None,
    ) -> BenchmarkSession:
        """Run a complete benchmark session.

        Args:
            game_name: Name of the game being benchmarked.
            preset: Graphics preset name (e.g., "high", "ultra").
            num_trials: Number of benchmark runs to perform.
            duration_seconds: Duration of each run in seconds.
            simulated: If True, use simulated capture backend.
            target_fps: Target FPS for simulated capture.
            seed: Random seed for reproducibility (simulated mode).
            variance: Frame time variance coefficient (simulated mode).
            stutter_injection: Dict mapping stutter severity to count.
            export: If True, export results to CSV files.
            output_dir: Optional specific output directory.
            charts: If True (and export=True), generate PNG charts.
            report: If True (and export=True), generate HTML report.
            bundle: If True (and export=True), create ZIP bundle.
            bundle_name: Optional custom name for ZIP bundle.

        Returns:
            BenchmarkSession containing all runs and aggregated metrics.
        """
        # Generate session ID
        session_id = self._generate_session_id(game_name, preset)

        # Create session
        session = BenchmarkSession(
            session_id=session_id,
            game_name=game_name,
            preset_name=preset,
            timestamp=datetime.now(),
            config={
                "num_trials": num_trials,
                "duration_seconds": duration_seconds,
                "simulated": simulated,
                "target_fps": target_fps,
                "variance": variance,
                "stutter_injection": stutter_injection,
            },
        )

        # Select capture backend
        backend = self._get_capture_backend(simulated)

        # Run each trial
        for trial_index in range(num_trials):
            # Generate per-trial seed for reproducibility with variation
            trial_seed = seed + trial_index if seed is not None else None

            run = self._run_single_trial(
                session_id=session_id,
                trial_index=trial_index,
                backend=backend,
                duration_seconds=duration_seconds,
                target_fps=target_fps,
                seed=trial_seed,
                variance=variance,
                stutter_injection=stutter_injection,
                game_name=game_name,
                preset=preset,
            )

            session.runs.append(run)

        # Perform variance check
        if len(session.runs) >= 2:
            session.variance_result = self.variance_checker.check_variance(session.runs)

        # Aggregate session-level metrics
        session.aggregate_metrics = self.metrics_collector.aggregate_session(session)

        # Export to CSV if requested
        if export:
            session_dir = self.exporter.export_session(session, output_dir=output_dir)

            # Generate charts if requested
            if charts:
                chart_paths = self.chart_generator.generate_session_charts(
                    session, session_dir
                )
                # Store chart paths in session for reference
                session.chart_paths = chart_paths

            # Generate HTML report if requested
            if report:
                report_path = self.html_reporter.generate_report(session, session_dir)
                session.report_path = report_path

            # Create ZIP bundle if requested
            if bundle:
                bundle_path = self.report_packager.package_session(
                    session, session_dir, bundle_name=bundle_name
                )
                session.bundle_path = bundle_path

        return session

    def _get_capture_backend(self, simulated: bool) -> CaptureBackend:
        """Get the appropriate capture backend.

        Args:
            simulated: If True, return simulated backend.

        Returns:
            CaptureBackend instance.
        """
        if self.capture_backend is not None:
            return self.capture_backend

        if simulated:
            return SimulatedCaptureBackend()

        # In the future, this would try real backends like PresentMon
        # For now, fall back to simulated
        return SimulatedCaptureBackend()

    def _run_single_trial(
        self,
        session_id: str,
        trial_index: int,
        backend: CaptureBackend,
        duration_seconds: int,
        target_fps: float,
        seed: Optional[int],
        variance: float,
        stutter_injection: Optional[Dict[str, int]],
        game_name: str,
        preset: str,
    ) -> BenchmarkRun:
        """Run a single benchmark trial.

        Args:
            session_id: Parent session ID.
            trial_index: 0-based trial index.
            backend: Capture backend to use.
            duration_seconds: Duration of the capture.
            target_fps: Target FPS for simulation.
            seed: Random seed for this trial.
            variance: Frame time variance.
            stutter_injection: Stutter injection config.
            game_name: Game name for run config.
            preset: Preset name for run config.

        Returns:
            BenchmarkRun with all computed metrics.
        """
        # Generate run ID
        run_id = f"{session_id}_run_{trial_index:02d}"

        # Capture frame data
        if isinstance(backend, SimulatedCaptureBackend):
            df = backend.capture_session(
                duration_seconds=duration_seconds,
                target_fps=target_fps,
                seed=seed,
                variance=variance,
                stutter_injection=stutter_injection,
            )
            timestamps_ms = df["timestamp_ms"].values
            frame_times_ms = df["frame_time_ms"].values
        else:
            # For real backends, would start/stop capture
            backend.start_capture(f"{game_name}.exe")
            # In real implementation, would wait for duration
            frames = backend.stop_capture()
            timestamps_ms = np.array([f.timestamp_ms for f in frames])
            frame_times_ms = np.array([f.frame_time_ms for f in frames])

        # Analyze frames
        analysis = self.frame_analyzer.analyze(timestamps_ms, frame_times_ms)

        # Build BenchmarkRun
        run = BenchmarkRun(
            run_id=run_id,
            timestamp=datetime.now(),
            trial_index=trial_index,
            config={
                "game_name": game_name,
                "preset": preset,
                "duration_seconds": duration_seconds,
                "target_fps": target_fps,
                "seed": seed,
                "variance": variance,
            },
            frame_times_ms=np.array(frame_times_ms, dtype=np.float64),
            timestamps_ms=np.array(timestamps_ms, dtype=np.float64),
            fps_metrics=analysis["fps_metrics"],
            percentiles=analysis["percentiles"],
            stutter_events=analysis["stutter_events"],
            stutter_summary=analysis["stutter_summary"],
            duration_seconds=analysis["fps_metrics"].get("duration_seconds", 0.0),
            total_frames=len(frame_times_ms),
        )

        return run

    @staticmethod
    def _generate_session_id(game_name: str, preset: str) -> str:
        """Generate a unique session ID.

        Args:
            game_name: Game name for ID prefix.
            preset: Preset name for ID.

        Returns:
            Unique session ID string.
        """
        # Clean game name (remove spaces, lowercase)
        clean_game = game_name.lower().replace(" ", "_")

        # Timestamp component
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Short UUID suffix for uniqueness
        short_uuid = uuid.uuid4().hex[:6]

        return f"{clean_game}_{preset}_{ts}_{short_uuid}"
