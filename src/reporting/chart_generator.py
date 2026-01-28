"""
Chart Generator Module.

Generates PNG chart artifacts for benchmark sessions using matplotlib.
Uses headless Agg backend for CI-friendly operation.
"""

# Force headless backend before importing pyplot
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict  # noqa: E402

import numpy as np  # noqa: E402

from src.core.models import (  # noqa: E402
    BenchmarkRun,
    BenchmarkSession,
    StutterSeverity,
)


class ChartGenerator:
    """Generate PNG charts for benchmark sessions.

    Creates visualizations of benchmark metrics including:
        - FPS by trial
        - Frame time percentiles by trial
        - Stutter events by severity
        - Session summary

    Uses matplotlib with Agg backend for headless operation.
    """

    def __init__(self, dpi: int = 100, figsize: tuple = (10, 6)):
        """Initialize the chart generator.

        Args:
            dpi: DPI for output images.
            figsize: Default figure size (width, height) in inches.
        """
        self.dpi = dpi
        self.figsize = figsize

    def generate_session_charts(
        self,
        session: BenchmarkSession,
        session_dir: Path,
    ) -> Dict[str, Path]:
        """Generate all charts for a benchmark session.

        Args:
            session: BenchmarkSession containing run data.
            session_dir: Directory to write chart files.

        Returns:
            Dict mapping chart name to file path.
        """
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)

        charts: Dict[str, Path] = {}

        # Generate each chart type
        chart_generators = [
            ("avg_fps_by_trial.png", self._generate_avg_fps_chart),
            ("one_percent_low_by_trial.png", self._generate_one_percent_low_chart),
            ("frametime_p99_by_trial.png", self._generate_p99_chart),
            ("stutter_events_by_severity.png", self._generate_stutter_chart),
            ("session_summary.png", self._generate_summary_chart),
        ]

        for filename, generator in chart_generators:
            filepath = session_dir / filename
            try:
                generator(session, filepath)
                charts[filename] = filepath
            except Exception as e:
                # Log error but continue with other charts
                print(f"Warning: Failed to generate {filename}: {e}")

        return charts

    def generate_run_charts(
        self,
        run: BenchmarkRun,
        run_dir: Path,
        session_id: str,
    ) -> Dict[str, Path]:
        """Generate per-run charts.

        Args:
            run: BenchmarkRun to visualize.
            run_dir: Directory to write chart files.
            session_id: Parent session ID for labeling.

        Returns:
            Dict mapping chart name to file path.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        charts: Dict[str, Path] = {}

        # Frame time series chart
        trial_idx = run.trial_index
        filename = f"trial_{trial_idx}_frametime_series.png"
        filepath = run_dir / filename

        try:
            self._generate_frametime_series(run, filepath, session_id)
            charts[filename] = filepath
        except Exception as e:
            print(f"Warning: Failed to generate {filename}: {e}")

        return charts

    def _generate_avg_fps_chart(
        self,
        session: BenchmarkSession,
        filepath: Path,
    ) -> None:
        """Generate average FPS by trial chart.

        Args:
            session: BenchmarkSession with run data.
            filepath: Output file path.
        """
        if not session.runs:
            self._generate_empty_chart(filepath, "No data available")
            return

        trials = [run.trial_index for run in session.runs]
        avg_fps = [run.fps_metrics.get("avg_fps", 0) for run in session.runs]

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.bar(trials, avg_fps)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Average FPS")
        ax.set_title(
            f"Average FPS by Trial - {session.game_name} ({session.preset_name})"
        )
        ax.set_xticks(trials)

        # Add value labels on bars
        for i, v in enumerate(avg_fps):
            ax.text(trials[i], v + 0.5, f"{v:.1f}", ha="center", va="bottom")

        plt.tight_layout()
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

    def _generate_one_percent_low_chart(
        self,
        session: BenchmarkSession,
        filepath: Path,
    ) -> None:
        """Generate 1% low FPS by trial chart.

        Args:
            session: BenchmarkSession with run data.
            filepath: Output file path.
        """
        if not session.runs:
            self._generate_empty_chart(filepath, "No data available")
            return

        trials = [run.trial_index for run in session.runs]
        one_pct_low = [
            run.fps_metrics.get("one_percent_low_fps", 0) for run in session.runs
        ]

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.bar(trials, one_pct_low)
        ax.set_xlabel("Trial")
        ax.set_ylabel("1% Low FPS")
        ax.set_title(
            f"1% Low FPS by Trial - {session.game_name} ({session.preset_name})"
        )
        ax.set_xticks(trials)

        # Add value labels on bars
        for i, v in enumerate(one_pct_low):
            ax.text(trials[i], v + 0.5, f"{v:.1f}", ha="center", va="bottom")

        plt.tight_layout()
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

    def _generate_p99_chart(
        self,
        session: BenchmarkSession,
        filepath: Path,
    ) -> None:
        """Generate P99 frame time by trial chart.

        Args:
            session: BenchmarkSession with run data.
            filepath: Output file path.
        """
        if not session.runs:
            self._generate_empty_chart(filepath, "No data available")
            return

        trials = [run.trial_index for run in session.runs]

        # Try different key formats for p99
        p99_values = []
        for run in session.runs:
            p99 = run.percentiles.get("p99_ms") or run.percentiles.get("p99_0_ms", 0)
            p99_values.append(p99)

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.bar(trials, p99_values)
        ax.set_xlabel("Trial")
        ax.set_ylabel("P99 Frame Time (ms)")
        ax.set_title(
            f"P99 Frame Time by Trial - {session.game_name} ({session.preset_name})"
        )
        ax.set_xticks(trials)

        # Add value labels on bars
        for i, v in enumerate(p99_values):
            ax.text(trials[i], v + 0.2, f"{v:.1f}", ha="center", va="bottom")

        plt.tight_layout()
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

    def _generate_stutter_chart(
        self,
        session: BenchmarkSession,
        filepath: Path,
    ) -> None:
        """Generate stutter events by severity chart.

        Args:
            session: BenchmarkSession with run data.
            filepath: Output file path.
        """
        # Aggregate stutter counts across all runs
        severity_counts: Dict[str, int] = {s.value: 0 for s in StutterSeverity}

        for run in session.runs:
            if run.stutter_summary:
                counts = run.stutter_summary.get("counts_by_severity", {})
                for severity, count in counts.items():
                    if severity in severity_counts:
                        severity_counts[severity] += count

        # Order by severity (micro to freeze)
        severity_order = ["micro", "minor", "major", "severe", "freeze"]
        labels = []
        values = []

        for severity in severity_order:
            if severity in severity_counts:
                labels.append(severity.capitalize())
                values.append(severity_counts[severity])

        fig, ax = plt.subplots(figsize=self.figsize)

        if sum(values) == 0:
            ax.text(
                0.5,
                0.5,
                "No stutters detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(
                f"Stutter Events by Severity - {session.game_name} ({session.preset_name})"
            )
        else:
            bars = ax.bar(labels, values)
            ax.set_xlabel("Severity")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Stutter Events by Severity - {session.game_name} ({session.preset_name})"
            )

            # Add value labels on bars
            for bar, v in zip(bars, values):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        str(v),
                        ha="center",
                        va="bottom",
                    )

        plt.tight_layout()
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

    def _generate_summary_chart(
        self,
        session: BenchmarkSession,
        filepath: Path,
    ) -> None:
        """Generate session summary chart with text info.

        Args:
            session: BenchmarkSession with aggregate data.
            filepath: Output file path.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis("off")

        # Build summary text
        agg = session.aggregate_metrics or {}
        variance = session.variance_result

        # Format timestamp
        ts_str = "N/A"
        if session.timestamp:
            ts_str = session.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "Session Summary",
            "=" * 50,
            "",
            f"Session ID:     {session.session_id}",
            f"Game:           {session.game_name}",
            f"Preset:         {session.preset_name}",
            f"Timestamp:      {ts_str}",
            f"Number of Runs: {len(session.runs)}",
            "",
            "Performance Metrics",
            "-" * 50,
            f"Average FPS:         {agg.get('avg_of_avg_fps', 0):.2f}",
            f"1% Low FPS:          {agg.get('avg_of_one_percent_low_fps', 0):.2f}",
            "",
        ]

        # Add worst stutter info
        worst_stutter = agg.get("worst_stutter_severity_over_session")
        if worst_stutter:
            lines.append(f"Worst Stutter:       {worst_stutter}")
        else:
            lines.append("Worst Stutter:       None")

        lines.append("")

        # Add variance check result
        if variance:
            status = "PASSED" if variance.passed else "FAILED"
            lines.extend(
                [
                    "Variance Check",
                    "-" * 50,
                    f"Status:              {status}",
                    f"CV:                  {variance.cv_percent:.2f}%",
                    f"Max Deviation:       {variance.max_deviation:.2f} FPS",
                ]
            )

        # Render text
        text = "\n".join(lines)
        ax.text(
            0.1,
            0.9,
            text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
        )

        ax.set_title(
            "Benchmark Session Summary", fontsize=14, fontweight="bold", pad=20
        )

        plt.tight_layout()
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

    def _generate_frametime_series(
        self,
        run: BenchmarkRun,
        filepath: Path,
        session_id: str,
    ) -> None:
        """Generate frame time series chart for a single run.

        Args:
            run: BenchmarkRun with frame data.
            filepath: Output file path.
            session_id: Parent session ID for labeling.
        """
        if len(run.timestamps_ms) == 0 or len(run.frame_times_ms) == 0:
            self._generate_empty_chart(filepath, "No frame data available")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        # Convert timestamps to seconds for readability
        timestamps_sec = np.array(run.timestamps_ms) / 1000.0

        ax.plot(timestamps_sec, run.frame_times_ms, linewidth=0.5)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frame Time (ms)")
        ax.set_title(f"Frame Time Series - Trial {run.trial_index}")

        # Add horizontal line for target frame time if avg_fps is available
        avg_fps = run.fps_metrics.get("avg_fps")
        if avg_fps and avg_fps > 0:
            target_ft = 1000.0 / avg_fps
            ax.axhline(
                y=target_ft,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Avg: {target_ft:.1f}ms",
            )
            ax.legend()

        plt.tight_layout()
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

    def _generate_empty_chart(self, filepath: Path, message: str) -> None:
        """Generate a placeholder chart with a message.

        Args:
            filepath: Output file path.
            message: Message to display.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)
