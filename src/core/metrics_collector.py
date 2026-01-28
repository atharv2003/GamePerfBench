"""
Metrics Collector Module.

Aggregates session-level metrics across multiple benchmark runs.
"""

from typing import Any, Dict, List, Optional

from src.core.models import BenchmarkSession, StutterSeverity

# Severity ordering for comparison (worst to least)
SEVERITY_ORDER = [
    StutterSeverity.FREEZE,
    StutterSeverity.SEVERE,
    StutterSeverity.MAJOR,
    StutterSeverity.MINOR,
    StutterSeverity.MICRO,
]


class MetricsCollector:
    """Aggregate metrics across a benchmark session.

    Computes session-level statistics from multiple runs including
    average FPS, average 1% low, and worst stutter severity.
    """

    def aggregate_session(self, session: BenchmarkSession) -> Dict[str, Any]:
        """Aggregate metrics across all runs in a session.

        Args:
            session: BenchmarkSession containing runs to aggregate.

        Returns:
            Dict containing:
                - num_runs: Number of runs in session
                - avg_of_avg_fps: Mean of per-run average FPS
                - avg_of_one_percent_low_fps: Mean of per-run 1% low FPS
                - worst_stutter_severity_over_session: Worst severity or None
                - total_stutter_count: Sum of stutters across runs
        """
        runs = session.runs

        if not runs:
            return {
                "num_runs": 0,
                "avg_of_avg_fps": 0.0,
                "avg_of_one_percent_low_fps": 0.0,
                "worst_stutter_severity_over_session": None,
                "total_stutter_count": 0,
            }

        # Collect per-run metrics
        avg_fps_values: List[float] = []
        one_percent_low_values: List[float] = []
        stutter_severities: List[StutterSeverity] = []
        total_stutters = 0

        for run in runs:
            # Get avg_fps
            avg_fps = run.fps_metrics.get("avg_fps")
            if avg_fps is not None:
                avg_fps_values.append(float(avg_fps))

            # Get one_percent_low_fps
            one_percent = run.fps_metrics.get("one_percent_low_fps")
            if one_percent is not None:
                one_percent_low_values.append(float(one_percent))

            # Collect stutter severities
            for event in run.stutter_events:
                stutter_severities.append(event.severity)
                total_stutters += 1

        # Calculate averages
        avg_of_avg_fps = (
            sum(avg_fps_values) / len(avg_fps_values) if avg_fps_values else 0.0
        )
        avg_of_one_percent_low = (
            sum(one_percent_low_values) / len(one_percent_low_values)
            if one_percent_low_values
            else 0.0
        )

        # Find worst severity
        worst_severity = self._find_worst_severity(stutter_severities)

        return {
            "num_runs": len(runs),
            "avg_of_avg_fps": avg_of_avg_fps,
            "avg_of_one_percent_low_fps": avg_of_one_percent_low,
            "worst_stutter_severity_over_session": (
                worst_severity.value if worst_severity else None
            ),
            "total_stutter_count": total_stutters,
        }

    def _find_worst_severity(
        self, severities: List[StutterSeverity]
    ) -> Optional[StutterSeverity]:
        """Find the worst (highest) severity from a list.

        Args:
            severities: List of StutterSeverity values.

        Returns:
            The worst severity, or None if list is empty.
        """
        if not severities:
            return None

        for severity in SEVERITY_ORDER:
            if severity in severities:
                return severity

        return severities[0] if severities else None
