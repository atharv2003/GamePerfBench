"""
Variance Checker Module.

Evaluates run-to-run stability of benchmark results by checking
the coefficient of variation (CV) of key metrics across runs.
"""

from typing import Any, Dict, List, Union

import numpy as np

from src.core.models import BenchmarkRun, VarianceCheckResult

# Default thresholds for variance checks
MAX_CV_AVG_FPS = 0.05  # 5% max CV for average FPS
MAX_CV_ONE_PERCENT_LOW = 0.10  # 10% max CV for 1% low FPS
MIN_RUNS_REQUIRED = 2  # Minimum runs needed for variance check


class VarianceChecker:
    """Evaluate run-to-run stability of benchmark results.

    Checks that the coefficient of variation (CV) of key metrics
    stays below configured thresholds to ensure reproducible results.

    Default thresholds:
        - avg_fps CV: <= 5%
        - one_percent_low_fps CV: <= 10%
    """

    def __init__(
        self,
        max_cv_avg_fps: float = MAX_CV_AVG_FPS,
        max_cv_one_percent_low: float = MAX_CV_ONE_PERCENT_LOW,
        min_runs: int = MIN_RUNS_REQUIRED,
    ):
        """Initialize the variance checker.

        Args:
            max_cv_avg_fps: Maximum allowed CV for average FPS.
            max_cv_one_percent_low: Maximum allowed CV for 1% low FPS.
            min_runs: Minimum number of runs required.
        """
        self.max_cv_avg_fps = max_cv_avg_fps
        self.max_cv_one_percent_low = max_cv_one_percent_low
        self.min_runs = min_runs

    def check_variance(
        self,
        runs: Union[List[BenchmarkRun], List[Dict[str, Any]]],
    ) -> VarianceCheckResult:
        """Check variance across benchmark runs.

        Args:
            runs: List of BenchmarkRun objects or dicts with metrics.
                Required dict keys: avg_fps, one_percent_low_fps

        Returns:
            VarianceCheckResult with pass/fail status and details.
        """
        # Extract metrics from runs
        metrics = self._extract_metrics(runs)

        if metrics is None:
            return VarianceCheckResult(
                passed=False,
                cv_percent=0.0,
                max_deviation=0.0,
                run_means=[],
                overall_mean=0.0,
                message="Invalid input: could not extract metrics from runs",
            )

        avg_fps_values = metrics["avg_fps"]
        one_percent_low_values = metrics["one_percent_low_fps"]

        # Check minimum runs
        n_runs = len(avg_fps_values)
        if n_runs < self.min_runs:
            return VarianceCheckResult(
                passed=False,
                cv_percent=0.0,
                max_deviation=0.0,
                run_means=avg_fps_values,
                overall_mean=np.mean(avg_fps_values) if avg_fps_values else 0.0,
                message=f"Insufficient runs: {n_runs} < {self.min_runs} required",
            )

        # Calculate CVs
        cv_avg_fps = self._calculate_cv(avg_fps_values)
        cv_one_percent = self._calculate_cv(one_percent_low_values)

        # Check against thresholds
        avg_fps_ok = cv_avg_fps <= self.max_cv_avg_fps
        one_percent_ok = cv_one_percent <= self.max_cv_one_percent_low

        passed = avg_fps_ok and one_percent_ok

        # Calculate overall stats
        overall_mean = float(np.mean(avg_fps_values))
        max_deviation = float(np.max(np.abs(np.array(avg_fps_values) - overall_mean)))
        cv_percent = cv_avg_fps * 100  # Convert to percentage for result

        # Build message
        if passed:
            message = (
                f"Variance check PASSED: "
                f"avg_fps CV={cv_avg_fps * 100:.2f}% "
                f"(max {self.max_cv_avg_fps * 100:.0f}%), "
                f"1% low CV={cv_one_percent * 100:.2f}% "
                f"(max {self.max_cv_one_percent_low * 100:.0f}%)"
            )
        else:
            failures = []
            if not avg_fps_ok:
                failures.append(
                    f"avg_fps CV={cv_avg_fps * 100:.2f}% "
                    f"exceeds {self.max_cv_avg_fps * 100:.0f}%"
                )
            if not one_percent_ok:
                failures.append(
                    f"1% low CV={cv_one_percent * 100:.2f}% "
                    f"exceeds {self.max_cv_one_percent_low * 100:.0f}%"
                )
            message = f"Variance check FAILED: {'; '.join(failures)}"

        return VarianceCheckResult(
            passed=passed,
            cv_percent=cv_percent,
            max_deviation=max_deviation,
            run_means=avg_fps_values,
            overall_mean=overall_mean,
            message=message,
        )

    def _extract_metrics(
        self,
        runs: Union[List[BenchmarkRun], List[Dict[str, Any]]],
    ) -> Dict[str, List[float]] | None:
        """Extract metrics from runs.

        Args:
            runs: List of BenchmarkRun objects or dicts.

        Returns:
            Dict with lists of metric values, or None if extraction fails.
        """
        if not runs:
            return None

        avg_fps_values = []
        one_percent_low_values = []

        for run in runs:
            if isinstance(run, BenchmarkRun):
                # Extract from BenchmarkRun
                fps_metrics = run.fps_metrics
                avg_fps = fps_metrics.get("avg_fps")
                one_percent = fps_metrics.get("one_percent_low_fps")
            elif isinstance(run, dict):
                # Extract from dict
                avg_fps = run.get("avg_fps")
                one_percent = run.get("one_percent_low_fps")
            else:
                return None

            if avg_fps is None or one_percent is None:
                return None

            avg_fps_values.append(float(avg_fps))
            one_percent_low_values.append(float(one_percent))

        return {
            "avg_fps": avg_fps_values,
            "one_percent_low_fps": one_percent_low_values,
        }

    @staticmethod
    def _calculate_cv(values: List[float]) -> float:
        """Calculate coefficient of variation.

        Args:
            values: List of values.

        Returns:
            CV as a decimal (0.05 = 5%).
        """
        if not values or len(values) < 2:
            return 0.0

        arr = np.array(values)
        mean = np.mean(arr)

        if mean == 0:
            return 0.0

        std = np.std(arr, ddof=1)
        return float(std / mean)
