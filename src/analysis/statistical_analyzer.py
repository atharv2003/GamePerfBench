"""
Statistical Analysis Module.

Provides distribution analysis for frame time data including
mean, median, standard deviation, skewness, and kurtosis.
"""

from typing import List, Union

import numpy as np

from src.core.models import DistributionStats

ArrayLike = Union[np.ndarray, List[float]]


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness using Fisher's definition.

    Args:
        data: Array of values.

    Returns:
        Skewness value. Returns 0.0 for insufficient data.
    """
    n = len(data)
    if n < 3:
        return 0.0

    mean = np.mean(data)
    std = np.std(data, ddof=0)

    if std == 0:
        return 0.0

    # Fisher's skewness (adjusted for sample)
    m3 = np.mean((data - mean) ** 3)
    skew = m3 / (std**3)

    # Adjustment for sample size
    adjustment = np.sqrt(n * (n - 1)) / (n - 2)
    return float(skew * adjustment)


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis using Fisher's definition.

    Args:
        data: Array of values.

    Returns:
        Excess kurtosis (0 for normal distribution).
        Returns 0.0 for insufficient data.
    """
    n = len(data)
    if n < 4:
        return 0.0

    mean = np.mean(data)
    std = np.std(data, ddof=0)

    if std == 0:
        return 0.0

    # Fourth moment
    m4 = np.mean((data - mean) ** 4)
    kurt = m4 / (std**4) - 3.0  # Excess kurtosis

    # Adjustment for sample size
    adjustment = (n - 1) / ((n - 2) * (n - 3))
    return float(((n + 1) * kurt + 6) * adjustment)


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for frame time data.

    Calculates distribution statistics including moments (mean, variance,
    skewness, kurtosis) and summary measures (median, IQR, CV).
    """

    def __init__(self, use_scipy: bool = True):
        """Initialize the analyzer.

        Args:
            use_scipy: Whether to use scipy.stats for skew/kurtosis.
                Falls back to numpy approximations if scipy unavailable.
        """
        self.use_scipy = use_scipy
        self._scipy_available = False

        if use_scipy:
            try:
                from scipy import stats

                self._scipy_stats = stats
                self._scipy_available = True
            except ImportError:
                self._scipy_available = False

    def analyze_distribution(self, frame_times_ms: ArrayLike) -> DistributionStats:
        """Analyze the distribution of frame time data.

        Args:
            frame_times_ms: Array of frame times in milliseconds.

        Returns:
            DistributionStats with all statistical measures.
        """
        ft = np.asarray(frame_times_ms, dtype=np.float64)

        if len(ft) == 0:
            return DistributionStats(
                mean=0.0,
                median=0.0,
                std=0.0,
                variance=0.0,
                skewness=0.0,
                kurtosis=0.0,
                iqr=0.0,
                cv=0.0,
            )

        mean = float(np.mean(ft))
        median = float(np.median(ft))
        std = float(np.std(ft, ddof=1)) if len(ft) > 1 else 0.0
        variance = float(np.var(ft, ddof=1)) if len(ft) > 1 else 0.0

        # Interquartile range
        q75, q25 = np.percentile(ft, [75, 25])
        iqr = float(q75 - q25)

        # Coefficient of variation (as percentage)
        cv = (std / mean * 100.0) if mean > 0 else 0.0

        # Skewness and kurtosis
        if self._scipy_available:
            skewness = float(self._scipy_stats.skew(ft, bias=False))
            kurtosis = float(self._scipy_stats.kurtosis(ft, bias=False))
        else:
            skewness = _calculate_skewness(ft)
            kurtosis = _calculate_kurtosis(ft)

        return DistributionStats(
            mean=mean,
            median=median,
            std=std,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            iqr=iqr,
            cv=cv,
        )

    def calculate_confidence_interval(
        self,
        data: ArrayLike,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate confidence interval for the mean.

        Args:
            data: Array of values.
            confidence: Confidence level (default 0.95 for 95% CI).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        arr = np.asarray(data, dtype=np.float64)

        if len(arr) < 2:
            mean = float(np.mean(arr)) if len(arr) > 0 else 0.0
            return (mean, mean)

        mean = float(np.mean(arr))
        std_err = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))

        if self._scipy_available:
            from scipy import stats

            t_value = stats.t.ppf((1 + confidence) / 2, len(arr) - 1)
        else:
            # Approximate t-value for 95% CI with large sample
            # For small samples this is less accurate
            t_value = 1.96 if confidence == 0.95 else 2.576

        margin = t_value * std_err
        return (mean - margin, mean + margin)

    def detect_outliers(
        self,
        data: ArrayLike,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> np.ndarray:
        """Detect outliers in the data.

        Args:
            data: Array of values.
            method: Detection method ('iqr' or 'zscore').
            threshold: Threshold multiplier (1.5 for IQR, 3.0 for zscore).

        Returns:
            Boolean array where True indicates an outlier.
        """
        arr = np.asarray(data, dtype=np.float64)

        if len(arr) == 0:
            return np.array([], dtype=bool)

        if method == "iqr":
            q75, q25 = np.percentile(arr, [75, 25])
            iqr = q75 - q25
            lower = q25 - threshold * iqr
            upper = q75 + threshold * iqr
            return (arr < lower) | (arr > upper)

        elif method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                return np.zeros(len(arr), dtype=bool)
            z_scores = np.abs((arr - mean) / std)
            return z_scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}")
