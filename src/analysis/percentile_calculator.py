"""
Percentile Calculator Module.

Calculates frame time percentiles for performance analysis with
deterministic results using explicit numpy percentile methods.
"""

from typing import Dict, List, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, List[float]]

# Default percentiles for gaming benchmarks
DEFAULT_PERCENTILES = [50, 90, 95, 99, 99.9]


class PercentileCalculator:
    """Calculate frame time percentiles.

    Standard percentiles for gaming benchmarks:
        - 50th (median): Typical frame time
        - 90th: Most frames below this
        - 95th: Performance floor for most frames
        - 99th: Worst 1% of frames
        - 99.9th: Extreme outliers
    """

    # Percentile method for deterministic results
    PERCENTILE_METHOD = "higher"

    def __init__(self, percentiles: Optional[List[float]] = None):
        """Initialize with percentiles to calculate.

        Args:
            percentiles: List of percentile values (0-100).
                Defaults to [50, 90, 95, 99, 99.9].
        """
        self.percentiles = (
            percentiles if percentiles is not None else DEFAULT_PERCENTILES
        )

    def calculate_frame_time_percentiles(
        self,
        frame_times_ms: ArrayLike,
        percentiles: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Calculate frame time percentiles.

        Args:
            frame_times_ms: Array of frame times in milliseconds.
            percentiles: Optional override of percentiles to calculate.

        Returns:
            Dict mapping percentile names to values.
            Keys are formatted as "p{value}_ms", e.g., "p50_ms", "p99_9_ms".
        """
        ft = np.asarray(frame_times_ms, dtype=np.float64)
        percs = percentiles if percentiles is not None else self.percentiles

        if len(ft) == 0:
            return {self._format_key(p): 0.0 for p in percs}

        results = {}
        for p in percs:
            value = float(np.percentile(ft, p, method=self.PERCENTILE_METHOD))
            key = self._format_key(p)
            results[key] = value

        return results

    def calculate_fps_percentiles(
        self,
        frame_times_ms: ArrayLike,
        percentiles: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Calculate FPS at each percentile.

        Note: Higher percentile frame time = lower percentile FPS.
        e.g., 99th percentile frame time corresponds to 1st percentile FPS.

        Args:
            frame_times_ms: Array of frame times in milliseconds.
            percentiles: Optional override of percentiles to calculate.

        Returns:
            Dict mapping percentile names to FPS values.
            Keys are formatted as "p{value}_fps".
        """
        ft_percentiles = self.calculate_frame_time_percentiles(
            frame_times_ms, percentiles
        )

        results = {}
        for key, value in ft_percentiles.items():
            # Convert key from p50_ms to p50_fps
            fps_key = key.replace("_ms", "_fps")
            results[fps_key] = 1000.0 / value if value > 0 else 0.0

        return results

    @staticmethod
    def _format_key(percentile: float) -> str:
        """Format percentile value as dict key.

        Args:
            percentile: Percentile value (e.g., 99.9).

        Returns:
            Formatted key (e.g., "p99_9_ms").
        """
        # Replace decimal point with underscore
        p_str = str(percentile).replace(".", "_")
        return f"p{p_str}_ms"
