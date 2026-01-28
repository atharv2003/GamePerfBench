"""
FPS Calculation Module.

Converts frame times to FPS metrics with support for percentile-based
"low" FPS calculations (1% low, 0.1% low).
"""

from typing import Dict, List, Union

import numpy as np

ArrayLike = Union[np.ndarray, List[float]]


class FPSCalculator:
    """Calculate FPS metrics from frame time data.

    Low FPS Definition (Option A - frame time based):
        - 1% low FPS = 1000 / 99th percentile frame time
        - 0.1% low FPS = 1000 / 99.9th percentile frame time

    This approach uses the slowest frames (highest frame times) to determine
    the "low" FPS values, which represents the worst-case performance.
    """

    # Percentile method for deterministic results
    PERCENTILE_METHOD = "higher"

    def compute_instantaneous_fps(self, frame_times_ms: ArrayLike) -> np.ndarray:
        """Compute instantaneous FPS for each frame.

        Args:
            frame_times_ms: Array of frame times in milliseconds.

        Returns:
            Array of FPS values (1000 / frame_time_ms).
        """
        ft = np.asarray(frame_times_ms, dtype=np.float64)
        # Avoid division by zero
        safe_ft = np.maximum(ft, 0.001)
        return 1000.0 / safe_ft

    def calculate_summary_metrics(self, frame_times_ms: ArrayLike) -> Dict[str, float]:
        """Calculate comprehensive FPS and frame time summary metrics.

        Args:
            frame_times_ms: Array of frame times in milliseconds.

        Returns:
            Dict containing:
                - avg_fps, min_fps, max_fps
                - avg_frame_time_ms, min_frame_time_ms, max_frame_time_ms
                - one_percent_low_fps, point_one_percent_low_fps
                - p99_frame_time_ms, p99_9_frame_time_ms
                - total_frames, duration_seconds
        """
        ft = np.asarray(frame_times_ms, dtype=np.float64)

        if len(ft) == 0:
            return {
                "avg_fps": 0.0,
                "min_fps": 0.0,
                "max_fps": 0.0,
                "avg_frame_time_ms": 0.0,
                "min_frame_time_ms": 0.0,
                "max_frame_time_ms": 0.0,
                "one_percent_low_fps": 0.0,
                "point_one_percent_low_fps": 0.0,
                "p99_frame_time_ms": 0.0,
                "p99_9_frame_time_ms": 0.0,
                "total_frames": 0,
                "duration_seconds": 0.0,
            }

        # Compute instantaneous FPS
        fps = self.compute_instantaneous_fps(ft)

        # Frame time percentiles (using higher method for determinism)
        p99_ft = float(np.percentile(ft, 99, method=self.PERCENTILE_METHOD))
        p99_9_ft = float(np.percentile(ft, 99.9, method=self.PERCENTILE_METHOD))

        # Low FPS from frame time percentiles
        one_percent_low = 1000.0 / p99_ft if p99_ft > 0 else 0.0
        point_one_percent_low = 1000.0 / p99_9_ft if p99_9_ft > 0 else 0.0

        return {
            "avg_fps": float(np.mean(fps)),
            "min_fps": float(np.min(fps)),
            "max_fps": float(np.max(fps)),
            "avg_frame_time_ms": float(np.mean(ft)),
            "min_frame_time_ms": float(np.min(ft)),
            "max_frame_time_ms": float(np.max(ft)),
            "one_percent_low_fps": one_percent_low,
            "point_one_percent_low_fps": point_one_percent_low,
            "p99_frame_time_ms": p99_ft,
            "p99_9_frame_time_ms": p99_9_ft,
            "total_frames": len(ft),
            "duration_seconds": float(np.sum(ft)) / 1000.0,
        }
