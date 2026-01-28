"""
Frame Analyzer Module.

Orchestrates frame-level analysis by combining FPS calculation,
percentile analysis, stutter detection, and statistical analysis.
"""

from typing import Any, Dict, List, Union

import numpy as np

from src.analysis.percentile_calculator import PercentileCalculator
from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.analysis.stutter_detector import StutterDetector
from src.capture.fps_capture import FPSCalculator
from src.core.models import DistributionStats, StutterEvent

ArrayLike = Union[np.ndarray, List[float]]


class FrameAnalyzer:
    """Analyze raw frame data to produce comprehensive metrics.

    Combines multiple analysis components:
        - FPSCalculator: FPS metrics and summary statistics
        - PercentileCalculator: Frame time percentiles
        - StutterDetector: Stutter event detection and classification
        - StatisticalAnalyzer: Distribution statistics
    """

    def __init__(
        self,
        fps_calculator: FPSCalculator | None = None,
        percentile_calculator: PercentileCalculator | None = None,
        stutter_detector: StutterDetector | None = None,
        statistical_analyzer: StatisticalAnalyzer | None = None,
    ):
        """Initialize the frame analyzer.

        Args:
            fps_calculator: Optional FPSCalculator instance.
            percentile_calculator: Optional PercentileCalculator instance.
            stutter_detector: Optional StutterDetector instance.
            statistical_analyzer: Optional StatisticalAnalyzer instance.
        """
        self.fps_calculator = fps_calculator or FPSCalculator()
        self.percentile_calculator = percentile_calculator or PercentileCalculator()
        self.stutter_detector = stutter_detector or StutterDetector()
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer()

    def analyze(
        self,
        timestamps_ms: ArrayLike,
        frame_times_ms: ArrayLike,
    ) -> Dict[str, Any]:
        """Analyze frame data and return comprehensive metrics.

        Args:
            timestamps_ms: Array of frame timestamps in milliseconds.
            frame_times_ms: Array of frame times in milliseconds.

        Returns:
            Dict containing:
                - fps_metrics: FPS summary metrics
                - percentiles: Frame time percentiles
                - stutter_events: List of detected StutterEvent objects
                - stutter_summary: Stutter statistics summary
                - distribution_stats: DistributionStats object
        """
        ts = np.asarray(timestamps_ms, dtype=np.float64)
        ft = np.asarray(frame_times_ms, dtype=np.float64)

        # Calculate FPS metrics
        fps_metrics = self.fps_calculator.calculate_summary_metrics(ft)

        # Calculate percentiles
        percentiles = self.percentile_calculator.calculate_frame_time_percentiles(ft)

        # Detect stutters
        stutter_events: List[StutterEvent] = self.stutter_detector.detect_stutters(
            ts, ft
        )

        # Calculate stutter summary
        total_duration_ms = float(np.sum(ft)) if len(ft) > 0 else 0.0
        stutter_summary = self.stutter_detector.calculate_summary(
            stutter_events, total_duration_ms=total_duration_ms
        )

        # Analyze distribution
        distribution_stats: DistributionStats = (
            self.statistical_analyzer.analyze_distribution(ft)
        )

        return {
            "fps_metrics": fps_metrics,
            "percentiles": percentiles,
            "stutter_events": stutter_events,
            "stutter_summary": stutter_summary,
            "distribution_stats": distribution_stats,
        }
