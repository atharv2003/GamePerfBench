"""
Core data models for GamePerfBench.

Contains dataclasses for frame data, hardware snapshots, benchmark runs,
and analysis results. All models are designed to be platform-independent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class StutterSeverity(Enum):
    """Stutter severity classification."""

    MICRO = "micro"  # 2-3x average frame time, >25ms
    MINOR = "minor"  # 3-5x average frame time, >50ms
    MAJOR = "major"  # 5-10x average frame time, >100ms
    SEVERE = "severe"  # 10-20x average frame time, >200ms
    FREEZE = "freeze"  # >20x average frame time, >500ms


@dataclass
class FrameData:
    """Single frame capture data."""

    timestamp_ms: float
    frame_time_ms: float
    present_time_ms: float
    gpu_busy_ms: Optional[float] = None
    display_latency_ms: Optional[float] = None


@dataclass
class HardwareSnapshot:
    """Point-in-time hardware state.

    Uses a flexible payload dict for platform-specific metrics.
    """

    timestamp_ms: float
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StutterEvent:
    """Stutter event spanning one or more frames.

    Attributes:
        start_index: Index of first stutter frame.
        start_time_ms: Timestamp when stutter started.
        end_time_ms: Timestamp when stutter ended.
        duration_ms: Total duration of the stutter event.
        max_frame_time_ms: Longest frame time within this event.
        severity: Classification based on max_frame_time_ms.
        frame_count: Number of frames in this stutter event.
    """

    start_index: int
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    max_frame_time_ms: float
    severity: StutterSeverity
    frame_count: int = 1

    # Legacy aliases for backward compatibility
    @property
    def index(self) -> int:
        """Alias for start_index (backward compatibility)."""
        return self.start_index

    @property
    def timestamp_ms(self) -> float:
        """Alias for start_time_ms (backward compatibility)."""
        return self.start_time_ms

    @property
    def frame_time_ms(self) -> float:
        """Alias for max_frame_time_ms (backward compatibility)."""
        return self.max_frame_time_ms

    @property
    def duration_frames(self) -> int:
        """Alias for frame_count (backward compatibility)."""
        return self.frame_count

    def multiplier(self, avg_frame_time: float) -> float:
        """Calculate how many times longer than average."""
        if avg_frame_time <= 0:
            return 0.0
        return self.max_frame_time_ms / avg_frame_time


@dataclass
class DistributionStats:
    """Statistical distribution properties."""

    mean: float
    median: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    iqr: float
    cv: float  # Coefficient of variation (%)


@dataclass
class VarianceCheckResult:
    """Result of variance check across benchmark runs."""

    passed: bool
    cv_percent: float
    max_deviation: float
    run_means: List[float] = field(default_factory=list)
    overall_mean: float = 0.0
    message: str = ""


@dataclass
class BenchmarkRun:
    """Data from a single benchmark run."""

    run_id: str
    timestamp: datetime
    config: Dict[str, Any] = field(default_factory=dict)

    # Trial index (0-based position in session)
    trial_index: int = 0

    # Raw data
    frame_times_ms: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    timestamps_ms: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    hardware_snapshots: List[HardwareSnapshot] = field(default_factory=list)

    # Calculated metrics
    fps_metrics: Dict[str, float] = field(default_factory=dict)
    percentiles: Dict[str, float] = field(default_factory=dict)
    stutter_summary: Dict[str, Any] = field(default_factory=dict)
    stutter_events: List["StutterEvent"] = field(default_factory=list)
    hardware_summary: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    duration_seconds: float = 0.0
    total_frames: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class BenchmarkSession:
    """Complete benchmark session with multiple runs."""

    session_id: str
    game_name: str
    preset_name: str
    timestamp: datetime
    config: Dict[str, Any] = field(default_factory=dict)

    runs: List[BenchmarkRun] = field(default_factory=list)
    variance_result: Optional[VarianceCheckResult] = None

    # Aggregated results
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)
    comparison_stats: Dict[str, Any] = field(default_factory=dict)

    # Chart output paths (populated after chart generation)
    chart_paths: Dict[str, Union[str, Path]] = field(default_factory=dict)

    # HTML report path (populated after report generation)
    report_path: Optional[Union[str, Path]] = None

    # ZIP bundle path (populated after bundle generation)
    bundle_path: Optional[Union[str, Path]] = None

