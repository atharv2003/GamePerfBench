"""
Stutter Detection Module.

Detects and classifies frame stutters using absolute thresholds:
- micro: >25ms
- minor: >50ms
- major: >100ms
- severe: >200ms
- freeze: >500ms

Consecutive stutter frames are merged into single events.
"""

from typing import Any, Dict, List, Union

import numpy as np

from src.core.models import StutterEvent, StutterSeverity

ArrayLike = Union[np.ndarray, List[float]]

# Absolute thresholds in milliseconds (ordered from highest to lowest)
STUTTER_THRESHOLDS = [
    (StutterSeverity.FREEZE, 500.0),
    (StutterSeverity.SEVERE, 200.0),
    (StutterSeverity.MAJOR, 100.0),
    (StutterSeverity.MINOR, 50.0),
    (StutterSeverity.MICRO, 25.0),
]


class StutterDetector:
    """Detect and classify stutter events in frame time data.

    Uses absolute thresholds for classification:
        - MICRO: >25ms
        - MINOR: >50ms
        - MAJOR: >100ms
        - SEVERE: >200ms
        - FREEZE: >500ms

    Adjacent stutter frames (gap of 0 or 1 frame) are merged into
    single events to avoid over-counting.
    """

    def __init__(self, merge_gap: int = 1):
        """Initialize the stutter detector.

        Args:
            merge_gap: Maximum frame gap to merge adjacent stutters.
                0 = only merge consecutive frames
                1 = merge if separated by at most 1 normal frame
        """
        self.merge_gap = merge_gap

    def classify_frame(self, frame_time_ms: float) -> StutterSeverity | None:
        """Classify a single frame's stutter severity.

        Args:
            frame_time_ms: Frame time in milliseconds.

        Returns:
            StutterSeverity if frame is a stutter, None otherwise.
        """
        for severity, threshold in STUTTER_THRESHOLDS:
            if frame_time_ms > threshold:
                return severity
        return None

    def detect_stutters(
        self,
        timestamps_ms: ArrayLike,
        frame_times_ms: ArrayLike,
    ) -> List[StutterEvent]:
        """Detect all stutter events in frame time data.

        Args:
            timestamps_ms: Array of frame timestamps in milliseconds.
            frame_times_ms: Array of frame times in milliseconds.

        Returns:
            List of StutterEvent objects, sorted by start time.
            Adjacent stutters are merged into single events.
        """
        ts = np.asarray(timestamps_ms, dtype=np.float64)
        ft = np.asarray(frame_times_ms, dtype=np.float64)

        if len(ft) == 0 or len(ts) == 0:
            return []

        # Find all stutter frames
        stutter_frames = []
        for i, frame_time in enumerate(ft):
            severity = self.classify_frame(frame_time)
            if severity is not None:
                stutter_frames.append((i, ts[i], frame_time, severity))

        if not stutter_frames:
            return []

        # Merge adjacent stutters
        return self._merge_stutters(stutter_frames, ts, ft)

    def _merge_stutters(
        self,
        stutter_frames: List[tuple],
        timestamps_ms: np.ndarray,
        frame_times_ms: np.ndarray,
    ) -> List[StutterEvent]:
        """Merge adjacent stutter frames into single events.

        Args:
            stutter_frames: List of (index, timestamp, frame_time, severity).
            timestamps_ms: Full timestamp array.
            frame_times_ms: Full frame time array.

        Returns:
            List of merged StutterEvent objects.
        """
        if not stutter_frames:
            return []

        events = []
        current_group = [stutter_frames[0]]

        for i in range(1, len(stutter_frames)):
            prev_idx = stutter_frames[i - 1][0]
            curr_idx = stutter_frames[i][0]

            # Check if frames are adjacent (within merge_gap)
            if curr_idx - prev_idx <= self.merge_gap + 1:
                current_group.append(stutter_frames[i])
            else:
                # Finalize current group and start new one
                events.append(self._create_event(current_group, frame_times_ms))
                current_group = [stutter_frames[i]]

        # Don't forget the last group
        events.append(self._create_event(current_group, frame_times_ms))

        return events

    def _create_event(
        self,
        group: List[tuple],
        frame_times_ms: np.ndarray,
    ) -> StutterEvent:
        """Create a StutterEvent from a group of stutter frames.

        Args:
            group: List of (index, timestamp, frame_time, severity) tuples.
            frame_times_ms: Full frame time array.

        Returns:
            StutterEvent representing the merged group.
        """
        start_idx = group[0][0]
        start_time = group[0][1]
        end_idx = group[-1][0]

        # Calculate end time (start_time of last frame + its duration)
        end_time = group[-1][1] + group[-1][2]

        # Sum up duration from all frames in the range
        duration = sum(frame_times_ms[start_idx : end_idx + 1])

        # Find max frame time and worst severity
        max_frame_time = max(f[2] for f in group)
        worst_severity = self._get_worst_severity([f[3] for f in group])

        return StutterEvent(
            start_index=start_idx,
            start_time_ms=start_time,
            end_time_ms=end_time,
            duration_ms=duration,
            max_frame_time_ms=max_frame_time,
            severity=worst_severity,
            frame_count=end_idx - start_idx + 1,
        )

    def _get_worst_severity(self, severities: List[StutterSeverity]) -> StutterSeverity:
        """Get the worst (highest) severity from a list.

        Args:
            severities: List of StutterSeverity values.

        Returns:
            The worst severity in the list.
        """
        # Severity order from worst to least
        severity_order = [
            StutterSeverity.FREEZE,
            StutterSeverity.SEVERE,
            StutterSeverity.MAJOR,
            StutterSeverity.MINOR,
            StutterSeverity.MICRO,
        ]

        for severity in severity_order:
            if severity in severities:
                return severity

        return severities[0]  # Fallback

    def calculate_summary(
        self,
        stutters: List[StutterEvent],
        total_duration_ms: float | None = None,
    ) -> Dict[str, Any]:
        """Calculate stutter summary statistics.

        Args:
            stutters: List of detected StutterEvent objects.
            total_duration_ms: Total capture duration for rate calculation.

        Returns:
            Dict with:
                - total_events: Total number of stutter events
                - counts_by_severity: Dict mapping severity to count
                - worst_severity: The worst severity encountered (or None)
                - max_frame_time_ms: Longest frame time across all stutters
                - total_stutter_time_ms: Sum of all stutter durations
                - stutters_per_minute: Stutter rate (if duration provided)
        """
        if not stutters:
            return {
                "total_events": 0,
                "counts_by_severity": {s.value: 0 for s in StutterSeverity},
                "worst_severity": None,
                "max_frame_time_ms": 0.0,
                "total_stutter_time_ms": 0.0,
                "stutters_per_minute": 0.0,
            }

        # Count by severity
        counts = {s.value: 0 for s in StutterSeverity}
        for event in stutters:
            counts[event.severity.value] += 1

        # Find worst severity
        worst = self._get_worst_severity([e.severity for e in stutters])

        # Calculate totals
        max_ft = max(e.max_frame_time_ms for e in stutters)
        total_stutter_time = sum(e.duration_ms for e in stutters)

        # Calculate rate
        rate = 0.0
        if total_duration_ms and total_duration_ms > 0:
            rate = len(stutters) / (total_duration_ms / 60000.0)

        return {
            "total_events": len(stutters),
            "counts_by_severity": counts,
            "worst_severity": worst.value,
            "max_frame_time_ms": max_ft,
            "total_stutter_time_ms": total_stutter_time,
            "stutters_per_minute": rate,
        }
