"""Unit tests for StutterDetector."""

from src.analysis.stutter_detector import StutterDetector
from src.core.models import StutterSeverity


class TestClassifyFrame:
    """Tests for classify_frame method."""

    def test_normal_frame(self):
        """Test normal frame (< 25ms) returns None."""
        detector = StutterDetector()
        assert detector.classify_frame(16.6667) is None
        assert detector.classify_frame(24.9) is None

    def test_micro_stutter(self):
        """Test micro stutter (> 25ms)."""
        detector = StutterDetector()
        assert detector.classify_frame(26.0) == StutterSeverity.MICRO
        assert detector.classify_frame(49.0) == StutterSeverity.MICRO

    def test_minor_stutter(self):
        """Test minor stutter (> 50ms)."""
        detector = StutterDetector()
        assert detector.classify_frame(51.0) == StutterSeverity.MINOR
        assert detector.classify_frame(99.0) == StutterSeverity.MINOR

    def test_major_stutter(self):
        """Test major stutter (> 100ms)."""
        detector = StutterDetector()
        assert detector.classify_frame(101.0) == StutterSeverity.MAJOR
        assert detector.classify_frame(199.0) == StutterSeverity.MAJOR

    def test_severe_stutter(self):
        """Test severe stutter (> 200ms)."""
        detector = StutterDetector()
        assert detector.classify_frame(201.0) == StutterSeverity.SEVERE
        assert detector.classify_frame(499.0) == StutterSeverity.SEVERE

    def test_freeze(self):
        """Test freeze (> 500ms)."""
        detector = StutterDetector()
        assert detector.classify_frame(501.0) == StutterSeverity.FREEZE
        assert detector.classify_frame(1000.0) == StutterSeverity.FREEZE

    def test_boundary_values(self):
        """Test exact boundary values (threshold is > not >=)."""
        detector = StutterDetector()
        # Exact threshold values should not trigger
        assert detector.classify_frame(25.0) is None
        assert detector.classify_frame(50.0) == StutterSeverity.MICRO
        assert detector.classify_frame(100.0) == StutterSeverity.MINOR
        assert detector.classify_frame(200.0) == StutterSeverity.MAJOR
        assert detector.classify_frame(500.0) == StutterSeverity.SEVERE


class TestDetectStutters:
    """Tests for detect_stutters method."""

    def test_no_stutters(self):
        """Test detection with no stutters."""
        detector = StutterDetector()
        timestamps = [0, 17, 33, 50, 67]
        frame_times = [16.6667] * 5

        stutters = detector.detect_stutters(timestamps, frame_times)
        assert len(stutters) == 0

    def test_single_stutter(self):
        """Test detection of single stutter."""
        detector = StutterDetector()
        timestamps = [0, 17, 33, 50, 117]
        frame_times = [16.6667, 16.6667, 16.6667, 60.0, 16.6667]

        stutters = detector.detect_stutters(timestamps, frame_times)

        assert len(stutters) == 1
        assert stutters[0].severity == StutterSeverity.MINOR
        assert stutters[0].max_frame_time_ms == 60.0
        assert stutters[0].start_index == 3

    def test_multiple_separate_stutters(self):
        """Test detection of multiple separate stutters."""
        detector = StutterDetector()
        # Stutters at index 2 and 6 (far apart)
        timestamps = [0, 17, 33, 150, 167, 183, 200, 330]
        frame_times = [
            16.6667,
            16.6667,
            120.0,
            16.6667,
            16.6667,
            16.6667,
            130.0,
            16.6667,
        ]

        stutters = detector.detect_stutters(timestamps, frame_times)

        assert len(stutters) == 2
        assert stutters[0].severity == StutterSeverity.MAJOR
        assert stutters[1].severity == StutterSeverity.MAJOR

    def test_different_severities(self):
        """Test detection of different severity stutters."""
        detector = StutterDetector()
        # Spread stutters apart so they don't merge (>2 frames between each)
        timestamps = [0, 17, 33, 150, 167, 183, 200, 350, 367, 383, 400, 650]
        frame_times = [
            16.6667,
            16.6667,  # Normal
            60.0,  # Minor at index 2
            16.6667,
            16.6667,
            16.6667,  # Normal gap
            120.0,  # Major at index 6
            16.6667,
            16.6667,
            16.6667,  # Normal gap
            250.0,  # Severe at index 10
            16.6667,  # Normal
        ]

        stutters = detector.detect_stutters(timestamps, frame_times)

        severities = [s.severity for s in stutters]
        assert StutterSeverity.MINOR in severities
        assert StutterSeverity.MAJOR in severities
        assert StutterSeverity.SEVERE in severities

    def test_empty_input(self):
        """Test with empty input."""
        detector = StutterDetector()
        stutters = detector.detect_stutters([], [])
        assert len(stutters) == 0


class TestMergeAdjacentStutters:
    """Tests for stutter merging behavior."""

    def test_merge_consecutive_stutters(self):
        """Test that consecutive stutter frames are merged."""
        detector = StutterDetector(merge_gap=0)
        # Two consecutive stutter frames
        timestamps = [0, 17, 33, 93, 153]
        frame_times = [16.6667, 16.6667, 60.0, 60.0, 16.6667]

        stutters = detector.detect_stutters(timestamps, frame_times)

        assert len(stutters) == 1
        assert stutters[0].frame_count == 2  # Two frames merged
        assert stutters[0].max_frame_time_ms == 60.0

    def test_merge_with_gap(self):
        """Test merging with 1-frame gap (default)."""
        detector = StutterDetector(merge_gap=1)
        # Two stutters with 1 normal frame between
        timestamps = [0, 17, 77, 93, 153]
        frame_times = [16.6667, 60.0, 16.6667, 60.0, 16.6667]

        stutters = detector.detect_stutters(timestamps, frame_times)

        # Should merge into one event with gap=1
        assert len(stutters) == 1
        assert stutters[0].frame_count == 3  # Includes the gap frame

    def test_no_merge_large_gap(self):
        """Test that stutters with large gap are not merged."""
        detector = StutterDetector(merge_gap=1)
        # Two stutters with 3 normal frames between
        timestamps = [0, 60, 77, 93, 110, 170]
        frame_times = [60.0, 16.6667, 16.6667, 16.6667, 60.0, 16.6667]

        stutters = detector.detect_stutters(timestamps, frame_times)

        assert len(stutters) == 2

    def test_merged_event_uses_worst_severity(self):
        """Test merged events use the worst severity."""
        detector = StutterDetector(merge_gap=0)
        # Minor followed by major (consecutive)
        timestamps = [0, 17, 77, 200]
        frame_times = [16.6667, 60.0, 120.0, 16.6667]

        stutters = detector.detect_stutters(timestamps, frame_times)

        assert len(stutters) == 1
        assert stutters[0].severity == StutterSeverity.MAJOR
        assert stutters[0].max_frame_time_ms == 120.0


class TestCalculateSummary:
    """Tests for calculate_summary method."""

    def test_empty_stutters(self):
        """Test summary with no stutters."""
        detector = StutterDetector()
        summary = detector.calculate_summary([])

        assert summary["total_events"] == 0
        assert summary["worst_severity"] is None
        assert summary["max_frame_time_ms"] == 0.0

    def test_counts_by_severity(self):
        """Test severity counts in summary."""
        detector = StutterDetector()
        # Spread stutters apart so they don't merge
        timestamps = [0, 17, 33, 150, 167, 183, 300, 317, 333, 450]
        frame_times = [
            16.6667,  # 0
            60.0,  # 1: minor
            16.6667,
            16.6667,
            16.6667,  # 2-4: gap
            120.0,  # 5: major
            16.6667,
            16.6667,
            16.6667,  # 6-8: gap
            120.0,  # 9: major
        ]

        stutters = detector.detect_stutters(timestamps, frame_times)
        summary = detector.calculate_summary(stutters)

        # 1 minor (60ms), 2 major (120ms each)
        assert summary["counts_by_severity"]["minor"] == 1
        assert summary["counts_by_severity"]["major"] == 2

    def test_worst_severity(self):
        """Test worst severity detection."""
        detector = StutterDetector()
        timestamps = [0, 17, 77, 200]
        frame_times = [16.6667, 60.0, 120.0, 16.6667]

        stutters = detector.detect_stutters(timestamps, frame_times)
        summary = detector.calculate_summary(stutters)

        assert summary["worst_severity"] == "major"

    def test_stutters_per_minute(self):
        """Test stutters per minute calculation."""
        detector = StutterDetector()
        timestamps = [0, 17, 77]
        frame_times = [16.6667, 60.0, 16.6667]

        stutters = detector.detect_stutters(timestamps, frame_times)
        # Total duration: ~93ms = 0.00155 minutes
        summary = detector.calculate_summary(stutters, total_duration_ms=60000.0)

        # 1 stutter in 1 minute = 1.0 stutters/min
        assert summary["stutters_per_minute"] == 1.0
