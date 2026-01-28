"""
Frame Time Capture Module.

Provides abstract capture backend interface and a simulated backend
for testing the full pipeline without Windows dependencies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.models import FrameData


class CaptureBackend(ABC):
    """Abstract base class for frame capture backends."""

    @abstractmethod
    def start_capture(self, process_name: str) -> bool:
        """Start frame capture for target process.

        Args:
            process_name: Name of the process to capture (e.g., 'game.exe')

        Returns:
            True if capture started successfully, False otherwise.
        """
        pass

    @abstractmethod
    def stop_capture(self) -> List[FrameData]:
        """Stop capture and return collected frames.

        Returns:
            List of FrameData objects captured during the session.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system.

        Returns:
            True if the backend can be used, False otherwise.
        """
        pass


class SimulatedCaptureBackend(CaptureBackend):
    """Simulated capture backend for testing.

    Generates realistic synthetic frame time data without requiring
    actual game capture or Windows dependencies.
    """

    # Absolute stutter thresholds in ms
    STUTTER_THRESHOLDS = {
        "micro": 25.0,
        "minor": 50.0,
        "major": 100.0,
        "severe": 200.0,
        "freeze": 500.0,
    }

    def __init__(self) -> None:
        """Initialize the simulated backend."""
        self._captured_frames: List[FrameData] = []
        self._is_capturing: bool = False
        self._process_name: Optional[str] = None

    def start_capture(self, process_name: str) -> bool:
        """Start simulated capture.

        Args:
            process_name: Name of the process (stored but not used).

        Returns:
            Always True for simulated backend.
        """
        self._process_name = process_name
        self._is_capturing = True
        self._captured_frames = []
        return True

    def stop_capture(self) -> List[FrameData]:
        """Stop simulated capture.

        Returns:
            List of captured FrameData (empty if capture_session not called).
        """
        self._is_capturing = False
        frames = self._captured_frames.copy()
        self._captured_frames = []
        return frames

    def is_available(self) -> bool:
        """Check availability.

        Returns:
            Always True for simulated backend.
        """
        return True

    def capture_session(
        self,
        duration_seconds: int,
        target_fps: float = 60.0,
        seed: Optional[int] = 123,
        variance: float = 0.1,
        stutter_injection: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Generate a simulated capture session.

        Args:
            duration_seconds: Duration of the simulated capture.
            target_fps: Target average FPS for the simulation.
            seed: Random seed for reproducibility. None for random.
            variance: Coefficient of variation for frame time jitter (0.0-1.0).
            stutter_injection: Dict mapping stutter severity to count.
                Example: {"micro": 5, "major": 2} injects 5 micro and 2 major
                stutters. If None, no stutters are injected.

        Returns:
            DataFrame with columns:
                - timestamp_ms: Cumulative timestamp
                - frame_time_ms: Time to render frame
                - present_time_ms: Time to present frame (same as frame_time_ms)
                - fps_instantaneous: 1000 / frame_time_ms
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Calculate base frame time from target FPS
        base_frame_time_ms = 1000.0 / target_fps

        # Estimate total frames
        estimated_frames = int(duration_seconds * target_fps * 1.1)  # 10% buffer

        # Generate frame times using lognormal distribution for realism
        # Lognormal naturally produces right-skewed positive values
        sigma = variance  # Shape parameter controls variance
        mu = np.log(base_frame_time_ms) - (sigma**2) / 2  # Adjust mean

        frame_times = rng.lognormal(mean=mu, sigma=sigma, size=estimated_frames)

        # Ensure all frame times are strictly positive
        frame_times = np.maximum(frame_times, 0.1)

        # Inject stutters if requested
        if stutter_injection:
            frame_times = self._inject_stutters(
                frame_times, stutter_injection, rng, base_frame_time_ms
            )

        # Trim to actual duration
        cumulative_time_ms = np.cumsum(frame_times)
        target_duration_ms = duration_seconds * 1000.0
        valid_mask = cumulative_time_ms <= target_duration_ms

        # Ensure we have at least some frames
        if not np.any(valid_mask):
            valid_mask[0] = True

        frame_times = frame_times[valid_mask]

        # Recalculate cumulative timestamps
        timestamps = np.cumsum(frame_times)
        timestamps = np.insert(timestamps[:-1], 0, 0.0)  # Start at 0

        # Calculate instantaneous FPS
        fps_instant = 1000.0 / frame_times

        # Store as FrameData for stop_capture compatibility
        self._captured_frames = [
            FrameData(
                timestamp_ms=ts,
                frame_time_ms=ft,
                present_time_ms=ft,
            )
            for ts, ft in zip(timestamps, frame_times)
        ]

        # Return as DataFrame
        return pd.DataFrame(
            {
                "timestamp_ms": timestamps,
                "frame_time_ms": frame_times,
                "present_time_ms": frame_times,
                "fps_instantaneous": fps_instant,
            }
        )

    def _inject_stutters(
        self,
        frame_times: np.ndarray,
        stutter_injection: Dict[str, int],
        rng: np.random.Generator,
        base_frame_time_ms: float,
    ) -> np.ndarray:
        """Inject stutter events into frame times.

        Args:
            frame_times: Array of frame times to modify.
            stutter_injection: Dict mapping severity to count.
            rng: NumPy random generator.
            base_frame_time_ms: Base frame time for reference.

        Returns:
            Modified frame times array with stutters injected.
        """
        frame_times = frame_times.copy()
        n_frames = len(frame_times)

        # Track used indices to avoid double injection
        used_indices: set = set()

        for severity, count in stutter_injection.items():
            if severity not in self.STUTTER_THRESHOLDS:
                continue

            threshold = self.STUTTER_THRESHOLDS[severity]

            # Inject in first 60% of frames to avoid trimming at duration boundary
            # Leave some margin at the start too
            start_idx = 5
            end_idx = max(start_idx + 1, int(n_frames * 0.6))

            # Get available indices (not already used)
            available = [i for i in range(start_idx, end_idx) if i not in used_indices]

            if len(available) == 0:
                continue

            # Select random indices
            num_to_inject = min(count, len(available))
            indices = rng.choice(available, size=num_to_inject, replace=False)

            # Inject stutters slightly above threshold for clear detection
            for idx in indices:
                # Add 10-50% above threshold for variability
                stutter_time = threshold * (1.0 + rng.uniform(0.1, 0.5))
                frame_times[idx] = stutter_time
                used_indices.add(idx)

        return frame_times
