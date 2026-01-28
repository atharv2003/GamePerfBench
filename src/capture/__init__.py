"""Data capture module."""

from .fps_capture import FPSCalculator
from .frametime_capture import (
    CaptureBackend,
    SimulatedCaptureBackend,
)

__all__ = [
    "CaptureBackend",
    "FPSCalculator",
    "SimulatedCaptureBackend",
]
