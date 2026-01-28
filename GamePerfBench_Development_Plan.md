# GamePerfBench - Comprehensive Development Plan

## Executive Summary

GamePerfBench is a repeatable PC game benchmarking harness designed to capture performance metrics (FPS, frame-time percentiles, 1% lows, stutter events), produce comparison charts, and automate end-to-end reporting with CSV artifacts and summary visualizations.

---

## Project Architecture Overview

```
GamePerfBench/
├── config/
│   ├── presets/
│   │   ├── low.yaml
│   │   ├── medium.yaml
│   │   ├── high.yaml
│   │   ├── ultra.yaml
│   │   └── custom.yaml
│   ├── games/
│   │   └── game_profiles.yaml
│   └── benchmark_config.yaml
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── benchmark_runner.py
│   │   ├── metrics_collector.py
│   │   ├── frame_analyzer.py
│   │   └── variance_checker.py
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── fps_capture.py
│   │   ├── frametime_capture.py
│   │   └── hardware_monitor.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── percentile_calculator.py
│   │   ├── stutter_detector.py
│   │   └── statistical_analyzer.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── csv_exporter.py
│   │   ├── chart_generator.py
│   │   ├── report_packager.py
│   │   └── templates/
│   │       └── report_template.html
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py
│       ├── logger.py
│       └── validators.py
├── tests/
│   ├── __init__.py
│   ├── test_metrics_collector.py
│   ├── test_frame_analyzer.py
│   ├── test_stutter_detector.py
│   └── test_csv_exporter.py
├── output/
│   ├── runs/
│   ├── charts/
│   └── reports/
├── main.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Phase 1: Foundation & Core Infrastructure (Week 1-2)

### 1.1 Project Setup & Dependencies

**Objective:** Establish project structure, virtual environment, and core dependencies.

**Tasks:**

| Task | Description | Priority | Est. Hours |
|------|-------------|----------|------------|
| 1.1.1 | Create project directory structure | High | 1 |
| 1.1.2 | Initialize Git repository with .gitignore | High | 0.5 |
| 1.1.3 | Create virtual environment | High | 0.5 |
| 1.1.4 | Install and configure dependencies | High | 1 |
| 1.1.5 | Create requirements.txt with version pinning | High | 0.5 |
| 1.1.6 | Set up logging infrastructure | Medium | 2 |

**Dependencies (requirements.txt):**

```
# Core
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration
pyyaml>=6.0
python-dotenv>=1.0.0

# Hardware Monitoring
psutil>=5.9.0
GPUtil>=1.4.0
py3nvml>=0.2.7  # NVIDIA GPU monitoring

# Data Capture
pywin32>=306  # Windows API access
comtypes>=1.2.0  # COM interface for PresentMon/RTSS

# Statistics
scipy>=1.10.0

# Reporting
jinja2>=3.1.0  # HTML report templates
openpyxl>=3.1.0  # Excel export option

# Development
pytest>=7.3.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
```

### 1.2 Configuration System

**Objective:** Build a flexible, config-driven system for benchmark presets and game profiles.

**File: `src/utils/config_loader.py`**

```python
"""
Configuration Loader Module

Responsibilities:
- Load YAML configuration files
- Validate configuration schema
- Merge preset configurations with overrides
- Provide runtime configuration access
"""

class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        pass
    
    def load_benchmark_config(self) -> dict:
        """Load main benchmark configuration"""
        pass
    
    def load_preset(self, preset_name: str) -> dict:
        """Load graphics preset configuration"""
        pass
    
    def load_game_profile(self, game_name: str) -> dict:
        """Load game-specific benchmark profile"""
        pass
    
    def merge_configs(self, base: dict, override: dict) -> dict:
        """Deep merge configurations with override priority"""
        pass
    
    def validate_config(self, config: dict, schema: dict) -> bool:
        """Validate configuration against schema"""
        pass
```

**File: `config/benchmark_config.yaml`**

```yaml
# Master Benchmark Configuration

benchmark:
  # Trial Configuration
  trials:
    count: 3                    # Number of benchmark runs per configuration
    warmup_runs: 1              # Warmup runs (discarded)
    cooldown_seconds: 30        # Cooldown between runs
  
  # Capture Settings
  capture:
    duration_seconds: 60        # Benchmark duration
    sample_rate_hz: 1000        # Frame capture rate
    include_loading: false      # Include loading screens
  
  # Variance Thresholds
  variance:
    max_cv_percent: 5.0         # Maximum coefficient of variation
    max_fps_deviation: 3.0      # Maximum FPS deviation between runs
    require_stable: true        # Require variance check pass

hardware:
  monitor_interval_ms: 100      # Hardware monitoring interval
  capture_temps: true           # Capture temperatures
  capture_power: true           # Capture power draw
  capture_clocks: true          # Capture clock speeds
  thermal_throttle_threshold: 90 # Celsius

output:
  base_dir: "output"
  timestamp_format: "%Y%m%d_%H%M%S"
  csv_precision: 3              # Decimal places
  chart_dpi: 150
  chart_format: "png"

logging:
  level: "INFO"
  file: "benchmark.log"
  console: true
```

**File: `config/presets/high.yaml`**

```yaml
# High Quality Preset

preset:
  name: "High"
  description: "High quality settings for mid-to-high-end systems"

graphics:
  resolution: "1920x1080"
  display_mode: "fullscreen"
  vsync: false
  frame_limit: 0                # Uncapped
  
  quality:
    texture: "high"
    shadows: "high"
    anti_aliasing: "TAA"
    ambient_occlusion: "HBAO+"
    reflections: "SSR"
    effects: "high"
    post_processing: "high"
    view_distance: "high"

# Override specific settings per game if needed
game_overrides:
  cyberpunk2077:
    ray_tracing: false
    dlss: "quality"
```

### 1.3 Logging Infrastructure

**File: `src/utils/logger.py`**

```python
"""
Logging Module

Features:
- Structured logging with context
- File and console handlers
- Performance timing decorators
- Run-specific log files
"""

import logging
import functools
import time
from pathlib import Path
from datetime import datetime

class BenchmarkLogger:
    def __init__(self, name: str, config: dict):
        pass
    
    def setup_handlers(self):
        """Configure file and console handlers"""
        pass
    
    def create_run_logger(self, run_id: str) -> logging.Logger:
        """Create a logger for a specific benchmark run"""
        pass
    
    @staticmethod
    def timed(func):
        """Decorator to log function execution time"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logging.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        return wrapper
```

---

## Phase 2: Data Capture Layer (Week 2-3)

### 2.1 Frame Time Capture

**Objective:** Capture frame timing data from games using multiple methods.

**Capture Methods (Priority Order):**

1. **PresentMon Integration** (Recommended)
   - Microsoft's open-source frame capture tool
   - Most accurate for modern games
   - Supports DXGI, D3D11, D3D12, Vulkan

2. **RTSS (RivaTuner Statistics Server)**
   - Industry standard for overlay/capture
   - COM interface available

3. **NVIDIA FrameView**
   - NVIDIA's official tool
   - CSV export capability

**File: `src/capture/frametime_capture.py`**

```python
"""
Frame Time Capture Module

Responsibilities:
- Capture frame presentation times
- Handle multiple capture backends
- Normalize data to common format
- Detect and handle capture errors
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class FrameData:
    """Single frame capture data"""
    timestamp_ms: float         # Absolute timestamp
    frame_time_ms: float        # Time to render frame
    present_time_ms: float      # Time to present frame
    gpu_busy_ms: Optional[float] = None
    display_latency_ms: Optional[float] = None

class CaptureBackend(ABC):
    """Abstract base class for capture backends"""
    
    @abstractmethod
    def start_capture(self, process_name: str) -> bool:
        """Start frame capture for target process"""
        pass
    
    @abstractmethod
    def stop_capture(self) -> List[FrameData]:
        """Stop capture and return collected frames"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available on system"""
        pass

class PresentMonCapture(CaptureBackend):
    """
    PresentMon-based frame capture
    
    Uses PresentMon CLI tool or direct ETW events
    """
    
    def __init__(self, presentmon_path: str = None):
        self.presentmon_path = presentmon_path or self._find_presentmon()
        self.process = None
        self.output_file = None
    
    def _find_presentmon(self) -> str:
        """Locate PresentMon executable"""
        pass
    
    def start_capture(self, process_name: str) -> bool:
        """
        Start PresentMon capture
        
        Command: PresentMon.exe -process_name {game.exe} 
                 -output_file {temp.csv} -timed {duration}
        """
        pass
    
    def stop_capture(self) -> List[FrameData]:
        """Parse PresentMon CSV output"""
        pass
    
    def _parse_presentmon_csv(self, filepath: str) -> pd.DataFrame:
        """
        Parse PresentMon output columns:
        - Application
        - ProcessID
        - SwapChainAddress
        - Runtime (DXGI, D3D11, D3D12, etc.)
        - SyncInterval
        - PresentFlags
        - AllowsTearing
        - PresentMode
        - MsBetweenPresents (frame time)
        - MsBetweenDisplayChange
        - MsInPresentAPI
        - MsUntilRenderComplete
        - MsUntilDisplayed
        """
        pass

class RTSSCapture(CaptureBackend):
    """RivaTuner Statistics Server capture via COM"""
    pass

class FrameTimeCaptureManager:
    """
    Manages frame capture across multiple backends
    
    Automatically selects best available backend
    Handles fallback if primary fails
    """
    
    def __init__(self, config: dict):
        self.backends = []
        self.active_backend = None
        self._init_backends(config)
    
    def _init_backends(self, config: dict):
        """Initialize available backends in priority order"""
        backend_classes = [PresentMonCapture, RTSSCapture]
        for cls in backend_classes:
            backend = cls()
            if backend.is_available():
                self.backends.append(backend)
    
    def capture_session(self, process_name: str, 
                        duration_seconds: int) -> pd.DataFrame:
        """
        Run a complete capture session
        
        Returns DataFrame with columns:
        - timestamp_ms
        - frame_time_ms
        - present_time_ms
        - fps_instantaneous
        """
        pass
```

### 2.2 Hardware Monitoring

**File: `src/capture/hardware_monitor.py`**

```python
"""
Hardware Monitoring Module

Captures:
- CPU usage, temperature, frequency
- GPU usage, temperature, frequency, VRAM
- RAM usage
- Power consumption (where available)
"""

import psutil
from dataclasses import dataclass
from typing import Dict, List, Optional
import threading
import time

@dataclass
class HardwareSnapshot:
    """Point-in-time hardware state"""
    timestamp_ms: float
    
    # CPU Metrics
    cpu_usage_percent: float
    cpu_temp_celsius: Optional[float]
    cpu_frequency_mhz: float
    cpu_power_watts: Optional[float]
    
    # GPU Metrics
    gpu_usage_percent: float
    gpu_temp_celsius: float
    gpu_frequency_mhz: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_power_watts: Optional[float]
    
    # Memory
    ram_used_mb: float
    ram_total_mb: float

class HardwareMonitor:
    """
    Continuous hardware monitoring with configurable interval
    
    Runs in background thread during benchmark
    """
    
    def __init__(self, config: dict):
        self.interval_ms = config.get('monitor_interval_ms', 100)
        self.capture_temps = config.get('capture_temps', True)
        self.capture_power = config.get('capture_power', True)
        self._running = False
        self._thread = None
        self._snapshots: List[HardwareSnapshot] = []
        self._lock = threading.Lock()
    
    def start(self):
        """Start background monitoring thread"""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> List[HardwareSnapshot]:
        """Stop monitoring and return collected data"""
        self._running = False
        self._thread.join(timeout=2.0)
        with self._lock:
            return self._snapshots.copy()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            snapshot = self._capture_snapshot()
            with self._lock:
                self._snapshots.append(snapshot)
            time.sleep(self.interval_ms / 1000.0)
    
    def _capture_snapshot(self) -> HardwareSnapshot:
        """Capture current hardware state"""
        pass
    
    def _get_nvidia_gpu_stats(self) -> Dict:
        """Get NVIDIA GPU stats via py3nvml"""
        pass
    
    def _get_amd_gpu_stats(self) -> Dict:
        """Get AMD GPU stats via ADL"""
        pass
    
    def detect_thermal_throttling(self, 
                                   snapshots: List[HardwareSnapshot],
                                   threshold: int = 90) -> List[dict]:
        """
        Detect thermal throttling events
        
        Returns list of throttling events with timestamps
        """
        pass
```

### 2.3 FPS Calculator

**File: `src/capture/fps_capture.py`**

```python
"""
FPS Calculation Module

Converts frame times to FPS metrics
Handles various edge cases (stutters, hangs, etc.)
"""

import numpy as np
import pandas as pd
from typing import Tuple

class FPSCalculator:
    """
    Calculate FPS metrics from frame time data
    """
    
    def __init__(self, config: dict):
        self.sample_window_ms = config.get('fps_sample_window_ms', 1000)
    
    def calculate_instant_fps(self, frame_times_ms: np.ndarray) -> np.ndarray:
        """
        Calculate instantaneous FPS for each frame
        
        FPS = 1000 / frame_time_ms
        """
        # Avoid division by zero
        safe_times = np.maximum(frame_times_ms, 0.001)
        return 1000.0 / safe_times
    
    def calculate_windowed_fps(self, frame_times_ms: np.ndarray,
                                window_ms: int = 1000) -> np.ndarray:
        """
        Calculate FPS using rolling window
        
        More stable than instantaneous FPS
        """
        pass
    
    def calculate_summary_metrics(self, 
                                   frame_times_ms: np.ndarray) -> dict:
        """
        Calculate comprehensive FPS summary
        
        Returns:
        - avg_fps: Mean FPS
        - min_fps: Minimum FPS
        - max_fps: Maximum FPS
        - fps_1_percent_low: 1% low FPS
        - fps_0_1_percent_low: 0.1% low FPS
        - fps_std: Standard deviation
        - total_frames: Frame count
        - duration_seconds: Capture duration
        """
        fps_instant = self.calculate_instant_fps(frame_times_ms)
        
        # Sort for percentile calculation
        fps_sorted = np.sort(fps_instant)
        
        return {
            'avg_fps': np.mean(fps_instant),
            'min_fps': np.min(fps_instant),
            'max_fps': np.max(fps_instant),
            'median_fps': np.median(fps_instant),
            'fps_1_percent_low': np.percentile(fps_sorted, 1),
            'fps_0_1_percent_low': np.percentile(fps_sorted, 0.1),
            'fps_5_percent_low': np.percentile(fps_sorted, 5),
            'fps_std': np.std(fps_instant),
            'fps_cv': (np.std(fps_instant) / np.mean(fps_instant)) * 100,
            'total_frames': len(frame_times_ms),
            'duration_seconds': np.sum(frame_times_ms) / 1000.0
        }
```

---

## Phase 3: Analysis Engine (Week 3-4)

### 3.1 Frame Time Analysis

**File: `src/analysis/percentile_calculator.py`**

```python
"""
Percentile Calculator Module

Calculates frame time percentiles for performance analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

class PercentileCalculator:
    """
    Calculate frame time percentiles
    
    Standard percentiles for gaming benchmarks:
    - 50th (median): Typical frame time
    - 90th: Most frames below this
    - 95th: Performance floor for most frames  
    - 99th: Worst 1% of frames
    - 99.9th: Extreme outliers
    """
    
    STANDARD_PERCENTILES = [50, 90, 95, 97, 99, 99.5, 99.9]
    
    def __init__(self, percentiles: List[float] = None):
        self.percentiles = percentiles or self.STANDARD_PERCENTILES
    
    def calculate_percentiles(self, 
                              frame_times_ms: np.ndarray) -> Dict[str, float]:
        """
        Calculate all configured percentiles
        
        Returns dict mapping percentile names to values
        Example: {'p50': 8.33, 'p90': 10.5, 'p99': 16.7}
        """
        results = {}
        for p in self.percentiles:
            key = f"p{p}".replace('.', '_')
            results[key] = np.percentile(frame_times_ms, p)
        return results
    
    def calculate_percentile_fps(self, 
                                  frame_times_ms: np.ndarray) -> Dict[str, float]:
        """
        Calculate percentiles as FPS values
        
        Note: Higher percentile frame time = lower percentile FPS
        e.g., 99th percentile frame time -> 1st percentile FPS
        """
        frame_percentiles = self.calculate_percentiles(frame_times_ms)
        return {k: 1000.0 / v for k, v in frame_percentiles.items()}
    
    def calculate_frame_time_histogram(self,
                                        frame_times_ms: np.ndarray,
                                        bin_width_ms: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create histogram of frame times
        
        Returns (bin_edges, counts)
        """
        max_time = np.percentile(frame_times_ms, 99.9)
        bins = np.arange(0, max_time + bin_width_ms, bin_width_ms)
        counts, edges = np.histogram(frame_times_ms, bins=bins)
        return edges[:-1], counts
```

### 3.2 Stutter Detection

**File: `src/analysis/stutter_detector.py`**

```python
"""
Stutter Detection Module

Detects and categorizes frame stutters:
- Micro-stutters: Small but noticeable hitches
- Major stutters: Significant frame drops
- Freezes: Extended hangs
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

class StutterSeverity(Enum):
    """Stutter severity classification"""
    MICRO = "micro"       # 2-3x average frame time
    MINOR = "minor"       # 3-5x average frame time
    MAJOR = "major"       # 5-10x average frame time
    SEVERE = "severe"     # 10-20x average frame time
    FREEZE = "freeze"     # >20x average frame time (>333ms at 60fps)

@dataclass
class StutterEvent:
    """Single stutter event"""
    index: int
    timestamp_ms: float
    frame_time_ms: float
    severity: StutterSeverity
    duration_frames: int  # Consecutive stutter frames
    
    @property
    def multiplier(self) -> float:
        """How many times longer than average"""
        return self.frame_time_ms / self._avg_frame_time
    
class StutterDetector:
    """
    Detect and classify stutter events
    
    Thresholds are configurable but defaults based on
    industry standards and perception research
    """
    
    DEFAULT_THRESHOLDS = {
        StutterSeverity.MICRO: 2.0,    # 2x average
        StutterSeverity.MINOR: 3.0,    # 3x average
        StutterSeverity.MAJOR: 5.0,    # 5x average
        StutterSeverity.SEVERE: 10.0,  # 10x average
        StutterSeverity.FREEZE: 20.0,  # 20x average
    }
    
    # Absolute thresholds (regardless of target FPS)
    ABSOLUTE_THRESHOLDS_MS = {
        StutterSeverity.MICRO: 25.0,   # >25ms noticeable
        StutterSeverity.MINOR: 50.0,   # >50ms jarring
        StutterSeverity.MAJOR: 100.0,  # >100ms disruptive
        StutterSeverity.SEVERE: 200.0, # >200ms very bad
        StutterSeverity.FREEZE: 500.0, # >500ms game freeze
    }
    
    def __init__(self, 
                 use_relative: bool = True,
                 custom_thresholds: dict = None):
        self.use_relative = use_relative
        self.thresholds = custom_thresholds or self.DEFAULT_THRESHOLDS
    
    def detect_stutters(self, 
                        frame_times_ms: np.ndarray) -> List[StutterEvent]:
        """
        Detect all stutter events in frame time data
        
        Returns list of StutterEvent objects sorted by timestamp
        """
        stutters = []
        avg_frame_time = np.median(frame_times_ms)  # Use median for robustness
        
        for i, ft in enumerate(frame_times_ms):
            severity = self._classify_stutter(ft, avg_frame_time)
            if severity:
                stutters.append(StutterEvent(
                    index=i,
                    timestamp_ms=np.sum(frame_times_ms[:i]),
                    frame_time_ms=ft,
                    severity=severity,
                    duration_frames=1
                ))
        
        # Merge consecutive stutters
        return self._merge_consecutive_stutters(stutters)
    
    def _classify_stutter(self, 
                          frame_time_ms: float,
                          avg_frame_time: float) -> StutterSeverity:
        """Classify a single frame's stutter severity"""
        if self.use_relative:
            multiplier = frame_time_ms / avg_frame_time
            for severity in reversed(list(StutterSeverity)):
                if multiplier >= self.thresholds[severity]:
                    return severity
        else:
            for severity in reversed(list(StutterSeverity)):
                if frame_time_ms >= self.ABSOLUTE_THRESHOLDS_MS[severity]:
                    return severity
        return None
    
    def _merge_consecutive_stutters(self, 
                                     stutters: List[StutterEvent]) -> List[StutterEvent]:
        """Merge stutters that occur in consecutive frames"""
        pass
    
    def calculate_stutter_summary(self, 
                                   stutters: List[StutterEvent],
                                   total_duration_ms: float) -> dict:
        """
        Calculate stutter statistics
        
        Returns:
        - total_stutter_count: Total number of stutters
        - stutters_per_minute: Stutter rate
        - stutter_time_percent: % of time in stutter
        - severity_breakdown: Count by severity
        - worst_stutter_ms: Longest stutter
        """
        if not stutters:
            return {
                'total_stutter_count': 0,
                'stutters_per_minute': 0.0,
                'stutter_time_percent': 0.0,
                'severity_breakdown': {},
                'worst_stutter_ms': 0.0
            }
        
        total_duration_min = total_duration_ms / 60000.0
        stutter_time = sum(s.frame_time_ms for s in stutters)
        
        severity_counts = {}
        for severity in StutterSeverity:
            count = sum(1 for s in stutters if s.severity == severity)
            severity_counts[severity.value] = count
        
        return {
            'total_stutter_count': len(stutters),
            'stutters_per_minute': len(stutters) / total_duration_min,
            'stutter_time_percent': (stutter_time / total_duration_ms) * 100,
            'severity_breakdown': severity_counts,
            'worst_stutter_ms': max(s.frame_time_ms for s in stutters),
            'avg_stutter_ms': np.mean([s.frame_time_ms for s in stutters])
        }
```

### 3.3 Statistical Analyzer

**File: `src/analysis/statistical_analyzer.py`**

```python
"""
Statistical Analysis Module

Provides statistical analysis for benchmark data:
- Distribution analysis
- Outlier detection
- Confidence intervals
- Comparison tests
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DistributionStats:
    """Statistical distribution properties"""
    mean: float
    median: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    iqr: float
    cv: float  # Coefficient of variation
    
class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for benchmark data
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def analyze_distribution(self, data: np.ndarray) -> DistributionStats:
        """Calculate distribution statistics"""
        return DistributionStats(
            mean=np.mean(data),
            median=np.median(data),
            std=np.std(data),
            variance=np.var(data),
            skewness=stats.skew(data),
            kurtosis=stats.kurtosis(data),
            iqr=np.percentile(data, 75) - np.percentile(data, 25),
            cv=(np.std(data) / np.mean(data)) * 100 if np.mean(data) > 0 else 0
        )
    
    def calculate_confidence_interval(self, 
                                       data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate confidence interval for the mean
        
        Returns (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        stderr = stats.sem(data)
        margin = stderr * stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        return (mean - margin, mean + margin)
    
    def detect_outliers(self, 
                        data: np.ndarray,
                        method: str = 'iqr') -> np.ndarray:
        """
        Detect outliers using specified method
        
        Methods:
        - 'iqr': Interquartile range (1.5 * IQR)
        - 'zscore': Z-score (> 3 std)
        - 'mad': Median absolute deviation
        
        Returns boolean mask of outliers
        """
        if method == 'iqr':
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (data < lower) | (data > upper)
        elif method == 'zscore':
            z = np.abs(stats.zscore(data))
            return z > 3
        elif method == 'mad':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z = 0.6745 * (data - median) / mad
            return np.abs(modified_z) > 3.5
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compare_runs(self, 
                     runs: List[np.ndarray]) -> Dict[str, any]:
        """
        Compare multiple benchmark runs for consistency
        
        Returns:
        - means: List of run means
        - overall_mean: Grand mean
        - between_run_variance: Variance between run means
        - within_run_variance: Average variance within runs
        - f_statistic: ANOVA F-statistic
        - p_value: ANOVA p-value
        - is_consistent: Whether runs are statistically similar
        """
        means = [np.mean(run) for run in runs]
        overall_mean = np.mean(means)
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*runs)
        
        return {
            'run_means': means,
            'overall_mean': overall_mean,
            'between_run_variance': np.var(means),
            'within_run_variance': np.mean([np.var(r) for r in runs]),
            'f_statistic': f_stat,
            'p_value': p_value,
            'is_consistent': p_value > 0.05  # Not significantly different
        }
```

### 3.4 Variance Checker

**File: `src/core/variance_checker.py`**

```python
"""
Variance Checker Module

Ensures benchmark results are reproducible by checking
variance across multiple runs
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class VarianceCheckResult:
    """Result of variance check"""
    passed: bool
    cv_percent: float          # Coefficient of variation
    max_deviation: float       # Max deviation from mean
    run_means: List[float]
    overall_mean: float
    message: str

class VarianceChecker:
    """
    Check variance across benchmark runs
    
    Ensures results are reproducible and not affected by
    background processes, thermal throttling, etc.
    """
    
    def __init__(self, config: dict):
        self.max_cv_percent = config.get('max_cv_percent', 5.0)
        self.max_fps_deviation = config.get('max_fps_deviation', 3.0)
        self.require_stable = config.get('require_stable', True)
    
    def check_fps_variance(self, 
                           run_fps_means: List[float]) -> VarianceCheckResult:
        """
        Check if FPS variance across runs is acceptable
        
        Criteria:
        1. Coefficient of variation < threshold
        2. Max deviation from mean < threshold
        """
        if len(run_fps_means) < 2:
            return VarianceCheckResult(
                passed=True,
                cv_percent=0.0,
                max_deviation=0.0,
                run_means=run_fps_means,
                overall_mean=run_fps_means[0] if run_fps_means else 0,
                message="Single run - variance check skipped"
            )
        
        means = np.array(run_fps_means)
        overall_mean = np.mean(means)
        std = np.std(means)
        cv = (std / overall_mean) * 100 if overall_mean > 0 else 0
        max_dev = np.max(np.abs(means - overall_mean))
        
        cv_ok = cv <= self.max_cv_percent
        dev_ok = max_dev <= self.max_fps_deviation
        passed = cv_ok and dev_ok
        
        if passed:
            message = f"Variance check PASSED (CV: {cv:.2f}%, Max Dev: {max_dev:.2f} FPS)"
        else:
            issues = []
            if not cv_ok:
                issues.append(f"CV {cv:.2f}% exceeds {self.max_cv_percent}%")
            if not dev_ok:
                issues.append(f"Deviation {max_dev:.2f} exceeds {self.max_fps_deviation}")
            message = f"Variance check FAILED: {'; '.join(issues)}"
        
        return VarianceCheckResult(
            passed=passed,
            cv_percent=cv,
            max_deviation=max_dev,
            run_means=list(run_fps_means),
            overall_mean=overall_mean,
            message=message
        )
    
    def suggest_additional_runs(self, 
                                 current_results: List[VarianceCheckResult]) -> int:
        """
        Suggest additional runs if variance is borderline
        
        Returns number of additional runs recommended (0 if stable)
        """
        if not current_results:
            return 3  # Default trial count
        
        latest = current_results[-1]
        if latest.passed:
            return 0
        
        # If close to threshold, suggest 1-2 more runs
        if latest.cv_percent < self.max_cv_percent * 1.5:
            return 2
        
        # If far from threshold, suggest investigation
        return 0  # Don't suggest more runs, suggest investigation
```

---

## Phase 4: Benchmark Runner (Week 4-5)

### 4.1 Core Benchmark Runner

**File: `src/core/benchmark_runner.py`**

```python
"""
Benchmark Runner Module

Orchestrates the complete benchmark workflow:
1. Load configuration and presets
2. Initialize capture systems
3. Execute benchmark runs
4. Collect and validate data
5. Trigger analysis and reporting
"""

import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class BenchmarkRun:
    """Data from a single benchmark run"""
    run_id: str
    timestamp: datetime
    config: dict
    
    # Raw data
    frame_times_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    hardware_snapshots: List = field(default_factory=list)
    
    # Calculated metrics
    fps_metrics: dict = field(default_factory=dict)
    percentiles: dict = field(default_factory=dict)
    stutter_summary: dict = field(default_factory=dict)
    hardware_summary: dict = field(default_factory=dict)
    
    # Metadata
    duration_seconds: float = 0.0
    total_frames: int = 0
    warnings: List[str] = field(default_factory=list)

@dataclass
class BenchmarkSession:
    """Complete benchmark session with multiple runs"""
    session_id: str
    game_name: str
    preset_name: str
    timestamp: datetime
    config: dict
    
    runs: List[BenchmarkRun] = field(default_factory=list)
    variance_result: Optional[VarianceCheckResult] = None
    
    # Aggregated results
    aggregate_metrics: dict = field(default_factory=dict)
    comparison_stats: dict = field(default_factory=dict)

class BenchmarkRunner:
    """
    Main benchmark execution engine
    
    Usage:
        runner = BenchmarkRunner(config)
        session = runner.run_benchmark(
            game_name="cyberpunk2077",
            preset="high",
            num_trials=3
        )
    """
    
    def __init__(self, config_path: str = "config/benchmark_config.yaml"):
        # Initialize all subsystems
        pass
    
    def run_benchmark(self,
                      game_name: str,
                      preset: str = "high",
                      num_trials: int = 3,
                      custom_config: dict = None) -> BenchmarkSession:
        """
        Execute complete benchmark session
        """
        pass
    
    def _execute_single_run(self, 
                            game_config: dict,
                            run_label: str) -> BenchmarkRun:
        """Execute a single benchmark run"""
        pass
    
    def _cooldown(self):
        """Wait for system to cool down between runs"""
        pass
    
    def _aggregate_runs(self, runs: List[BenchmarkRun]) -> dict:
        """Aggregate metrics across all runs"""
        pass
```

---

## Phase 5: Reporting & Visualization (Week 5-6)

### 5.1 CSV Exporter

**File: `src/reporting/csv_exporter.py`**

```python
"""
CSV Export Module

Exports benchmark data to CSV format for:
- Raw frame time data
- Summary metrics
- Hardware monitoring data
- Comparison tables
"""

class CSVExporter:
    """
    Export benchmark data to CSV files
    
    Output structure:
    output/
    └── {session_id}/
        ├── raw_frametimes.csv      # All frame times
        ├── run_summaries.csv       # Per-run metrics
        ├── hardware_data.csv       # Hardware snapshots
        ├── stutter_events.csv      # Detected stutters
        └── aggregate_metrics.csv   # Final aggregates
    """
    pass
```

### 5.2 Chart Generator

**File: `src/reporting/chart_generator.py`**

Chart types to implement:
1. **FPS Timeline** - FPS over benchmark duration
2. **Frame Time Distribution** - Histogram of frame times
3. **Percentile Bar Chart** - Key percentiles comparison
4. **Stutter Timeline** - Stutter events over time
5. **Hardware Utilization** - CPU/GPU usage over time
6. **Run Comparison** - Multiple runs side-by-side
7. **Settings Comparison** - Different presets/configs
8. **Summary Dashboard** - All-in-one overview

### 5.3 Report Packager

Creates ZIP archives with:
- All CSV data files
- Generated charts
- HTML summary report
- Metadata JSON/YAML

---

## Phase 6: CLI & Integration (Week 6-7)

### 6.1 Command Line Interface

```bash
# Run benchmark
python main.py benchmark --game cyberpunk2077 --preset high --trials 3

# Compare sessions
python main.py compare --sessions session1 session2 --output comparison.png

# Generate report
python main.py report --session session_id --format html

# List sessions
python main.py list --game cyberpunk2077 --limit 10
```

---

## Phase 7: Testing & Quality Assurance (Week 7-8)

### 7.1 Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures
├── test_data/                     # Sample test data
├── unit/
│   ├── test_fps_calculator.py
│   ├── test_percentile_calculator.py
│   ├── test_stutter_detector.py
│   ├── test_variance_checker.py
│   └── test_config_loader.py
├── integration/
│   ├── test_benchmark_runner.py
│   ├── test_csv_exporter.py
│   └── test_chart_generator.py
└── e2e/
    └── test_full_workflow.py
```

---

## Phase 8: Documentation & Polish (Week 8)

### 8.1 Documentation Deliverables

1. **README.md** - Project overview, quick start
2. **CONTRIBUTING.md** - Development guidelines
3. **docs/configuration.md** - Config file reference
4. **docs/metrics.md** - Metric definitions
5. **docs/api.md** - Python API documentation

---

## Implementation Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1. Foundation | Week 1-2 | Project structure, config system, logging |
| 2. Data Capture | Week 2-3 | Frame capture, hardware monitoring, FPS calc |
| 3. Analysis | Week 3-4 | Percentiles, stutter detection, statistics |
| 4. Runner | Week 4-5 | Benchmark orchestration, variance checking |
| 5. Reporting | Week 5-6 | CSV export, charts, report packaging |
| 6. CLI | Week 6-7 | Command line interface, integration |
| 7. Testing | Week 7-8 | Unit tests, integration tests, E2E tests |
| 8. Documentation | Week 8 | README, API docs, user guides |

---

## Key Metrics Captured

| Metric | Description | Unit |
|--------|-------------|------|
| avg_fps | Mean FPS | fps |
| min_fps | Minimum FPS | fps |
| max_fps | Maximum FPS | fps |
| fps_1_percent_low | 1st percentile FPS | fps |
| fps_0_1_percent_low | 0.1st percentile FPS | fps |
| fps_std | FPS standard deviation | fps |
| fps_cv | Coefficient of variation | % |
| p50_frametime | Median frame time | ms |
| p90_frametime | 90th percentile frame time | ms |
| p95_frametime | 95th percentile frame time | ms |
| p99_frametime | 99th percentile frame time | ms |
| p99_9_frametime | 99.9th percentile frame time | ms |
| stutter_count | Total stutter events | count |
| stutters_per_minute | Stutter frequency | /min |
| stutter_time_percent | Time spent in stutter | % |
| worst_stutter_ms | Longest single stutter | ms |
| avg_cpu_usage | Mean CPU utilization | % |
| avg_gpu_usage | Mean GPU utilization | % |
| max_gpu_temp | Peak GPU temperature | °C |
| gpu_memory_used | VRAM consumption | MB |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| PresentMon unavailable | Fallback to RTSS, manual CSV import |
| Thermal throttling | Cooldown periods, throttle detection |
| Background interference | Warmup runs, variance checking |
| Game crashes | Exception handling, partial data recovery |
| Large data files | Optional raw data export, compression |

---

## Future Enhancements

1. **GUI Application** - Desktop app with real-time monitoring
2. **Web Dashboard** - Browser-based results viewer
3. **Database Backend** - SQLite/PostgreSQL for historical tracking
4. **Game Integration** - Auto-detect running games
5. **Cloud Sync** - Share results across machines
6. **ML Analysis** - Anomaly detection, performance prediction
