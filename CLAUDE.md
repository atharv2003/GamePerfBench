# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GamePerfBench is a Windows PC game benchmarking harness that captures performance metrics (FPS, frame-time percentiles, 1% lows, stutter events), produces comparison charts, and automates end-to-end reporting with CSV artifacts and summary visualizations.

**Platform**: Windows-only (uses PresentMon, RTSS, pywin32)

## Build & Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
pytest --cov=src/           # With coverage

# Code formatting and linting
black src/
flake8 src/

# CLI usage (when implemented)
python main.py benchmark --game cyberpunk2077 --preset high --trials 3
python main.py compare --sessions session1 session2 --output comparison.png
python main.py report --session session_id --format html
python main.py list --game cyberpunk2077 --limit 10
```

## Architecture

```
GamePerfBench/
├── config/                 # YAML configuration files
│   ├── benchmark_config.yaml    # Master settings (trials, capture, variance thresholds)
│   ├── presets/                 # Graphics quality presets (low/medium/high/ultra/custom)
│   └── games/                   # Game-specific profiles
├── src/
│   ├── core/               # Orchestration layer
│   │   ├── benchmark_runner.py    # Main execution engine (BenchmarkRunner class)
│   │   ├── metrics_collector.py   # Aggregate metrics collection
│   │   ├── frame_analyzer.py      # Frame-level analysis
│   │   └── variance_checker.py    # Multi-run consistency validation
│   ├── capture/            # Data capture layer
│   │   ├── frametime_capture.py   # PresentMon/RTSS integration (abstract CaptureBackend)
│   │   ├── fps_capture.py         # FPS calculation from frame times
│   │   └── hardware_monitor.py    # Background CPU/GPU/RAM monitoring thread
│   ├── analysis/           # Statistical analysis
│   │   ├── percentile_calculator.py  # Frame time percentiles (p50/p90/p95/p99/p99.9)
│   │   ├── stutter_detector.py       # Stutter classification (micro/minor/major/severe/freeze)
│   │   └── statistical_analyzer.py   # Distribution analysis, outlier detection, ANOVA
│   ├── reporting/          # Output generation
│   │   ├── csv_exporter.py       # Per-run metrics, raw data, hardware logs
│   │   ├── chart_generator.py    # 8 chart types (timelines, distributions, comparisons)
│   │   ├── report_packager.py    # ZIP with CSV + charts + HTML
│   │   └── templates/            # HTML report templates
│   └── utils/              # Shared utilities
│       ├── config_loader.py      # YAML loading + validation + merging
│       ├── logger.py             # Structured logging + perf decorators
│       └── validators.py         # Config schema validation
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── output/                 # Runtime output (runs/, charts/, reports/)
└── main.py                 # CLI entry point
```

## Key Design Patterns

- **Backend abstraction**: `CaptureBackend` abstract base class with implementations for PresentMon → RTSS → FrameView (fallback chain)
- **Configuration merging**: Presets + game overrides + CLI args (deep merge with override priority)
- **Dataclasses**: `FrameData`, `HardwareSnapshot`, `BenchmarkRun`, `BenchmarkSession`, `VarianceCheckResult`
- **Thread-safe monitoring**: `HardwareMonitor` runs in background thread with locks

## Key Dependencies

- **Frame Capture**: PresentMon (primary), RTSS (fallback) via pywin32/comtypes
- **GPU Monitoring**: py3nvml (NVIDIA), ADL (AMD fallback)
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn
- **Templating**: jinja2 for HTML reports

## Stutter Classification Thresholds

| Severity | Relative (×avg) | Absolute |
|----------|-----------------|----------|
| Micro    | 2×              | >25ms    |
| Minor    | 3×              | >50ms    |
| Major    | 5×              | >100ms   |
| Severe   | 10×             | >200ms   |
| Freeze   | 20×             | >500ms   |

## Variance Check Defaults

- Max coefficient of variation: 5%
- Max FPS deviation between runs: 3 FPS
- Default trial count: 3 (with 1 warmup run discarded)
- Cooldown between runs: 30 seconds

## Development Reference

Full specification with implementation details: `GamePerfBench_Development_Plan.md`
