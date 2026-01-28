# GamePerfBench

A Windows PC game benchmarking harness that captures performance metrics (FPS, frame-time percentiles, 1% lows, stutter events), produces comparison charts, and automates end-to-end reporting with CSV artifacts and summary visualizations.

## Features

- **Frame Time Analysis**: Capture and analyze frame times with percentile calculations (p50/p90/p95/p99/p99.9)
- **Stutter Detection**: Automatic classification of stutters (micro/minor/major/severe/freeze)
- **Variance Checking**: Multi-run consistency validation to ensure reproducible results
- **CSV Export**: Detailed per-run metrics, raw frame data, and stutter event logs
- **Simulated Mode**: Test the full pipeline without Windows dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gameperfbench.git
cd gameperfbench

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run a simulated benchmark:

```bash
python main.py benchmark --game test --preset high --trials 3 --simulated
```

List previous sessions:

```bash
python main.py list
```

## CLI Usage

### Benchmark Command

Run a benchmark session:

```bash
python main.py benchmark --game <name> --preset <preset> --trials <n> [options]
```

**Required arguments:**
- `--game`: Game name (e.g., 'test', 'cyberpunk2077')
- `--preset`: Graphics preset (low, medium, high, ultra, custom)
- `--trials`: Number of benchmark trials to run

**Optional arguments:**
- `--simulated`: Use simulated capture backend (default: True)
- `--duration <seconds>`: Duration of each trial (default: from config)
- `--target-fps <float>`: Target FPS for simulation (default: from config)
- `--seed <int>`: Random seed for reproducibility (default: from config)
- `--variance <float>`: Frame time variance coefficient (default: from config)
- `--output <path>`: Output directory for session files
- `--no-export`: Skip CSV export
- `--stutter <spec>`: Stutter injection (e.g., 'minor=2,major=1')
- `--charts`: Generate PNG charts (default: True when export enabled)
- `--no-charts`: Disable chart generation
- `--report`: Generate HTML report (default: True when export enabled)
- `--no-report`: Disable HTML report generation
- `--bundle`: Generate ZIP bundle (default: True when export enabled)
- `--no-bundle`: Disable ZIP bundle generation
- `--bundle-name <name.zip>`: Custom name for ZIP bundle

**Examples:**

```bash
# Basic benchmark with high preset
python main.py benchmark --game test --preset high --trials 3 --simulated

# Custom duration and target FPS
python main.py benchmark --game cyberpunk2077 --preset ultra --trials 5 --duration 30 --target-fps 45

# With stutter injection
python main.py benchmark --game test --preset medium --trials 2 --stutter "minor=3,major=1"

# Custom output directory
python main.py benchmark --game test --preset high --trials 2 --output ./my_results
```

### List Command

List previous benchmark sessions:

```bash
python main.py list [options]
```

**Optional arguments:**
- `--limit <n>`: Maximum number of sessions to show (default: 10)
- `--output-root <path>`: Root directory for output files (default: output)
- `--game <name>`: Filter by game name

**Examples:**

```bash
# List recent sessions
python main.py list

# List more sessions
python main.py list --limit 20

# Filter by game
python main.py list --game cyberpunk2077
```

## Configuration

Configuration files are located in the `config/` directory:

- `benchmark_config.yaml`: Global defaults
- `presets/*.yaml`: Graphics preset configurations
- `games/game_profiles.yaml`: Game-specific settings

### Configuration Precedence

Settings are merged with the following precedence (highest wins):

1. CLI arguments
2. Preset YAML
3. Game profile
4. Global benchmark_config.yaml

## Output Files

Each benchmark session creates a directory with:

**CSV Files:**
- `raw_frametimes.csv`: Frame-by-frame timing data
- `run_summaries.csv`: Per-run aggregate metrics
- `stutter_events.csv`: Detected stutter events

**PNG Charts (when `--charts` is enabled, default):**
- `avg_fps_by_trial.png`: Bar chart of average FPS per trial
- `one_percent_low_by_trial.png`: Bar chart of 1% low FPS per trial
- `frametime_p99_by_trial.png`: Bar chart of P99 frame time per trial
- `stutter_events_by_severity.png`: Bar chart of stutter counts by severity
- `session_summary.png`: Text summary with key metrics

**HTML Report (when `--report` is enabled, default):**
- `report.html`: Self-contained HTML report with embedded charts

**ZIP Bundle (when `--bundle` is enabled, default):**
- `GamePerfBench_<session_id>.zip`: Portable archive containing:
  - All CSV files
  - PNG charts (if generated)
  - HTML report (if generated)
  - `metadata.json`: Session info and aggregate metrics
  - `manifest.json`: File list with sizes and SHA256 hashes

Use `--no-charts`, `--no-report`, or `--no-bundle` to disable individual outputs.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/

# Run specific test file
pytest tests/e2e/test_cli.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/ main.py

# Lint code
flake8 src/ tests/ main.py
```

## Stutter Classification

| Severity | Threshold |
|----------|-----------|
| Micro    | >25ms     |
| Minor    | >50ms     |
| Major    | >100ms    |
| Severe   | >200ms    |
| Freeze   | >500ms    |

## License

MIT License
