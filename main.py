#!/usr/bin/env python
"""
GamePerfBench CLI

Command-line interface for running benchmarks and managing sessions.

Usage:
    python main.py benchmark --game <name> --preset <preset> --trials <n>
    python main.py list [--limit <n>] [--output-root <path>]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def parse_stutter_injection(value: str) -> Dict[str, int]:
    """Parse stutter injection string into dict.

    Args:
        value: String like 'minor=2,major=1'

    Returns:
        Dict mapping severity to count.
    """
    if not value:
        return {}

    result = {}
    for part in value.split(","):
        part = part.strip()
        if "=" in part:
            severity, count = part.split("=", 1)
            result[severity.strip().lower()] = int(count.strip())

    return result


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="gameperfbench",
        description="GamePerfBench - PC Game Benchmarking Tool",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run a benchmark session",
    )
    bench_parser.add_argument(
        "--game",
        required=True,
        help="Game name (e.g., 'test', 'cyberpunk2077')",
    )
    bench_parser.add_argument(
        "--preset",
        required=True,
        help="Graphics preset (low, medium, high, ultra, custom)",
    )
    bench_parser.add_argument(
        "--trials",
        type=int,
        required=True,
        help="Number of benchmark trials to run",
    )
    bench_parser.add_argument(
        "--simulated",
        action="store_true",
        default=True,
        help="Use simulated capture backend (default: True)",
    )
    bench_parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration of each trial in seconds (default: from config)",
    )
    bench_parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target FPS for simulation (default: from config)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: from config)",
    )
    bench_parser.add_argument(
        "--variance",
        type=float,
        default=None,
        help="Frame time variance coefficient (default: from config)",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for session files",
    )
    bench_parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip CSV export",
    )
    bench_parser.add_argument(
        "--stutter",
        type=str,
        default=None,
        help="Stutter injection (e.g., 'minor=2,major=1')",
    )
    bench_parser.add_argument(
        "--charts",
        action="store_true",
        default=None,
        help="Generate PNG charts (default: True when export is enabled)",
    )
    bench_parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Disable chart generation",
    )
    bench_parser.add_argument(
        "--report",
        action="store_true",
        default=None,
        help="Generate HTML report (default: True when export is enabled)",
    )
    bench_parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable HTML report generation",
    )
    bench_parser.add_argument(
        "--bundle",
        action="store_true",
        default=None,
        help="Generate ZIP bundle (default: True when export is enabled)",
    )
    bench_parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="Disable ZIP bundle generation",
    )
    bench_parser.add_argument(
        "--bundle-name",
        type=str,
        default=None,
        help="Custom name for ZIP bundle (must end with .zip)",
    )

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List previous benchmark sessions",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of sessions to show (default: 10)",
    )
    list_parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Root directory for output files (default: output)",
    )
    list_parser.add_argument(
        "--game",
        type=str,
        default=None,
        help="Filter by game name",
    )

    return parser


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Import here to avoid slow startup
    from src.core.benchmark_runner import BenchmarkRunner
    from src.utils.config_loader import ConfigLoader

    # Load and merge configuration
    loader = ConfigLoader()

    # Check if preset exists
    available_presets = loader.get_available_presets()
    if args.preset not in available_presets:
        print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
        print(f"Available presets: {', '.join(available_presets)}", file=sys.stderr)
        return 1

    # Build CLI overrides dict (only include non-None values)
    cli_overrides: Dict[str, object] = {}
    if args.duration is not None:
        cli_overrides["duration_seconds"] = args.duration
    if args.target_fps is not None:
        cli_overrides["target_fps"] = args.target_fps
    if args.seed is not None:
        cli_overrides["seed"] = args.seed
    if args.variance is not None:
        cli_overrides["variance"] = args.variance
    if args.stutter:
        cli_overrides["stutter_injection"] = parse_stutter_injection(args.stutter)

    # Merge configs
    try:
        config = loader.build_benchmark_config(
            game_name=args.game,
            preset_name=args.preset,
            cli_overrides=cli_overrides,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Extract final parameters
    output_root = config.get("output_root", "output")
    duration_seconds = config.get("duration_seconds", 10)
    target_fps = config.get("target_fps", 60.0)
    seed = config.get("seed", 123)
    variance = config.get("variance", 0.05)
    stutter_injection = config.get("stutter_injection")

    # Create runner and execute
    runner = BenchmarkRunner(output_root=output_root)

    output_dir: Optional[Path] = None
    if args.output:
        output_dir = Path(args.output)

    # Determine chart generation setting
    # Default: charts enabled when export is enabled
    # --no-charts explicitly disables
    # --charts explicitly enables (even without export, though that's unusual)
    export_enabled = not args.no_export
    if args.no_charts:
        generate_charts = False
    elif args.charts is not None:
        generate_charts = args.charts
    else:
        generate_charts = export_enabled

    # Determine report generation setting
    # Default: report enabled when export is enabled
    # --no-report explicitly disables
    # --report explicitly enables
    if args.no_report:
        generate_report = False
    elif args.report is not None:
        generate_report = args.report
    else:
        generate_report = export_enabled

    # Determine bundle generation setting
    # Default: bundle enabled when export is enabled
    # --no-bundle explicitly disables
    # --bundle explicitly enables
    if args.no_bundle:
        generate_bundle = False
    elif args.bundle is not None:
        generate_bundle = args.bundle
    else:
        generate_bundle = export_enabled

    bundle_name = args.bundle_name

    try:
        session = runner.run_benchmark(
            game_name=args.game,
            preset=args.preset,
            num_trials=args.trials,
            duration_seconds=duration_seconds,
            simulated=args.simulated,
            target_fps=target_fps,
            seed=seed,
            variance=variance,
            stutter_injection=stutter_injection,
            export=export_enabled,
            output_dir=output_dir,
            charts=generate_charts,
            report=generate_report,
            bundle=generate_bundle,
            bundle_name=bundle_name,
        )
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        return 1

    # Print summary
    agg = session.aggregate_metrics or {}
    variance_result = session.variance_result

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Session ID:        {session.session_id}")
    print(f"Game:              {session.game_name}")
    print(f"Preset:            {session.preset_name}")
    print(f"Trials:            {len(session.runs)}")
    print("-" * 60)
    print(f"Avg FPS:           {agg.get('avg_of_avg_fps', 0):.2f}")
    print(f"1% Low FPS:        {agg.get('avg_of_one_percent_low_fps', 0):.2f}")

    if variance_result:
        status = "PASSED" if variance_result.passed else "FAILED"
        print(f"Variance Check:    {status} (CV: {variance_result.cv_percent:.2f}%)")

    if export_enabled:
        if output_dir:
            print(f"Output Directory:  {output_dir}")
        else:
            runs_dir = Path(output_root) / "runs" / session.session_id
            print(f"Output Directory:  {runs_dir}")

        # Show chart info if charts were generated
        if generate_charts and session.chart_paths:
            print(f"Charts Generated:  {len(session.chart_paths)} files")
            for chart_name in sorted(session.chart_paths.keys()):
                print(f"  - {chart_name}")

        # Show report info if report was generated
        if generate_report and session.report_path:
            print("HTML Report:       report.html")

        # Show bundle info if bundle was generated
        if generate_bundle and session.bundle_path:
            bundle_filename = Path(session.bundle_path).name
            print(f"ZIP Bundle:        {bundle_filename}")

    print("=" * 60 + "\n")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Execute list command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    output_root = Path(args.output_root)
    runs_dir = output_root / "runs"

    if not runs_dir.exists():
        print("No sessions found.")
        return 0

    # Find all session directories
    sessions = []
    for session_dir in runs_dir.iterdir():
        if not session_dir.is_dir():
            continue

        # Try to read run_summaries.csv
        summary_path = session_dir / "run_summaries.csv"
        if not summary_path.exists():
            continue

        try:
            df = pd.read_csv(summary_path)
        except Exception:
            continue

        # Extract session info
        session_id = session_dir.name

        # Filter by game if specified
        if args.game:
            if args.game.lower() not in session_id.lower():
                continue

        # Parse session_id for info (format: game_preset_timestamp_uuid)
        parts = session_id.split("_")
        game = parts[0] if parts else "unknown"
        preset = parts[1] if len(parts) > 1 else "unknown"

        # Get creation time from directory
        try:
            created = datetime.fromtimestamp(session_dir.stat().st_mtime)
        except Exception:
            created = None

        # Calculate average FPS across runs
        avg_fps = df["avg_fps"].mean() if "avg_fps" in df.columns else 0
        num_runs = len(df)

        sessions.append(
            {
                "session_id": session_id,
                "created": created,
                "game": game,
                "preset": preset,
                "num_runs": num_runs,
                "avg_fps": avg_fps,
            }
        )

    if not sessions:
        print("No sessions found.")
        return 0

    # Sort by creation time (newest first)
    sessions.sort(key=lambda x: x["created"] or datetime.min, reverse=True)

    # Limit results
    sessions = sessions[: args.limit]

    # Print table
    print("\n" + "=" * 90)
    print("BENCHMARK SESSIONS")
    print("=" * 90)
    print(f"{'Session ID':<45} {'Created':<20} {'Runs':<6} {'Avg FPS':<10}")
    print("-" * 90)

    for s in sessions:
        created_str = s["created"].strftime("%Y-%m-%d %H:%M") if s["created"] else "N/A"
        sid = s["session_id"]
        runs = s["num_runs"]
        fps = s["avg_fps"]
        print(f"{sid:<45} {created_str:<20} {runs:<6} {fps:<10.2f}")

    print("=" * 90)
    print(f"Showing {len(sessions)} session(s)")
    print()

    return 0


def main() -> int:
    """Main entry point.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "list":
        return cmd_list(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
