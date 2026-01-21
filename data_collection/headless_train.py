"""
Automated headless data collection for ML-Agents training. Runs training using a pre-built Unity executable
"""

import argparse
import csv
import os
import subprocess
import sys
import time

from simplified_collectors import collect_all_hardware, collect_system_data
from config_collector import ConfigCollector
from training_results_collector import TrainingResultsCollector
from csv_schema import get_csv_headers, get_empty_row


def generate_run_id(env_name: str, trainer: str) -> str:
    timestamp = int(time.time())
    return f"{env_name}_{trainer}_{timestamp}"


def get_build_path(env_name: str, builds_dir: str = None) -> str:
    if builds_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        builds_dir = os.path.join(script_dir, "builds")

    platform = sys.platform

    if platform == "darwin":
        # macOS: look for {env_name}.app
        platform_dir = os.path.join(builds_dir, "macos")
        app_path = os.path.join(platform_dir, f"{env_name}.app")
        if os.path.exists(app_path):
            # path without .app extension because mlagents env_utils appends it
            return os.path.join(platform_dir, env_name)

    elif platform == "win32":
        # windows: look for {env_name}/{env_name}.exe or {env_name}.exe
        platform_dir = os.path.join(builds_dir, "windows")

        # check if its directory
        dir_path = os.path.join(platform_dir, env_name)
        if os.path.isdir(dir_path):
            exe_path = os.path.join(dir_path, f"{env_name}.exe")
            if os.path.exists(exe_path):
                return os.path.join(dir_path, env_name)

        exe_path = os.path.join(platform_dir, f"{env_name}.exe")
        if os.path.exists(exe_path):
            return os.path.join(platform_dir, env_name)

    elif platform.startswith("linux"):
        # linux: look for {env_name}.x86_64
        platform_dir = os.path.join(builds_dir, "linux")
        for ext in [".x86_64", ".x86", ""]:
            path = os.path.join(platform_dir, f"{env_name}{ext}")
            if os.path.exists(path):
                return os.path.join(platform_dir, env_name)
    raise FileNotFoundError(f"No build found for '{env_name}' on platform '{platform}'")


def run_headless_training(
    env_path: str,
    config_path: str,
    run_id: str,
    no_graphics: bool = True,
    time_scale: float = 20.0,
    force: bool = False,
    resume: bool = False
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "mlagents.trainers.learn",
        config_path,
        "--run-id",
        run_id,
        "--env",
        env_path,
        "--time-scale",
        str(time_scale),
    ]

    if no_graphics: cmd.append("--no-graphics")
    if force: cmd.append("--force")
    if resume: cmd.append("--resume")

    print(f"Starting headless training: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def write_csv_row(output_path: str, row: dict):
    headers = get_csv_headers()
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run headless ML-Agents training and collect data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "env_name",
        type=str,
        nargs="?",
        default="3DBall",
        help="Environment name (default: 3DBall)",
    )

    # training configuration
    parser.add_argument(
        "--trainer",
        type=str,
        choices=["ppo", "sac", "poca"],
        default="ppo",
        help="Trainer type (default: ppo)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (auto-generated if not specified)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing run if it exists",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a checkpoint",
    )

    # headless/graphics options
    parser.add_argument(
        "--no-graphics",
        action="store_true",
        help="Run Unity in no-graphics mode (faster, no visual observations)",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=20.0,
        help="Unity time scale (default: 20.0)",
    )

    # path config
    parser.add_argument(
        "--builds-dir",
        type=str,
        default=None,
        help="Directory containing Unity builds (default: data_collection/builds/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data.csv",
        help="Output CSV file path (default: training_data.csv)",
    )

    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training, only collect data from existing results",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Custom config file path (overrides auto-generated path from trainer/env_name)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    env_name = args.env_name
    trainer = args.trainer
    builds_dir = args.builds_dir or os.path.join(script_dir, "builds")
    config_dir = os.path.join(project_root, "config")
    results_dir = os.path.abspath("results")

    run_id = args.run_id or generate_run_id(env_name, trainer)

    # Use custom config path if provided, otherwise auto-generate from trainer/env_name
    if args.config_path:
        config_path = args.config_path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
    else:
        config_path = os.path.join(config_dir, trainer, f"{env_name}.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # find build (skip if --no-train)
    build_path = None
    if not args.no_train:
        try:
            build_path = get_build_path(env_name, builds_dir)
            print(f"Found build: {build_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    row = get_empty_row()
    row["run_id"] = run_id

    print("Collecting hardware data...")
    hardware_data = collect_all_hardware()
    row.update(hardware_data)

    system_data = collect_system_data(False)
    row.update(system_data)

    print("Collecting config data...")
    config_collector = ConfigCollector(config_dir)
    # Use collect_from_path if custom config provided, otherwise use trainer/env_name
    if args.config_path:
        config_data = config_collector.collect_from_path(config_path)
    else:
        config_data = config_collector.collect_data(trainer, env_name)
    row.update(config_data)

    # run training (if not selected --no-train)
    if not args.no_train:
        print(f"\nStarting headless training for {env_name}...")
        print(f"  Build: {build_path}")
        print(f"  Config: {config_path}")
        print(f"  Run ID: {run_id}")
        print(f"  No graphics: {args.no_graphics}")
        print(f"  Time scale: {args.time_scale}")
        print()

        return_code = run_headless_training(
            env_path=build_path,
            config_path=config_path,
            run_id=run_id,
            no_graphics=args.no_graphics,
            time_scale=args.time_scale,
            force=args.force,
            resume=args.resume,
        )

        if return_code != 0:
            print(f"Warning: Training exited with code {return_code}")

    print("\nCollecting training results...")
    results_collector = TrainingResultsCollector(results_dir)
    results_data = results_collector.collect_data(run_id)
    row.update(results_data)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)

    print(f"Writing data to {output_path}...")
    write_csv_row(output_path, row)

    print("\nDone!")
    print(f"  Run ID: {run_id}")
    print(f"  Total steps: {row.get('total_steps')}")
    print(f"  Duration: {row.get('total_duration')} seconds")


if __name__ == "__main__":
    main()
