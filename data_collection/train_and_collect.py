"""
1. Collects hardware specs before training
2. Parses training config from YAML
3. Runs mlagents-learn and waits for completion
4. Parse training results from TensorBoard logs
5. Outputs one row per training run to training_data.csv
"""

import argparse
import csv
import os
import subprocess
import time

from csv_schema import get_csv_headers, get_empty_row
from simplified_collectors import collect_all_hardware, collect_system_data
from config_collector import ConfigCollector
from training_results_collector import TrainingResultsCollector


def generate_run_id(env_name: str, trainer: str) -> str:
    timestamp = int(time.time())
    return f"{env_name}_{trainer}_{timestamp}"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ML-Agents training and collect data",
        usage="%(prog)s <docker_used> <trainer_type> <env_name> [options]"
    )
    parser.add_argument(
        "docker_used",
        type=str,
        choices=["true", "false"],
        help="Whether running in Docker (true/false)",
    )
    parser.add_argument(
        "trainer_type",
        type=str,
        choices=["ppo", "sac", "poca", "imitation"],
        help="Trainer type",
    )
    parser.add_argument(
        "env_name",
        type=str,
        help="Environment name (e.g., 3DBall, Crawler)",
    )


    # optional flags
    parser.add_argument(
        "--output",
        type=str,
        default="training_data.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing run if it exists",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training, only collect data from existing results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (auto-generated if not specified)",
    )
    return parser.parse_args()


def run_training(run_id: str, trainer: str, env: str, config_dir: str, force: bool):
    """Run mlagents-learn training"""
    config_path = os.path.join(config_dir, trainer, f"{env}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cmd = ["mlagents-learn", config_path, "--run-id", run_id]
    if force:
        cmd.append("--force")

    print(f"Starting training: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")


def write_csv_row(output_path: str, row: dict):
    """Append a row to the CSV file"""
    headers = get_csv_headers()
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # generate or use provided run_id
    run_id = args.run_id or generate_run_id(args.env_name, args.trainer_type)

    config_dir = os.path.join(project_root, "config")
    results_dir = os.path.abspath("results")

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)

    row = get_empty_row()
    row["run_id"] = run_id

    print(f"Run ID: {run_id}")
    print("Collecting hardware data...")
    hardware_data = collect_all_hardware()
    row.update(hardware_data)

    is_docker = args.docker_used.lower() == "true"
    system_data = collect_system_data(is_docker)
    row.update(system_data)

    print("Collecting config data...")
    config_collector = ConfigCollector(config_dir)
    config_data = config_collector.collect_data(args.trainer_type, args.env_name)
    row.update(config_data)

    if not args.no_train:
        print("Running training...")
        run_training(run_id, args.trainer_type, args.env_name, config_dir, args.force)

    print("Collecting training results...")
    results_collector = TrainingResultsCollector(results_dir)
    results_data = results_collector.collect_data(run_id)
    row.update(results_data)

    print(f"Writing data to {output_path}...")
    write_csv_row(output_path, row)

    print("Done!")
    print(f"  Run ID: {run_id}")
    print(f"  Total steps: {row.get('total_steps')}")
    print(f"  Duration: {row.get('total_duration')} seconds")


if __name__ == "__main__":
    main()
