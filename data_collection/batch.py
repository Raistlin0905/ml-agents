import argparse
import hashlib
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import yaml

PARAM_GRID = {
    # batch_size: powers of 2
    "batch_size": [512, 1024, 2048],
    # buffer_size: should be >= batch_size
    "buffer_size": [6000, 12000, 24000],
    # learning_rate: logarithmic scale
    "learning_rate": [0.005, 0.05, 0.1, 1],
    "beta": [0.001, 0.005],
    "epsilon": [0.1, 0.2],
    "lambd": [0.95, 0.99],
    # num_epoch: training epochs per update
    "num_epoch": [3, 5, 7, 9, 21],
    "learning_rate_schedule": ["linear", "constant"],
}

TARGET_SAMPLES = 20

RANDOM_SEED = 42

# between runs in seconds (0 = no sleep, 300 = 5 min sleep)
SLEEP_BETWEEN_RUNS = 60

ENV_NAME = "3DBall"
TRAINER_TYPE = "ppo"


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
BASE_CONFIG_PATH = os.path.join(CONFIG_DIR, TRAINER_TYPE, f"{ENV_NAME}.yaml")
TEMP_CONFIG_DIR = os.path.join(SCRIPT_DIR, "temp_configs")
PROGRESS_FILE = os.path.join(SCRIPT_DIR, "batch_progress.json")


def get_param_hash(params: Dict) -> str:
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()[:8]


def validate_params(params: Dict) -> bool:
    if params["buffer_size"] < params["batch_size"]:
        return False
    return True


def generate_combinations() -> List[Dict]:
    """Generate combinations with grid search"""
    keys = list(PARAM_GRID.keys())
    all_combos = list(itertools.product(*PARAM_GRID.values()))

    all_params = [dict(zip(keys, combo)) for combo in all_combos]

    valid_params = [p for p in all_params if validate_params(p)]

    print(f"Total grid combinations: {len(all_combos)}")
    print(f"Valid combinations (buffer >= batch): {len(valid_params)}")

    random.seed(RANDOM_SEED)
    if len(valid_params) > TARGET_SAMPLES:
        selected = random.sample(valid_params, TARGET_SAMPLES)
    else:
        selected = valid_params

    print(f"Selected combinations: {len(selected)}")
    return selected


def create_temp_config(params: Dict, index: int) -> str:
    """Create a temp YAML config with modified parameters"""
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

    with open(BASE_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    behavior_name = list(config["behaviors"].keys())[0]

    hyperparams = config["behaviors"][behavior_name]["hyperparameters"]
    for key, value in params.items():
        hyperparams[key] = value

    param_hash = get_param_hash(params)
    temp_filename = f"batch_{index:03d}_{param_hash}.yaml"
    temp_config_path = os.path.join(TEMP_CONFIG_DIR, temp_filename)

    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return temp_config_path


def cleanup_temp_config(temp_config_path: str) -> None:
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)


def load_progress() -> Optional[Dict]:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_progress(progress: Dict) -> None:
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def init_progress(combinations: List[Dict]) -> Dict:
    return {
        "started_at": datetime.now().isoformat(),
        "total_combinations": len(combinations),
        "completed": [],
        "failed": [],
        "param_grid": PARAM_GRID,
        "combinations": combinations,
        "target_samples": TARGET_SAMPLES,
        "random_seed": RANDOM_SEED,
    }


def run_single_training(params: Dict, index: int) -> bool:
    param_hash = get_param_hash(params)
    temp_config_path = None

    try:
        temp_config_path = create_temp_config(params, index)

        run_id = f"batch_{index:03d}_{param_hash}"

        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "headless_train.py"),
            ENV_NAME,
            "--trainer",
            TRAINER_TYPE,
            "--config-path",
            temp_config_path,
            "--no-graphics",
            "--force",
            "--run-id",
            run_id,
        ]

        print(f"Run {index + 1}: {param_hash}")
        print(f"Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print(f"Config: {temp_config_path}")
        print(f"Run ID: {run_id}")

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0

    except Exception as e:
        print(f"Error in run {index}: {e}")
        return False

    finally:
        if temp_config_path:
            cleanup_temp_config(temp_config_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch hyperparameter training for ML-Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress file",
    )
    parser.add_argument(
        "--list-progress",
        action="store_true",
        help="Show current progress and exit",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up temp configs and progress file, then exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --clean
    if args.clean:
        if os.path.exists(TEMP_CONFIG_DIR):
            shutil.rmtree(TEMP_CONFIG_DIR)
            print(f"Removed temp config directory: {TEMP_CONFIG_DIR}")
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print(f"Removed progress file: {PROGRESS_FILE}")
        print("Cleanup complete.")
        return

    # --list-progress
    if args.list_progress:
        progress = load_progress()
        if progress:
            total = progress["total_combinations"]
            completed = len(progress["completed"])
            failed = len(progress["failed"])
            remaining = total - completed - failed
            print(f"\nBatch Progress:")
            print(f"  Started: {progress.get('started_at', 'Unknown')}")
            print(f"  Last updated: {progress.get('last_updated', 'Unknown')}")
            print(f"  Total combinations: {total}")
            print(f"  Completed: {completed}")
            print(f"  Failed: {failed}")
            print(f"  Remaining: {remaining}")
            print(f"  Progress: {completed / total * 100:.1f}%")
        else:
            print("No progress file found. Start a new batch with: python batch.py")
        return

    progress = load_progress()

    if args.resume and progress:
        combinations = progress["combinations"]
        completed_hashes = set(progress["completed"])
        failed_hashes = set(progress["failed"])
        print(f"\nResuming batch run...")
        print(f"  Total: {len(combinations)}")
        print(f"  Already completed: {len(completed_hashes)}")
        print(f"  Previously failed: {len(failed_hashes)}")
    else:
        if progress and not args.resume:
            print("\nExisting progress file found.")
            print("Use --resume to continue, or --clean to start fresh.")
            response = input("Start new batch and overwrite? (y/N): ")
            if response.lower() != "y":
                return

        print("\nGenerating parameter combinations...")
        combinations = generate_combinations()
        progress = init_progress(combinations)
        completed_hashes = set()
        failed_hashes = set()
        save_progress(progress)
        print(f"\nStarting new batch run with {len(combinations)} combinations.")

    print("\n")
    print("Starting batch training...")
    print("\n")

    for i, params in enumerate(combinations):
        param_hash = get_param_hash(params)
        if param_hash in completed_hashes:
            print(f"Skipping run {i + 1} ({param_hash}) - already completed")
            continue

        if param_hash in failed_hashes:
            print(f"Skipping run {i + 1} ({param_hash}) - previously failed")
            continue

        # Sleep before run (skip sleep for first run)
        if SLEEP_BETWEEN_RUNS > 0 and len(progress["completed"]) > 0:
            print(f"Sleeping for {SLEEP_BETWEEN_RUNS} seconds...")
            time.sleep(SLEEP_BETWEEN_RUNS)

        success = run_single_training(params, i)

        if success:
            progress["completed"].append(param_hash)
            completed_hashes.add(param_hash)
        else:
            progress["failed"].append(param_hash)
            failed_hashes.add(param_hash)

        save_progress(progress)

        total = len(combinations)
        done = len(progress["completed"])
        fail = len(progress["failed"])
        print(f"\nProgress: {done}/{total} completed, {fail} failed")

    print("\n")
    print("Batch training complete!")
    print("\n")
    print(f"  Total combinations: {len(combinations)}")
    print(f"  Completed: {len(progress['completed'])}")
    print(f"  Failed: {len(progress['failed'])}")
    print(f"\nResults saved to training_data.csv")
    print(f"Progress file: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
