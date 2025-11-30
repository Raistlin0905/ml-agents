import subprocess
import uuid
import sys
import os
import pandas
import csv
from csv_writer import CSVWriter
from cpu_collector import CPUCollector
from ram_collector import RAMCollector
from env_collector import EnvCollector
from gpu_adapter import GPUAdapter
from disk_adapter import DiskAdapter
from yaml_config_collector import YAMLConfigCollector


def get_row(docker_used: bool, trainer_type: str, env_name: str, yaml_path: str):
    main = {
        "run_id": uuid.uuid4(),
        "host_os_name": sys.platform,
        "docker_used": docker_used,
        "trainer_type": trainer_type,
        "time_to_threshold_class_label": -999,
    }
    cpu = CPUCollector().collect_data()
    ram = RAMCollector().collect_data()
    gpu = GPUAdapter().collect_data()
    disk = DiskAdapter().collect_data()

    env = EnvCollector(run_id=main["run_id"], env_name=env_name).collect_data()

    yaml_config = YAMLConfigCollector().collect_data(
        yaml_path=yaml_path, env_name=env_name
    )

    return main | cpu | ram | gpu | disk | env | yaml_config


def is_valid_trainer(trainer: str):
    return trainer in ["imitation", "poca", "ppo", "sac"]


def is_valid_name(trainer: str, env_name: str):
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(parent_path, "config", trainer, f"{env_name}.yaml")
    return yaml_path if os.path.exists(yaml_path) else ""


def string_to_bool(string: str):
    return string == "true"


def print_usage_and_exit():
    print(
        "\nUsage:\n"
        "  !!!WORKING DIRECTORY IS ROOT DIRECTORY!!!\n"
        "  python data_collection/main.py <docker_used> <trainer_type> <env_name>\n\n"
        "Arguments:\n"
        "  docker_used    true | false\n"
        "  trainer_type   imitation | ppo | sac | poca\n"
        "  env_name       Name of environment YAML (e.g., Crawler, Hallway, 3DBall)\n",
        file=sys.stderr,
    )
    sys.exit(1)


def launch_mlagents_training(yaml_path: str, run_id: str):
    arg = [
        sys.executable,
        "-m",
        "mlagents.trainers.learn",
        yaml_path,
        f"--run-id={run_id}",
    ]

    try:
        subprocess.run(arg)
    except KeyboardInterrupt:
        print("Training results saved.\n" "Returning to caller.")
        subprocess.Popen(["tensorboard", "--logdir=results"])
        return

    subprocess.Popen(["tensorboard", "--logdir=results"])


def main():
    args = sys.argv
    if len(args) < 4:
        print_usage_and_exit()

    docker_used = string_to_bool(args[1].lower())
    trainer_type = args[2].lower()
    env_name = args[3]

    yaml_path = is_valid_name(trainer_type, env_name)
    if not is_valid_trainer(trainer_type) or not yaml_path:
        print_usage_and_exit()

    row = get_row(
        docker_used=docker_used,
        trainer_type=trainer_type,
        env_name=env_name,
        yaml_path=yaml_path,
    )
    CSVWriter.append_row(csv_file_path="human_readable.csv", row=row)

    launch_mlagents_training(yaml_path=yaml_path, run_id=row["run_id"])

    try:
        with open("policy_trainer.csv", "r") as f:
            reader = csv.DictReader(f)
            threshold_row = next(reader)

        dataframe = pandas.read_csv("human_readable.csv")
        dataframe["time_to_threshold_class_label"] = dataframe[
            "time_to_threshold_class_label"
        ].astype(float)
        dataframe.loc[
            dataframe["run_id"] == str(row["run_id"]), "time_to_threshold_class_label"
        ] = float(threshold_row["time_to_threshold_class_label"])
        dataframe.to_csv("human_readable.csv", index=False)

        print("Updated human_readable.csv with threshold time.")
    except Exception as e:
        print("Could not update human_readable.csv: ", e)


if __name__ == "__main__":
    main()
