import uuid
import sys
from csv_writer import CSVWriter
from cpu_collector import CPUCollector
from ram_collector import RAMCollector
from env_collector import EnvCollector


def get_row():
    main = {
        "run_id": uuid.uuid4(),
        "host_os_name": sys.platform,
    }
    cpu = CPUCollector().collect_features()
    ram = RAMCollector().collect_features()

    env = EnvCollector(
        run_id=main.get("run_id"), env_name="placeholder_env_name"
    ).collect_features()

    return main | cpu | ram | env


def main():
    row = get_row()
    CSVWriter.append_row(
        csv_file_path="human_readable.csv", row=row, human_readable=True
    )


if __name__ == "__main__":
    main()
