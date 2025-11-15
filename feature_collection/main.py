import uuid
import sys
from csv_writer import CSVWriter
from cpu_collector import CPUCollector
from ram_collector import RAMCollector
from env_collector import EnvCollector
from gpu_adapter import GPUAdapter
from disk_adapter import DiskAdapter


def get_row():
    main = {
        "run_id": uuid.uuid4(),
        "host_os_name": sys.platform,
    }
    cpu = CPUCollector().collect_features()
    ram = RAMCollector().collect_features()
    gpu = GPUAdapter().collect_features()
    disk = DiskAdapter().collect_features()

    print(gpu)

    env = EnvCollector(
        run_id=main.get("run_id"), env_name="placeholder_env_name"
    ).collect_features()

    return main | cpu | ram | gpu | disk | env


def main():
    row = get_row()
    CSVWriter.append_row(csv_file_path="human_readable.csv", row=row)


if __name__ == "__main__":
    main()
