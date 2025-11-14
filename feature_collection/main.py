import uuid
import sys
from csv_writer import CSVWriter
from cpu_collector import CPUCollector
from ram_collector import RAMCollector


def get_row():
    main = {
        "run_id": uuid.uuid4(),
        "host_os_name": sys.platform,
    }
    cpu = CPUCollector().collect_features()
    ram = RAMCollector().collect_features()

    return main | cpu | ram


def main():
    row = get_row()
    CSVWriter.append_row(csv_file_path="test.csv", row=row)


if __name__ == "__main__":
    main()
