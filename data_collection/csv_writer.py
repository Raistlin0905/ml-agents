import csv
import os
from attributes import Attributes


class CSVWriter:

    def append_row(csv_file_path: str, row: dict):
        header_written = os.path.exists(csv_file_path)

        csv_headers = Attributes().get_csv_headers()

        with open(file=csv_file_path, mode="a", newline="") as file:
            writer = csv.DictWriter(f=file, fieldnames=csv_headers)
            if not header_written:
                writer.writeheader()

            writer.writerow(row)
