from gpu_collector import GPUCollector
from base_collector import DataCollector
from typing import Dict


class GPUAdapter(DataCollector):

    def flatten_dict(self, old_dict: Dict[any, any]) -> dict:
        flat_dict = {}
        stack = [(old_dict, "")]

        while stack:
            current_dict, prefix = stack.pop()

            for key, val in current_dict.items():
                new_key = f"{prefix}_{key}" if prefix else key

                if isinstance(val, list):
                    for index, item in enumerate(val):
                        if isinstance(item, dict):
                            stack.append((item, f"{new_key}{index}"))
                elif isinstance(val, dict):
                    stack.append((val, new_key))
                else:
                    flat_dict[new_key] = val

        return flat_dict

    def collect_from_windows(self) -> dict:
        return self.flatten_dict(GPUCollector().to_dict())

    def collect_from_mac(self) -> dict:
        return self.flatten_dict(GPUCollector().to_dict())

    def collect_from_linux(self) -> dict:
        return self.flatten_dict(GPUCollector().to_dict())
