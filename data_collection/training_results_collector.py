"""
Parse timers.json and training_status.json from the results directory
to extract training timing and step information
"""

import json
import os


class TrainingResultsCollector:
    """Collects training results from TensorBoard log files."""

    def __init__(self, results_base_dir: str = "results"):
        self.results_base_dir = results_base_dir

    def collect_data(self, run_id: str) -> dict:
        """
        Collect training results for a specific run,
        returns Dictionary with time_start, time_end, total_steps, total_duration
        """
        run_logs_dir = os.path.join(self.results_base_dir, run_id, "run_logs")

        timers_data = self._parse_timers(run_logs_dir)
        status_data = self._parse_training_status(run_logs_dir)

        time_start = timers_data.get("time_start")
        time_end = timers_data.get("time_end")

        total_duration = None
        if time_start is not None and time_end is not None:
            total_duration = time_end - time_start

        total_steps = status_data.get("total_steps") or timers_data.get("total_steps")

        return {
            "time_start": time_start,
            "time_end": time_end,
            "total_steps": total_steps,
            "total_duration": total_duration,
        }

    def _parse_timers(self, run_logs_dir: str) -> dict:
        """Parse timers.json for timing information"""
        timers_path = os.path.join(run_logs_dir, "timers.json")
        result = {
            "time_start": None,
            "time_end": None,
            "total_steps": None,
        }

        if not os.path.exists(timers_path):
            return result

        try:
            with open(timers_path, "r") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            start_str = metadata.get("start_time_seconds")
            end_str = metadata.get("end_time_seconds")

            if start_str:
                result["time_start"] = int(start_str)
            if end_str:
                result["time_end"] = float(end_str)

            gauges = data.get("gauges", {})
            for key, value in gauges.items():
                if ".Step.sum" in key or ".Step.mean" in key:
                    step_value = value.get("value")
                    if step_value is not None:
                        result["total_steps"] = int(step_value)
                        break

        except (json.JSONDecodeError, IOError, KeyError):
            pass

        return result

    def _parse_training_status(self, run_logs_dir: str) -> dict:
        """Parse training_status.json for final step count"""
        status_path = os.path.join(run_logs_dir, "training_status.json")
        result = {"total_steps": None}

        if not os.path.exists(status_path):
            return result

        try:
            with open(status_path, "r") as f:
                data = json.load(f)

            for key, value in data.items():
                if key == "metadata":
                    continue
                if isinstance(value, dict):
                    final_checkpoint = value.get("final_checkpoint", {})
                    steps = final_checkpoint.get("steps")
                    if steps is not None:
                        result["total_steps"] = int(steps)
                        break

        except (json.JSONDecodeError, IOError, KeyError):
            pass

        return result
