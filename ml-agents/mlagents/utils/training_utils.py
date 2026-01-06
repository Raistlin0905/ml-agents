import csv
import time
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


def treat_threshold(duration):
    row = {"time_to_threshold_class_label": duration}
    with open("policy_trainer.csv", mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    # Force flush TensorBoard writers so stats appear immediately
    raise KeyboardInterrupt("Training stopped. Threshold met")


"""
def check_threshold(mean_reward, class_label, _start_time, _stats_reporter):
    threshold = 100.0
    # Check if performance threshold is met (mean_reward = 100 and std_reward = 0)
    # retired test for now: print(f"{mean_reward} and {elapsed_time}")
    if mean_reward >= threshold and class_label is None:
        # Calculate elapsed time using the already tracked start time
        elapsed_time = time.time() - _start_time  # Using the existing start time
        class_label = elapsed_time

        row = {"time_to_threshold_class_label": class_label}
        with open("policy_trainer.csv", mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        # Force flush TensorBoard writers so stats appear immediately
        for writer in _stats_reporter.writers:
            if hasattr(writer, "writer") and writer.writer is not None:
                writer.writer.flush()
        logger.info("Threshold reached, elapsed time saved: " + str(class_label))
        logger.info("Stopping training as threshold reached.")
        raise KeyboardInterrupt("Training stopped. Threshold met")
"""
