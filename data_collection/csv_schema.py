from typing import List

# column definitions by category
CSV_COLUMNS: List[str] = [
    # Identifiers
    "run_id",

    # CPU Features
    "cpu_vendor",
    "cpu_cores",
    "cpu_threads",
    "cpu_frequency_khz",
    "cpu_architecture",

    # RAM Features
    "ram_total_bytes",
    "ram_available_bytes",
    "ram_type",

    # GPU Features
    "gpu_count",
    "gpu_0_name",
    "gpu_0_vendor",
    "gpu_0_vram_gb",
    "gpu_1_name",
    "gpu_1_vendor",
    "gpu_1_vram_gb",

    # Disk Features
    "disk_total_gb",
    "disk_free_gb",
    "disk_filesystem",

    # System
    "host_os_name",
    "is_docker_used",

    # Core Hyperparameters
    "trainer_type",
    "batch_size",
    "buffer_size",
    "learning_rate",
    "beta",
    "epsilon",
    "lambd",
    "num_epoch",
    "shared_critic",

    # Schedules
    "learning_rate_schedule",
    "beta_schedule",
    "epsilon_schedule",

    # Checkpointing
    "checkpoint_interval",
    "keep_checkpoints",
    "even_checkpoints",

    # Network Architecture
    "normalize",
    "hidden_units",
    "num_layers",
    "vis_encode_type",

    # Advanced
    "memory",
    "goal_conditioning_type",
    "deterministic",
    "extrinsic_gamma",
    "extrinsic_strength",
    "init_path",
    "threaded",

    # Training Config
    "max_steps",
    "time_horizon",
    "summary_freq",
    "self_play",
    "behavioral_cloning",

    # Training Results
    "time_start",
    "time_end",
    "total_steps",
    "total_duration",
]


def get_csv_headers() -> List[str]:
    return CSV_COLUMNS.copy()


def get_empty_row() -> dict:
    """return a dict with all columns set to None."""
    return {col: None for col in CSV_COLUMNS}
