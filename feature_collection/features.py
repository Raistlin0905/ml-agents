class Features:

    MAIN = [
        "run_id",
        "host_os_name",
    ]

    CPU = [
        "host_cpu_cores",
    ]

    RAM = [
        "host_ram_total_bytes",
        "host_ram_available_bytes",
        "host_ram_used_bytes",
        "host_ram_type",
        "host_swap_total_bytes",
        "host_swap_used_bytes",
    ]

    GPU = [
        "gpu_count",
        "gpus",
        "gpus0_index",
        "gpus0_name",
        "gpus0_vendor",
        "gpus0_vram_total_gb",
        "gpus0_vram_used_gb",
        "gpus0_vram_free_gb",
        "gpus0_vram_free_pct",
        "gpus0_utilization_pct",
        "gpus0_memory_util_pct",
        "gpus0_temperature_C",
    ]

    DISK = [
        "mount",
        "disk_total_gb",
        "disk_used_gb",
        "disk_free_gb",
        "disk_percent_used",
        "filesystem",
        "read_bytes_total",
        "write_bytes_total",
        "read_count_total",
        "write_count_total",
    ]

    ENV = [
        "behavior_name",
        "actions_continuous_actions",
        "actions_discrete_size",
        "actions_discrete_branches",
        "model",
        "inference_device",
        "deterministic_inference",
        "behavior_type",
        "team_id",
        "use_child_actuators",
        "use_child_sensors",
        "observational_attribute_handling",
        "max_step",
        "decision_period",
        "decision_step",
        "take_actions_between_decisions",
    ]

    @classmethod
    def get_csv_headers(cls) -> list[str]:
        return cls.MAIN + cls.CPU + cls.RAM + cls.GPU + cls.DISK + cls.ENV
