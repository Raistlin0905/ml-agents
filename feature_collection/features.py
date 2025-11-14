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
        "gpu_feature_one",
        "gpu_feature_two",
    ]

    DISK = [
        "disk_feature_one",
        "disk_feature_two",
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
        return cls.MAIN + cls.CPU + cls.RAM + cls.ENV  # + self.GPU + self.DISK
