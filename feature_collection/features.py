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

    @classmethod
    def get_csv_headers(cls) -> list[str]:
        return cls.MAIN + cls.CPU + cls.RAM  # + self.GPU + self.DISK
