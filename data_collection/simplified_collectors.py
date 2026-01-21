import sys

from cpu_collector import CPUCollector
from ram_collector import RAMCollector
from gpu_collector import GPUCollector
from disk_collector import DiskCollector


# from cpu_collector.py
VENDOR_MAP = {0: "Intel", 1: "Apple", 2: "AMD", 3: "Other"}
ARCH_MAP = {0: "x86", 1: "ARM", 2: "Other"}


def collect_cpu_data() -> dict:
    """Collect simplified CPU data."""
    collector = CPUCollector()
    raw = collector.collect_data()

    vendor_enum = raw.get("host_cpu_vendor", 3)
    arch_enum = raw.get("host_cpu_architecture", 2)

    return {
        "cpu_vendor": VENDOR_MAP.get(vendor_enum, "Other"),
        "cpu_cores": raw.get("host_cpu_physical_core_count"),
        "cpu_threads": raw.get("host_cpu_thread_count"),
        "cpu_frequency_khz": raw.get("host_cpu_clock_frequency"),
        "cpu_architecture": ARCH_MAP.get(arch_enum, "Other"),
    }


def collect_ram_data() -> dict:
    """Collect simplified RAM data."""
    collector = RAMCollector()
    raw = collector.collect_data()

    return {
        "ram_total_bytes": raw.get("host_ram_total_bytes"),
        "ram_available_bytes": raw.get("host_ram_available_bytes"),
        "ram_type": raw.get("host_ram_type", "Unknown"),
    }


def collect_gpu_data() -> dict:
    """Collect simplified GPU data."""
    collector = GPUCollector()
    raw = collector.to_dict()

    result = {
        "gpu_count": raw.get("gpu_count", 0),
        "gpu_0_name": None,
        "gpu_0_vendor": None,
        "gpu_0_vram_gb": None,
        "gpu_1_name": None,
        "gpu_1_vendor": None,
        "gpu_1_vram_gb": None,
    }

    gpus = raw.get("gpus", [])
    if len(gpus) > 0:
        result["gpu_0_name"] = gpus[0].get("name")
        result["gpu_0_vendor"] = gpus[0].get("vendor")
        result["gpu_0_vram_gb"] = gpus[0].get("vram_total_gb")

    if len(gpus) > 1:
        result["gpu_1_name"] = gpus[1].get("name")
        result["gpu_1_vendor"] = gpus[1].get("vendor")
        result["gpu_1_vram_gb"] = gpus[1].get("vram_total_gb")

    return result


def collect_disk_data() -> dict:
    """Collect simplified disk data."""
    collector = DiskCollector()
    raw = collector.to_dict()

    return {
        "disk_total_gb": raw.get("disk_total_gb"),
        "disk_free_gb": raw.get("disk_free_gb"),
        "disk_filesystem": raw.get("filesystem"),
    }


def collect_system_data(is_docker_used: bool) -> dict:
    """Collect system info."""
    return {
        "host_os_name": sys.platform,
        "is_docker_used": is_docker_used,
    }


def collect_all_hardware() -> dict:
    result = {}
    result.update(collect_cpu_data())
    result.update(collect_ram_data())
    result.update(collect_gpu_data())
    result.update(collect_disk_data())
    return result
