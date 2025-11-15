from disk_collector import DiskCollector
from gpu_collector import GPUCollector


def print_hardware_summary() -> None:
    try:
        disk = DiskInfoCollector().to_dict()
    except Exception as e:
        disk = None
        print("\nStorage Information: unavailable:", str(e))

    try:
        gpu = GPUInfoCollector().to_dict()
    except Exception as e:
        gpu = {"gpu_count": 0, "gpus": []}
        print("\nGPU Information: unavailable:", str(e))

    print("\nHardware features detected before training start:")

    # Disk
    if disk:
        print("\nStorage Information:")
        print(f"Mount: {disk.get('mount')}")
        print(f"Filesystem: {disk.get('filesystem') or 'Unknown'}")
        print("Capacity:")
        print(f"Total: {disk.get('disk_total_gb', 0):.1f} GB")
        print(
            f"Used: {disk.get('disk_used_gb', 0):.1f} GB "
            f"({disk.get('disk_percent_used', 0)}%)"
        )
        print(f"Free: {disk.get('disk_free_gb', 0):.1f} GB")
    else:
        print("\nStorage Information: unavailable")

    # GPU
    print("\nGPU Information:")
    gpu_count = (gpu or {}).get("gpu_count", 0)
    print(f"GPU count: {gpu_count}")
    if gpu_count == 0:
        print("(No GPU detected or no supported backend found)")
    else:
        drv = gpu.get("nvidia_driver_version") or gpu.get("cuda_version")
        if drv:
            print(f"Driver/CUDA: {drv}")
        for g in gpu.get("gpus", []):
            print(f"- GPU {g.get('index')}: {g.get('name')}")
            if g.get("vendor"):
                print(f"Vendor: {g.get('vendor')}")
            if g.get("vram_total_gb") is not None:
                print(
                    f"VRAM: {g.get('vram_total_gb')} GB "
                    f"(used {g.get('vram_used_gb')})"
                )
            if g.get("utilization_pct") is not None:
                print(f"Util: {g.get('utilization_pct')}%")
            if g.get("memory_util_pct") is not None:
                print(f"Mem %: {g.get('memory_util_pct')}%")
            if g.get("temperature_C") is not None:
                print(f"Temp: {g.get('temperature_C')}°C")
