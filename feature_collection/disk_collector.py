from __future__ import annotations
from typing import Any, Dict, Optional

try:
    import psutil  # disk details + IO counters
except Exception:
    psutil = None  # still works with shutil fallback


class DiskCollector:
    """
    Disk info:
      - total/used/free (GB)
      - percent used
      - filesystem type for the target mount (if psutil available)
      - cumulative IO counters since boot (if psutil available)
    """

    def __init__(self, mount: Optional[str] = None) -> None:
        import os

        # default to OS root ("/" on Unix, drive root on Windows)
        self.root = mount or os.path.abspath(os.sep)

    def to_dict(self) -> Dict[str, Any]:
        import shutil, os

        info: Dict[str, Any] = {"mount": self.root}

        # Core space stats (always available)
        usage = shutil.disk_usage(self.root)
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        free_gb = usage.free / (1024**3)
        info.update(
            {
                "disk_total_gb": round(total_gb, 3),
                "disk_used_gb": round(used_gb, 3),
                "disk_free_gb": round(free_gb, 3),
                "disk_percent_used": (
                    round((used_gb / total_gb) * 100, 2) if total_gb > 0 else None
                ),
            }
        )

        # filesystem type for this mount (best-effort if psutil is present)
        fs_type = None
        if psutil is not None:
            try:
                for p in psutil.disk_partitions(all=False):
                    if os.path.abspath(p.mountpoint) == os.path.abspath(self.root):
                        fs_type = p.fstype
                        break
            except Exception:
                pass
        info["filesystem"] = fs_type

        # Cumulative IO counters since boot (optional)
        if psutil is not None:
            try:
                io = psutil.disk_io_counters()
                if io:
                    info.update(
                        {
                            "read_bytes_total": int(io.read_bytes),
                            "write_bytes_total": int(io.write_bytes),
                            "read_count_total": int(io.read_count),
                            "write_count_total": int(io.write_count),
                        }
                    )
            except Exception:
                pass

        return info
