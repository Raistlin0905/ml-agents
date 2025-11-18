import json
import psutil
import subprocess
from base_collector import DataCollector


class RAMCollector(DataCollector):
    @staticmethod
    def get_attributes():
        v_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()

        return {
            "host_ram_total_bytes": v_mem.total,
            "host_ram_available_bytes": v_mem.available,
            "host_ram_used_bytes": v_mem.used,
            "host_ram_type": "Unknown",
            "host_swap_total_bytes": swap_mem.total,
            "host_swap_used_bytes": swap_mem.used,
        }

    def collect_from_windows(self):
        symbios_mem_types = {
            1: "Other",
            2: "Unknown",
            3: "DRAM",
            4: "EDRAM",
            5: "VRAM",
            6: "SRAM",
            7: "RAM",
            8: "ROM",
            9: "FLASH",
            10: "EEPROM",
            11: "FEPROM",
            12: "EPROM",
            13: "CDRAM",
            14: "3DRAM",
            15: "SDRAM",
            16: "SGRAM",
            17: "RDRAM",
            18: "DDR",
            19: "DDR2",
            20: "DDR2 FB-DIMM",
            21: "RESERVED",
            22: "RESERVED",
            23: "RESERVED",
            24: "DDR3",
            25: "FBD2",
            26: "DDR4",
            27: "LPDDR",
            28: "LPDDR2",
            29: "LPDDR3",
            30: "LPDDR4",
            31: "LOGICAL NON-VOLATILE DEVICE",
            32: "HBM",
            33: "HBM2",
            34: "DDR5",
            35: "LPDDR5",
            36: "HBM3",
        }
        attributes = self.get_attributes()

        output = subprocess.run(
            args=[
                "powershell",
                "-Command",
                "Get-CimInstance -ClassName Win32_PhysicalMemory | Select-Object -ExpandProperty SMBIOSMemoryType | ConvertTo-Json",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        data = json.loads(output.stdout)
        if isinstance(data, int):
            attributes["host_ram_type"] = symbios_mem_types[data]
        elif isinstance(data, list):
            mem_type_nums = [num for num in data if isinstance(num, int)]
            for num in mem_type_nums:
                if symbios_mem_types.get(num, "Unknown") != "Unknown":
                    attributes["host_ram_type"] = symbios_mem_types.get(num)
                    break

        return attributes

    def collect_from_mac(self):
        attributes = self.get_attributes()

        output = subprocess.run(
            args=["system_profiler", "SPMemoryDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        data = json.loads(output.stdout)
        ram_type = data.get("SPMemoryDataType", [{}])[0].get("dimm_type")
        attributes["host_ram_type"] = ram_type

        return attributes

    def collect_from_linux(self):
        attributes = self.get_attributes()

        return attributes
