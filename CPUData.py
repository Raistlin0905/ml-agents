from feature_collection import Model_Feature_Collector, Data_Object
import platform
import subprocess
import re
import json
from typing import Optional

POSSIBLE_ARCHITECTURES = [0, 1, 2]  # mapping: 0->x86, 1->arm, 2->other
POSSIBLE_VENDORS = [0, 1, 2, 3]     # mapping: 0->Intel, 1->Apple, 2->AMD, 3->other

class CPUDataObject(Data_Object):
    def __init__(self):
        self.architecture = POSSIBLE_ARCHITECTURES[0]  # default x86
        self.physical_core_count = -1  # default: non-existent
        self.thread_count = -1         # default: non-existent [TOTAL thread count over all cpu cores]
        self.freq = -1                 # unit: kHz, sometimes can not exist
        self.vendor = POSSIBLE_VENDORS[0]  # default: Intel

class CPUFeatureExtractor(Model_Feature_Collector):
    # helpers
    def run_command(self, c: str) -> str:
        try:
            return subprocess.check_output(c, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""

    def sysctl_int(self, key: str) -> Optional[int]:
        val = self.run_command(f"sysctl -n {key}")
        if val.isdigit():
            return int(val)
        else:
            try:
                return int(float(val.split()[0]))
            except Exception:
                return None

    def map_to_arch_enum(self, machine: str) -> int:
        m = machine.lower()
        if m in ("x86_64", "amd64", "i386", "i686"):
            return 0
        elif m in ("arm64", "aarch64"):
            return 1
        else:
            return 2

    def map_to_vendor_enum(self, vendor: str, arch_enum: int) -> int:
        v = (vendor or "").lower()
        if "intel" in v or "genuineintel" in v:
            return 0
        elif "amd" in v or "advanced micro" in v or "authenticamd" in v:
            return 2
        elif "apple" in v:
            return 1
        else:
            return 3

    def collectFromMac(self) -> CPUDataObject:
        obj = CPUDataObject()

        machine = platform.machine()
        arch_enum = self.map_to_arch_enum(machine=machine)
        obj.architecture = arch_enum

        vendor = self.run_command("sysctl -n machdep.cpu.vendor")
        vendor_enum = self.map_to_vendor_enum(vendor=vendor, arch_enum=obj.architecture)
        # infer Apple on macOS ARM when vendor string is empty
        if vendor_enum == 3 and obj.architecture == 1 and not (vendor or "").strip():
            vendor_enum = 1
        obj.vendor = vendor_enum

        core_count = self.sysctl_int("hw.physicalcpu")
        thread_count = self.sysctl_int("hw.logicalcpu")

        if core_count and core_count > 0:
            obj.physical_core_count = core_count
        if thread_count and thread_count > 0:
            obj.thread_count = thread_count

        freq = self.sysctl_int("hw.cpufrequency_max")
        if freq and freq > 0:
            freq = freq // 1000  # to kHz (int)
        else:
            freq = -1
        obj.freq = freq

        return obj

    def collectFromLinux(self):
        obj = CPUDataObject()
        obj.architecture = self.map_to_arch_enum(platform.machine())

        lscpu = self.run_command("LC_ALL=C lscpu")

        def kv(key: str) -> str:
            m = re.search(rf"^{re.escape(key)}:\s*(.+)$", lscpu, re.M)
            return m.group(1).strip() if m else ""

        vendor = kv("Vendor ID")
        if not vendor:
            vendor = self.run_command("grep -m1 -E 'vendor_id|Hardware' /proc/cpuinfo | awk -F: '{print $2}'").strip()
        obj.vendor = self.map_to_vendor_enum(vendor, obj.architecture)

        cps = kv("Core(s) per socket")
        socks = kv("Socket(s)")
        phys = None
        try:
            if cps and socks:
                phys = int(float(cps)) * int(float(socks))
        except Exception:
            phys = None
        if phys is None or phys <= 0:
            core_ids = self.run_command("awk -F: '/^core id|^cpu cores/ {print $2}' /proc/cpuinfo | sort -u | wc -l")
            if core_ids.isdigit() and int(core_ids) > 0:
                phys = int(core_ids)
        if phys is None or phys <= 0:
            tpc = kv("Thread(s) per core")
            logical = kv("CPU(s)") or self.run_command("nproc")
            if tpc and logical and tpc.replace('.', '', 1).isdigit() and logical.isdigit():
                tpcv = float(tpc)
                if tpcv > 0:
                    phys = int(int(logical) / tpcv)
        if phys is None or phys <= 0:
            phys = 1
        obj.physical_core_count = phys

        logical = kv("CPU(s)")
        if not logical.isdigit():
            logical = self.run_command("nproc")
        obj.thread_count = int(logical) if logical.isdigit() else max(phys, 1)

        maxmhz = kv("CPU max MHz")
        if maxmhz:
            try:
                obj.freq = int(round(float(maxmhz) * 1000))
            except Exception:
                obj.freq = -1
        else:
            khz = self.run_command("cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null")
            if khz.isdigit():
                obj.freq = int(khz)
            else:
                khz2 = self.run_command("grep -hoE 'cpu MHz\\s*:\\s*[0-9.]+' /proc/cpuinfo | awk '{print $4}' | sort -nr | head -1")
                if khz2:
                    try:
                        obj.freq = int(round(float(khz2) * 1000))
                    except Exception:
                        obj.freq = -1
                else:
                    obj.freq = -1
        return obj

    def collectFromWindows(self):
        obj = CPUDataObject()
        obj.architecture = self.map_to_arch_enum(platform.machine())

        ps = r"""$cpu=Get-CimInstance Win32_Processor | Select-Object -First 1 *;
$vendor=$cpu.Manufacturer
$phys=[int]$cpu.NumberOfCores
$logi=[int]$cpu.NumberOfLogicalProcessors
$max=[int]$cpu.MaxClockSpeed
@{vendor=$vendor;phys=$phys;logi=$logi;max=$max} | ConvertTo-Json -Compress"""
        raw = self.run_command(f'powershell -NoProfile -Command "{ps}"')

        vendor, phys, logi, max_mhz = "", None, None, None
        if raw:
            try:
                data = json.loads(raw)
                vendor = data.get("vendor") or ""
                phys = data.get("phys")
                logi = data.get("logi")
                max_mhz = data.get("max")
            except Exception:
                pass
        else:
            # Fallback to WMIC (deprecated but often present)
            vendor  = self.run_command('wmic cpu get Manufacturer /value').split('=')[-1].strip()
            phys_s  = self.run_command('wmic cpu get NumberOfCores /value').split('=')[-1].strip()
            logi_s  = self.run_command('wmic cpu get NumberOfLogicalProcessors /value').split('=')[-1].strip()
            max_s   = self.run_command('wmic cpu get MaxClockSpeed /value').split('=')[-1].strip()
            phys    = int(phys_s) if phys_s.isdigit() else None
            logi    = int(logi_s) if logi_s.isdigit() else None
            max_mhz = int(max_s)  if max_s.isdigit()  else None

        obj.vendor = self.map_to_vendor_enum(vendor, obj.architecture)
        obj.physical_core_count = int(phys) if isinstance(phys, int) and phys > 0 else 1
        obj.thread_count = int(logi) if isinstance(logi, int) and logi > 0 else obj.physical_core_count
        obj.freq = (int(max_mhz) * 1000) if (isinstance(max_mhz, int) and max_mhz > 0) else -1

        return obj
