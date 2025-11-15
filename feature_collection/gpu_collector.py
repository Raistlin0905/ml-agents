"""
figures out what gpus you have and how they're doing

this class tries different ways to detect gpus, starting with the best methods
and falling back to simpler ones if needed. works with:
- nvidia gpus (using nvml api or nvidia-smi)
- amd gpus (using rocm-smi on linux)
- intel gpus (using intel_gpu_top on linux)
- any gpu on windows (using wmi or dxdiag)

it tries to tell you things like:
- how many gpus you have
- what kind they are
- how much memory they have and are using
- how hard they're working
- how hot they are (not always available though))
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import os
import shutil
import subprocess

try:
    import pynvml  # NVIDIA NVML (pip install nvidia-ml-py3)
except Exception:
    pynvml = None

try:
    import GPUtil  # pip install gputil
except Exception:
    GPUtil = None

try:
    import torch  # pip install torch
except Exception:
    torch = None

try:
    import wmi  # pip install WMI  (Windows only)
except Exception:
    wmi = None


class GPUInfoCollector:
    def __init__(self) -> None:
        """initialize the collector and try to enable nvml if available

        this sets an internal flag so nvml calls are only attempted when
        initialization succeeded
        """
        # see if we can use nvidia's fancy api
        self._nvml_ok = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_ok = True
            except Exception:
                self._nvml_ok = False

    # NVIDIA
    def _from_nvml(self) -> Optional[Dict[str, Any]]:
        """get detailed info from nvidia gpus via nvml
        returns a dict with per-gpu stats or None if nvml is unavailable
        """
        # get all the details from nvidia gpus using their official api
        if not self._nvml_ok:
            return None
        try:
            out: Dict[str, Any] = {}
            try:
                out["nvidia_driver_version"] = pynvml.nvmlSystemGetDriverVersion().decode()
            except Exception:
                out["nvidia_driver_version"] = None

            count = pynvml.nvmlDeviceGetCount()
            gpus: List[Dict[str, Any]] = []
            for i in range(count):
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(h).decode()
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    util = None
                            # see if we can use nvidia's fancy api
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    except Exception:
                        pass
                    temp = None
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        pass
                            # get all the details from nvidia gpus using their official api
                    gpus.append({
                        "index": i,
                        "name": name,
                        "vendor": "NVIDIA",
                            # backup plan - use nvidia-smi command line tool if the api didn't work
                        "vram_total_gb": round(mem.total / (1024**3), 2),
                        "vram_used_gb": round(mem.used  / (1024**3), 2),
                        "utilization_pct": (util.gpu if util else None),
                        "memory_util_pct": (util.memory if util else None),
                            # check intel gpus on linux using their monitoring tool
                        "temperature_C": temp,
                    })
                except Exception:
                    continue
                            # handle amd gpus on linux - tries json first, then falls back to text
            return {"gpu_count": len(gpus), "gpus": gpus}
        except Exception:
            return None


    #  NVIDIA CLI fallback 
    def _from_nvidia_smi(self) -> Optional[Dict[str, Any]]:
        # fallback to parsing nvidia-smi output for gpu info when nvml isn't available
        
        # backup plan - use nvidia-smi command line tool if the api didn't work
        if not shutil.which("nvidia-smi"):
            return None
                            # ask pytorch what it knows about the gpu - usually just basic info
        try:
            q = "name,memory.total,memory.used,utilization.gpu,temperature.gpu"
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                            # windows-specific way to get gpu info (no usage stats though)
                stderr=subprocess.DEVNULL, text=True, timeout=2.0
            )
            gpus: List[Dict[str, Any]] = []
            for idx, line in enumerate(l for l in out.splitlines() if l.strip()):
                            # last resort on windows - parse dxdiag output for basic gpu info
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5: continue
                name, mem_total_mb, mem_used_mb, util_pct, temp_c = parts[:5]
                gpus.append({
                            # add some useful calculated fields like free memory percentage
                    "index": idx,
                    "name": name,
                    "vendor": "NVIDIA",
                    "vram_total_gb": round(float(mem_total_mb)/1024.0, 2),
                    "vram_used_gb": round(float(mem_used_mb)/1024.0,  2),
                            # try all our detection methods in order, from best to basic
                    "utilization_pct": int(util_pct),
                    "temperature_C": int(temp_c),
                })
            return {"gpu_count": len(gpus), "gpus": gpus}
        except Exception:
            return None

    # Intel (Linux) 
    def _from_intel_gpu_top(self) -> Optional[Dict[str, Any]]:
        """collect intel gpu stats on linux via intel_gpu_top
        returns basic usage and memory info when the tool is present
        """
        # check intel gpus on linux using their monitoring tool
        if os.name != "posix" or not shutil.which("intel_gpu_top"):
            return None
        try:
            out = subprocess.check_output(["intel_gpu_top", "-J"], stderr=subprocess.DEVNULL, text=True, timeout=2.0)
            import json
            data = json.loads(out)
            eng = data.get("engines", {})
            util = None
            if isinstance(eng, dict):
                for v in eng.values():
                    if isinstance(v, dict) and "busy" in v:
                        util = v["busy"]; break
            mem = data.get("memory", {})
            used_gb  = round(float(mem.get("used", 0))/1024.0, 2) if mem else None
            total_gb = round(float(mem.get("total",0))/1024.0, 2) if mem else None
            return {"gpu_count": 1, "gpus": [{
                "index": 0, "name": "Intel Graphics", "vendor": "Intel",
                "vram_total_gb": total_gb, "vram_used_gb": used_gb,
                "utilization_pct": util, "temperature_C": None
            }]}
        except Exception:
            return None

    # AMD (Linux ROCm)
    def _from_rocm_smi(self) -> Optional[Dict[str, Any]]:
        """collect amd gpu info on linux using rocm-smi
        prefers json output but will parse plain text if necessary
        """
        # handle amd gpus on linux - tries json first, then falls back to text
        if os.name != "posix" or not shutil.which("rocm-smi"):
            return None
        # Try JSON first
        try:
            out = subprocess.check_output(["rocm-smi", "--json"], stderr=subprocess.DEVNULL, text=True, timeout=2.0)
            import json
            data = json.loads(out)
            gpus: List[Dict[str, Any]] = []
            cards = list(data.values()) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for idx, card in enumerate(cards):
                if not isinstance(card, dict): continue
                name = card.get("Card series") or card.get("Card SKU") or "AMD GPU"
                util = card.get("GPU use (%)")
                temp = card.get("Temperature (Sensor #1) (C)") or card.get("Temperature (Sensor #1)")
                totB = card.get("VRAM Total (B)"); usedB = card.get("VRAM Used (B)")
                gpus.append({
                    "index": idx, "name": name, "vendor": "AMD",
                    "vram_total_gb": round(float(totB)/(1024**3), 2) if totB else None,
                    "vram_used_gb": round(float(usedB)/(1024**3), 2) if usedB else None,
                    "utilization_pct": float(util) if util is not None else None,
                    "temperature_C": float(temp) if temp is not None else None,
                })
            if gpus: return {"gpu_count": len(gpus), "gpus": gpus}
        except Exception:
            pass
        # Plain text fallback
        try:
            out = subprocess.check_output(["rocm-smi"], stderr=subprocess.DEVNULL, text=True, timeout=2.0)
            gpus: List[Dict[str, Any]] = []; idx = 0; util = None; temp = None
            for line in out.splitlines():
                s = line.strip()
                if s.startswith("GPU"):  # new section
                    if idx > 0:
                        gpus.append({"index": idx-1, "name": "AMD GPU", "vendor": "AMD",
                                     "vram_total_gb": None, "vram_used_gb": None,
                                     "utilization_pct": util, "temperature_C": temp})
                        util = temp = None
                    idx += 1
                if "GPU use (%)" in s:
                    try: util = float(s.split(":")[1].strip().rstrip("%"))
                    except: pass
                if "Temperature" in s and "(C)" in s:
                    try: temp = float(s.split(":")[1].strip().rstrip("cC").strip())
                    except: pass
            if idx > 0:
                gpus.append({"index": idx-1, "name": "AMD GPU", "vendor": "AMD",
                             "vram_total_gb": None, "vram_used_gb": None,
                             "utilization_pct": util, "temperature_C": temp})
            if gpus: return {"gpu_count": len(gpus), "gpus": gpus}
        except Exception:
            return None
        return None

    # generic fallbacks 
    def _from_gputil(self) -> Optional[Dict[str, Any]]:
        """use the GPUtil package to obtain generic gpu statistics
        this gives basic fields
        """
        # try to get basic info using a generic gpu library
        if GPUtil is None:
            return None
        try:
            devices = GPUtil.getGPUs()
        except Exception:
            return None
        gpus: List[Dict[str, Any]] = []
        for d in devices:
            try:
                gpus.append({
                    "index": d.id, "name": d.name, "vendor": None,
                    "vram_total_gb": round(d.memoryTotal/1024.0, 2),
                    "vram_used_gb": round(d.memoryUsed /1024.0, 2),
                    "utilization_pct": int(d.load*100) if d.load is not None else None,
                    "temperature_C": int(d.temperature) if d.temperature is not None else None,
                })
            except Exception:
                continue
        return {"gpu_count": len(gpus), "gpus": gpus}

    def _from_torch(self) -> Optional[Dict[str, Any]]:
        """ask pytorch for basic cuda device info
        lightweight option when torch is installed; doesn't provide runtime stats
        """
        # ask pytorch what it knows about the gpu - usually just basic info
        if torch is None or not hasattr(torch, "cuda"):
            return None
        try:
            if not torch.cuda.is_available():
                return None
            count = torch.cuda.device_count()
            gpus: List[Dict[str, Any]] = []
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                total_gb = None
                try:
                    props = torch.cuda.get_device_properties(i)
                    total_gb = props.total_memory / (1024**3)
                except Exception:
                    pass
                gpus.append({
                    "index": i, "name": name, "vendor": None,
                    "vram_total_gb": round(total_gb, 2) if total_gb else None,
                    "vram_used_gb": None, "utilization_pct": None, "temperature_C": None,
                })
            return {"gpu_count": len(gpus), "gpus": gpus, "cuda_version": getattr(torch.version, "cuda", None)}
        except Exception:
            return None

    # Windows Intel/AMD/NVIDIA (static) 
    def _from_windows_wmi(self) -> Optional[Dict[str, Any]]:
        """use wmi on windows to list installed display adapters
        provides static info such as name and memory but no runtime stats or anything cool like that
        """
        # windows-specific way to get gpu info (no usage stats though)
        if os.name != "nt" or wmi is None:
            return None
        try:
            c = wmi.WMI()
            gpus: List[Dict[str, Any]] = []
            for idx, adapter in enumerate(c.Win32_VideoController()):
                name = adapter.Name
                vendor = adapter.AdapterCompatibility
                vram = None
                try:
                    if getattr(adapter, "AdapterRAM", None):
                        vram = round(float(adapter.AdapterRAM)/(1024**3), 2)
                except Exception:
                    pass
                gpus.append({
                    "index": idx, "name": name, "vendor": vendor,
                    "vram_total_gb": vram, "vram_used_gb": None,
                    "utilization_pct": None, "temperature_C": None,
                })
            return {"gpu_count": len(gpus), "gpus": gpus}
        except Exception:
            return None

    def _from_windows_dxdiag(self) -> Optional[Dict[str, Any]]:
        """run dxdiag and parse its report for gpu details as a fallback
        slower but available on many windows installs without extra python packages
        """
        # last resort on windows - parse dxdiag output for basic gpu info
        if os.name != "nt" or not shutil.which("dxdiag"):
            return None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                path = tmp.name
            try:
                subprocess.check_output(["dxdiag", "/t", path], stderr=subprocess.DEVNULL, timeout=5.0)
                with open(path, "r", encoding="utf-16", errors="ignore") as f:
                    text = f.read()
            finally:
                try: os.remove(path)
                except Exception: pass

            lines = [l.strip() for l in text.splitlines()]
            gpus: List[Dict[str, Any]] = []; cur = {}
            for ln in lines:
                if ln.startswith("Card name:"):
                    if cur: gpus.append(cur); cur = {}
                    cur = {"index": len(gpus), "name": ln.split(":",1)[1].strip(),
                           "vendor": None, "vram_total_gb": None,
                           "vram_used_gb": None, "utilization_pct": None,
                           "temperature_C": None}
                elif ln.startswith("Manufacturer:") and cur and not cur.get("vendor"):
                    cur["vendor"] = ln.split(":",1)[1].strip()
                elif "Dedicated Memory:" in ln and cur:
                    try:
                        mb = float(ln.split(":",1)[1].strip().split(" ")[0])
                        cur["vram_total_gb"] = round(mb/1024.0, 2)
                    except Exception:
                        pass
            if cur: gpus.append(cur)
            if gpus: return {"gpu_count": len(gpus), "gpus": gpus}
        except Exception:
            return None
        return None

    # Public API
    def _postprocess(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """add derived fields such as free vram and top-level aggregates
        mutates the per-gpu dicts to include vram_free_gb and vram_free_pct
        """
        # add some useful calculated fields like free memory percentage
        try:
            if not isinstance(res, dict):
                return res
            gpus = res.get("gpus") if isinstance(res.get("gpus"), list) else []
            for gpu in gpus:
                total = gpu.get("vram_total_gb")
                used = gpu.get("vram_used_gb")
                if total is not None and used is not None:
                    try:
                        free = round(float(total) - float(used), 2)
                    except Exception:
                        free = None
                    gpu["vram_free_gb"] = free
                    try:
                        gpu["vram_free_pct"] = round((free / float(total)) * 100, 2) if free is not None and float(total) != 0 else None
                    except Exception:
                        gpu["vram_free_pct"] = None
                else:
                    gpu.setdefault("vram_free_gb", None)
                    gpu.setdefault("vram_free_pct", None)

            # top-level aggregates
            try:
                totals = [p for p in gpus if p.get("vram_total_gb") is not None]
                if totals:
                    total_sum = sum(p.get("vram_total_gb", 0) for p in totals)
                    used_sum = sum(p.get("vram_used_gb", 0) for p in gpus if p.get("vram_used_gb") is not None)
                    res["vram_total_gb"] = round(total_sum, 2)
                    res["vram_used_gb"] = round(used_sum, 2) if used_sum is not None else None
                    res["vram_free_gb"] = round(total_sum - used_sum, 2) if used_sum is not None else None
            except Exception:
                pass
            return res
        except Exception:
            return res

    def to_dict(self) -> Dict[str, Any]:
        """run detection passers until one returns meaningful gpu data
        returns a normalized dictionary with a gpu_count and a list under 'gpus'
        """
        for getter in (
            self._from_nvml,# NVIDIA rich
            self._from_nvidia_smi,# NVIDIA CLI
            self._from_rocm_smi, # AMD Linux
            self._from_intel_gpu_top, # Intel Linux
            self._from_gputil, # generic NVIDIA
            self._from_torch, # minimal CUDA info
            self._from_windows_wmi, # Windows (Intel/AMD/NVIDIA) static
            self._from_windows_dxdiag, # Windows fallback
        ):
            try:
                res = getter()
                if not res:
                    continue
                # prefer explicit positive GPU counts
                if int(res.get("gpu_count", 0)) > 0:
                    return self._postprocess(res)
                # or a non-empty 'gpus' array
                g = res.get("gpus") if isinstance(res, dict) else None
                if isinstance(g, list) and len(g) > 0:
                    return self._postprocess(res)
                # otherwise treat as no-detection and continue
            except Exception:
                continue
        return {"gpu_count": 0, "gpus": []}
    

    ''' the dictionary looks like this:

    {
        "gpu_count": int,
        "gpus": [
            {
                "index": int,
                "name": str,
                "vendor": str,
                "vram_total_gb": float,
                "vram_used_gb": float,
                "vram_free_gb": float,
                "vram_free_pct": float,
                "utilization_pct": int,
                "memory_util_pct": int,
                "temperature_C": int
            }
        ]
    }

    I believe this should be "flattened" when we put it in CSV. 
    idk like for example: 
    gpu_count | gpu_index | gpu_name | gpu_vendor | gpu_vram_total_gb | gpu_vram_used_gb | gpu_vram_free_gb | gpu_vram_free_pct | gpu_utilization_pct | gpu_memory_util_pct | gpu_temperature_C


    '''
    

    def __del__(self):
        """shut down nvml if we initialized it earlier
        garbage collection! whippy
        """
        try:
            if self._nvml_ok and pynvml is not None:
                pynvml.nvmlShutdown()
        except Exception:
            pass
