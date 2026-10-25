"""Hardware / software description recorded alongside every benchmark."""
from __future__ import annotations

import os
import platform
import subprocess
import sys


def cpu_name() -> str:
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "") or platform.processor()
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor()


def system_info() -> dict:
    import onnxruntime as ort
    info = {"cpu": cpu_name(), "logical_cores": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "onnxruntime": ort.__version__,
            "machine": platform.machine(), "in_colab": "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        f = psutil.cpu_freq()
        if f:
            info["cpu_freq_mhz_max"] = f.max
    except Exception:
        pass
    return info
