from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess

import psutil


@dataclass(frozen=True)
class TelemetrySnapshot:
    ram_available_mb: int
    ram_used_percent: float
    cpu_used_percent: float = 0.0
    swap_used_percent: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    gpu_used_mb: int | None = None
    gpu_total_mb: int | None = None
    gpu_util_percent: int | None = None
    gpu_power_watts: float | None = None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def classify_failure(text: str) -> str:
    lowered = text.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if (
        "segmentation fault" in lowered
        or "access violation" in lowered
        or "exception code" in lowered
        or "core dumped" in lowered
    ):
        return "crash"
    if "failed to load model" in lowered or "error loading model" in lowered:
        return "model_load"
    if "cuda" in lowered and "out of memory" in lowered:
        return "gpu_oom"
    if "failed to allocate" in lowered or "kv cache" in lowered and "memory" in lowered:
        return "memory_allocation"
    if "out of memory" in lowered:
        return "memory_allocation"
    return "unknown"


def sample_telemetry() -> TelemetrySnapshot:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_io_counters()
    gpu = _sample_gpu_with_nvidia_smi()
    return TelemetrySnapshot(
        ram_available_mb=int(memory.available / 1024 / 1024),
        ram_used_percent=float(memory.percent),
        cpu_used_percent=float(psutil.cpu_percent(interval=None)),
        swap_used_percent=float(swap.percent),
        disk_read_mb=float((disk.read_bytes if disk else 0) / 1024 / 1024),
        disk_write_mb=float((disk.write_bytes if disk else 0) / 1024 / 1024),
        gpu_used_mb=gpu.get("used"),
        gpu_total_mb=gpu.get("total"),
        gpu_util_percent=gpu.get("util"),
        gpu_power_watts=gpu.get("power"),
    )


def _sample_gpu_with_nvidia_smi() -> dict[str, int | None]:
    query = "memory.used,memory.total,utilization.gpu,power.draw"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"used": None, "total": None, "util": None}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"used": None, "total": None, "util": None}
    parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    if len(parts) != 4:
        return {"used": None, "total": None, "util": None, "power": None}
    return {
        "used": int(float(parts[0])),
        "total": int(float(parts[1])),
        "util": int(float(parts[2])),
        "power": float(parts[3]) if parts[3] not in {"[N/A]", "N/A"} else None,
    }
