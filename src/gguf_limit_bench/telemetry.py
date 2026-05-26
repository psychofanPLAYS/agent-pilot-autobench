from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess

import psutil


@dataclass(frozen=True)
class TelemetrySnapshot:
    ram_available_mb: int
    ram_used_percent: float
    gpu_used_mb: int | None = None
    gpu_total_mb: int | None = None
    gpu_util_percent: int | None = None

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
    gpu = _sample_gpu_with_nvidia_smi()
    return TelemetrySnapshot(
        ram_available_mb=int(memory.available / 1024 / 1024),
        ram_used_percent=float(memory.percent),
        gpu_used_mb=gpu.get("used"),
        gpu_total_mb=gpu.get("total"),
        gpu_util_percent=gpu.get("util"),
    )


def _sample_gpu_with_nvidia_smi() -> dict[str, int | None]:
    query = "memory.used,memory.total,utilization.gpu"
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
    if len(parts) != 3:
        return {"used": None, "total": None, "util": None}
    return {"used": int(parts[0]), "total": int(parts[1]), "util": int(parts[2])}
