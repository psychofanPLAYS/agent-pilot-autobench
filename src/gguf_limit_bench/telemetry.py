from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess
import time
from typing import TypedDict

import psutil


class _GpuSample(TypedDict, total=False):
    used: int | None
    total: int | None
    util: int | None
    power: float | None


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


def _sample_gpu_with_nvidia_smi() -> _GpuSample:
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


class PeakEnergySampler:
    """Track peak VRAM and integrate GPU power into Joules over a window.

    Call :meth:`sample` repeatedly during a generation window. Energy is a
    trapezoidal integral of power over wall-clock time. Inject ``sampler`` and
    ``clock`` for tests; defaults read real telemetry and a monotonic clock.
    """

    def __init__(self, sampler=sample_telemetry, clock=time.monotonic) -> None:
        self._sampler = sampler
        self._clock = clock
        self._peak_vram_mb: int | None = None
        self._energy_joules = 0.0
        self._saw_power = False
        self._last_time: float | None = None
        self._last_power: float | None = None
        self._start_time: float | None = None
        self._end_time: float | None = None

    def sample(self) -> None:
        snap = self._sampler()
        now = self._clock()
        if self._start_time is None:
            self._start_time = now
        self._end_time = now
        if snap.gpu_used_mb is not None:
            self._peak_vram_mb = (
                snap.gpu_used_mb
                if self._peak_vram_mb is None
                else max(self._peak_vram_mb, snap.gpu_used_mb)
            )
        power = snap.gpu_power_watts
        if power is not None:
            self._saw_power = True
            if self._last_time is not None and self._last_power is not None:
                dt = now - self._last_time
                self._energy_joules += (power + self._last_power) / 2.0 * dt
            self._last_time = now
            self._last_power = power

    @property
    def peak_vram_mb(self) -> int | None:
        return self._peak_vram_mb

    @property
    def energy_joules(self) -> float | None:
        return self._energy_joules if self._saw_power else None

    @property
    def duration_s(self) -> float:
        if self._start_time is None or self._end_time is None:
            return 0.0
        return self._end_time - self._start_time
