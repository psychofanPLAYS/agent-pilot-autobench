"""GPU-specific flag profiles for llama.cpp inference.

Each profile is matched by case-insensitive substring against the GPU name
returned by the system (e.g. nvidia-smi or pynvml).  The first matching entry
wins, so order matters – put more specific substrings first.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class _GpuProfile:
    substring: str  # lower-case match substring
    always_on_flags: tuple[str, ...]
    parallel_slots: int
    description_template: str  # {gpu_name} is substituted at call-time


_PROFILES: list[_GpuProfile] = [
    _GpuProfile(
        substring="4090",
        always_on_flags=(
            "--flash-attn",
            "on",
            "--kv-unified",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--jinja",
            "--gpu-layers",
            "99",
        ),
        parallel_slots=4,
        description_template=(
            "RTX 4090 (Ada, 24 GB): flash-attn + unified q8_0 KV cache +"
            " Jinja chat templating, 4 parallel slots"
        ),
    ),
]

_FALLBACK = _GpuProfile(
    substring="",
    always_on_flags=(
        "--flash-attn",
        "on",
        "--jinja",
        "--gpu-layers",
        "99",
    ),
    parallel_slots=1,
    description_template=(
        "{gpu_name}: tuned recommendations not available yet;"
        " using conservative defaults (flash-attn + full GPU offload)"
    ),
)


def _match(gpu_name: str) -> _GpuProfile:
    lower = gpu_name.lower()
    for profile in _PROFILES:
        if profile.substring in lower:
            return profile
    return _FALLBACK


def recommended_always_on(gpu_name: str) -> tuple[str, ...]:
    """Return a tuple of llama.cpp flags that should always be passed for *gpu_name*.

    The tuple is ordered so that each flag is immediately followed by its value
    (e.g. ``("--flash-attn", "on", "--gpu-layers", "99", ...)``).
    Unknown GPUs get a conservative set of flags.
    """
    return _match(gpu_name).always_on_flags


def recommended_parallel(gpu_name: str) -> int:
    """Return the recommended number of parallel inference slots for *gpu_name*.

    Returns 4 for an RTX 4090, 1 for any unknown GPU.
    """
    return _match(gpu_name).parallel_slots


def describe(gpu_name: str) -> str:
    """Return a human-readable one-liner describing the recommended settings."""
    profile = _match(gpu_name)
    return profile.description_template.format(gpu_name=gpu_name)


def detect_gpu_name() -> str:
    """Best-effort detection of the primary GPU name via ``nvidia-smi``.

    Returns the first GPU's name (e.g. ``"NVIDIA GeForce RTX 4090"``) or an empty
    string when ``nvidia-smi`` is unavailable or fails. Never raises.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    first_line = completed.stdout.strip().splitlines()
    return first_line[0].strip() if first_line else ""
