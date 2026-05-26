from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BenchProfile:
    name: str
    repetitions: int = 1
    prompt_gen: str = "512,128"
    gpu_layers: int = 99
    flash_attention: bool = True
    batch_size: int = 2048
    ubatch_size: int = 512
    depths: list[int] = field(default_factory=lambda: [0])
    timeout_seconds: int = 300

    @classmethod
    def quick(cls) -> "BenchProfile":
        return cls(name="quick", timeout_seconds=300)

    @classmethod
    def baseline(cls) -> "BenchProfile":
        return cls(name="baseline", repetitions=3, depths=[0, 4096, 8192], timeout_seconds=900)

    @classmethod
    def limit(cls, max_depth: int = 131072) -> "BenchProfile":
        depths = [0]
        value = 4096
        while value <= max_depth:
            depths.append(value)
            value *= 2
        return cls(name="limit", repetitions=1, depths=depths, timeout_seconds=1200)


def build_llama_bench_command(
    llama_bench: Path,
    model: Path,
    profile: BenchProfile,
    depth: int | None = None,
) -> list[str]:
    command = [
        str(llama_bench),
        "--model",
        str(model),
        "-o",
        "jsonl",
        "-r",
        str(profile.repetitions),
        "-pg",
        profile.prompt_gen,
        "-ngl",
        str(profile.gpu_layers),
        "-fa",
        "1" if profile.flash_attention else "0",
        "-b",
        str(profile.batch_size),
        "-ub",
        str(profile.ubatch_size),
    ]
    active_depth = profile.depths[0] if depth is None else depth
    if active_depth:
        command.extend(["-d", str(active_depth)])
    return command

