from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


DEFAULT_MODEL_ROOT = Path("G:/AI/models")
DEFAULT_MODEL_ROOTS = (Path("G:/AI/models"), Path("G:/AI/models/LM_Studio-gguf"))
DEFAULT_LLAMA_BENCH = Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-bench.exe")
DEFAULT_LLAMA_CLI = Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-cli.exe")
DEFAULT_LLAMA_SERVER = Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-server.exe")
DEFAULT_RUNS_ROOT = Path("runs")
DEFAULT_DB_PATH = Path("db/agentpilot.sqlite")
DEFAULT_PARALLEL_MAX = 4
DEFAULT_PRESET = "quick"
DEFAULT_CONFIG_PATH = Path("pilotbench.toml")


@dataclass(frozen=True)
class PathSettings:
    model_roots: tuple[Path, ...] = DEFAULT_MODEL_ROOTS
    llama_bench: Path = DEFAULT_LLAMA_BENCH
    llama_cli: Path = DEFAULT_LLAMA_CLI
    llama_server: Path = DEFAULT_LLAMA_SERVER
    runs_root: Path = DEFAULT_RUNS_ROOT


@dataclass(frozen=True)
class BenchmarkSettings:
    default_preset: str = DEFAULT_PRESET
    parallel_max: int = DEFAULT_PARALLEL_MAX


@dataclass(frozen=True)
class PilotbenchConfig:
    paths: PathSettings = PathSettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()

    def to_dict(self) -> dict:
        return {
            "paths": {
                "model_roots": [str(path) for path in self.paths.model_roots],
                "llama_bench": str(self.paths.llama_bench),
                "llama_cli": str(self.paths.llama_cli),
                "llama_server": str(self.paths.llama_server),
                "runs_root": str(self.paths.runs_root),
            },
            "benchmark": {
                "default_preset": self.benchmark.default_preset,
                "parallel_max": self.benchmark.parallel_max,
            },
        }


def load_config(config_path: Path | None = None) -> PilotbenchConfig:
    path = config_path or find_config_path()
    if path is None or not path.exists():
        return apply_env_overrides(PilotbenchConfig())
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = payload.get("paths", {})
    benchmark = payload.get("benchmark", {})
    config = PilotbenchConfig(
        paths=PathSettings(
            model_roots=_paths(paths.get("model_roots"), DEFAULT_MODEL_ROOTS),
            llama_bench=_path(paths.get("llama_bench"), DEFAULT_LLAMA_BENCH),
            llama_cli=_path(paths.get("llama_cli"), DEFAULT_LLAMA_CLI),
            llama_server=_path(paths.get("llama_server"), DEFAULT_LLAMA_SERVER),
            runs_root=_path(paths.get("runs_root"), DEFAULT_RUNS_ROOT),
        ),
        benchmark=BenchmarkSettings(
            default_preset=str(benchmark.get("default_preset", DEFAULT_PRESET)),
            parallel_max=int(benchmark.get("parallel_max", DEFAULT_PARALLEL_MAX)),
        ),
    )
    return apply_env_overrides(config)


def find_config_path(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for folder in (current, *current.parents):
        candidate = folder / DEFAULT_CONFIG_PATH
        if candidate.exists():
            return candidate
    return None


def apply_env_overrides(config: PilotbenchConfig) -> PilotbenchConfig:
    paths = config.paths
    benchmark = config.benchmark
    return PilotbenchConfig(
        paths=PathSettings(
            model_roots=_env_paths("PILOTBENCH_MODEL_ROOTS", paths.model_roots),
            llama_bench=_env_path("PILOTBENCH_LLAMA_BENCH", paths.llama_bench),
            llama_cli=_env_path("PILOTBENCH_LLAMA_CLI", paths.llama_cli),
            llama_server=_env_path("PILOTBENCH_LLAMA_SERVER", paths.llama_server),
            runs_root=_env_path("PILOTBENCH_RUNS_ROOT", paths.runs_root),
        ),
        benchmark=BenchmarkSettings(
            default_preset=os.environ.get("PILOTBENCH_DEFAULT_PRESET", benchmark.default_preset),
            parallel_max=int(os.environ.get("PILOTBENCH_PARALLEL_MAX", benchmark.parallel_max)),
        ),
    )


def with_cli_overrides(
    config: PilotbenchConfig,
    *,
    model_roots: list[Path] | tuple[Path, ...] | None = None,
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    runs_root: Path | None = None,
    default_preset: str | None = None,
    parallel_max: int | None = None,
) -> PilotbenchConfig:
    config = PilotbenchConfig(
        paths=PathSettings(
            model_roots=tuple(model_roots) if model_roots else config.paths.model_roots,
            llama_bench=llama_bench or config.paths.llama_bench,
            llama_cli=llama_cli or config.paths.llama_cli,
            llama_server=llama_server or config.paths.llama_server,
            runs_root=runs_root or config.paths.runs_root,
        ),
        benchmark=BenchmarkSettings(
            default_preset=default_preset or config.benchmark.default_preset,
            parallel_max=parallel_max
            if parallel_max is not None
            else config.benchmark.parallel_max,
        ),
    )
    return apply_env_overrides(config)


def _path(value: object, default: Path) -> Path:
    return default if value in (None, "") else Path(str(value))


def _paths(value: object, default: tuple[Path, ...]) -> tuple[Path, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return tuple(Path(item.strip()) for item in value.split(os.pathsep) if item.strip())
    return tuple(Path(str(item)) for item in value)


def _env_path(name: str, default: Path) -> Path:
    return _path(os.environ.get(name), default)


def _env_paths(name: str, default: tuple[Path, ...]) -> tuple[Path, ...]:
    return _paths(os.environ.get(name), default)
