from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


DEFAULT_MODEL_ROOT = Path("_models")
DEFAULT_MODEL_ROOTS = (DEFAULT_MODEL_ROOT,)
DEFAULT_LLAMA_BENCH = Path("_llama/llama-bench.exe")
DEFAULT_LLAMA_CLI = Path("_llama/llama-cli.exe")
DEFAULT_LLAMA_SERVER = Path("_llama/llama-server.exe")
DEFAULT_LLAMA_PERPLEXITY = Path("_llama/llama-perplexity.exe")
DEFAULT_RUNS_ROOT = Path("_runs")
DEFAULT_DB_PATH = Path("_db/agentpilot.sqlite")
DEFAULT_PARALLEL_MAX = 4
DEFAULT_PRESET = "deep"
DEFAULT_CONFIG_PATH = Path("_CONFIG.toml")
DEFAULT_PERPLEXITY_CONTEXTS = (4096, 8192, 16384)


@dataclass(frozen=True)
class PathSettings:
    model_roots: tuple[Path, ...] = DEFAULT_MODEL_ROOTS
    llama_bench: Path = DEFAULT_LLAMA_BENCH
    llama_cli: Path = DEFAULT_LLAMA_CLI
    llama_server: Path = DEFAULT_LLAMA_SERVER
    llama_perplexity: Path = DEFAULT_LLAMA_PERPLEXITY
    runs_root: Path = DEFAULT_RUNS_ROOT


@dataclass(frozen=True)
class BenchmarkSettings:
    default_preset: str = DEFAULT_PRESET
    parallel_max: int = DEFAULT_PARALLEL_MAX
    learning: bool = True
    workflow_eval: bool = True
    ttft_probe: bool = True
    perplexity_corpus: Path | None = None
    perplexity_contexts: tuple[int, ...] = DEFAULT_PERPLEXITY_CONTEXTS


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
                "llama_perplexity": str(self.paths.llama_perplexity),
                "runs_root": str(self.paths.runs_root),
            },
            "benchmark": {
                "default_preset": self.benchmark.default_preset,
                "parallel_max": self.benchmark.parallel_max,
                "learning": self.benchmark.learning,
                "workflow_eval": self.benchmark.workflow_eval,
                "ttft_probe": self.benchmark.ttft_probe,
                "perplexity_corpus": (
                    None
                    if self.benchmark.perplexity_corpus is None
                    else str(self.benchmark.perplexity_corpus)
                ),
                "perplexity_contexts": list(self.benchmark.perplexity_contexts),
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
            llama_perplexity=_path(paths.get("llama_perplexity"), DEFAULT_LLAMA_PERPLEXITY),
            runs_root=_path(paths.get("runs_root"), DEFAULT_RUNS_ROOT),
        ),
        benchmark=BenchmarkSettings(
            default_preset=str(benchmark.get("default_preset", DEFAULT_PRESET)),
            parallel_max=int(benchmark.get("parallel_max", DEFAULT_PARALLEL_MAX)),
            learning=_bool(benchmark.get("learning"), True),
            workflow_eval=_bool(benchmark.get("workflow_eval"), True),
            ttft_probe=_bool(benchmark.get("ttft_probe"), True),
            perplexity_corpus=_optional_path(benchmark.get("perplexity_corpus")),
            perplexity_contexts=_ints(
                benchmark.get("perplexity_contexts"), DEFAULT_PERPLEXITY_CONTEXTS
            ),
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
            llama_perplexity=_env_path("PILOTBENCH_LLAMA_PERPLEXITY", paths.llama_perplexity),
            runs_root=_env_path("PILOTBENCH_RUNS_ROOT", paths.runs_root),
        ),
        benchmark=BenchmarkSettings(
            default_preset=os.environ.get("PILOTBENCH_DEFAULT_PRESET", benchmark.default_preset),
            parallel_max=int(os.environ.get("PILOTBENCH_PARALLEL_MAX", benchmark.parallel_max)),
            learning=_env_bool("PILOTBENCH_LEARNING", benchmark.learning),
            workflow_eval=_env_bool("PILOTBENCH_WORKFLOW_EVAL", benchmark.workflow_eval),
            ttft_probe=_env_bool("PILOTBENCH_TTFT_PROBE", benchmark.ttft_probe),
            perplexity_corpus=_env_optional_path(
                "PILOTBENCH_PERPLEXITY_CORPUS", benchmark.perplexity_corpus
            ),
            perplexity_contexts=_env_ints(
                "PILOTBENCH_PERPLEXITY_CONTEXTS", benchmark.perplexity_contexts
            ),
        ),
    )


def with_cli_overrides(
    config: PilotbenchConfig,
    *,
    model_roots: list[Path] | tuple[Path, ...] | None = None,
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    llama_perplexity: Path | None = None,
    runs_root: Path | None = None,
    default_preset: str | None = None,
    parallel_max: int | None = None,
    learning: bool | None = None,
    workflow_eval: bool | None = None,
    ttft_probe: bool | None = None,
    perplexity_corpus: Path | None = None,
    perplexity_contexts: tuple[int, ...] | list[int] | None = None,
) -> PilotbenchConfig:
    config = PilotbenchConfig(
        paths=PathSettings(
            model_roots=tuple(model_roots) if model_roots else config.paths.model_roots,
            llama_bench=llama_bench or config.paths.llama_bench,
            llama_cli=llama_cli or config.paths.llama_cli,
            llama_server=llama_server or config.paths.llama_server,
            llama_perplexity=llama_perplexity or config.paths.llama_perplexity,
            runs_root=runs_root or config.paths.runs_root,
        ),
        benchmark=BenchmarkSettings(
            default_preset=default_preset or config.benchmark.default_preset,
            parallel_max=parallel_max
            if parallel_max is not None
            else config.benchmark.parallel_max,
            learning=learning if learning is not None else config.benchmark.learning,
            workflow_eval=workflow_eval
            if workflow_eval is not None
            else config.benchmark.workflow_eval,
            ttft_probe=ttft_probe if ttft_probe is not None else config.benchmark.ttft_probe,
            perplexity_corpus=perplexity_corpus or config.benchmark.perplexity_corpus,
            perplexity_contexts=(
                tuple(perplexity_contexts)
                if perplexity_contexts is not None
                else config.benchmark.perplexity_contexts
            ),
        ),
    )
    return apply_env_overrides(config)


def _path(value: object, default: Path) -> Path:
    return default if value in (None, "") else Path(str(value))


def _optional_path(value: object) -> Path | None:
    return None if value in (None, "") else Path(str(value))


def _paths(value: object, default: tuple[Path, ...]) -> tuple[Path, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return tuple(Path(item.strip()) for item in value.split(os.pathsep) if item.strip())
    return tuple(Path(str(item)) for item in value)


def _env_path(name: str, default: Path) -> Path:
    return _path(os.environ.get(name), default)


def _env_optional_path(name: str, default: Path | None) -> Path | None:
    value = os.environ.get(name)
    return default if value is None else _optional_path(value)


def _env_paths(name: str, default: tuple[Path, ...]) -> tuple[Path, ...]:
    return _paths(os.environ.get(name), default)


def _bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    return _bool(os.environ.get(name), default)


def _ints(value: object, default: tuple[int, ...]) -> tuple[int, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(os.pathsep) if item.strip())
    return tuple(int(item) for item in value)


def _env_ints(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    return _ints(os.environ.get(name), default)
