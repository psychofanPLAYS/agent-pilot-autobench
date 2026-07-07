from __future__ import annotations

from collections.abc import Iterable
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
DEFAULT_QUESTION_SAMPLE_SIZE = 5
DEFAULT_QUESTION_SELECTION = "sequential"


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
    forced_server_args: tuple[str, ...] = ()
    question_sample_size: int = DEFAULT_QUESTION_SAMPLE_SIZE
    question_selection: str = DEFAULT_QUESTION_SELECTION


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
                "forced_server_args": list(self.benchmark.forced_server_args),
                "question_sample_size": self.benchmark.question_sample_size,
                "question_selection": self.benchmark.question_selection,
            },
        }


def load_config(config_path: Path | None = None) -> PilotbenchConfig:
    path = config_path or find_config_path()
    if path is None or not path.exists():
        return apply_env_overrides(PilotbenchConfig())
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = payload.get("paths", {})
    benchmark = payload.get("benchmark", {})
    # Relative paths in the config file mean "next to the config file", not
    # "wherever the process happens to be started from" — otherwise running
    # apb from another directory silently scatters _runs/_db into that cwd.
    base = path.parent

    def _anchored(value: object, default: Path) -> Path:
        return _anchor(_path(value, default), base)

    config = PilotbenchConfig(
        paths=PathSettings(
            model_roots=tuple(
                _anchor(root, base)
                for root in _paths(paths.get("model_roots"), DEFAULT_MODEL_ROOTS)
            ),
            llama_bench=_anchored(paths.get("llama_bench"), DEFAULT_LLAMA_BENCH),
            llama_cli=_anchored(paths.get("llama_cli"), DEFAULT_LLAMA_CLI),
            llama_server=_anchored(paths.get("llama_server"), DEFAULT_LLAMA_SERVER),
            llama_perplexity=_anchored(paths.get("llama_perplexity"), DEFAULT_LLAMA_PERPLEXITY),
            runs_root=_anchored(paths.get("runs_root"), DEFAULT_RUNS_ROOT),
        ),
        benchmark=BenchmarkSettings(
            default_preset=str(benchmark.get("default_preset", DEFAULT_PRESET)),
            parallel_max=int(benchmark.get("parallel_max", DEFAULT_PARALLEL_MAX)),
            learning=_bool(benchmark.get("learning"), True),
            workflow_eval=_bool(benchmark.get("workflow_eval"), True),
            ttft_probe=_bool(benchmark.get("ttft_probe"), True),
            perplexity_corpus=(
                None
                if (corpus := _optional_path(benchmark.get("perplexity_corpus"))) is None
                else _anchor(corpus, base)
            ),
            perplexity_contexts=_ints(
                benchmark.get("perplexity_contexts"), DEFAULT_PERPLEXITY_CONTEXTS
            ),
            forced_server_args=_strings(benchmark.get("forced_server_args"), ()),
            question_sample_size=int(
                benchmark.get("question_sample_size", DEFAULT_QUESTION_SAMPLE_SIZE)
            ),
            question_selection=str(benchmark.get("question_selection", DEFAULT_QUESTION_SELECTION)),
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
            forced_server_args=_strings(
                os.environ.get("PILOTBENCH_FORCED_SERVER_ARGS"), benchmark.forced_server_args
            ),
            question_sample_size=int(
                os.environ.get("PILOTBENCH_QUESTION_SAMPLE_SIZE", benchmark.question_sample_size)
            ),
            question_selection=str(
                os.environ.get("PILOTBENCH_QUESTION_SELECTION", benchmark.question_selection)
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
    question_sample_size: int | None = None,
    question_selection: str | None = None,
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
            forced_server_args=config.benchmark.forced_server_args,
            question_sample_size=(
                question_sample_size
                if question_sample_size is not None
                else config.benchmark.question_sample_size
            ),
            question_selection=(
                question_selection
                if question_selection is not None
                else config.benchmark.question_selection
            ),
        ),
    )
    return apply_env_overrides(config)


def _path(value: object, default: Path) -> Path:
    return default if value in (None, "") else Path(str(value))


def _anchor(value: Path, base: Path) -> Path:
    return value if value.is_absolute() else base / value


def _optional_path(value: object) -> Path | None:
    return None if value in (None, "") else Path(str(value))


def _paths(value: object, default: tuple[Path, ...]) -> tuple[Path, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return tuple(Path(item.strip()) for item in value.split(os.pathsep) if item.strip())
    if not isinstance(value, Iterable):
        raise TypeError("Expected paths to be a string or iterable")
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
    if not isinstance(value, Iterable):
        raise TypeError("Expected integers to be a string or iterable")
    return tuple(int(item) for item in value)


def _env_ints(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    return _ints(os.environ.get(name), default)


def _strings(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable):
        raise TypeError("Expected a string or iterable of strings")
    return tuple(str(item) for item in value)
