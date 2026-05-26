from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from gguf_limit_bench.autoresearch import AutoresearchLoop, LlamaBenchAttemptRunner
from gguf_limit_bench.bench_plan import BenchProfile
from gguf_limit_bench.discovery import discover_models
from gguf_limit_bench.doctor import DoctorReport, build_doctor_report
from gguf_limit_bench.learning import OptunaSettingsLearner
from gguf_limit_bench.runner import BenchmarkRunner
from gguf_limit_bench.tui import BenchTui
from gguf_limit_bench.workflows import WorkflowAugmentedAttemptRunner, WorkflowEvaluator


DEFAULT_MODEL_ROOT = Path("G:/AI/models")
DEFAULT_MODEL_ROOTS = (Path("G:/AI/models"), Path("G:/AI/models/LM_Studio-gguf"))
DEFAULT_LLAMA_BENCH = Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-bench.exe")
DEFAULT_LLAMA_CLI = Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-cli.exe")
DEFAULT_RUNS_ROOT = Path("runs")

app = typer.Typer(
    help="Local-first GGUF autoresearch lab for finding useful agent pilot settings.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def doctor(
    root: list[Path] | None = typer.Option(
        None,
        "--root",
        help="Model root to check. Repeat this option for multiple folders.",
    ),
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    strict: bool = False,
    json_out: bool = False,
) -> None:
    """Check local paths before spending time on benchmark runs."""
    roots = root if root else list(DEFAULT_MODEL_ROOTS)
    report = build_doctor_report(
        model_roots=roots,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        runs_root=runs_root,
    )
    if json_out:
        console.print_json(json.dumps(report.to_dict()))
    else:
        _print_doctor_report(report)
    if strict and not report.ready:
        typer.echo("Required checks failed.", err=True)
        raise typer.Exit(1)


@app.command()
def survey(
    root: Path | None = None,
    qwen_only: bool = False,
    qwen_35b_only: bool = False,
    mtp_only: bool = False,
    json_out: bool = False,
) -> None:
    roots = [root] if root is not None else list(DEFAULT_MODEL_ROOTS)
    models = discover_models(roots)
    models = _filter_models(models, qwen_only=qwen_only, qwen_35b_only=qwen_35b_only, mtp_only=mtp_only)
    if json_out:
        console.print_json(
            json.dumps(
                [
                    {
                        "path": str(model.path),
                        "family": model.family,
                        "parameters": model.parameters,
                        "quant": model.quant,
                        "has_mtp": model.has_mtp,
                        "has_vision": model.has_vision,
                    }
                    for model in models
                ]
            )
        )
        return
    table = Table(title=f"Discovered GGUF models under {', '.join(str(path) for path in roots)}")
    table.add_column("#", justify="right")
    table.add_column("Family")
    table.add_column("Params")
    table.add_column("Quant")
    table.add_column("MTP")
    table.add_column("Vision")
    table.add_column("Path")
    for index, model in enumerate(models, start=1):
        table.add_row(
            str(index),
            model.family,
            model.parameters,
            model.quant,
            "yes" if model.has_mtp else "",
            "yes" if model.has_vision else "",
            str(model.path),
        )
    console.print(table)


@app.command()
def quick(
    model: Annotated[Path, typer.Option(help="GGUF model path.")],
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    runs_root: Path = DEFAULT_RUNS_ROOT,
) -> None:
    runner = BenchmarkRunner(llama_bench=llama_bench, runs_root=runs_root)
    receipt = runner.run_model(model=model, profile=BenchProfile.quick())
    console.print(f"Receipt: {receipt.path}")


@app.command()
def autoresearch(
    model: Annotated[Path, typer.Option(help="GGUF model path.")],
    budget_minutes: int = 5,
    parallel_max: int = 4,
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    max_attempts: int | None = None,
    learning: bool = True,
    workflow_eval: bool = False,
) -> None:
    receipt = _run_one_autoresearch(
        model=model,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        runs_root=runs_root,
        budget_seconds=budget_minutes * 60,
        parallel_max=parallel_max,
        max_attempts=max_attempts,
        learning=learning,
        workflow_eval=workflow_eval,
        enable_mtp=_is_mtp_model(model),
    )
    console.print(f"Receipt: {receipt.path}")


@app.command("autoresearch-all")
def autoresearch_all(
    root: Path | None = None,
    qwen_only: bool = False,
    qwen_35b_only: bool = False,
    mtp_only: bool = False,
    budget_minutes: int = 5,
    total_budget_minutes: int | None = None,
    parallel_max: int = 4,
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    max_attempts: int | None = None,
    learning: bool = True,
    workflow_eval: bool = False,
    finish_early_on: bool = False,
    target_score: float = 100.0,
) -> None:
    roots = [root] if root is not None else list(DEFAULT_MODEL_ROOTS)
    models = discover_models(roots)
    models = _filter_models(models, qwen_only=qwen_only, qwen_35b_only=qwen_35b_only, mtp_only=mtp_only)
    console.print(f"Autoresearch queue: {len(models)} model(s)")
    deadline = time.monotonic() + total_budget_minutes * 60 if total_budget_minutes else None
    for model in models:
        budget_seconds = budget_minutes * 60
        if deadline is not None:
            remaining_seconds = int(deadline - time.monotonic())
            if remaining_seconds <= 0:
                console.print("Total budget exhausted.")
                break
            budget_seconds = min(budget_seconds, remaining_seconds)
        receipt = _run_one_autoresearch(
            model=model.path,
            llama_bench=llama_bench,
            llama_cli=llama_cli,
            runs_root=runs_root,
            budget_seconds=budget_seconds,
            parallel_max=parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            enable_mtp=model.has_mtp,
        )
        console.print(f"{model.name}: {receipt.path}")
        score = _receipt_score(receipt.path)
        if finish_early_on and score is not None and score >= target_score:
            console.print(f"Finish-early target met: score {score:.2f} >= {target_score:.2f}")
            break


@app.command()
def tui(root: Path = DEFAULT_MODEL_ROOT) -> None:
    BenchTui(root=root).run()


def _build_learner(
    enabled: bool,
    runs_root: Path,
    model: Path,
    parallel_max: int,
) -> OptunaSettingsLearner | None:
    if not enabled:
        return None
    return OptunaSettingsLearner(
        storage_path=runs_root / "learning" / "optuna.sqlite3",
        model=model,
        parallel_max=parallel_max,
    )


def _run_one_autoresearch(
    model: Path,
    llama_bench: Path,
    llama_cli: Path,
    runs_root: Path,
    budget_seconds: int,
    parallel_max: int,
    max_attempts: int | None,
    learning: bool,
    workflow_eval: bool,
    enable_mtp: bool = False,
):
    attempt_runner = LlamaBenchAttemptRunner(
        llama_bench=llama_bench,
        model=model,
        timeout_seconds=max(30, budget_seconds),
    )
    if workflow_eval:
        attempt_runner = WorkflowAugmentedAttemptRunner(
            bench_runner=attempt_runner,
            evaluator=WorkflowEvaluator(
                llama_cli=llama_cli,
                model=model,
                timeout_seconds=max(30, min(120, budget_seconds)),
                enable_mtp=enable_mtp,
            ),
        )
    loop = AutoresearchLoop(
        model=model,
        runs_root=runs_root,
        attempt_runner=attempt_runner,
        budget_seconds=budget_seconds,
        parallel_max=parallel_max,
        max_attempts=max_attempts,
        learner=_build_learner(learning, runs_root, model, parallel_max),
    )
    return loop.run()


def _receipt_score(receipt_path: Path) -> float | None:
    best_path = receipt_path / "best-settings.json"
    if not best_path.exists():
        return None
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    return float(payload["score"])


def _filter_models(
    models,
    qwen_only: bool,
    qwen_35b_only: bool,
    mtp_only: bool,
):
    if qwen_only or qwen_35b_only:
        models = [model for model in models if model.family == "qwen"]
    if qwen_35b_only:
        models = [model for model in models if model.parameters.startswith("35B")]
    if mtp_only:
        models = [model for model in models if model.has_mtp]
    return models


def _print_doctor_report(report: DoctorReport) -> None:
    table = Table(title="GGUF Limit Bench Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Path")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row(check.name, check.status, check.path, check.detail)
    console.print(table)
    if report.ready:
        console.print("Ready for benchmark runs.")
    else:
        console.print("Some required paths are missing. Use --strict in scripts to fail fast.")


def _is_mtp_model(model: Path) -> bool:
    return "mtp" in model.name.lower()
