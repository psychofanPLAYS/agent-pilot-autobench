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
from gguf_limit_bench.deployment import export_champion_profile
from gguf_limit_bench.discovery import discover_models
from gguf_limit_bench.doctor import DoctorReport, build_doctor_report
from gguf_limit_bench.learning import OptunaSettingsLearner
from gguf_limit_bench.packs import load_benchmark_packs
from gguf_limit_bench.reports import write_leaderboard
from gguf_limit_bench.runner import BenchmarkRunner
from gguf_limit_bench.run_config import PRESETS, RunConfig
from gguf_limit_bench.state_db import init_state_db
from gguf_limit_bench.tui import BenchTui
from gguf_limit_bench.workflows import WorkflowAugmentedAttemptRunner, WorkflowEvaluator


DEFAULT_MODEL_ROOT = Path("G:/AI/models")
DEFAULT_MODEL_ROOTS = (Path("G:/AI/models"), Path("G:/AI/models/LM_Studio-gguf"))
DEFAULT_LLAMA_BENCH = Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-bench.exe")
DEFAULT_LLAMA_CLI = Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-cli.exe")
DEFAULT_LLAMA_SERVER = Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-server.exe")
DEFAULT_RUNS_ROOT = Path("runs")
DEFAULT_DB_PATH = Path("db/agentpilot.sqlite")

app = typer.Typer(
    help="Local-first autobenchmarking for choosing the best LLLM agent pilot.",
    no_args_is_help=True,
    invoke_without_command=True,
    rich_markup_mode="rich",
)
console = Console()


@app.callback()
def main(
    start_now: bool = typer.Option(
        False,
        "--start",
        help="Open the easy model picker.",
    ),
    check_only: bool = typer.Option(
        False,
        "--check-only",
        help="Only check the computer. Do not open the picker.",
    ),
) -> None:
    """Local-first LLLM agent-pilot autobench."""
    if not start_now:
        return
    _start_app(root=DEFAULT_MODEL_ROOT, check_only=check_only)
    raise typer.Exit()


@app.command()
def start(
    root: Path = typer.Option(
        DEFAULT_MODEL_ROOT,
        help="Folder where your GGUF models live.",
    ),
    check_only: bool = typer.Option(
        False,
        "--check-only",
        help="Only check the computer. Do not open the picker.",
    ),
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    budget_minutes: int | None = None,
    parallel_max: int = 4,
    max_attempts: int | None = 1,
    learning: bool = True,
    workflow_eval: bool = False,
    preset: str = "quick",
) -> None:
    """Beginner start button: check paths, then open the model picker."""
    _start_app(
        root=root,
        check_only=check_only,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        runs_root=runs_root,
        budget_minutes=budget_minutes,
        parallel_max=parallel_max,
        max_attempts=max_attempts,
        learning=learning,
        workflow_eval=workflow_eval,
        preset=preset,
    )


def _start_app(
    root: Path,
    check_only: bool,
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    budget_minutes: int | None = None,
    parallel_max: int = 4,
    max_attempts: int | None = 1,
    learning: bool = True,
    workflow_eval: bool = False,
    preset: str = "quick",
) -> None:
    report = build_doctor_report(
        model_roots=[root],
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        runs_root=runs_root,
    )
    if not report.ready:
        console.print("Something is missing. No benchmark was started.")
        _print_doctor_report(report)
        console.print("Run this first: uv run --extra dev pilotbench doctor")
        raise typer.Exit(1)
    console.print("Everything looks ready.")
    if check_only:
        console.print("Remove --check-only to open the picker.")
        return
    console.print("Opening the model picker.")
    run_config = _run_config_from_inputs(preset=preset, budget_minutes=budget_minutes, max_attempts=max_attempts)
    picker = BenchTui(
        root=root,
        run_model=lambda model: _run_one_autoresearch(
            model=model.path,
            llama_bench=llama_bench,
            llama_cli=llama_cli,
            runs_root=runs_root,
            budget_seconds=run_config.budget_minutes * 60,
            parallel_max=parallel_max,
            max_attempts=run_config.max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            enable_mtp=model.has_mtp,
        ).path,
    )
    picker.run_config = run_config
    picker.run()
    if getattr(picker, "ran_inside_tui", False):
        leaderboard = write_leaderboard(runs_root)
        if leaderboard.entries:
            console.print(f"Champion: {leaderboard.champion.model_name} ({leaderboard.champion.score:.2f})")
            console.print(f"Leaderboard: {runs_root / 'leaderboard.md'}")
    else:
        _run_tui_selection(
            selected_models=picker.models_to_run,
            llama_bench=llama_bench,
            llama_cli=llama_cli,
            runs_root=runs_root,
            budget_minutes=run_config.budget_minutes,
            parallel_max=parallel_max,
            max_attempts=run_config.max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
        )


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
        _print_json(report.to_dict())
    else:
        _print_doctor_report(report)
    if strict and not report.ready:
        typer.echo("Required checks failed.", err=True)
        raise typer.Exit(1)


@app.command("first-run")
def first_run(
    root: Path = typer.Option(
        DEFAULT_MODEL_ROOT,
        help="Folder where your GGUF models live.",
    ),
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    db_path: Path = DEFAULT_DB_PATH,
    json_out: bool = False,
) -> None:
    """Prepare local state and tell a first-time user exactly what to run next."""
    report = build_doctor_report(
        model_roots=[root],
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        runs_root=runs_root,
    )
    init_state_db(db_path)
    write_leaderboard(runs_root)
    payload = {
        **report.to_dict(),
        "db_path": str(db_path),
        "runs_root": str(runs_root),
        "next_command": "agent-autobench --start" if report.ready else "agent-autobench doctor",
    }
    if json_out:
        _print_json(payload)
    else:
        _print_first_run_report(report=report, db_path=db_path, runs_root=runs_root)
    if not report.ready:
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
        _print_json(
            [
                {
                    "path": str(model.path),
                    "family": model.family,
                    "parameters": model.parameters,
                    "quant": model.quant,
                    "size_bytes": model.size_bytes,
                    "size_gb": round(model.size_gb, 3),
                    "has_mtp": model.has_mtp,
                    "has_vision": model.has_vision,
                }
                for model in models
            ]
        )
        return
    table = Table(title=f"Discovered GGUF models under {', '.join(str(path) for path in roots)}")
    table.add_column("#", justify="right")
    table.add_column("Family")
    table.add_column("Params")
    table.add_column("Quant")
    table.add_column("GB", justify="right")
    table.add_column("MTP")
    table.add_column("Vision")
    table.add_column("Path")
    for index, model in enumerate(models, start=1):
        table.add_row(
            str(index),
            model.family,
            model.parameters,
            model.quant,
            f"{model.size_gb:.2f}",
            "yes" if model.has_mtp else "",
            "yes" if model.has_vision else "",
            str(model.path),
        )
    console.print(table)


@app.command()
def results(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    json_out: bool = False,
) -> None:
    """Show the latest leaderboard and champion in normal language."""
    leaderboard = write_leaderboard(runs_root)
    if not leaderboard.entries:
        console.print("No benchmark receipts found yet.")
        return
    if json_out:
        typer.echo((runs_root / "champion.json").read_text(encoding="utf-8"))
        return
    champion = leaderboard.champion
    console.print(f"Champion: {champion.model_name}")
    console.print(f"Score: {champion.score:.2f} | Status: {champion.status}")
    console.print(
        f"Speed: {champion.generation_tps:.2f} tok/s generation, "
        f"{champion.prompt_tps:.2f} tok/s prompt"
    )
    console.print(f"Context: {champion.context_label}")
    console.print(f"Receipt: {champion.receipt_path}")
    console.print(f"Leaderboard written: {runs_root / 'leaderboard.md'}")
    console.print(f"HTML report: {runs_root / 'results.html'}")


@app.command()
def packs(plugin_dir: Path = Path("plugins/benchmarks"), json_out: bool = False) -> None:
    """List built-in and local benchmark packs."""
    available = load_benchmark_packs(plugin_dir)
    if json_out:
        _print_json(
            {
                pack_id: {
                    "version": pack.version,
                    "description": pack.description,
                    "tasks": list(pack.tasks),
                    "scoring_categories": list(pack.scoring_categories),
                    "safety_policy": pack.safety_policy,
                }
                for pack_id, pack in sorted(available.items())
            }
        )
        return
    table = Table(title="Benchmark Packs")
    table.add_column("Pack")
    table.add_column("Version")
    table.add_column("Safety")
    table.add_column("Description")
    for pack in sorted(available.values(), key=lambda item: item.id):
        table.add_row(pack.id, pack.version, pack.safety_policy, pack.description)
    console.print(table)


@app.command("init-db")
def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the local SQLite experiment memory."""
    init_state_db(db_path)
    console.print(f"Experiment memory ready: {db_path}")


@app.command("export-profile")
def export_profile(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    output_dir: Path = Path("results/champions"),
    llama_server: Path = DEFAULT_LLAMA_SERVER,
    lane: str = "hermes_pilot",
) -> None:
    """Export the latest champion as a ready-to-edit deployment profile."""
    leaderboard = write_leaderboard(runs_root)
    if not leaderboard.entries:
        console.print("No champion found yet. Run a benchmark first.")
        raise typer.Exit(1)
    outputs = export_champion_profile(
        champion_path=runs_root / "champion.json",
        output_dir=output_dir,
        llama_server=str(llama_server),
        lane=lane,
    )
    console.print(f"YAML: {outputs.yaml_path}")
    console.print(f"PowerShell: {outputs.powershell_path}")
    console.print(f"Hermes note: {outputs.note_path}")


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
    leaderboard = write_leaderboard(runs_root)
    if leaderboard.entries:
        console.print(f"Champion: {leaderboard.champion.model_name} ({leaderboard.champion.score:.2f})")
        console.print(f"Leaderboard: {runs_root / 'leaderboard.md'}")


@app.command()
def tui(
    root: Path = DEFAULT_MODEL_ROOT,
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    budget_minutes: int = 5,
    parallel_max: int = 4,
    max_attempts: int | None = 1,
    learning: bool = True,
    workflow_eval: bool = False,
) -> None:
    picker = BenchTui(
        root=root,
        run_model=lambda model: _run_one_autoresearch(
            model=model.path,
            llama_bench=llama_bench,
            llama_cli=llama_cli,
            runs_root=runs_root,
            budget_seconds=budget_minutes * 60,
            parallel_max=parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            enable_mtp=model.has_mtp,
        ).path,
    )
    picker.run()
    if getattr(picker, "ran_inside_tui", False):
        leaderboard = write_leaderboard(runs_root)
        if leaderboard.entries:
            console.print(f"Champion: {leaderboard.champion.model_name} ({leaderboard.champion.score:.2f})")
            console.print(f"Leaderboard: {runs_root / 'leaderboard.md'}")
    else:
        _run_tui_selection(
            selected_models=picker.models_to_run,
            llama_bench=llama_bench,
            llama_cli=llama_cli,
            runs_root=runs_root,
            budget_minutes=budget_minutes,
            parallel_max=parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
        )


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


def _run_tui_selection(
    selected_models,
    llama_bench: Path,
    llama_cli: Path,
    runs_root: Path,
    budget_minutes: int,
    parallel_max: int,
    max_attempts: int | None,
    learning: bool,
    workflow_eval: bool,
) -> None:
    if not selected_models:
        console.print("No models selected. Nothing was started.")
        return
    console.print(f"Starting research loop for {len(selected_models)} selected model(s).")
    for model in selected_models:
        console.print(f"Running: {model.name} ({model.size_gb:.2f} GB)")
        receipt = _run_one_autoresearch(
            model=model.path,
            llama_bench=llama_bench,
            llama_cli=llama_cli,
            runs_root=runs_root,
            budget_seconds=budget_minutes * 60,
            parallel_max=parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            enable_mtp=model.has_mtp,
        )
        console.print(f"Receipt: {receipt.path}")
    leaderboard = write_leaderboard(runs_root)
    if leaderboard.entries:
        console.print(f"Champion: {leaderboard.champion.model_name} ({leaderboard.champion.score:.2f})")
        console.print(f"Leaderboard: {runs_root / 'leaderboard.md'}")


def _run_config_from_inputs(preset: str, budget_minutes: int | None, max_attempts: int | None) -> RunConfig:
    preset_id = preset if preset in PRESETS else "quick"
    config = RunConfig.from_preset(preset_id)
    return RunConfig(
        preset_id=config.preset_id,
        budget_minutes=budget_minutes if budget_minutes is not None else config.budget_minutes,
        max_extra_minutes=config.max_extra_minutes,
        total_session_cap_minutes=config.total_session_cap_minutes,
        max_attempts=max_attempts if max_attempts is not None else config.max_attempts,
        packs=config.packs,
        adaptive=config.adaptive,
        min_ttft_target_ms=config.min_ttft_target_ms,
        min_generation_tps=config.min_generation_tps,
        require_full_gpu_offload=config.require_full_gpu_offload,
        require_no_swap=config.require_no_swap,
    )


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
    table = Table(title="Agent Pilot Autobench Doctor")
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


def _print_first_run_report(report: DoctorReport, db_path: Path, runs_root: Path) -> None:
    console.print("First-time setup check")
    _print_doctor_report(report)
    console.print(f"Experiment memory: {db_path}")
    console.print(f"Results folder: {runs_root}")
    if report.ready:
        console.print("First-time setup is ready.")
        console.print("Next command: agent-autobench --start")
    else:
        console.print("First-time setup needs one or more missing paths fixed.")
        console.print("Next command: agent-autobench doctor")


def _print_json(payload) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, indent=2))


def _is_mtp_model(model: Path) -> bool:
    return "mtp" in model.name.lower()
