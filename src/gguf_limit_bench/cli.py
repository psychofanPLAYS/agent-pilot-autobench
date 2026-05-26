from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import time
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from gguf_limit_bench.autoresearch import AutoresearchLoop, LlamaBenchAttemptRunner
from gguf_limit_bench.bench_plan import BenchProfile
from gguf_limit_bench.benchmark_suite import (
    BenchmarkSuitePlan,
    benchmark_suite_run_to_dict,
    run_benchmark_suite,
)
from gguf_limit_bench.config import (
    DEFAULT_DB_PATH,
    DEFAULT_LLAMA_BENCH,
    DEFAULT_LLAMA_CLI,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_RUNS_ROOT,
    load_config,
    with_cli_overrides,
)
from gguf_limit_bench.deployment import export_champion_profile
from gguf_limit_bench.discovery import discover_models
from gguf_limit_bench.doctor import DoctorReport, build_doctor_report
from gguf_limit_bench.installer import (
    DEFAULT_SHIM_DIR,
    add_shim_dir_to_user_path,
    check_user_path,
    install_command_shims,
    project_root,
    sync_project_environment,
)
from gguf_limit_bench.learning import OptunaSettingsLearner
from gguf_limit_bench.packs import load_benchmark_packs
from gguf_limit_bench.reports import write_leaderboard
from gguf_limit_bench.runner import BenchmarkRunner
from gguf_limit_bench.run_config import PRESETS, RunConfig
from gguf_limit_bench.server_probe import DEFAULT_AGENT_TTFT_PROMPT, probe_llama_server_ttft
from gguf_limit_bench.state_db import init_state_db
from gguf_limit_bench.tui import BenchTui
from gguf_limit_bench.workflows import WorkflowAugmentedAttemptRunner, WorkflowEvaluator


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
    config = load_config()
    _start_app(root=config.paths.model_roots[0], check_only=check_only)
    raise typer.Exit()


@app.command()
def start(
    root: Path | None = typer.Option(
        None,
        help="Folder where your GGUF models live.",
    ),
    check_only: bool = typer.Option(
        False,
        "--check-only",
        help="Only check the computer. Do not open the picker.",
    ),
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    runs_root: Path | None = None,
    budget_minutes: int | None = None,
    parallel_max: int | None = None,
    max_attempts: int | None = 1,
    learning: bool = True,
    workflow_eval: bool = False,
    ttft_probe: bool = True,
    benchmark_suite_plan: Path | None = typer.Option(
        None,
        "--benchmark-suite-plan",
        help="Run a benchmark-suite plan for selected models and optimize by agent_bench_score.",
    ),
    preset: str | None = None,
) -> None:
    """Beginner start button: check paths, then open the model picker."""
    config = with_cli_overrides(
        load_config(),
        model_roots=[root] if root is not None else None,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        runs_root=runs_root,
        default_preset=preset,
        parallel_max=parallel_max,
    )
    _start_app(
        root=config.paths.model_roots[0],
        check_only=check_only,
        llama_bench=config.paths.llama_bench,
        llama_cli=config.paths.llama_cli,
        llama_server=config.paths.llama_server,
        runs_root=config.paths.runs_root,
        budget_minutes=budget_minutes,
        parallel_max=config.benchmark.parallel_max,
        max_attempts=max_attempts,
        learning=learning,
        workflow_eval=workflow_eval,
        ttft_probe=ttft_probe,
        benchmark_suite_plan=benchmark_suite_plan,
        preset=config.benchmark.default_preset,
    )


def _start_app(
    root: Path,
    check_only: bool,
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    llama_server: Path = DEFAULT_LLAMA_SERVER,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    budget_minutes: int | None = None,
    parallel_max: int = 4,
    max_attempts: int | None = 1,
    learning: bool = True,
    workflow_eval: bool = False,
    ttft_probe: bool = True,
    benchmark_suite_plan: Path | None = None,
    preset: str = "quick",
) -> None:
    report = build_doctor_report(
        model_roots=[root],
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
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
    run_config = _run_config_from_inputs(
        preset=preset, budget_minutes=budget_minutes, max_attempts=max_attempts
    )
    picker = BenchTui(
        root=root,
        run_model=lambda model: (
            _run_one_autoresearch(
                model=model.path,
                llama_bench=llama_bench,
                llama_cli=llama_cli,
                llama_server=llama_server,
                runs_root=runs_root,
                budget_seconds=run_config.budget_minutes * 60,
                parallel_max=parallel_max,
                max_attempts=run_config.max_attempts,
                learning=learning,
                workflow_eval=workflow_eval,
                ttft_probe=ttft_probe,
                benchmark_suite_plan=benchmark_suite_plan,
                enable_mtp=model.has_mtp,
            ).path
        ),
    )
    picker.run_config = run_config
    picker.run()
    if getattr(picker, "ran_inside_tui", False):
        leaderboard = write_leaderboard(runs_root)
        if leaderboard.entries:
            console.print(
                f"Champion: {leaderboard.champion.model_name} ({leaderboard.champion.score:.2f})"
            )
            console.print(f"Leaderboard: {runs_root / 'leaderboard.md'}")
    else:
        _run_tui_selection(
            selected_models=picker.models_to_run,
            llama_bench=llama_bench,
            llama_cli=llama_cli,
            llama_server=llama_server,
            runs_root=runs_root,
            budget_minutes=run_config.budget_minutes,
            parallel_max=parallel_max,
            max_attempts=run_config.max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            ttft_probe=ttft_probe,
            benchmark_suite_plan=benchmark_suite_plan,
        )


@app.command()
def doctor(
    root: list[Path] | None = typer.Option(
        None,
        "--root",
        help="Model root to check. Repeat this option for multiple folders.",
    ),
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    runs_root: Path | None = None,
    strict: bool = False,
    json_out: bool = False,
) -> None:
    """Check local paths before spending time on benchmark runs."""
    config = with_cli_overrides(
        load_config(),
        model_roots=root,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        runs_root=runs_root,
    )
    roots = list(config.paths.model_roots)
    report = build_doctor_report(
        model_roots=roots,
        llama_bench=config.paths.llama_bench,
        llama_cli=config.paths.llama_cli,
        llama_server=config.paths.llama_server,
        runs_root=config.paths.runs_root,
    )
    if json_out:
        _print_json({**report.to_dict(), "resolved_config": config.to_dict()})
    else:
        _print_doctor_report(report)
    if strict and not report.ready:
        typer.echo("Required checks failed.", err=True)
        raise typer.Exit(1)


@app.command("first-run")
def first_run(
    root: Path | None = typer.Option(
        None,
        help="Folder where your GGUF models live.",
    ),
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    runs_root: Path | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    shim_dir: Path = typer.Option(
        DEFAULT_SHIM_DIR,
        help="Folder where Windows command shims are installed.",
    ),
    skip_env_sync: bool = typer.Option(
        False,
        "--skip-env-sync",
        help="Do not run uv sync. Useful for tests or offline checks.",
    ),
    install_command: bool = typer.Option(
        True,
        "--install-command/--no-install-command",
        help="Create agent-autobench and apb command shims.",
    ),
    add_to_path: bool = typer.Option(
        False,
        "--add-to-path",
        help="Add the command shim folder to the Windows user PATH.",
    ),
    json_out: bool = False,
) -> None:
    """Install the local command, sync dependencies, and prepare first-run state."""
    config = with_cli_overrides(
        load_config(),
        model_roots=[root] if root is not None else None,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        runs_root=runs_root,
    )
    repo_root = project_root()
    install_steps = [sync_project_environment(repo_root=repo_root, skip=skip_env_sync)]
    if install_command:
        install_steps.extend(install_command_shims(repo_root=repo_root, shim_dir=shim_dir))
        install_steps.append(
            add_shim_dir_to_user_path(shim_dir) if add_to_path else check_user_path(shim_dir)
        )

    report = build_doctor_report(
        model_roots=[config.paths.model_roots[0]],
        llama_bench=config.paths.llama_bench,
        llama_cli=config.paths.llama_cli,
        llama_server=config.paths.llama_server,
        runs_root=config.paths.runs_root,
    )
    init_state_db(db_path)
    write_leaderboard(config.paths.runs_root)
    payload = {
        **report.to_dict(),
        "install_ready": all(step.ok for step in install_steps if step.required),
        "install_steps": [step.to_dict() for step in install_steps],
        "db_path": str(db_path),
        "runs_root": str(config.paths.runs_root),
        "resolved_config": config.to_dict(),
        "next_command": (
            "agent-autobench --start"
            if report.ready and all(step.ok for step in install_steps if step.required)
            else "agent-autobench doctor"
        ),
    }
    if json_out:
        _print_json(payload)
    else:
        _print_first_run_report(
            report=report,
            db_path=db_path,
            runs_root=config.paths.runs_root,
            install_steps=install_steps,
        )
    if not report.ready or not all(step.ok for step in install_steps if step.required):
        raise typer.Exit(1)


@app.command()
def survey(
    root: Path | None = None,
    qwen_only: bool = False,
    qwen_35b_only: bool = False,
    mtp_only: bool = False,
    json_out: bool = False,
) -> None:
    config = with_cli_overrides(load_config(), model_roots=[root] if root is not None else None)
    roots = [root] if root is not None else list(config.paths.model_roots)
    models = discover_models(roots)
    models = _filter_models(
        models, qwen_only=qwen_only, qwen_35b_only=qwen_35b_only, mtp_only=mtp_only
    )
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
    runs_root: Path | None = None,
    json_out: bool = False,
) -> None:
    """Show the latest leaderboard and champion in normal language."""
    config = with_cli_overrides(load_config(), runs_root=runs_root)
    runs_root = config.paths.runs_root
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
        f"Bench speed: {champion.generation_tps:.2f} tok/s generation, "
        f"{champion.prompt_tps:.2f} tok/s prompt"
    )
    console.print(f"Cold TTFT: {_format_optional_ms(champion.serving_ttft_ms)}")
    console.print(f"Warm TTFT: {_format_optional_ms(champion.serving_warm_ttft_ms)}")
    console.print(f"Warmup penalty: {_format_optional_ms(champion.serving_warmup_penalty_ms)}")
    console.print(f"Serving speed: {_format_optional_tps(champion.serving_tps)}")
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


@app.command("benchmark-suite-template")
def benchmark_suite_template(
    output: Path = Path("benchmark-suite.plan.json"),
    model: str = "local-model",
    base_url: str = "http://127.0.0.1:8080/v1",
    context: int = 4096,
) -> None:
    """Write an editable plan that wraps real external benchmark harnesses."""
    plan = {
        "model": model,
        "context": context,
        "settings": {"base_url": base_url, "score_contract": "agent_bench_score"},
        "tasks": [
            {
                "id": "gsm8k_cot_zeroshot_smoke",
                "phase": "general",
                "harness": "lm-evaluation-harness",
                "env": {"PYTHONIOENCODING": "utf-8"},
                "commands": [
                    [
                        "uv",
                        "run",
                        "--extra",
                        "bench",
                        "lm-eval",
                        "run",
                        "--model",
                        "local-chat-completions",
                        "--model_args",
                        f"model={{model}},base_url={base_url}/chat/completions,eos_string=<|im_end|>",
                        "--tasks",
                        "gsm8k_cot_zeroshot",
                        "--apply_chat_template",
                        "--limit",
                        "10",
                        "--output_path",
                        "{task_dir}",
                        "--log_samples",
                    ],
                    [
                        "uv",
                        "run",
                        "--extra",
                        "bench",
                        "python",
                        "-m",
                        "gguf_limit_bench.score_extract",
                        "--root",
                        "{task_dir}",
                        "--out",
                        "{task_dir}/score.json",
                        "--keys",
                        "exact_match,accuracy,score",
                    ],
                ],
                "score_file": "{task_dir}/score.json",
                "min_score": 0.01,
                "timeout_seconds": 1800,
            },
            {
                "id": "inspect_agentic_smoke",
                "phase": "agentic",
                "harness": "inspect-ai",
                "env": {
                    "LOCAL_API_KEY": "local-no-key",
                    "LOCAL_BASE_URL": base_url,
                    "PYTHONIOENCODING": "utf-8",
                },
                "commands": [
                    [
                        "uv",
                        "run",
                        "--extra",
                        "bench",
                        "inspect",
                        "eval",
                        "benchmarks/inspect_tasks/json_repair.py",
                        "--model",
                        "openai-api/local/{model}",
                        "--model-base-url",
                        base_url,
                        "--log-dir",
                        "{task_dir}",
                        "--log-format",
                        "json",
                        "--display",
                        "none",
                        "--max-connections",
                        "1",
                        "--max-tokens",
                        "128",
                        "--temperature",
                        "0",
                    ],
                    [
                        "uv",
                        "run",
                        "--extra",
                        "bench",
                        "python",
                        "-m",
                        "gguf_limit_bench.inspect_score",
                        "--log-dir",
                        "{task_dir}",
                        "--out",
                        "{task_dir}/score.json",
                    ],
                ],
                "score_file": "{task_dir}/score.json",
                "min_score": 0.01,
                "timeout_seconds": 1800,
            },
        ],
    }
    output.write_text(json.dumps(plan, ensure_ascii=True, indent=2), encoding="utf-8")
    console.print(f"Benchmark-suite plan written: {output}")
    console.print("Edit the task commands, then run: agent-autobench benchmark-suite --plan PLAN")


@app.command("benchmark-suite")
def benchmark_suite(
    plan: Annotated[Path, typer.Option(help="Benchmark-suite plan JSON.")],
    runs_root: Path | None = None,
    allow_partial: bool = typer.Option(
        False,
        "--allow-partial",
        help="Write receipts but do not fail the command when one suite task fails.",
    ),
    json_out: bool = False,
) -> None:
    """Run Phase 1/2 benchmark-suite tasks and append benchmark TSV evidence."""
    config = with_cli_overrides(load_config(), runs_root=runs_root)
    try:
        suite_plan = BenchmarkSuitePlan.from_path(plan)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"Invalid benchmark-suite plan: {exc}", err=True)
        raise typer.Exit(2) from exc

    suite_run = run_benchmark_suite(suite_plan, config.paths.runs_root)
    payload = benchmark_suite_run_to_dict(suite_run)
    if json_out:
        _print_json(payload)
    else:
        console.print(f"Benchmark suite receipt: {suite_run.receipt_path}")
        console.print(f"General score: {_format_optional_float(suite_run.general_score)}")
        console.print(f"Agentic score: {_format_optional_float(suite_run.agentic_score)}")
        console.print(f"agent_bench_score: {suite_run.agent_bench_score:.6f}")
        console.print(
            "Suite status: pass"
            if suite_run.ok
            else "Suite status: fail (not production-ready evidence)"
        )
    if not suite_run.ok and not allow_partial:
        raise typer.Exit(1)


@app.command("benchmark-suite-plans")
def benchmark_suite_plans(json_out: bool = False) -> None:
    """List bundled benchmark-suite plan files."""
    plans_dir = project_root() / "benchmarks" / "plans"
    plans = sorted(plans_dir.glob("*.plan.json"))
    if json_out:
        _print_json([str(path) for path in plans])
        return
    if not plans:
        console.print("No bundled benchmark-suite plans found.")
        return
    table = Table(title="Benchmark Suite Plans")
    table.add_column("Plan")
    table.add_column("Kind")
    table.add_column("Requires")
    for path in plans:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            table.add_row(str(path), "invalid", "could not read JSON")
            continue
        settings = dict(payload.get("settings", {}))
        table.add_row(
            str(path.relative_to(project_root())),
            str(settings.get("plan_kind", "unknown")),
            str(settings.get("requires", "")),
        )
    console.print(table)


@app.command("init-db")
def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the local SQLite experiment memory."""
    init_state_db(db_path)
    console.print(f"Experiment memory ready: {db_path}")


@app.command("export-profile")
def export_profile(
    runs_root: Path | None = None,
    output_dir: Path = Path("results/champions"),
    llama_server: Path | None = None,
    lane: str = "hermes_pilot",
) -> None:
    """Export the latest champion as a ready-to-edit deployment profile."""
    config = with_cli_overrides(load_config(), runs_root=runs_root, llama_server=llama_server)
    runs_root = config.paths.runs_root
    leaderboard = write_leaderboard(runs_root)
    if not leaderboard.entries:
        console.print("No champion found yet. Run a benchmark first.")
        raise typer.Exit(1)
    outputs = export_champion_profile(
        champion_path=runs_root / "champion.json",
        output_dir=output_dir,
        llama_server=str(config.paths.llama_server),
        lane=lane,
    )
    console.print(f"YAML: {outputs.yaml_path}")
    console.print(f"PowerShell: {outputs.powershell_path}")
    console.print(f"Hermes note: {outputs.note_path}")


@app.command()
def quick(
    model: Annotated[Path, typer.Option(help="GGUF model path.")],
    llama_bench: Path | None = None,
    runs_root: Path | None = None,
) -> None:
    config = with_cli_overrides(load_config(), llama_bench=llama_bench, runs_root=runs_root)
    runner = BenchmarkRunner(llama_bench=config.paths.llama_bench, runs_root=config.paths.runs_root)
    receipt = runner.run_model(model=model, profile=BenchProfile.quick())
    console.print(f"Receipt: {receipt.path}")


@app.command("serve-probe")
def serve_probe(
    model: Annotated[Path, typer.Option(help="GGUF model path.")],
    llama_server: Path | None = None,
    context_size: int = 4096,
    parallel: int = 1,
    gpu_layers: int = 99,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    flash_attention: bool = True,
    timeout_seconds: int = 180,
    max_tokens: int = 64,
    samples: int = 0,
    cache_prompt: bool = True,
    prompt: str | None = None,
    json_out: bool = False,
) -> None:
    """Start llama-server once and measure cold/warm streaming TTFT plus serving speed."""
    from gguf_limit_bench.autoresearch import AutoresearchSettings

    config = with_cli_overrides(load_config(), llama_server=llama_server)

    settings = AutoresearchSettings(
        context_size=context_size,
        parallel=parallel,
        gpu_layers=gpu_layers,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        flash_attention=flash_attention,
    )
    result = probe_llama_server_ttft(
        llama_server=config.paths.llama_server,
        model=model,
        settings=settings,
        prompt=prompt or DEFAULT_AGENT_TTFT_PROMPT,
        max_tokens=max_tokens,
        samples=samples,
        cache_prompt=cache_prompt,
        timeout_seconds=timeout_seconds,
    )
    if json_out:
        _print_json(result.to_dict())
        return
    if result.ok:
        console.print(f"Server ready: {_format_optional_ms(result.server_ready_ms)}")
        console.print(
            f"Server start to first token: {_format_optional_ms(result.cold_start_to_first_token_ms)}"
        )
        console.print(f"Cold TTFT: {result.ttft_ms:.0f} ms")
        console.print(f"Warm TTFT: {_format_optional_ms(result.warm_ttft_ms)}")
        console.print(f"Warmup penalty: {_format_optional_ms(result.warmup_penalty_ms)}")
        console.print(f"Serving speed: {result.tokens_per_second:.2f} tok/s")
        if result.warm_tokens_per_second is not None:
            console.print(f"Warm serving speed: {result.warm_tokens_per_second:.2f} tok/s")
        console.print(f"Generated tokens: {result.generated_tokens}")
        console.print(
            f"TTFT samples: {', '.join(f'{sample:.0f} ms' for sample in result.ttft_samples_ms)}"
        )
        console.print(f"Tokens cached: {result.tokens_cached_samples}")
        console.print(f"Tokens evaluated: {result.tokens_evaluated_samples}")
        console.print(f"Output chars: {result.output_chars}")
    else:
        console.print(f"Serving probe failed: {result.failure}")
        if result.stderr_tail:
            console.print(result.stderr_tail)
        raise typer.Exit(1)


@app.command()
def autoresearch(
    model: Annotated[Path, typer.Option(help="GGUF model path.")],
    budget_minutes: int = 5,
    parallel_max: int | None = None,
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    runs_root: Path | None = None,
    max_attempts: int | None = None,
    learning: bool = True,
    workflow_eval: bool = False,
    ttft_probe: bool = True,
    benchmark_suite_plan: Path | None = typer.Option(
        None,
        "--benchmark-suite-plan",
        help=(
            "Run this benchmark-suite plan for each successful attempt and optimize "
            "by agent_bench_score."
        ),
    ),
) -> None:
    config = with_cli_overrides(
        load_config(),
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        runs_root=runs_root,
        parallel_max=parallel_max,
    )
    receipt = _run_one_autoresearch(
        model=model,
        llama_bench=config.paths.llama_bench,
        llama_cli=config.paths.llama_cli,
        llama_server=config.paths.llama_server,
        runs_root=config.paths.runs_root,
        budget_seconds=budget_minutes * 60,
        parallel_max=config.benchmark.parallel_max,
        max_attempts=max_attempts,
        learning=learning,
        workflow_eval=workflow_eval,
        ttft_probe=ttft_probe,
        benchmark_suite_plan=benchmark_suite_plan,
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
    parallel_max: int | None = None,
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    runs_root: Path | None = None,
    max_attempts: int | None = None,
    learning: bool = True,
    workflow_eval: bool = False,
    ttft_probe: bool = True,
    benchmark_suite_plan: Path | None = typer.Option(
        None,
        "--benchmark-suite-plan",
        help=(
            "Run this benchmark-suite plan for each successful attempt and optimize "
            "by agent_bench_score."
        ),
    ),
    finish_early_on: bool = False,
    target_score: float = 100.0,
) -> None:
    config = with_cli_overrides(
        load_config(),
        model_roots=[root] if root is not None else None,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        runs_root=runs_root,
        parallel_max=parallel_max,
    )
    roots = list(config.paths.model_roots)
    models = discover_models(roots)
    models = _filter_models(
        models, qwen_only=qwen_only, qwen_35b_only=qwen_35b_only, mtp_only=mtp_only
    )
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
            llama_bench=config.paths.llama_bench,
            llama_cli=config.paths.llama_cli,
            llama_server=config.paths.llama_server,
            runs_root=config.paths.runs_root,
            budget_seconds=budget_seconds,
            parallel_max=config.benchmark.parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            ttft_probe=ttft_probe,
            benchmark_suite_plan=benchmark_suite_plan,
            enable_mtp=model.has_mtp,
        )
        console.print(f"{model.name}: {receipt.path}")
        score = _receipt_score(receipt.path)
        if finish_early_on and score is not None and score >= target_score:
            console.print(f"Finish-early target met: score {score:.2f} >= {target_score:.2f}")
            break
    leaderboard = write_leaderboard(config.paths.runs_root)
    if leaderboard.entries:
        console.print(
            f"Champion: {leaderboard.champion.model_name} ({leaderboard.champion.score:.2f})"
        )
        console.print(f"Leaderboard: {config.paths.runs_root / 'leaderboard.md'}")


@app.command()
def tui(
    root: Path | None = None,
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    runs_root: Path | None = None,
    budget_minutes: int = 5,
    parallel_max: int | None = None,
    max_attempts: int | None = 1,
    learning: bool = True,
    workflow_eval: bool = False,
    ttft_probe: bool = True,
    benchmark_suite_plan: Path | None = typer.Option(
        None,
        "--benchmark-suite-plan",
        help="Run a benchmark-suite plan for selected models and optimize by agent_bench_score.",
    ),
) -> None:
    config = with_cli_overrides(
        load_config(),
        model_roots=[root] if root is not None else None,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        runs_root=runs_root,
        parallel_max=parallel_max,
    )
    picker = BenchTui(
        root=config.paths.model_roots[0],
        run_model=lambda model: (
            _run_one_autoresearch(
                model=model.path,
                llama_bench=config.paths.llama_bench,
                llama_cli=config.paths.llama_cli,
                llama_server=config.paths.llama_server,
                runs_root=config.paths.runs_root,
                budget_seconds=budget_minutes * 60,
                parallel_max=config.benchmark.parallel_max,
                max_attempts=max_attempts,
                learning=learning,
                workflow_eval=workflow_eval,
                ttft_probe=ttft_probe,
                benchmark_suite_plan=benchmark_suite_plan,
                enable_mtp=model.has_mtp,
            ).path
        ),
    )
    picker.run()
    if getattr(picker, "ran_inside_tui", False):
        leaderboard = write_leaderboard(runs_root)
        if leaderboard.entries:
            console.print(
                f"Champion: {leaderboard.champion.model_name} ({leaderboard.champion.score:.2f})"
            )
            console.print(f"Leaderboard: {config.paths.runs_root / 'leaderboard.md'}")
    else:
        _run_tui_selection(
            selected_models=picker.models_to_run,
            llama_bench=config.paths.llama_bench,
            llama_cli=config.paths.llama_cli,
            llama_server=config.paths.llama_server,
            runs_root=config.paths.runs_root,
            budget_minutes=budget_minutes,
            parallel_max=config.benchmark.parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            ttft_probe=ttft_probe,
            benchmark_suite_plan=benchmark_suite_plan,
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
    llama_server: Path,
    runs_root: Path,
    budget_seconds: int,
    parallel_max: int,
    max_attempts: int | None,
    learning: bool,
    workflow_eval: bool,
    ttft_probe: bool,
    benchmark_suite_plan: Path | None = None,
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
    if ttft_probe:
        base_runner = attempt_runner

        def attempt_runner(settings):
            result = base_runner(settings)
            if not result.ok:
                return result
            if not llama_server.exists():
                return replace(result, serving_failure=f"llama_server_missing: {llama_server}")
            serving = probe_llama_server_ttft(
                llama_server=llama_server,
                model=model,
                settings=settings,
                timeout_seconds=max(60, min(240, budget_seconds)),
            )
            if not serving.ok:
                return replace(result, serving_failure=serving.failure)
            return replace(
                result,
                serving_ttft_ms=serving.ttft_ms,
                serving_tokens_per_second=serving.tokens_per_second,
                serving_warm_ttft_ms=serving.warm_ttft_ms,
                serving_warmup_penalty_ms=serving.warmup_penalty_ms,
                serving_server_ready_ms=serving.server_ready_ms,
                serving_cold_start_to_first_token_ms=serving.cold_start_to_first_token_ms,
                serving_question_results=serving.question_results,
                serving_failure=None,
            )

    loop = AutoresearchLoop(
        model=model,
        runs_root=runs_root,
        attempt_runner=attempt_runner,
        budget_seconds=budget_seconds,
        parallel_max=parallel_max,
        max_attempts=max_attempts,
        learner=_build_learner(learning, runs_root, model, parallel_max),
        benchmark_suite_plan=(
            BenchmarkSuitePlan.from_path(benchmark_suite_plan)
            if benchmark_suite_plan is not None
            else None
        ),
    )
    return loop.run()


def _run_tui_selection(
    selected_models,
    llama_bench: Path,
    llama_cli: Path,
    llama_server: Path,
    runs_root: Path,
    budget_minutes: int,
    parallel_max: int,
    max_attempts: int | None,
    learning: bool,
    workflow_eval: bool,
    ttft_probe: bool,
    benchmark_suite_plan: Path | None = None,
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
            llama_server=llama_server,
            runs_root=runs_root,
            budget_seconds=budget_minutes * 60,
            parallel_max=parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            ttft_probe=ttft_probe,
            benchmark_suite_plan=benchmark_suite_plan,
            enable_mtp=model.has_mtp,
        )
        console.print(f"Receipt: {receipt.path}")
    leaderboard = write_leaderboard(runs_root)
    if leaderboard.entries:
        console.print(
            f"Champion: {leaderboard.champion.model_name} ({leaderboard.champion.score:.2f})"
        )
        console.print(f"Leaderboard: {runs_root / 'leaderboard.md'}")


def _run_config_from_inputs(
    preset: str, budget_minutes: int | None, max_attempts: int | None
) -> RunConfig:
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


def _print_first_run_report(
    report: DoctorReport,
    db_path: Path,
    runs_root: Path,
    install_steps,
) -> None:
    console.print("First-time installer")
    install_table = Table(title="Install Steps")
    install_table.add_column("Step")
    install_table.add_column("Status")
    install_table.add_column("Path")
    install_table.add_column("Detail")
    for step in install_steps:
        install_table.add_row(step.name, step.status, step.path, step.detail)
    console.print(install_table)

    console.print("Machine readiness check")
    _print_doctor_report(report)
    console.print(f"Experiment memory: {db_path}")
    console.print(f"Results folder: {runs_root}")
    install_ready = all(step.ok for step in install_steps if step.required)
    if report.ready and install_ready:
        console.print("First-time setup is ready.")
        console.print("Next command: agent-autobench --start")
        console.print("Short command: apb --start")
    else:
        console.print("First-time setup needs one or more missing items fixed.")
        console.print("Next command: agent-autobench doctor")


def _print_json(payload) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, indent=2))


def _format_optional_ms(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.0f} ms"


def _format_optional_tps(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.2f} tok/s"


def _format_optional_float(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.6f}"


def _is_mtp_model(model: Path) -> bool:
    return "mtp" in model.name.lower()
