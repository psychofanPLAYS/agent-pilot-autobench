from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import signal
import sqlite3
import subprocess
import time
from typing import Annotated, Any, Callable, cast
import webbrowser

import typer
from rich.console import Console
from rich.table import Table

from gguf_limit_bench.autoresearch import (
    AttemptResult,
    AttemptRunner,
    AutoresearchLoop,
    AutoresearchSettings,
    LlamaBenchAttemptRunner,
    LlamaPerplexityRunner,
)
from gguf_limit_bench.bench_plan import BenchProfile
from gguf_limit_bench.context_limit import (
    DEFAULT_MIN_CONTEXT,
    LaunchOutcome,
    find_context_limit,
)
from gguf_limit_bench.context_search import context_ladder
from gguf_limit_bench.gguf_metadata import read_model_arch
from gguf_limit_bench.vram import (
    detect_vram_mb,
    max_fitting_context,
    plan_context_fit,
)
from gguf_limit_bench.benchmark_suite import (
    BenchmarkSuitePlan,
    benchmark_suite_preflight_to_dict,
    benchmark_suite_run_to_dict,
    preflight_benchmark_suite,
    run_benchmark_suite,
    suite_verdict,
)
from gguf_limit_bench.config import (
    DEFAULT_DB_PATH,
    DEFAULT_LLAMA_BENCH,
    DEFAULT_LLAMA_CLI,
    DEFAULT_LLAMA_PERPLEXITY,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_RUNS_ROOT,
    find_config_path,
    load_config,
    with_cli_overrides,
)
from gguf_limit_bench.deployment_readiness import write_deployment_readiness
from gguf_limit_bench.deployment_proof import (
    BenchmarkSuitePreflightError,
    DEFAULT_DEPLOYMENT_SIMPLE_BENCH_MAX_TOKENS,
    run_deployment_proof,
)
from gguf_limit_bench.deployment import export_champion_profile
from gguf_limit_bench.flight_plans import flight_plan_payloads
from gguf_limit_bench.gpu_profiles import detect_gpu_name, recommended_always_on
from gguf_limit_bench.discovery import discover_models
from gguf_limit_bench.doctor import DoctorReport, build_doctor_report
from gguf_limit_bench.evaluation_mode import (
    EvaluationMode,
    asks_questions,
    resolve_evaluation_mode,
)
from gguf_limit_bench.flag_recommendations import write_flag_recommendations
from gguf_limit_bench.hard_recommendations import write_hard_recommendations
from gguf_limit_bench.autodetect import (
    LLAMA_ENV_VARS,
    default_llama_search_roots,
    default_model_search_roots,
    find_llama_binaries,
    find_model_roots,
)
from gguf_limit_bench.installer import (
    DEFAULT_SHIM_DIR,
    add_shim_dir_to_user_path,
    check_user_path,
    install_command_shims,
    is_setup_complete,
    mark_setup_complete,
    persist_user_env,
    project_root,
    resolved_shim_dir,
    sync_project_environment,
)
from gguf_limit_bench.hf_catalog import HubCatalog, HuggingFaceGateway
from gguf_limit_bench.hf_recommended_settings import (
    recommended_sampler_flags,
    recommended_sampler_presets,
    sampler_flags_from_values,
)
from gguf_limit_bench.learning import OptunaSettingsLearner
from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS
from gguf_limit_bench.template_recommend import merge_flags, recommended_model_flags
from gguf_limit_bench.model_catalog import (
    ModelCatalog,
    find_catalog_entry,
    load_catalog,
    write_catalog,
)
from gguf_limit_bench.modes import KARPATHY_ROUND_SECONDS, mode_by_id
from gguf_limit_bench.packs import available_packs, load_benchmark_packs, load_pack
from gguf_limit_bench.programs import (
    MIN_SERIOUS_CONTEXT_SIZE,
    ProgramId,
    enforce_min_context,
    fit_probe_prompt,
    speed_probe_prompt,
)
from gguf_limit_bench.qe_results import write_qe_leaderboard
from gguf_limit_bench.qe_suite import run_qe_format_suite
from gguf_limit_bench.flag_ladder import (
    build_core_flag_ladder,
    build_flag_ladder_plan,
    filter_unsupported_profiles,
    validate_extra_server_args,
)
from gguf_limit_bench.receipts import RunReceipt
from gguf_limit_bench.runtime_capabilities import (
    LlamaRuntimeCapabilities,
    collect_llama_capabilities,
    inspect_llama_executable,
)
from gguf_limit_bench.reports import (
    build_report_audit,
    build_verdict,
    score_summary_for_entry,
    write_leaderboard,
)
from gguf_limit_bench.runner import BenchmarkRunner
from gguf_limit_bench.run_config import PRESETS, RunConfig
from gguf_limit_bench.server_probe import ServingProbeResult, probe_llama_server_ttft
from gguf_limit_bench.simple_bench import (
    DEFAULT_SIMPLE_BENCH_PATH,
    DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT,
)
from gguf_limit_bench.simple_bench_runner import LlamaServerSimpleBenchAttemptRunner
from gguf_limit_bench.state_db import (
    get_context_limit,
    init_state_db,
    record_context_limit,
)
from gguf_limit_bench.tui import BenchTui
from gguf_limit_bench.webui import benchmark_suite_plan_payloads, serve_webui
from gguf_limit_bench.workflows import WorkflowAugmentedAttemptRunner, WorkflowEvaluator


app = typer.Typer(
    help="Local-first GGUF and llama.cpp benchmarking for agent workloads.",
    no_args_is_help=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
)
models_app = typer.Typer(help="Discover and explain local model evidence.", no_args_is_help=True)
app.add_typer(models_app, name="models")
console = Console()


@models_app.command("scan")
def models_scan(
    model_roots: list[Path] | None = typer.Option(None, "--model-root"),
    cache_root: Path = typer.Option(Path("_db/catalog"), "--cache-root"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Discover local GGUF files without contacting Hugging Face."""
    roots = model_roots or list(load_config().paths.model_roots)
    snapshot = ModelCatalog(cache_root=cache_root).build(discover_models(roots), enrich=False)
    paths = write_catalog(snapshot, cache_root)
    if json_out:
        _print_json(snapshot.to_dict())
        return
    console.print(f"Cataloged {len(snapshot.entries)} models without network access.")
    console.print(f"JSON: {paths.json}")
    console.print(f"Markdown: {paths.markdown}")
    console.print(f"Recommendations DB: {paths.recommendations}")


@models_app.command("enrich")
def models_enrich(
    model_roots: list[Path] | None = typer.Option(None, "--model-root"),
    cache_root: Path = typer.Option(Path("_db/catalog"), "--cache-root"),
    offline: bool = typer.Option(False, "--offline"),
    llama_server: Path | None = typer.Option(None, "--llama-server"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Retrieve revision-pinned Hugging Face evidence or use the offline cache."""
    roots = model_roots or list(load_config().paths.model_roots)
    gateway = None if offline else HuggingFaceGateway(cache_dir=cache_root / "transport")
    hub = HubCatalog(
        gateway=gateway,
        cache_root=cache_root / "hub",
        offline=offline,
    )
    capabilities = inspect_llama_executable(llama_server) if llama_server is not None else None
    snapshot = ModelCatalog(
        cache_root=cache_root,
        hub=hub,
        capabilities=capabilities,
    ).build(discover_models(roots), enrich=True)
    paths = write_catalog(snapshot, cache_root)
    if json_out:
        _print_json(snapshot.to_dict())
        return
    console.print(f"Enriched {len(snapshot.entries)} models.")
    console.print(f"JSON: {paths.json}")
    console.print(f"Markdown: {paths.markdown}")
    console.print(f"Recommendations DB: {paths.recommendations}")


@models_app.command("list")
def models_list(
    cache_root: Path = typer.Option(Path("_db/catalog"), "--cache-root"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """List the latest persisted catalog."""
    snapshot = load_catalog(cache_root)
    if json_out:
        _print_json(snapshot.to_dict())
        return
    table = Table(title="PilotBENCHY Model Catalog")
    for column in ("Model", "Repository", "Quant", "Identity", "Documents"):
        table.add_column(column)
    for entry in snapshot.entries:
        table.add_row(
            entry.name,
            entry.repo_id or "unresolved",
            entry.quant,
            entry.identity_confidence,
            entry.document_confidence,
        )
    console.print(table)


@models_app.command("show")
def models_show(
    selector: str,
    cache_root: Path = typer.Option(Path("_db/catalog"), "--cache-root"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Explain one model's provenance and evidence."""
    entry = find_catalog_entry(load_catalog(cache_root), selector)
    if json_out:
        _print_json(entry.to_dict())
        return
    console.print_json(data=entry.to_dict())


@models_app.command("recommendations")
def models_recommendations(
    selector: str,
    cache_root: Path = typer.Option(Path("_db/catalog"), "--cache-root"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show publisher claims and local validation separately."""
    entry = find_catalog_entry(load_catalog(cache_root), selector)
    payload = [asdict(item) for item in entry.recommendations]
    if json_out:
        _print_json(payload)
        return
    if not payload:
        console.print("No recommendations are available for this catalog entry.")
        return
    console.print_json(data=payload)


@models_app.command("export")
def models_export(
    output_dir: Path,
    cache_root: Path = typer.Option(Path("_db/catalog"), "--cache-root"),
) -> None:
    """Export the latest catalog as deterministic JSON and Markdown."""
    paths = write_catalog(load_catalog(cache_root), output_dir)
    console.print(f"JSON: {paths.json}")
    console.print(f"Markdown: {paths.markdown}")
    console.print(f"Recommendations DB: {paths.recommendations}")


def _effective_forced_server_args(custom_args: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Return standard always-on llama-server flags plus user-locked extras.

    User extras are additive so template choices such as ``--jinja`` or
    ``--chat-template-file`` stay locked across every profile without dropping
    the current GPU's standard baseline.
    """
    standard_args = recommended_always_on(detect_gpu_name())
    return validate_extra_server_args(tuple(standard_args) + tuple(custom_args))


@app.callback()
def main(
    ctx: typer.Context,
    start_now: bool = typer.Option(
        False,
        "--start",
        help="Open the easy model picker.",
    ),
    first_run_now: bool = typer.Option(
        False,
        "--first-run",
        help="Set up the app, install apb, then open the model picker.",
    ),
    check_only: bool = typer.Option(
        False,
        "--check-only",
        help="Only check the computer. Do not open the picker.",
    ),
) -> None:
    """Open the app. The first run sets itself up; after that it just launches.

    Power-user subcommands (doctor, autoresearch, results, ...) still work; run
    `apb --help` to see them.
    """
    # A subcommand was given (e.g. `apb doctor`): let it handle everything.
    if ctx.invoked_subcommand is not None:
        return
    # Bare `apb` is the front door. Set up on the very first run, then launch.
    # Explicit --first-run always re-runs setup; --start always skips it.
    if find_config_path() is None:
        console.print(
            f"[yellow]No _CONFIG.toml found from {Path.cwd()} — receipts and app "
            "state will be created under this directory. Run apb from the "
            "pilotBENCHY folder (or via the installed apb shim) to keep results "
            "in one place.[/yellow]"
        )
    config = load_config()
    needs_setup = first_run_now or (not start_now and not is_setup_complete(project_root()))
    if needs_setup:
        if not first_run_now:
            console.print("First run detected. Setting up Agent Pilot (one time)...")
        _setup_app(
            root=config.paths.model_roots[0],
            llama_bench=config.paths.llama_bench,
            llama_cli=config.paths.llama_cli,
            llama_server=config.paths.llama_server,
            llama_perplexity=config.paths.llama_perplexity,
            runs_root=config.paths.runs_root,
            db_path=DEFAULT_DB_PATH,
            shim_dir=DEFAULT_SHIM_DIR,
            skip_env_sync=False,
            install_command=True,
            add_to_path=True,
            json_out=False,
        )
    _start_app(
        root=config.paths.model_roots[0],
        check_only=check_only,
        llama_bench=config.paths.llama_bench,
        llama_cli=config.paths.llama_cli,
        llama_server=config.paths.llama_server,
        llama_perplexity=config.paths.llama_perplexity,
        runs_root=config.paths.runs_root,
        preset=config.benchmark.default_preset,
        parallel_max=config.benchmark.parallel_max,
        learning=config.benchmark.learning,
        workflow_eval=config.benchmark.workflow_eval,
        ttft_probe=config.benchmark.ttft_probe,
    )
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
    llama_perplexity: Path | None = None,
    runs_root: Path | None = None,
    budget_minutes: int | None = None,
    parallel_max: int | None = None,
    max_attempts: int | None = None,
    learning: bool = True,
    workflow_eval: bool = True,
    ttft_probe: bool = True,
    context_ladder: list[int] | None = typer.Option(
        None,
        "--context-ladder",
        help="Add a fixed context tier to profile after the best settings are found. Repeatable.",
    ),
    benchmark_suite_plan: Path | None = typer.Option(
        None,
        "--benchmark-suite-plan",
        help="Run a benchmark-suite plan for selected models and optimize by agent_bench_score.",
    ),
    required_context: int | None = typer.Option(
        None,
        "--required-context",
        min=1,
        help="Minimum context that must be proven before Web recommendations can call the stack ready.",
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
        llama_perplexity=llama_perplexity,
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
        context_ladder=context_ladder,
        benchmark_suite_plan=benchmark_suite_plan,
        required_context=required_context,
        preset=config.benchmark.default_preset,
    )


def _start_app(
    root: Path,
    check_only: bool,
    llama_bench: Path = DEFAULT_LLAMA_BENCH,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    llama_server: Path = DEFAULT_LLAMA_SERVER,
    llama_perplexity: Path = DEFAULT_LLAMA_PERPLEXITY,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    budget_minutes: int | None = None,
    parallel_max: int = 4,
    max_attempts: int | None = None,
    learning: bool = True,
    workflow_eval: bool = True,
    ttft_probe: bool = True,
    context_ladder: list[int] | None = None,
    benchmark_suite_plan: Path | None = None,
    required_context: int | None = None,
    preset: str = "deep",
) -> None:
    report = build_doctor_report(
        model_roots=[root],
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        llama_perplexity=llama_perplexity,
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
    console.print("Opening the browser cockpit.")
    # The web server is a thin client: it only passes instructions to a detached
    # engine process (spawned per run) and renders the run directory. All
    # evaluation logic lives in the `engine` command's run_model.
    serve_webui(
        root=root,
        runs_root=runs_root,
        llama_server=llama_server,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_perplexity=llama_perplexity,
        required_context=required_context,
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
    llama_perplexity: Path | None = None,
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
        llama_perplexity=llama_perplexity,
        runs_root=runs_root,
    )
    roots = list(config.paths.model_roots)
    report = build_doctor_report(
        model_roots=roots,
        llama_bench=config.paths.llama_bench,
        llama_cli=config.paths.llama_cli,
        llama_server=config.paths.llama_server,
        llama_perplexity=config.paths.llama_perplexity,
        runs_root=config.paths.runs_root,
    )
    if json_out:
        _print_json({**report.to_dict(), "resolved_config": config.to_dict()})
    else:
        _print_doctor_report(report)
    if strict and not report.ready:
        typer.echo("Required checks failed.", err=True)
        raise typer.Exit(1)


def _autoconfigure_paths(
    config,
    *,
    detect_models: Callable[[], list[Path]] = lambda: find_model_roots(
        default_model_search_roots()
    ),
    detect_binaries: Callable[[], dict[str, Path]] = lambda: find_llama_binaries(
        default_llama_search_roots()
    ),
    persist: Callable[[dict[str, str]], object] = persist_user_env,
):
    """Detect missing model/llama paths, save them, and return updated config.

    Only paths that are not already present are detected, so a machine that
    already has working env vars (or a real ``_CONFIG.toml``) is left untouched.
    """
    detected: dict[str, str] = {}

    if not any(root.exists() for root in config.paths.model_roots):
        model_roots = detect_models()
        if model_roots:
            detected["PILOTBENCH_MODEL_ROOTS"] = os.pathsep.join(str(path) for path in model_roots)

    configured_binaries = {
        "llama-server": config.paths.llama_server,
        "llama-bench": config.paths.llama_bench,
        "llama-cli": config.paths.llama_cli,
        "llama-perplexity": config.paths.llama_perplexity,
    }
    missing_binaries = {stem for stem, path in configured_binaries.items() if not path.exists()}
    if missing_binaries:
        found = detect_binaries()
        for stem in missing_binaries:
            if stem in found:
                detected[LLAMA_ENV_VARS[stem]] = str(found[stem])

    if not detected:
        return config, []

    step = persist(detected)
    # Re-load so the freshly persisted env vars flow through resolution.
    return load_config(), [step]


def _setup_app(
    *,
    root: Path | None,
    llama_bench: Path | None,
    llama_cli: Path | None,
    llama_server: Path | None,
    llama_perplexity: Path | None,
    runs_root: Path | None,
    db_path: Path,
    shim_dir: Path,
    skip_env_sync: bool,
    install_command: bool,
    add_to_path: bool,
    json_out: bool,
) -> None:
    """Install the local command, sync dependencies, and prepare app state."""
    config = with_cli_overrides(
        load_config(),
        model_roots=[root] if root is not None else None,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        llama_perplexity=llama_perplexity,
        runs_root=runs_root,
    )
    repo_root = project_root()
    actual_shim_dir = resolved_shim_dir(repo_root, shim_dir)
    install_steps = [sync_project_environment(repo_root=repo_root, skip=skip_env_sync)]
    if install_command:
        install_steps.extend(install_command_shims(repo_root=repo_root, shim_dir=shim_dir))
        install_steps.append(
            add_shim_dir_to_user_path(actual_shim_dir)
            if add_to_path
            else check_user_path(actual_shim_dir)
        )

    # Auto-detect model folders and llama.cpp binaries for a fresh machine and
    # save them, so `apb` opens against a real model instead of dead-ending.
    # Gated on add_to_path: that is the "real install" flag (tests pass
    # --no-add-to-path), and it persists user env vars just like the PATH step.
    if add_to_path:
        config, detect_steps = _autoconfigure_paths(config)
        install_steps.extend(detect_steps)

    report = build_doctor_report(
        model_roots=[config.paths.model_roots[0]],
        llama_bench=config.paths.llama_bench,
        llama_cli=config.paths.llama_cli,
        llama_server=config.paths.llama_server,
        llama_perplexity=config.paths.llama_perplexity,
        runs_root=config.paths.runs_root,
    )
    init_state_db(db_path)
    write_leaderboard(config.paths.runs_root)
    install_ready = all(step.ok for step in install_steps if step.required)
    if install_ready:
        # Record that the app is installed so future bare `apb` calls launch
        # straight away instead of re-running first-run setup.
        mark_setup_complete(repo_root)
    payload = {
        **report.to_dict(),
        "install_ready": install_ready,
        "install_steps": [step.to_dict() for step in install_steps],
        "db_path": str(db_path),
        "runs_root": str(config.paths.runs_root),
        "resolved_config": config.to_dict(),
        "next_command": ("apb" if report.ready and install_ready else "apb doctor"),
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
    if not report.ready or not install_ready:
        raise typer.Exit(1)


@app.command("setup")
def setup(
    root: Path | None = typer.Option(
        None,
        help="Folder where your GGUF models live.",
    ),
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    llama_perplexity: Path | None = None,
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
        True,
        "--add-to-path/--no-add-to-path",
        help="Add the command shim folder to the Windows user PATH.",
    ),
    json_out: bool = False,
) -> None:
    """Set up the app, command shims, local state, and readiness checks."""
    _setup_app(
        root=root,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        llama_perplexity=llama_perplexity,
        runs_root=runs_root,
        db_path=db_path,
        shim_dir=shim_dir,
        skip_env_sync=skip_env_sync,
        install_command=install_command,
        add_to_path=add_to_path,
        json_out=json_out,
    )


@app.command("first-run", hidden=True)
def first_run(
    root: Path | None = typer.Option(None, help="Folder where your GGUF models live."),
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    llama_perplexity: Path | None = None,
    runs_root: Path | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    shim_dir: Path = typer.Option(DEFAULT_SHIM_DIR, help="Folder where shims are installed."),
    skip_env_sync: bool = typer.Option(False, "--skip-env-sync"),
    install_command: bool = typer.Option(True, "--install-command/--no-install-command"),
    add_to_path: bool = typer.Option(True, "--add-to-path/--no-add-to-path"),
    json_out: bool = False,
) -> None:
    """Compatibility alias for setup."""
    setup(
        root=root,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        runs_root=runs_root,
        db_path=db_path,
        shim_dir=shim_dir,
        skip_env_sync=skip_env_sync,
        install_command=install_command,
        add_to_path=add_to_path,
        json_out=json_out,
    )


@app.command()
def survey(
    root: Path | None = None,
    qwen_only: bool = False,
    qwen_35b_only: bool = False,
    mtp_only: bool = False,
    json_out: bool = False,
) -> None:
    """List every GGUF model found under the model root(s)."""
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
    target_model: str | None = typer.Option(
        None,
        "--target-model",
        help="Scope the displayed decision packet to one intended model name or GGUF basename.",
    ),
    target_model_path: Path | None = typer.Option(
        None,
        "--target-model-path",
        help="GGUF path to use for target deployment flag proof commands.",
    ),
    required_context: int | None = typer.Option(
        None,
        "--required-context",
        min=1,
        help="Minimum context that must be proven before the result can call a stack ready.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open-browser",
        help="Open the generated browser report.",
    ),
    serve: bool = typer.Option(
        False,
        "--serve",
        help="Serve the results folder on localhost until Ctrl+C.",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        help="Localhost port for --serve.",
    ),
) -> None:
    """Show the latest leaderboard and verdict in normal language."""
    config = with_cli_overrides(load_config(), runs_root=runs_root)
    runs_root = config.paths.runs_root
    leaderboard = write_leaderboard(runs_root)
    hard_outputs = write_hard_recommendations(
        runs_root,
        target_model=target_model,
        target_model_path=str(target_model_path) if target_model_path is not None else None,
        required_context=required_context,
    )
    hard_payload = hard_outputs.payload
    target_scope = hard_payload.get("target_scope") or {}
    if target_model and target_scope.get("status") == "NO_TARGET_EVIDENCE":
        if json_out:
            _print_json(
                {
                    "schema_version": 2,
                    "result_label": "no_target_evidence",
                    "model": None,
                    "top_candidate": None,
                    "verdict": hard_payload.get("model_gate"),
                    "decision_packet": hard_payload,
                    "artifacts": {
                        "hard_recommendations": str(hard_outputs.markdown_path),
                    },
                }
            )
            return
        console.print(f"No benchmark receipts found for target model: {target_model}")
        console.print(
            "Target scope: "
            f"{target_scope['target_model']} | {target_scope['status']} | "
            f"matched {target_scope['matched_receipt_count']}, "
            f"ignored {target_scope['ignored_receipt_count']}"
        )
        operator = _operator_verdict_payload(hard_payload)
        console.print(f"Operator verdict: {operator['status']} - {operator['headline']}")
        if hard_payload.get("proof_runbook"):
            console.print("Proof runbook:")
            for step in hard_payload["proof_runbook"]:
                console.print(
                    f"{step['step']}. [{step['gate']}/{step['id']}] "
                    f"{step['status']} -> {step['proves']}",
                    markup=False,
                )
        for command in hard_payload.get("proof_commands", []):
            command_id = command.get("id", command["gate"])
            typer.echo(f"Proof command ({command['gate']}/{command_id}): {command['command']}")
        console.print(f"Hard recommendations: {hard_outputs.markdown_path}")
        return
    if not leaderboard.entries:
        console.print("No benchmark receipts found yet.")
        return
    champion = leaderboard.champion
    verdict = build_verdict(leaderboard)
    audit = build_report_audit(leaderboard)
    if json_out:
        _print_json(
            {
                "schema_version": 2,
                "result_label": (
                    "recommended_model" if verdict.action == "PROMOTE" else "top_candidate"
                ),
                "model": asdict(champion),
                "top_candidate": asdict(champion),
                "verdict": asdict(verdict),
                "report_audit": asdict(audit),
                "decision_packet": {
                    "operator_verdict": hard_payload.get("operator_verdict"),
                    "score_evidence": hard_payload.get("score_evidence"),
                    "performance_prediction": hard_payload.get("performance_prediction"),
                    "candidate_assessment": hard_payload.get("candidate_assessment"),
                    "candidate_rankings": hard_payload.get("candidate_rankings", []),
                    "settings_candidates": hard_payload.get("settings_candidates", []),
                    "repeatability": hard_payload.get("repeatability"),
                    "context_gate": hard_payload.get("context_gate"),
                    "resource_gate": hard_payload.get("resource_gate"),
                    "stability_gate": hard_payload.get("stability_gate"),
                    "proof_runbook": hard_payload.get("proof_runbook", []),
                    "proof_commands": hard_payload.get("proof_commands", []),
                    "hard_recommendations": hard_payload.get("hard_recommendations", []),
                    "proven_components": hard_payload.get("proven_components", []),
                    "overall_action": hard_payload.get("overall_action"),
                },
                "artifacts": {
                    "leaderboard": str(runs_root / "leaderboard.md"),
                    "verdict": str(runs_root / "verdict.md"),
                    "hard_recommendations": str(hard_outputs.markdown_path),
                    "report_audit": str(runs_root / "report-audit.md"),
                    "html": str(runs_root / "results.html"),
                    "legacy_champion": str(runs_root / "champion.json"),
                },
            }
        )
        return
    model_label = "Recommended model" if verdict.action == "PROMOTE" else "Top candidate"
    console.print(f"{model_label}: {champion.model_name}")
    _print_score_summary(score_summary_for_entry(champion))
    console.print(f"Verdict: {verdict.action} ({verdict.confidence} confidence)")
    console.print(f"Report audit: {audit.status} ({audit.warning_count} warning(s))")
    for warning in audit.warnings:
        console.print(f"Audit warning: {warning['code']} in {warning['run_id']}")
    console.print(f"Predicted quality: {verdict.prediction['quality']}")
    console.print(f"Predicted speed: {verdict.prediction['speed']}")
    console.print(f"Predicted context: {verdict.prediction['context']}")
    console.print(f"Recommendation class: {verdict.prediction['recommendation']}")
    assessment = hard_payload.get("candidate_assessment") or {}
    performance = assessment.get("known_performance") or {}
    console.print(
        "Candidate readiness: "
        f"{assessment.get('readiness', 'unknown')} "
        f"({int(assessment.get('readiness_score') or 0)}/100)"
    )
    console.print(
        "Candidate performance: "
        f"quality={performance.get('quality', 'unmeasured')} "
        f"speed={performance.get('speed', 'unmeasured')} "
        f"context={performance.get('context_class', 'unmeasured')}"
    )
    if hard_payload.get("candidate_rankings"):
        console.print("Candidate rankings:")
        for candidate in hard_payload["candidate_rankings"][:3]:
            prediction = candidate.get("prediction") or {}
            gaps = ", ".join(candidate.get("evidence_gaps") or []) or "none"
            console.print(
                f"#{candidate['rank']} {candidate['model']} | "
                f"{candidate['status']} | "
                f"agent={_format_optional_float(candidate.get('agent_quality_score'))} | "
                f"{prediction.get('quality', 'unmeasured')}/"
                f"{prediction.get('speed', 'unmeasured')}/"
                f"{prediction.get('context', 'unmeasured')} | "
                f"gaps={gaps}"
            )
    for line in _settings_candidate_lines(hard_payload.get("settings_candidates", [])):
        console.print(line)
    if hard_payload.get("repeatability"):
        console.print(_repeatability_cli_line(hard_payload["repeatability"]))
    console.print(_context_gate_cli_line(hard_payload.get("context_gate")))
    console.print(_resource_gate_cli_line(hard_payload.get("resource_gate")))
    if hard_payload.get("proof_runbook"):
        console.print("Proof runbook:")
        for step in hard_payload["proof_runbook"]:
            console.print(
                f"{step['step']}. [{step['gate']}/{step['id']}] "
                f"{step['status']} -> {step['proves']}",
                markup=False,
            )
    console.print(verdict.summary)
    console.print(f"Next run: {verdict.next_run}")
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
    console.print(f"Verdict report: {runs_root / 'verdict.md'}")
    console.print(f"Hard recommendations: {hard_outputs.markdown_path}")
    console.print(f"Report audit: {runs_root / 'report-audit.md'}")
    report_path = runs_root / "results.html"
    console.print(f"HTML report: {report_path}")
    if open_browser:
        url = report_path.resolve().as_uri()
        webbrowser.open(url)
        console.print(f"Opened browser report: {url}")
    if serve:
        serve_root = runs_root.resolve()
        url = f"http://127.0.0.1:{port}/results.html"
        console.print(f"Serving report: {url}")
        console.print("Press Ctrl+C to stop the report server.")
        _serve_report_directory(serve_root, port)


def _leaderboard_summary_line(leaderboard) -> str:
    verdict = build_verdict(leaderboard)
    label = "Recommended model" if verdict.action == "PROMOTE" else "Top candidate"
    return f"{label}: {leaderboard.champion.model_name} ({leaderboard.champion.score:.2f})"


@app.command("export-plan")
def export_plan(
    run: Annotated[
        Path,
        typer.Option(
            "--run",
            help="Receipt folder, or a direct path to resolved-plan.json.",
        ),
    ],
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional file to write the resolved plan JSON to.",
    ),
    json_out: bool = False,
) -> None:
    """Print or copy the resolved plan saved by a benchmark receipt."""
    plan_path = run if run.name == "resolved-plan.json" else run / "resolved-plan.json"
    if not plan_path.exists():
        typer.echo(f"Resolved plan not found: {plan_path}", err=True)
        raise typer.Exit(2)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"Resolved plan is not valid JSON: {exc}", err=True)
        raise typer.Exit(2) from exc
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        console.print(f"Resolved plan exported: {output}")
        return
    if json_out:
        _print_json(payload)
        return
    console.print(f"Resolved plan: {plan_path}")
    command_path = plan_path.parent / "command.txt"
    if command_path.exists():
        first_command = command_path.read_text(encoding="utf-8").splitlines()[0]
        console.print(f"Command: {first_command}")
    console.print("Use --json-out to print the full plan JSON.")


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


@app.command("question-packs")
def question_packs(
    json_out: bool = False,
    librarian_only: bool = typer.Option(
        False,
        "--librarian-only",
        help="Show only the memory/RAG librarian packs used by librarian-bench.",
    ),
) -> None:
    """List question packs that can be used for champion evaluation."""
    ids = tuple(LIBRARIAN_PACK_IDS) if librarian_only else available_packs()
    rows = []
    for pack_id in ids:
        pack = load_pack(pack_id)
        rows.append(
            {
                "pack_id": pack.pack_id,
                "title": pack.title,
                "tier": pack.tier,
                "answer_type": pack.answer_type.value,
                "questions": str(len(pack.questions)),
            }
        )

    if json_out:
        _print_json(rows)
        return

    table = Table(title="Question Packs")
    table.add_column("Pack")
    table.add_column("Tier")
    table.add_column("Answer")
    table.add_column("Questions", justify="right")
    table.add_column("Title")
    for row in rows:
        table.add_row(
            row["pack_id"],
            row["tier"],
            row["answer_type"],
            row["questions"],
            row["title"],
        )
    console.print(table)


@app.command("qe-format")
def qe_format(
    model: Annotated[str, typer.Option(help="Model label to record in the QE receipt.")],
    base_url: Annotated[
        str,
        typer.Option(help="Base URL of the running QE llama.cpp server."),
    ] = "http://127.0.0.1:8080",
    runs_root: Annotated[
        Path, typer.Option(help="Directory where QE receipts are written.")
    ] = DEFAULT_RUNS_ROOT,
    repeats: Annotated[
        int,
        typer.Option(min=1, help="Fresh sessions per QE prompt."),
    ] = 10,
    max_tokens: Annotated[
        int,
        typer.Option(min=1, help="Maximum generated tokens per QE attempt."),
    ] = 128,
    timeout_seconds: Annotated[
        int,
        typer.Option(min=1, help="Per-request timeout for the QE endpoint."),
    ] = 600,
    temperature: Annotated[
        float,
        typer.Option(min=0.0, help="QE sampling temperature."),
    ] = 0.1,
    top_p: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="QE nucleus sampling top-p."),
    ] = 0.8,
    min_p: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="QE min-p sampling cutoff."),
    ] = 0.02,
    repeat_penalty: Annotated[
        float,
        typer.Option(min=0.0, help="QE repeat penalty."),
    ] = 1.05,
    dry_multiplier: Annotated[
        float,
        typer.Option(min=0.0, help="QE DRY repetition penalty multiplier."),
    ] = 0.6,
) -> None:
    """Run QE fresh-session format checks against a live endpoint."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = runs_root / f"{stamp}-qe-format-{_safe_receipt_slug(model)}"
    sampling: dict[str, object] = {
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
        "dry_multiplier": dry_multiplier,
    }
    summary = run_qe_format_suite(
        model=model,
        base_url=base_url,
        out_dir=out_dir,
        repeats=repeats,
        answer_max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        sampling=sampling,
    )
    console.print(f"QE format receipt: {out_dir}")
    console.print(f"Score: {_format_optional_float(_float_from_summary(summary, 'score'))}")
    console.print(
        f"Format rate: {_format_optional_float(_float_from_summary(summary, 'format_rate'))}"
    )
    console.print(
        "Direct-answer rate: "
        f"{_format_optional_float(_float_from_summary(summary, 'direct_answer_rate'))}"
    )
    console.print(f"Attempts: {summary['attempts']}")


@app.command("qe-results")
def qe_results(
    runs_root: Annotated[
        Path, typer.Option(help="Directory where QE receipts are stored.")
    ] = DEFAULT_RUNS_ROOT,
    json_out: bool = False,
) -> None:
    """Rank QE fresh-session receipts and print the current QE gate result."""
    leaderboard = write_qe_leaderboard(runs_root)
    if json_out:
        typer.echo((runs_root / "qe-format-leaderboard.json").read_text(encoding="utf-8"))
        return
    if leaderboard.champion is None:
        console.print("No QE format receipts found yet.")
        console.print("Next run: apb qe-format --model MODEL --base-url http://127.0.0.1:PORT")
        return
    champion = leaderboard.champion
    label = "QE champion" if champion.action == "PROMOTE_QE_PROFILE" else "QE top candidate"
    console.print(f"{label}: {champion.model}")
    console.print(f"Action: {champion.action}")
    console.print(f"Recommendation: {champion.recommendation}")
    console.print(f"Next run: {champion.next_run}")
    console.print(
        "Score: "
        f"{champion.score:.6f} | Format: {champion.format_rate:.6f} | "
        f"Direct-answer: {champion.direct_answer_rate:.6f}"
    )
    console.print(f"Attempts: {champion.attempts}")
    console.print(f"Median generation: {_format_optional_tps(champion.median_tps)}")
    console.print(f"Median TTFT: {_format_optional_ms(champion.median_ttft_ms)}")
    console.print(f"Receipt: {champion.receipt_path}")
    console.print(f"Leaderboard written: {runs_root / 'qe-format-leaderboard.md'}")


@app.command("flag-recommendations")
def flag_recommendations(
    model: Annotated[Path, typer.Option(help="GGUF model to generate llama-server flags for.")],
    llama_server: Path | None = None,
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where flag-recommendations artifacts are written."),
    ] = DEFAULT_RUNS_ROOT,
    root: list[Path] | None = typer.Option(
        None,
        "--root",
        help="Optional model/template search root. Repeatable.",
    ),
    host: str = "127.0.0.1",
    port: int = 8080,
    json_out: bool = False,
) -> None:
    """Write recommended llama.cpp serving modes for a model or QE helper."""
    config = with_cli_overrides(load_config(), llama_server=llama_server, model_roots=root)
    outputs = write_flag_recommendations(
        model=model,
        llama_server=config.paths.llama_server,
        output_dir=output_dir,
        gpu_name=detect_gpu_name(),
        search_roots=tuple(config.paths.model_roots),
        host=host,
        port=port,
    )
    if json_out:
        typer.echo(outputs.json_path.read_text(encoding="utf-8"))
        return
    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    console.print(f"Flag recommendations written: {outputs.markdown_path}")
    console.print(f"JSON: {outputs.json_path}")
    console.print(f"Lane type: {payload['lane_type']}")
    for profile in payload["profiles"]:
        console.print(
            f"{profile['label']}: ctx {profile['context_size']} | "
            f"KV K={profile['kv_cache']['k']} V={profile['kv_cache']['v']} | "
            f"parallel {profile['parallel']}"
        )


@app.command("deployment-readiness")
def deployment_readiness(
    runs_root: Path | None = None,
    json_out: bool = False,
) -> None:
    """Gate flag recommendations against score, context, and serving evidence."""
    config = with_cli_overrides(load_config(), runs_root=runs_root)
    outputs = write_deployment_readiness(config.paths.runs_root)
    if json_out:
        typer.echo(outputs.json_path.read_text(encoding="utf-8"))
        return
    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    console.print(f"Deployment readiness: {payload['action']}")
    console.print(f"Recommended profile: {payload['recommended_profile_id'] or 'none'}")
    console.print(str(payload["summary"]))
    console.print(f"Next run: {payload['next_run']}")
    console.print(f"Report: {outputs.markdown_path}")
    for profile in payload["profiles"]:
        console.print(
            f"{profile['label']}: {profile['status']} at ctx {profile['context_size']} "
            f"({profile['reason']})"
        )


@app.command("deployment-proof")
def deployment_proof(
    runs_root: Path | None = None,
    llama_server: Path | None = None,
    profile: str = typer.Option(
        "standard", "--profile", help="Profile id from flag-recommendations.json."
    ),
    flag_recommendations: Path | None = typer.Option(
        None,
        "--flag-recommendations",
        help="Path to flag-recommendations.json. Defaults to RUNS_ROOT/flag-recommendations.json.",
    ),
    benchmark_suite_plan: Path = typer.Option(
        Path("benchmark-suite.plan.json"),
        "--benchmark-suite-plan",
        help="Scored benchmark-suite plan required for deployment proof.",
    ),
    budget_minutes: int = typer.Option(30, "--budget-minutes", min=1),
    simple_bench: Path = typer.Option(
        DEFAULT_SIMPLE_BENCH_PATH,
        "--simple-bench",
        help="SimpleBench JSON file used for the serving pass.",
    ),
    simple_bench_system_prompt: Path = typer.Option(
        DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT,
        "--simple-bench-system-prompt",
        help="System prompt prepended before SimpleBench questions.",
    ),
    simple_bench_max_tokens: int = typer.Option(
        DEFAULT_DEPLOYMENT_SIMPLE_BENCH_MAX_TOKENS,
        "--simple-bench-max-tokens",
        min=1,
        help="Maximum generated tokens per SimpleBench question.",
    ),
) -> None:
    """Run one exact flag-recommendation profile and write score plus serving evidence."""
    config = with_cli_overrides(load_config(), runs_root=runs_root, llama_server=llama_server)
    try:
        receipt = run_deployment_proof(
            runs_root=config.paths.runs_root,
            profile_id=profile,
            flag_recommendations_path=flag_recommendations,
            benchmark_suite_plan=benchmark_suite_plan,
            llama_server=config.paths.llama_server,
            simple_bench=simple_bench,
            simple_bench_system_prompt=simple_bench_system_prompt,
            budget_seconds=budget_minutes * 60,
            simple_bench_max_tokens=simple_bench_max_tokens,
        )
    except BenchmarkSuitePreflightError as exc:
        console.print(f"Deployment proof preflight failed: {exc}")
        console.print(f"Preflight receipt: {exc.receipt_path}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"Deployment proof failed: {exc}")
        raise typer.Exit(1) from exc
    readiness_outputs = write_deployment_readiness(config.paths.runs_root)
    receipt_target = _deployment_proof_target(receipt.path)
    hard_outputs = write_hard_recommendations(
        config.paths.runs_root,
        target_model_path=receipt_target.get("model"),
        required_context=receipt_target.get("context_size"),
    )
    readiness_payload = json.loads(readiness_outputs.json_path.read_text(encoding="utf-8"))
    hard_payload = hard_outputs.payload
    typer.echo(f"Deployment proof receipt: {receipt.path}")
    typer.echo(f"Deployment readiness: {readiness_payload['action']}")
    typer.echo(f"Recommended profile: {readiness_payload.get('recommended_profile_id') or 'none'}")
    typer.echo(f"Readiness report: {readiness_outputs.markdown_path}")
    typer.echo(f"Hard recommendations: {hard_payload['overall_action']}")
    operator = _operator_verdict_payload(hard_payload)
    score_evidence = _score_evidence_payload(hard_payload)
    typer.echo(f"Operator verdict: {operator['status']} - {operator['headline']}")
    prediction = _performance_prediction_payload(hard_payload)
    typer.echo(f"Performance prediction: {prediction['status']} ({prediction['risk']} risk)")
    typer.echo(f"Deployment expectation: {prediction['deployment_expectation']}")
    typer.echo(
        "Scored candidates: "
        f"{score_evidence['scored_candidate_count']}/{score_evidence['candidate_count']}"
    )
    typer.echo(f"Proven recommendations: {len(hard_payload['hard_recommendations'])}")
    for line in _settings_candidate_lines(hard_payload.get("settings_candidates", [])):
        typer.echo(line)
    typer.echo(f"Hard recommendation report: {hard_outputs.markdown_path}")


def _deployment_proof_target(receipt_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads((receipt_path / "best-settings.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw_settings = payload.get("settings")
    settings = raw_settings if isinstance(raw_settings, dict) else {}
    target: dict[str, Any] = {}
    model = str(payload.get("model") or "")
    if model:
        target["model"] = model
    context_size = _int_from_object(settings.get("context_size"))
    if context_size > 0:
        target["context_size"] = context_size
    return target


@app.command("hard-recommendations")
def hard_recommendations(
    runs_root: Path | None = None,
    json_out: bool = False,
    target_model: str | None = typer.Option(
        None,
        "--target-model",
        help="Scope the recommendation report to one intended model name or GGUF basename.",
    ),
    target_model_path: Path | None = typer.Option(
        None,
        "--target-model-path",
        help="GGUF path to use for target deployment flag proof commands.",
    ),
    required_context: int | None = typer.Option(
        None,
        "--required-context",
        min=1,
        help="Minimum context that must be proven before the stack is ready.",
    ),
) -> None:
    """Write one consolidated hard-recommendation report from all evidence gates."""
    config = with_cli_overrides(load_config(), runs_root=runs_root)
    outputs = write_hard_recommendations(
        config.paths.runs_root,
        target_model=target_model,
        target_model_path=str(target_model_path) if target_model_path is not None else None,
        required_context=required_context,
    )
    if json_out:
        _print_json(outputs.payload)
        return
    payload = outputs.payload
    console.print(f"Hard recommendations: {payload['overall_action']}")
    target_scope = payload.get("target_scope") or {}
    if target_scope.get("target_model"):
        console.print(
            "Target scope: "
            f"{target_scope['target_model']} | {target_scope['status']} | "
            f"matched {target_scope['matched_receipt_count']}, "
            f"ignored {target_scope['ignored_receipt_count']}"
        )
    operator = _operator_verdict_payload(payload)
    score_evidence = _score_evidence_payload(payload)
    console.print(f"Operator verdict: {operator['status']} - {operator['headline']}")
    console.print(f"Operator reason: {operator['why']}")
    prediction = _performance_prediction_payload(payload)
    console.print(f"Performance prediction: {prediction['status']} ({prediction['risk']} risk)")
    console.print(f"Deployment expectation: {prediction['deployment_expectation']}")
    console.print(
        "Scored candidates: "
        f"{score_evidence['scored_candidate_count']}/{score_evidence['candidate_count']}"
    )
    console.print(f"Proven recommendations: {len(payload['hard_recommendations'])}")
    for line in _settings_candidate_lines(payload.get("settings_candidates", [])):
        console.print(line)
    console.print(f"Proof commands: {len(payload['proof_commands'])}")
    console.print(f"Model gate: {payload['model_gate']['action']}")
    console.print(f"Deployment gate: {payload['deployment_gate']['action']}")
    console.print(_context_gate_cli_line(payload.get("context_gate")))
    console.print(_resource_gate_cli_line(payload.get("resource_gate")))
    console.print(f"QE gate: {payload['qe_gate']['action']}")
    console.print(f"Stability gate: {payload['stability_gate']['action']}")
    assessment = payload["candidate_assessment"]
    performance = assessment["known_performance"]
    console.print(
        f"Candidate readiness: {assessment['readiness']} ({assessment['readiness_score']}/100)"
    )
    console.print(
        "Candidate performance: "
        f"quality={performance['quality']} "
        f"speed={performance['speed']} "
        f"context={performance['context_class']}"
    )
    if payload.get("candidate_rankings"):
        console.print("Candidate rankings:")
        for candidate in payload["candidate_rankings"][:3]:
            prediction = candidate.get("prediction") or {}
            gaps = ", ".join(candidate.get("evidence_gaps") or []) or "none"
            console.print(
                f"#{candidate['rank']} {candidate['model']} | "
                f"{candidate['status']} | "
                f"agent={_format_optional_float(candidate.get('agent_quality_score'))} | "
                f"{prediction.get('quality', 'unmeasured')}/"
                f"{prediction.get('speed', 'unmeasured')}/"
                f"{prediction.get('context', 'unmeasured')} | "
                f"gaps={gaps}"
            )
    if payload.get("repeatability"):
        console.print(_repeatability_cli_line(payload["repeatability"]))
    console.print(f"Quality: {payload['scorecard']['quality']}")
    console.print(f"Speed: {payload['scorecard']['speed']}")
    console.print(f"Context: {payload['scorecard']['context']}")
    console.print(f"Report: {outputs.markdown_path}")
    for action in payload["next_actions"]:
        console.print(f"Next action: {action}")
    if payload.get("proof_runbook"):
        console.print("Proof runbook:")
        for step in payload["proof_runbook"]:
            console.print(
                f"{step['step']}. [{step['gate']}/{step['id']}] "
                f"{step['status']} -> {step['proves']}",
                markup=False,
            )
    for command in payload["proof_commands"]:
        command_id = command.get("id", command["gate"])
        typer.echo(f"Proof command ({command['gate']}/{command_id}): {command['command']}")


@app.command("benchmark-suite-template")
def benchmark_suite_template(
    output: Path = Path("benchmark-suite.plan.json"),
    model: str = "local-model",
    base_url: str = "http://127.0.0.1:8080/v1",
    context: int = 131072,
    template_kind: str = typer.Option(
        "local_librarian",
        "--template-kind",
        help="Plan kind: local_librarian (default) or external.",
    ),
) -> None:
    """Write an editable recommendation-grade benchmark-suite plan."""
    if template_kind == "local_librarian":
        plan = _local_librarian_benchmark_suite_plan(
            model=model,
            base_url=base_url,
            context=context,
        )
    elif template_kind == "external":
        plan = _external_benchmark_suite_plan(model=model, base_url=base_url, context=context)
    else:
        raise typer.BadParameter("template_kind must be local_librarian or external")
    output.write_text(json.dumps(plan, ensure_ascii=True, indent=2), encoding="utf-8")
    console.print(f"Benchmark-suite plan written: {output}")
    console.print("Edit the task commands, then run: agent-autobench benchmark-suite --plan PLAN")


def _local_librarian_benchmark_suite_plan(*, model: str, base_url: str, context: int) -> dict:
    librarian_base_url = base_url.removesuffix("/v1")
    common_command = [
        "uv",
        "run",
        "--extra",
        "dev",
        "python",
        "-m",
        "gguf_limit_bench.librarian_suite",
        "--model",
        "{model}",
        "--base-url",
        "{base_url}",
        "--out-dir",
        "{task_dir}",
        "--score-out",
        "{task_dir}/score.json",
        "--settings-json",
        "{settings_json}",
        "--sample-size",
        "0",
        "--repeats",
        "3",
    ]
    return {
        "model": model,
        "context": context,
        "settings": {
            "base_url": librarian_base_url,
            "score_contract": "agent_bench_score",
            "context_target": f"required_context_{context}",
            "plan_kind": "local_librarian_template",
            "extra_server_args": ["--jinja"],
            "answer_max_tokens": DEFAULT_DEPLOYMENT_SIMPLE_BENCH_MAX_TOKENS,
            "requires": (
                "A live llama.cpp OpenAI-compatible endpoint plus the repo-local "
                "librarian-suite. No uv/uvx external harness command is required."
            ),
        },
        "tasks": [
            {
                "id": "local_librarian_general",
                "phase": "general",
                "harness": "librarian-suite",
                "commands": [
                    [
                        *common_command,
                        "--pack",
                        "librarian-write-entry",
                        "--pack",
                        "librarian-triage",
                        "--pack",
                        "librarian-dedupe",
                    ]
                ],
                "score_file": "{task_dir}/score.json",
                "min_score": 0.01,
                "timeout_seconds": 3600,
            },
            {
                "id": "local_librarian_agentic",
                "phase": "agentic",
                "harness": "librarian-suite",
                "commands": [
                    [
                        *common_command,
                        "--pack",
                        "librarian-gate",
                        "--pack",
                        "librarian-rerank",
                        "--pack",
                        "librarian-query",
                        "--pack",
                        "librarian-compress",
                        "--pack",
                        "librarian-contradiction",
                    ]
                ],
                "score_file": "{task_dir}/score.json",
                "min_score": 0.01,
                "timeout_seconds": 3600,
            },
        ],
    }


def _external_benchmark_suite_plan(*, model: str, base_url: str, context: int) -> dict:
    return {
        "model": model,
        "context": context,
        "settings": {
            "base_url": base_url,
            "score_contract": "agent_bench_score",
            "context_target": f"required_context_{context}",
        },
        "tasks": [
            {
                "id": "gsm8k_cot_zeroshot_smoke",
                "phase": "general",
                "harness": "lm-evaluation-harness",
                "env": {"PYTHONIOENCODING": "utf-8"},
                "commands": [
                    [
                        "uvx",
                        "--from",
                        "lm-eval",
                        "lm-eval",
                        "run",
                        "--model",
                        "local-chat-completions",
                        "--model_args",
                        "model={model},base_url={base_url}/chat/completions,eos_string=<|im_end|>",
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
                    "LOCAL_BASE_URL": "{base_url}",
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
                        "{base_url}",
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


@app.command("benchmark-suite-preflight")
def benchmark_suite_preflight(
    plan: Path = typer.Option(
        Path("benchmark-suite.plan.json"),
        "--plan",
        help="Benchmark-suite plan to preflight before launching a model.",
    ),
    runs_root: Path | None = None,
    json_out: bool = False,
) -> None:
    """Check benchmark-suite harness commands before spending GPU time."""
    config = with_cli_overrides(load_config(), runs_root=runs_root)
    try:
        suite_plan = BenchmarkSuitePlan.from_path(plan)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"Invalid benchmark-suite plan: {exc}", err=True)
        raise typer.Exit(1) from exc
    preflight = preflight_benchmark_suite(
        suite_plan,
        config.paths.runs_root,
        plan_path=plan,
    )
    payload = benchmark_suite_preflight_to_dict(preflight)
    payload["plan_path"] = str(plan)
    if json_out:
        typer.echo(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        typer.echo(f"Benchmark-suite preflight: {preflight.status}")
        typer.echo(f"Issues: {preflight.issue_count}")
        typer.echo(f"Next action: {preflight.next_action}")
        typer.echo(f"Receipt: {preflight.receipt_path}")
    if not preflight.ok:
        raise typer.Exit(1)


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
        verdict = suite_verdict(suite_run)
        console.print(f"Benchmark suite receipt: {suite_run.receipt_path}")
        console.print(f"Verdict: {verdict['action']} ({verdict['confidence']} confidence)")
        console.print(str(verdict["summary"]))
        console.print(f"Next run: {verdict['next_run']}")
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
    root = project_root()
    plans = benchmark_suite_plan_payloads(root)
    if json_out:
        _print_json(plans)
        return
    if not plans:
        console.print("No bundled benchmark-suite plans found.")
        return
    table = Table(title="Benchmark Suite Plans")
    table.add_column("Plan")
    table.add_column("Kind")
    table.add_column("Context")
    table.add_column("Requires")
    for plan in plans:
        plan_path = Path(str(plan["path"]))
        try:
            label = str(plan_path.relative_to(root))
        except ValueError:
            label = str(plan_path)
        table.add_row(
            label,
            str(plan.get("plan_kind") or "unknown"),
            str(plan.get("context") or "unknown"),
            str(plan.get("requires", "")),
        )
    console.print(table)


@app.command("flight-plans")
def flight_plans(json_out: bool = False) -> None:
    """List beginner-friendly benchmark Flight Plans."""
    plans = flight_plan_payloads(project_root())
    if json_out:
        _print_json(plans)
        return
    table = Table(title="pilotBENCHY Flight Plans")
    table.add_column("Plan", no_wrap=True)
    table.add_column("Recommended", no_wrap=True)
    table.add_column("Mode", no_wrap=True)
    table.add_column("Budget", no_wrap=True)
    table.add_column("Class", no_wrap=True)
    table.add_column("Score", no_wrap=True)
    table.add_column("Evidence")
    for plan in plans:
        table.add_row(
            str(plan["label"]),
            "yes" if plan["recommended"] else "",
            str(plan["mode_id"]),
            f"{plan['budget_minutes']} min/model",
            str(plan["evidence_class"]),
            str(plan["score_contract"]),
            str(plan["evidence_goal"]),
        )
    console.print(table)
    for plan in plans:
        if plan["recommended"]:
            console.print(
                "Recommended plan: "
                f"{plan['label']} | Class: {plan['evidence_class']} | "
                f"Score: {plan['score_contract']}"
            )
        elif plan["evidence_class"] == "speed_only":
            console.print(
                "Speed-only plan: "
                f"{plan['label']} | Class: {plan['evidence_class']} | "
                "not a recommendation"
            )


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
    allow_unproven: Annotated[
        bool,
        typer.Option(
            "--allow-unproven",
            help="Export even when deployment-readiness has not proven a deployable profile.",
        ),
    ] = False,
) -> None:
    """Export the latest champion as a ready-to-edit deployment profile."""
    config = with_cli_overrides(load_config(), runs_root=runs_root, llama_server=llama_server)
    runs_root = config.paths.runs_root
    leaderboard = write_leaderboard(runs_root)
    if not leaderboard.entries:
        console.print("No champion found yet. Run a benchmark first.")
        raise typer.Exit(1)
    readiness = write_deployment_readiness(runs_root)
    readiness_payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    readiness_action = str(readiness_payload.get("action") or "UNKNOWN")
    if readiness_action != "PROMOTE_DEPLOYMENT_PROFILE" and not allow_unproven:
        console.print(f"Deployment readiness: {readiness_action}")
        console.print(str(readiness_payload.get("summary") or "Deployment is not proven."))
        console.print("Refusing to export unproven champion.")
        console.print(f"Next run: {readiness_payload.get('next_run')}")
        console.print("Override only for manual lab work: --allow-unproven")
        raise typer.Exit(1)
    if readiness_action != "PROMOTE_DEPLOYMENT_PROFILE":
        console.print(
            "WARNING: exporting an unproven champion. Do not treat this as a deployment "
            "recommendation."
        )
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
    """Run a quick llama-bench speed probe on a single model."""
    config = with_cli_overrides(load_config(), llama_bench=llama_bench, runs_root=runs_root)
    runner = BenchmarkRunner(llama_bench=config.paths.llama_bench, runs_root=config.paths.runs_root)
    receipt = runner.run_model(model=model, profile=BenchProfile.quick())
    console.print(f"Receipt: {receipt.path}")


@app.command("vram-plan")
def vram_plan(
    model: Annotated[Path, typer.Option(help="GGUF model path.")],
    min_context: int = typer.Option(16_384, "--min-context", min=256),
    max_context: int = typer.Option(262_144, "--max-context", min=256),
    kv_bits: int = typer.Option(
        8, "--kv-bits", help="KV cache bits per element: 8 (q8_0, recommended) or 16 (f16)."
    ),
    budget_mb: int | None = typer.Option(
        None, "--budget-mb", help="VRAM budget in MB (default: detected total GPU VRAM)."
    ),
    json_out: bool = False,
) -> None:
    """Predict which context sizes fit in VRAM for a model, before running it.

    A conservative (dense upper-bound) guard so the context ladder skips tiers
    that would OOM-crash llama-server. Sliding-window models will use less.
    """
    arch = read_model_arch(model)
    if arch is None:
        console.print(f"Could not read GGUF architecture metadata from: {model}")
        raise typer.Exit(1)
    try:
        size_bytes = model.stat().st_size
    except OSError as exc:
        console.print(f"Could not read model file: {exc}")
        raise typer.Exit(1) from exc

    vram = detect_vram_mb()
    budget = budget_mb if budget_mb is not None else (vram.total_mb if vram else None)
    if budget is None:
        console.print("Could not detect GPU VRAM. Pass --budget-mb to plan anyway.")
        raise typer.Exit(1)

    ladder = [tier for tier in context_ladder(max_context) if tier >= min_context]
    plan = plan_context_fit(arch, size_bytes, ladder, budget, k_bits=kv_bits, v_bits=kv_bits)
    best = max_fitting_context(plan)

    if json_out:
        _print_json(
            {
                "model": str(model),
                "architecture": arch.architecture,
                "weights_mb": int(size_bytes / (1024 * 1024)),
                "vram_total_mb": vram.total_mb if vram else None,
                "vram_free_mb": vram.free_mb if vram else None,
                "budget_mb": budget,
                "kv_bits": kv_bits,
                "max_fitting_context": best,
                "tiers": [
                    {"context": fit.context_size, "needed_mb": fit.needed_mb, "fits": fit.fits}
                    for fit in plan
                ],
            }
        )
        return

    console.print(
        f"{arch.architecture}: {arch.n_layers} layers, {arch.n_heads_kv} KV heads, "
        f"k/v length {arch.key_length}/{arch.value_length}"
    )
    if vram:
        console.print(f"GPU VRAM: {vram.total_mb} MB total, {vram.free_mb} MB free")
    console.print(f"Budget: {budget} MB | KV cache: {kv_bits}-bit")
    table = Table(title=f"Context fit for {model.name}")
    table.add_column("Context", justify="right")
    table.add_column("Fits")
    table.add_column("VRAM needed", justify="right")
    for fit in plan:
        table.add_row(
            f"{fit.context_size // 1024}k",
            "yes" if fit.fits else "no",
            f"~{fit.needed_mb} MB",
        )
    console.print(table)
    if best is not None:
        console.print(f"Largest context predicted to fit: {best // 1024}k")
    else:
        console.print("No context tier in range is predicted to fit. Try --kv-bits 8.")


@app.command("context-limit")
def context_limit_command(
    model: Annotated[Path, typer.Option(help="GGUF model path.")],
    llama_server: Path | None = None,
    min_context: int = typer.Option(
        DEFAULT_MIN_CONTEXT, "--min-context", min=MIN_SERIOUS_CONTEXT_SIZE
    ),
    max_context: int = typer.Option(262_144, "--max-context", min=256),
    parallel: int = typer.Option(1, "--parallel", min=1),
    kv_cache_type: str = typer.Option(
        "q8_0", "--kv-cache-type", help="KV cache type (default q8_0; nobody benchmarks f16)."
    ),
    timeout_seconds: int = typer.Option(240, "--timeout-seconds", min=10),
    no_vram_guard: bool = typer.Option(
        False, "--no-vram-guard", help="Launch every tier even if the VRAM estimate says skip."
    ),
    no_refine: bool = typer.Option(False, "--no-refine"),
    json_out: bool = False,
) -> None:
    """Find the largest context the model can actually serve, from 32k upward.

    Starts at a useful 32k, climbs by 32k, uses q8_0 KV cache, recognises an
    out-of-memory crash, then backs off/refines instead of aborting the run.
    """
    config = with_cli_overrides(load_config(), llama_server=llama_server)
    server = config.paths.llama_server
    if not server.exists():
        console.print(f"llama-server not found: {server}")
        raise typer.Exit(1)

    # Recall what we already learned about this model in past runs.
    init_state_db(DEFAULT_DB_PATH)
    with sqlite3.connect(DEFAULT_DB_PATH) as conn:
        remembered = get_context_limit(conn, model.name, kv_cache_type)
    if remembered and not json_out:
        console.print(
            f"Previously found max context for this model (KV {kv_cache_type}): "
            f"{cast(int, remembered['max_context']) // 1024}k"
        )

    # Optional pre-flight VRAM guard so we never even launch a doomed tier.
    fits_vram = None
    if not no_vram_guard:
        arch = read_model_arch(model)
        vram = detect_vram_mb()
        if arch is not None and vram is not None:
            size_bytes = model.stat().st_size
            kv_bits = 8 if kv_cache_type.lower().startswith("q8") else 16

            def fits_vram(context: int) -> bool:
                plan = plan_context_fit(
                    arch, size_bytes, [context], vram.total_mb, k_bits=kv_bits, v_bits=kv_bits
                )
                return plan[0].fits

    def attempt(context: int) -> LaunchOutcome:
        settings = AutoresearchSettings(
            context_size=context,
            parallel=parallel,
            gpu_layers=99,
            flash_attention=True,
            kv_unified=True,
            cache_type_k=kv_cache_type,
            cache_type_v=kv_cache_type,
        )
        result = probe_llama_server_ttft(
            llama_server=server,
            model=model,
            settings=settings,
            prompt=fit_probe_prompt(),
            max_tokens=256,
            samples=1,
            timeout_seconds=timeout_seconds,
        )
        return LaunchOutcome(ok=result.ok, stderr=result.stderr_tail, detail=result.failure)

    result = find_context_limit(
        attempt,
        min_context=min_context,
        max_context=max_context,
        fits_vram=fits_vram,
        refine=not no_refine,
        log=None if json_out else console.print,
    )

    # Remember the discovered ceiling so future runs/sessions can warm-start.
    if result.max_context is not None:
        with sqlite3.connect(DEFAULT_DB_PATH) as conn:
            record_context_limit(
                conn,
                model.name,
                kv_cache_type,
                result.max_context,
                result.hit_oom,
                datetime.now(timezone.utc).isoformat(),
            )

    if json_out:
        _print_json(
            {
                "model": str(model),
                "kv_cache_type": kv_cache_type,
                "max_context": result.max_context,
                "hit_oom": result.hit_oom,
                "attempts": [
                    {"context": a.context_size, "outcome": a.outcome, "note": a.note}
                    for a in result.attempts
                ],
            }
        )
        return

    table = Table(title=f"Context limit for {model.name} (KV {kv_cache_type})")
    table.add_column("Context", justify="right")
    table.add_column("Result")
    table.add_column("Note")
    for a in result.attempts:
        table.add_row(f"{a.context_size // 1024}k", a.outcome, a.note)
    console.print(table)
    if result.max_context is not None:
        console.print(f"Largest context that actually served: {result.max_context // 1024}k")
    else:
        console.print("No context tier served successfully.")


@app.command("serve-probe")
def serve_probe(
    model: Annotated[Path, typer.Option(help="GGUF model path.")],
    llama_server: Path | None = None,
    runs_root: Path | None = None,
    context_size: int = 4096,
    parallel: int = 1,
    gpu_layers: int = 99,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    flash_attention: bool = True,
    timeout_seconds: int = 180,
    max_tokens: int = 768,
    samples: int = 0,
    cache_prompt: bool = True,
    prompt: str | None = None,
    llama_server_extra_arg: list[str] | None = typer.Option(
        None,
        "--llama-server-extra-arg",
        help="Extra raw llama-server argument locked into this speed probe. Repeatable.",
    ),
    json_out: bool = False,
) -> None:
    """Run the repeatable 16k+ speed probe and write a receipt."""
    from gguf_limit_bench.autoresearch import AutoresearchSettings

    config = with_cli_overrides(load_config(), llama_server=llama_server, runs_root=runs_root)
    requested_context_size = context_size
    effective_context_size = enforce_min_context(context_size, ProgramId.SPEED)
    forced_args = _effective_forced_server_args(tuple(llama_server_extra_arg or ()))
    probe_prompt = prompt or speed_probe_prompt()

    settings = AutoresearchSettings(
        profile_name="speed-probe",
        context_size=effective_context_size,
        parallel=parallel,
        gpu_layers=gpu_layers,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        flash_attention=flash_attention,
        kv_unified=True,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        extra_server_args=forced_args,
    )
    result = probe_llama_server_ttft(
        llama_server=config.paths.llama_server,
        model=model,
        settings=settings,
        prompt=probe_prompt,
        max_tokens=max_tokens,
        samples=samples,
        cache_prompt=cache_prompt,
        timeout_seconds=timeout_seconds,
    )
    receipt = _write_speed_probe_receipt(
        runs_root=config.paths.runs_root,
        model=model,
        prompt=probe_prompt,
        requested_context_size=requested_context_size,
        settings=settings,
        result=result,
    )
    if json_out:
        _print_json(
            {
                "program": ProgramId.SPEED.value,
                "receipt": str(receipt.path),
                "result": result.to_dict(),
            }
        )
        return
    if result.ok:
        if effective_context_size != requested_context_size:
            console.print(f"Context bumped to {effective_context_size // 1024}k.")
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
        console.print(f"Receipt: {receipt.path}")
    else:
        console.print(f"Serving probe failed: {result.failure}")
        console.print(f"Receipt: {receipt.path}")
        if result.stderr_tail:
            console.print(result.stderr_tail)
        raise typer.Exit(1)


def _write_speed_probe_receipt(
    *,
    runs_root: Path,
    model: Path,
    prompt: str,
    requested_context_size: int,
    settings: AutoresearchSettings,
    result: ServingProbeResult,
) -> RunReceipt:
    receipt = RunReceipt.create(runs_root, slug=f"{_safe_receipt_slug(model.stem)}-speed")
    payload = {
        "program": ProgramId.SPEED.value,
        "model": str(model),
        "prompt": prompt,
        "requested_context_size": requested_context_size,
        "settings": settings.to_dict(),
        "result": result.to_dict(),
    }
    receipt.write_resolved_plan(
        {
            "schema_version": 1,
            "program": ProgramId.SPEED.value,
            "model": str(model),
            "requested_context_size": requested_context_size,
            "settings": settings.to_dict(),
            "prompt": prompt,
        },
        [
            _command_record(
                [
                    "agent-autobench",
                    "serve-probe",
                    "--model",
                    str(model),
                    "--context-size",
                    str(requested_context_size),
                ]
            )
        ],
    )
    receipt.write_json("speed-probe.json", payload)
    receipt.write_summary(
        [
            f"# Speed Probe - {model.name}",
            "",
            f"- Program: `{ProgramId.SPEED.value}`",
            f"- Requested context: `{requested_context_size}`",
            f"- Effective context: `{settings.context_size}`",
            f"- KV cache: `{settings.cache_type_k}/{settings.cache_type_v}`",
            f"- Unified KV: `{settings.kv_unified}`",
            f"- Extra server args: `{list(settings.extra_server_args)}`",
            f"- OK: `{result.ok}`",
            f"- Decode TPS: `{result.tokens_per_second:.2f}`",
            f"- Generated tokens: `{result.generated_tokens}`",
            f"- Failure: `{result.failure}`",
        ]
    )
    receipt.mark_recovery(step="speed-probe", status="finished" if result.ok else "failed")
    receipt.write_status(
        "finished" if result.ok else "failed",
        step="speed-probe",
        detail=result.failure,
    )
    return receipt


def _safe_receipt_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)[:80]


def _display_command(parts: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in parts])


def _command_record(parts: list[str], *, cwd: Path | None = None) -> dict:
    argv = [str(part) for part in parts]
    payload = {
        "argv": argv,
        "display_command": subprocess.list2cmdline(argv),
    }
    if cwd is not None:
        payload["cwd"] = str(cwd)
    return payload


def _autoresearch_command_record(
    *,
    model: Path,
    budget_seconds: int,
    parallel_max: int,
    max_attempts: int | None,
    evaluation: EvaluationMode,
    flag_ladder: bool,
    dry_run: bool,
    flag_context_size: int,
    benchmark_suite_plan: Path | None,
    context_ladder: tuple[int, ...] | None,
    perplexity_corpus: Path | None,
    perplexity_context: tuple[int, ...] | None,
    llama_server_extra_args: tuple[str, ...],
) -> dict:
    command = [
        "agent-autobench",
        "autoresearch",
        "--model",
        str(model),
        "--budget-minutes",
        str(max(1, math.ceil(budget_seconds / 60))),
        "--parallel-max",
        str(parallel_max),
    ]
    if max_attempts is not None:
        command.extend(["--max-attempts", str(max_attempts)])
    if evaluation is EvaluationMode.SPEED_SCOUT:
        command.append("--speed-scout")
    if flag_ladder:
        command.append("--flag-ladder")
    if dry_run:
        command.append("--dry-run")
    if flag_context_size != MIN_SERIOUS_CONTEXT_SIZE:
        command.extend(["--flag-context-size", str(flag_context_size)])
    if benchmark_suite_plan is not None:
        command.extend(["--benchmark-suite-plan", str(benchmark_suite_plan)])
    for context in context_ladder or ():
        command.extend(["--context-ladder", str(context)])
    if perplexity_corpus is not None:
        command.extend(["--perplexity-corpus", str(perplexity_corpus)])
    for context in perplexity_context or ():
        command.extend(["--perplexity-context", str(context)])
    for arg in llama_server_extra_args:
        command.append(f"--llama-server-extra-arg={arg}")
    return _command_record(command, cwd=Path.cwd())


@app.command()
def autoresearch(
    model: Annotated[Path, typer.Option(help="GGUF model path.")],
    budget_minutes: int = typer.Option(5, min=1),
    parallel_max: int | None = None,
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    llama_perplexity: Path | None = None,
    runs_root: Path | None = None,
    max_attempts: int | None = typer.Option(None, min=1),
    learning: bool = True,
    workflow_eval: bool = True,
    ttft_probe: bool = True,
    context_ladder: list[int] | None = typer.Option(
        None,
        "--context-ladder",
        help="Add a fixed context tier to profile after the best settings are found. Repeatable.",
    ),
    perplexity_corpus: Path | None = typer.Option(
        None,
        "--perplexity-corpus",
        help="Text corpus for optional llama-perplexity quality falloff profiling.",
    ),
    perplexity_context: list[int] | None = typer.Option(
        None,
        "--perplexity-context",
        help="Context tier for optional perplexity profiling. Repeatable.",
    ),
    benchmark_suite_plan: Path | None = typer.Option(
        None,
        "--benchmark-suite-plan",
        help=(
            "Run this benchmark-suite plan for each successful attempt and optimize "
            "by agent_bench_score."
        ),
    ),
    flag_ladder: bool = typer.Option(
        False,
        "--flag-ladder",
        help="Run a fixed llama-server flag ladder through the 10-question SimpleBench batch.",
    ),
    speed_scout: bool = typer.Option(
        False,
        "--speed-scout",
        help="Fast synthetic llama-bench scout (does NOT ask the benchmark questions).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Write the flag-ladder launch plan without starting llama-server.",
    ),
    flag_context_size: int = typer.Option(
        MIN_SERIOUS_CONTEXT_SIZE,
        "--flag-context-size",
        min=MIN_SERIOUS_CONTEXT_SIZE,
        help="Context size used by every rung in the flag ladder.",
    ),
    simple_bench: Path = typer.Option(
        DEFAULT_SIMPLE_BENCH_PATH,
        "--simple-bench",
        help="SimpleBench JSON file with eval_data rows.",
    ),
    simple_bench_system_prompt: Path = typer.Option(
        DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT,
        "--simple-bench-system-prompt",
        help="System prompt prepended before each SimpleBench question.",
    ),
    simple_bench_max_tokens: int = typer.Option(
        DEFAULT_DEPLOYMENT_SIMPLE_BENCH_MAX_TOKENS,
        "--simple-bench-max-tokens",
        min=1,
        help="Maximum generated tokens per SimpleBench question (room for reasoning models to finish).",
    ),
    llama_server_extra_arg: list[str] | None = typer.Option(
        None,
        "--llama-server-extra-arg",
        help="Extra raw llama-server argument appended to every flag-ladder rung. Repeatable.",
    ),
    sample_size: int = typer.Option(
        None,
        "--sample-size",
        min=1,
        help="Questions sampled per pack in the post-run champion eval (default: from config).",
    ),
    selection: str = typer.Option(
        None,
        "--selection",
        help="Question selection mode for champion eval: sequential or random (default: from config).",
    ),
) -> None:
    """Run the flag-ladder benchmark loop to find optimal server settings for a model."""
    evaluation = resolve_evaluation_mode(speed_scout=speed_scout, flag_ladder=flag_ladder)
    if dry_run and evaluation is EvaluationMode.SPEED_SCOUT:
        raise typer.BadParameter(
            "--dry-run needs benchmark mode; remove --speed-scout",
            param_hint="--dry-run",
        )
    try:
        extra_server_args = validate_extra_server_args(tuple(llama_server_extra_arg or ()))
    except ValueError as exc:
        raise typer.BadParameter(
            str(exc),
            param_hint="--llama-server-extra-arg",
        ) from exc
    config = with_cli_overrides(
        load_config(),
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_server=llama_server,
        llama_perplexity=llama_perplexity,
        runs_root=runs_root,
        parallel_max=parallel_max,
    )
    resolved_sample_size = (
        sample_size if sample_size is not None else config.benchmark.question_sample_size
    )
    resolved_selection = selection if selection is not None else config.benchmark.question_selection
    receipt = _run_one_autoresearch(
        model=model,
        llama_bench=config.paths.llama_bench,
        llama_cli=config.paths.llama_cli,
        llama_server=config.paths.llama_server,
        llama_perplexity=config.paths.llama_perplexity,
        runs_root=config.paths.runs_root,
        budget_seconds=budget_minutes * 60,
        parallel_max=config.benchmark.parallel_max,
        max_attempts=max_attempts,
        learning=learning,
        workflow_eval=workflow_eval,
        ttft_probe=ttft_probe,
        context_ladder=_context_ladder_or_none(context_ladder),
        perplexity_corpus=perplexity_corpus,
        perplexity_context=_context_ladder_or_none(perplexity_context),
        benchmark_suite_plan=benchmark_suite_plan,
        enable_mtp=_is_mtp_model(model),
        flag_ladder=flag_ladder,
        dry_run=dry_run,
        flag_context_size=flag_context_size,
        simple_bench=simple_bench,
        simple_bench_system_prompt=simple_bench_system_prompt,
        simple_bench_max_tokens=simple_bench_max_tokens,
        llama_server_extra_args=extra_server_args,
        evaluation=evaluation,
        forced_server_args=_effective_forced_server_args(config.benchmark.forced_server_args),
        champion_sample_size=resolved_sample_size,
        champion_selection=resolved_selection,
    )
    _print_receipt_outputs(receipt.path)


@app.command()
def engine(
    run_dir: Annotated[
        Path, typer.Option("--run-dir", help="Run directory holding run-spec.json.")
    ],
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_server: Path | None = None,
    llama_perplexity: Path | None = None,
    runs_root: Path | None = None,
    parallel_max: int | None = None,
) -> None:
    """Detached engine: run the models in <run-dir>/run-spec.json sequentially.

    This is the process the web UI launches. It owns the run, writes the live
    stream + status heartbeat, and obeys control.json (stop/abort). The web UI
    never evaluates in-process; it only reads this run directory.
    """
    from gguf_limit_bench import engine as engine_runner
    from gguf_limit_bench import run_dir as run_dir_io

    rd = Path(run_dir)
    spec = run_dir_io.read_spec(rd)
    # The web UI carries the resolved llama.cpp paths through run-spec.json so the
    # detached engine can find the real binaries. Explicit --llama-* flags on this
    # command still win; otherwise a non-null spec path overrides the config default.
    spec_paths = spec.get("paths") or {}

    def _spec_path(value: object) -> Path | None:
        return Path(str(value)) if value else None

    config = with_cli_overrides(
        load_config(),
        llama_bench=llama_bench or _spec_path(spec_paths.get("llama_bench")),
        llama_cli=llama_cli or _spec_path(spec_paths.get("llama_cli")),
        llama_server=llama_server or _spec_path(spec_paths.get("llama_server")),
        llama_perplexity=llama_perplexity or _spec_path(spec_paths.get("llama_perplexity")),
        runs_root=runs_root or _spec_path(spec_paths.get("runs_root")),
        parallel_max=parallel_max,
    )
    paths = config.paths
    bench = config.benchmark
    mode_id = str(spec.get("mode", "librarian_bench"))

    def _on_signal(signum, frame):  # noqa: ANN001 - stdlib signal handler API
        run_dir_io.write_status(rd, phase="aborted", pid=os.getpid())
        run_dir_io.release_lock(rd)
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    model_root = paths.model_roots[0] if paths.model_roots else Path.cwd()
    mode = mode_by_id(mode_id)

    def run_model(model_item: object, options: dict, emit) -> object:
        if isinstance(model_item, dict):
            model_path = Path(str(model_item["path"]))
            has_mtp = bool(model_item.get("has_mtp"))
        else:
            model_path = Path(str(model_item))
            has_mtp = False
        budget_minutes = int(options.get("budget_minutes") or 5)
        plan = options.get("benchmark_suite_plan") or None
        sampler_policy = str(options.get("sampler_policy") or "hf_recommended")
        try:
            model_flags = recommended_model_flags(model_path, search_roots=(model_root,))
        except Exception:  # noqa: BLE001 - missing/unreadable model: fall back to none
            model_flags = ()
        sampler_flags = _sampler_flags_for_policy(model_path, sampler_policy)
        forced = merge_flags(
            merge_flags(tuple(options.get("forced_server_args", ())), sampler_flags),
            model_flags,
        )
        receipt = _run_one_autoresearch(
            model=model_path,
            llama_bench=paths.llama_bench,
            llama_cli=paths.llama_cli,
            llama_server=paths.llama_server,
            llama_perplexity=paths.llama_perplexity,
            runs_root=paths.runs_root,
            budget_seconds=budget_minutes * 60,
            parallel_max=bench.parallel_max,
            max_attempts=options.get("max_attempts"),
            learning=bench.learning,
            workflow_eval=bench.workflow_eval,
            ttft_probe=bench.ttft_probe,
            context_ladder=_context_ladder_or_none(mode.context_ladder),
            benchmark_suite_plan=Path(plan) if plan else None,
            enable_mtp=has_mtp,
            evaluation=mode.evaluation,
            forced_server_args=forced,
            champion_pack_ids=(tuple(LIBRARIAN_PACK_IDS) if mode_id == "librarian_bench" else None),
            champion_sample_size=int(options.get("sample_size") or 5),
            champion_repeats=int(options.get("repeats") or 3),
            sampler_policy=sampler_policy,
            run_mode_id=mode_id,
            flight_plan_id=(
                str(options.get("flight_plan_id")) if options.get("flight_plan_id") else None
            ),
        )
        emit("receipt_ready", {"model": str(model_path), "path": str(receipt.path)})
        return receipt

    engine_runner.run_engine(rd, run_model)


@app.command("engine-replay")
def engine_replay_cmd(
    run_dir: Annotated[Path, typer.Option("--run-dir", help="Run directory to replay into.")],
    source: Annotated[Path, typer.Option("--source", help="Recorded live.jsonl to replay.")],
    delay: float = typer.Option(0.1, "--delay", help="Seconds between replayed events."),
) -> None:
    """Replay a recorded live.jsonl into a run dir (drives the cockpit, no GPU)."""
    from gguf_limit_bench import engine_replay

    source_path = Path(source)
    if not source_path.is_file():
        console.print(f"Source live.jsonl not found: {source_path}")
        raise typer.Exit(1)

    engine_replay.replay(Path(run_dir), source_path, delay=delay)


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
    workflow_eval: bool = True,
    ttft_probe: bool = True,
    context_ladder: list[int] | None = typer.Option(
        None,
        "--context-ladder",
        help="Add a fixed context tier to profile after the best settings are found. Repeatable.",
    ),
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
    speed_scout: bool = typer.Option(
        False,
        "--speed-scout",
        help="Fast synthetic llama-bench scout (does NOT ask the benchmark questions).",
    ),
) -> None:
    """Run the autoresearch loop on every model found under the model root(s)."""
    evaluation = resolve_evaluation_mode(speed_scout=speed_scout, flag_ladder=False)
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
            llama_perplexity=config.paths.llama_perplexity,
            runs_root=config.paths.runs_root,
            budget_seconds=budget_seconds,
            parallel_max=config.benchmark.parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            ttft_probe=ttft_probe,
            context_ladder=_context_ladder_or_none(context_ladder),
            benchmark_suite_plan=benchmark_suite_plan,
            enable_mtp=model.has_mtp,
            evaluation=evaluation,
            forced_server_args=_effective_forced_server_args(config.benchmark.forced_server_args),
        )
        console.print(f"{model.name}: {receipt.path}")
        _print_receipt_outputs(receipt.path)
        score = _receipt_score(receipt.path)
        if finish_early_on and score is not None and score >= target_score:
            console.print(f"Finish-early target met: score {score:.2f} >= {target_score:.2f}")
            break
    leaderboard = write_leaderboard(config.paths.runs_root)
    if leaderboard.entries:
        console.print(_leaderboard_summary_line(leaderboard))
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
    max_attempts: int | None = None,
    learning: bool = True,
    workflow_eval: bool = True,
    ttft_probe: bool = True,
    context_ladder: list[int] | None = typer.Option(
        None,
        "--context-ladder",
        help="Add a fixed context tier to profile after the best settings are found. Repeatable.",
    ),
    benchmark_suite_plan: Path | None = typer.Option(
        None,
        "--benchmark-suite-plan",
        help="Run a benchmark-suite plan for selected models and optimize by agent_bench_score.",
    ),
    target_model: str | None = typer.Option(
        None,
        "--target-model",
        help="Scope post-run TUI decision lines to one intended model name or GGUF basename.",
    ),
    target_model_path: Path | None = typer.Option(
        None,
        "--target-model-path",
        help="GGUF path to use for target deployment flag proof commands.",
    ),
    required_context: int | None = typer.Option(
        None,
        "--required-context",
        min=1,
        help="Minimum context that must be proven before TUI recommendations can call the stack ready.",
    ),
) -> None:
    """Open the interactive TUI model picker and run the autoresearch loop."""
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
        runs_root=config.paths.runs_root,
        target_model=target_model,
        target_model_path=str(target_model_path) if target_model_path is not None else None,
        required_context=required_context,
    )
    picker.run_model = lambda model: (
        _run_one_autoresearch(
            model=model.path,
            llama_bench=config.paths.llama_bench,
            llama_cli=config.paths.llama_cli,
            llama_server=config.paths.llama_server,
            runs_root=config.paths.runs_root,
            budget_seconds=picker.run_mode.budget_minutes * 60,
            parallel_max=config.benchmark.parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            ttft_probe=ttft_probe,
            context_ladder=(
                picker.run_mode.context_ladder or _context_ladder_or_none(context_ladder)
            ),
            benchmark_suite_plan=benchmark_suite_plan,
            enable_mtp=model.has_mtp,
            evaluation=picker.evaluation_mode,
            forced_server_args=_effective_forced_server_args(config.benchmark.forced_server_args),
        ).path
    )
    picker.run()
    if getattr(picker, "ran_inside_tui", False):
        leaderboard = write_leaderboard(config.paths.runs_root)
        if leaderboard.entries:
            console.print(_leaderboard_summary_line(leaderboard))
            console.print(f"Leaderboard: {config.paths.runs_root / 'leaderboard.md'}")
    else:
        _run_tui_selection(
            selected_models=picker.models_to_run,
            llama_bench=config.paths.llama_bench,
            llama_cli=config.paths.llama_cli,
            llama_server=config.paths.llama_server,
            llama_perplexity=config.paths.llama_perplexity,
            runs_root=config.paths.runs_root,
            budget_minutes=budget_minutes,
            parallel_max=config.benchmark.parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            ttft_probe=ttft_probe,
            context_ladder=_context_ladder_or_none(context_ladder),
            benchmark_suite_plan=benchmark_suite_plan,
        )


def _build_learner(
    enabled: bool,
    runs_root: Path,
    model: Path,
    parallel_max: int,
    objective: str = "throughput",
) -> OptunaSettingsLearner | None:
    if not enabled:
        return None
    learner = OptunaSettingsLearner(
        storage_path=runs_root / "learning" / "optuna.sqlite3",
        model=model,
        parallel_max=parallel_max,
        objective=objective,
    )
    previous = _previous_successful_settings(model, runs_root)
    if previous is not None:
        learner.enqueue_settings(previous)
    return learner


def _sampler_flags_for_policy(model: Path, sampler_policy: str) -> tuple[str, ...]:
    if sampler_policy == "runtime_defaults":
        return ()
    if sampler_policy.startswith("hf:"):
        preset_name = sampler_policy.split(":", 1)[1]
        for preset in recommended_sampler_presets(model):
            if preset["name"] == preset_name:
                return sampler_flags_from_values(preset["values"])
        return ()
    return recommended_sampler_flags(model)


def _previous_successful_settings(model: Path, runs_root: Path):
    from gguf_limit_bench.autoresearch import AutoresearchSettings

    candidate_roots = [runs_root]
    if runs_root.name == "_runs" and Path("runs").exists():
        candidate_roots.append(Path("runs"))
    best_score = None
    best_settings = None
    for root in candidate_roots:
        for best_path in root.glob("*/best-settings.json"):
            try:
                payload = json.loads(best_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if Path(str(payload.get("model", ""))).name != model.name:
                continue
            result = payload.get("result", {})
            if result.get("ok") is not True:
                continue
            score = float(payload.get("score") or -10_000.0)
            if best_score is not None and score <= best_score:
                continue
            settings = payload.get("settings", {})
            best_score = score
            best_settings = AutoresearchSettings(
                context_size=int(settings.get("context_size") or 4096),
                parallel=int(settings.get("parallel") or 1),
                gpu_layers=int(settings.get("gpu_layers") or 99),
                batch_size=int(settings.get("batch_size") or 2048),
                ubatch_size=int(settings.get("ubatch_size") or 512),
                flash_attention=bool(settings.get("flash_attention", True)),
                kv_unified=bool(settings.get("kv_unified", True)),
            )
    return best_settings


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
    context_ladder: tuple[int, ...] | None = None,
    llama_perplexity: Path = DEFAULT_LLAMA_PERPLEXITY,
    perplexity_corpus: Path | None = None,
    perplexity_context: tuple[int, ...] | None = None,
    benchmark_suite_plan: Path | None = None,
    enable_mtp: bool = False,
    flag_ladder: bool = False,
    dry_run: bool = False,
    flag_context_size: int = MIN_SERIOUS_CONTEXT_SIZE,
    simple_bench: Path = DEFAULT_SIMPLE_BENCH_PATH,
    simple_bench_system_prompt: Path = DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT,
    simple_bench_max_tokens: int = DEFAULT_DEPLOYMENT_SIMPLE_BENCH_MAX_TOKENS,
    llama_server_extra_args: tuple[str, ...] = (),
    capability_collector: Callable[[Path], LlamaRuntimeCapabilities] = collect_llama_capabilities,
    flag_ladder_attempt_runner: AttemptRunner | None = None,
    evaluation: EvaluationMode = EvaluationMode.BENCHMARK,
    forced_server_args: tuple[str, ...] = (),
    champion_sample_size: int = 5,
    champion_selection: str = "sequential",
    champion_pack_ids: tuple[str, ...] | None = None,
    champion_state_db_path: Path | None = None,
    champion_gpu_name: str = "",
    champion_repeats: int = 3,
    sampler_policy: str = "hf_recommended",
    run_mode_id: str | None = None,
    flight_plan_id: str | None = None,
):
    # Benchmark mode asks the real questions via the flag-ladder SimpleBench engine.
    # The legacy --flag-ladder flag forces the same path.
    flag_ladder = flag_ladder or asks_questions(evaluation)
    # Forced flags are applied on top of every profile's flags.
    sampler_args = _sampler_flags_for_policy(model, sampler_policy)
    if sampler_args:
        forced_server_args = merge_flags(tuple(forced_server_args), sampler_args)
    if forced_server_args:
        llama_server_extra_args = validate_extra_server_args(
            tuple(forced_server_args) + tuple(llama_server_extra_args)
        )
    candidate_sequence = None
    skipped_profiles: tuple[dict, ...] = ()
    attempt_runner: AttemptRunner
    if flag_ladder:
        candidate_sequence = build_core_flag_ladder(
            context_size=flag_context_size,
            parallel_max=parallel_max,
            extra_server_args=llama_server_extra_args,
            enable_mtp=enable_mtp,
        )
        if dry_run:
            return _write_flag_ladder_dry_run(
                model=model,
                llama_server=llama_server,
                runs_root=runs_root,
                context_size=flag_context_size,
                parallel_max=parallel_max,
                extra_server_args=llama_server_extra_args,
                enable_mtp=enable_mtp,
                capability_collector=capability_collector,
            )
        if enable_mtp:
            candidate_sequence, skipped_profiles = filter_unsupported_profiles(
                candidate_sequence,
                capability_collector(llama_server),
            )
        attempt_runner = flag_ladder_attempt_runner or LlamaServerSimpleBenchAttemptRunner(
            llama_server=llama_server,
            model=model,
            benchmark_path=simple_bench,
            system_prompt_path=simple_bench_system_prompt,
            timeout_seconds=max(60, budget_seconds),
            max_tokens=simple_bench_max_tokens,
        )
        workflow_eval = False
        ttft_probe = False
        # Walk the ordered ladder first, then (when learning is on) keep searching
        # with the persistent learner until the time budget runs out. This is what
        # makes a long/overnight run converge instead of stopping at the ladder.
    else:
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

        def ttft_attempt_runner(settings: AutoresearchSettings) -> AttemptResult:
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

        attempt_runner = ttft_attempt_runner

    # Resolve champion-eval defaults: detect the GPU for recommended flags and
    # persist lifetime stats to the shared experiment DB (not an in-memory one).
    resolved_gpu_name = champion_gpu_name or detect_gpu_name()
    resolved_state_db_path = champion_state_db_path or DEFAULT_DB_PATH
    resolved_benchmark_suite_plan = (
        BenchmarkSuitePlan.from_path(benchmark_suite_plan)
        if benchmark_suite_plan is not None
        else None
    )
    resolved_plan = {
        "schema_version": 1,
        "program": "autoresearch",
        "model": str(model),
        "mode_id": run_mode_id,
        "flight_plan_id": flight_plan_id,
        "evaluation": evaluation.value,
        "flag_ladder": flag_ladder,
        "dry_run": dry_run,
        "budget_seconds": budget_seconds,
        "parallel_max": parallel_max,
        "max_attempts": max_attempts,
        "llama_bench": str(llama_bench),
        "llama_cli": str(llama_cli),
        "llama_server": str(llama_server),
        "llama_perplexity": str(llama_perplexity),
        "context_ladder": list(context_ladder or ()),
        "perplexity_corpus": None if perplexity_corpus is None else str(perplexity_corpus),
        "perplexity_context": list(perplexity_context or ()),
        "benchmark_suite_plan": None if benchmark_suite_plan is None else str(benchmark_suite_plan),
        "enable_mtp": enable_mtp,
        "flag_context_size": flag_context_size,
        "simple_bench": str(simple_bench),
        "simple_bench_system_prompt": str(simple_bench_system_prompt),
        "simple_bench_max_tokens": simple_bench_max_tokens,
        "llama_server_extra_args": list(llama_server_extra_args),
        "forced_server_args": list(forced_server_args),
        "sampler_policy": sampler_policy,
        "candidate_sequence": [settings.to_dict() for settings in candidate_sequence or ()],
        "skipped_profiles": list(skipped_profiles),
        "champion_eval": {
            "pack_ids": list(champion_pack_ids or ()),
            "sample_size": champion_sample_size,
            "repeats": champion_repeats,
            "selection": champion_selection,
            "state_db_path": str(resolved_state_db_path),
            "gpu_name": resolved_gpu_name,
        },
    }
    command = _autoresearch_command_record(
        model=model,
        budget_seconds=budget_seconds,
        parallel_max=parallel_max,
        max_attempts=max_attempts,
        evaluation=evaluation,
        flag_ladder=flag_ladder,
        dry_run=dry_run,
        flag_context_size=flag_context_size,
        benchmark_suite_plan=benchmark_suite_plan,
        context_ladder=context_ladder,
        perplexity_corpus=perplexity_corpus,
        perplexity_context=perplexity_context,
        llama_server_extra_args=llama_server_extra_args,
    )

    loop = AutoresearchLoop(
        model=model,
        runs_root=runs_root,
        attempt_runner=attempt_runner,
        budget_seconds=budget_seconds,
        parallel_max=parallel_max,
        max_attempts=max_attempts,
        learner=_build_learner(
            learning,
            runs_root,
            model,
            parallel_max,
            objective="accuracy" if flag_ladder else "throughput",
        ),
        benchmark_suite_plan=resolved_benchmark_suite_plan,
        context_ladder=context_ladder,
        perplexity_runner=(
            LlamaPerplexityRunner(
                llama_perplexity=llama_perplexity,
                model=model,
                corpus=perplexity_corpus,
                timeout_seconds=max(60, budget_seconds),
            )
            if perplexity_corpus is not None and llama_perplexity.exists()
            else None
        ),
        perplexity_contexts=perplexity_context,
        candidate_sequence=candidate_sequence,
        skipped_profiles=skipped_profiles,
        round_seconds=KARPATHY_ROUND_SECONDS if flag_ladder else None,
        # Champion pack eval
        llama_server=llama_server,
        champion_pack_ids=champion_pack_ids,
        champion_sample_size=champion_sample_size,
        champion_repeats=champion_repeats,
        champion_selection=champion_selection,
        champion_state_db_path=resolved_state_db_path,
        champion_gpu_name=resolved_gpu_name,
        is_benchmark_mode=flag_ladder,
        resolved_plan=resolved_plan,
        commands=(command,),
    )
    return loop.run()


def _write_flag_ladder_dry_run(
    *,
    model: Path,
    llama_server: Path,
    runs_root: Path,
    context_size: int,
    parallel_max: int,
    extra_server_args: tuple[str, ...],
    enable_mtp: bool,
    capability_collector: Callable[[Path], LlamaRuntimeCapabilities] = collect_llama_capabilities,
) -> RunReceipt:
    receipt = RunReceipt.create(runs_root, slug=f"{model.stem}-flag-ladder-dry-run")
    runtime_capabilities = capability_collector(llama_server)
    plan = build_flag_ladder_plan(
        llama_server=llama_server,
        model=model,
        host="127.0.0.1",
        port=6939,
        context_size=context_size,
        parallel_max=parallel_max,
        extra_server_args=extra_server_args,
        enable_mtp=enable_mtp,
        runtime_capabilities=runtime_capabilities,
    )
    receipt.write_resolved_plan(
        {
            "schema_version": 1,
            "program": "autoresearch",
            "model": str(model),
            "evaluation": EvaluationMode.BENCHMARK.value,
            "flag_ladder": True,
            "dry_run": True,
            "llama_server": str(llama_server),
            "context_size": context_size,
            "parallel_max": parallel_max,
            "extra_server_args": list(extra_server_args),
            "enable_mtp": enable_mtp,
            "runtime_capabilities": runtime_capabilities.to_dict(),
            "profiles": plan,
        },
        [
            _command_record(
                [
                    "agent-autobench",
                    "autoresearch",
                    "--model",
                    str(model),
                    "--flag-ladder",
                    "--dry-run",
                    "--flag-context-size",
                    str(context_size),
                    "--parallel-max",
                    str(parallel_max),
                ]
            )
        ],
    )
    receipt.write_json(
        "flag-ladder-plan.json",
        {
            "model": str(model),
            "llama_server": str(llama_server),
            "context_size": context_size,
            "parallel_max": parallel_max,
            "extra_server_args": list(extra_server_args),
            "mtp_heads_detected": enable_mtp,
            "mtp_detection": "model filename contains MTP" if enable_mtp else "not detected",
            "runtime_capabilities": runtime_capabilities.to_dict(),
            "profiles": plan,
            "dry_run": True,
        },
    )
    (receipt.path / "flag-ladder-plan.md").write_text(
        _flag_ladder_plan_markdown(model=model, plan=plan),
        encoding="utf-8",
    )
    receipt.write_summary(
        [
            f"# Flag Ladder Dry Run: {model.name}",
            "",
            "No llama-server process was started.",
            "",
            f"- Plan JSON: `{receipt.path / 'flag-ladder-plan.json'}`",
            f"- Plan Markdown: `{receipt.path / 'flag-ladder-plan.md'}`",
        ]
    )
    receipt.event("flag_ladder_dry_run_written", {"profiles": plan})
    receipt.mark_recovery(step="flag-ladder-dry-run", status="finished")
    receipt.write_status("finished", step="flag-ladder-dry-run")
    return receipt


def _flag_ladder_plan_markdown(*, model: Path, plan: list[dict]) -> str:
    lines = [
        f"# Flag Ladder Plan: {model.name}",
        "",
        "| Profile | Status | Reason | Hypothesis | Command |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in plan:
        supported = row.get("supported", True)
        status = "Supported" if supported else "Unsupported"
        reason = str(row.get("unsupported_reason") or "")
        command_parts = row.get("command")
        command = " ".join(str(part) for part in command_parts) if command_parts else ""
        rendered_command = f"`{command}`" if command else "-"
        lines.append(
            f"| {row['name']} | {status} | {reason} | {row['hypothesis']} | {rendered_command} |"
        )
    return "\n".join(lines) + "\n"


def _run_tui_selection(
    selected_models,
    llama_bench: Path,
    llama_cli: Path,
    llama_server: Path,
    llama_perplexity: Path,
    runs_root: Path,
    budget_minutes: int,
    parallel_max: int,
    max_attempts: int | None,
    learning: bool,
    workflow_eval: bool,
    ttft_probe: bool,
    context_ladder: tuple[int, ...] | None = None,
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
            llama_perplexity=llama_perplexity,
            runs_root=runs_root,
            budget_seconds=budget_minutes * 60,
            parallel_max=parallel_max,
            max_attempts=max_attempts,
            learning=learning,
            workflow_eval=workflow_eval,
            ttft_probe=ttft_probe,
            context_ladder=context_ladder,
            benchmark_suite_plan=benchmark_suite_plan,
            enable_mtp=model.has_mtp,
        )
        _print_receipt_outputs(receipt.path)
    leaderboard = write_leaderboard(runs_root)
    if leaderboard.entries:
        console.print(_leaderboard_summary_line(leaderboard))
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
        context_ladder=config.context_ladder,
        packs=config.packs,
        adaptive=config.adaptive,
        min_ttft_target_ms=config.min_ttft_target_ms,
        min_generation_tps=config.min_generation_tps,
        require_full_gpu_offload=config.require_full_gpu_offload,
        require_no_swap=config.require_no_swap,
    )


def _context_ladder_or_none(values) -> tuple[int, ...] | None:
    if not values:
        return None
    cleaned = tuple(sorted({int(value) for value in values if int(value) > 0}))
    return cleaned or None


def _serve_report_directory(directory: Path, port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("Report server stopped.")
    finally:
        server.server_close()


def _print_receipt_outputs(receipt_path: Path) -> None:
    console.print(f"Receipt: {receipt_path}")
    report_path = receipt_path / "itemized-report.md"
    browser_report = receipt_path / "report.html"
    context_profile = receipt_path / "context-profile.md"
    perplexity_profile = receipt_path / "perplexity-profile.md"
    flag_ladder_plan = receipt_path / "flag-ladder-plan.md"
    flag_ladder_results = receipt_path / "flag-ladder-results.md"
    if report_path.exists():
        console.print(f"Itemized report: {report_path}")
    if browser_report.exists():
        console.print(f"Browser report: {browser_report}")
    if context_profile.exists():
        console.print(f"Context profile: {context_profile}")
    if perplexity_profile.exists():
        console.print(f"Perplexity profile: {perplexity_profile}")
    if flag_ladder_plan.exists():
        console.print(f"Flag ladder plan: {flag_ladder_plan}")
    if flag_ladder_results.exists():
        console.print(f"Flag ladder results: {flag_ladder_results}")


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
    console.print("Setup wizard")
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
        console.print("Setup is ready.")
        console.print("Next command: apb")
    else:
        console.print("Setup needs one or more missing items fixed.")
        console.print("Next command: apb doctor")


def _print_json(payload) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, indent=2))


def _format_optional_ms(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.0f} ms"


def _format_optional_tps(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.2f} tok/s"


def _format_optional_float(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.6f}"


def _print_score_summary(score_summary: dict[str, object]) -> None:
    console.print("Benchmark scores:")
    console.print(f"Score contract: {score_summary.get('score_contract') or 'unknown'}")
    console.print(
        "Agent bench score: "
        f"{_format_optional_float(_float_from_summary(score_summary, 'agent_bench_score'))}"
    )
    console.print(
        "General score: "
        f"{_format_optional_float(_float_from_summary(score_summary, 'general_score'))}"
    )
    console.print(
        "Agentic score: "
        f"{_format_optional_float(_float_from_summary(score_summary, 'agentic_score'))}"
    )
    console.print(
        "Generation speed: "
        f"{_format_optional_tps(_float_from_summary(score_summary, 'generation_tps'))}"
    )
    console.print(
        f"Serving speed: {_format_optional_tps(_float_from_summary(score_summary, 'serving_tps'))}"
    )
    context = score_summary.get("context")
    console.print(f"Context: {context if context is not None else 'not measured'}")


def _float_from_summary(score_summary: dict[str, object], key: str) -> float | None:
    value = score_summary.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_from_object(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if not isinstance(value, int | float | str | bytes | bytearray):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _operator_verdict_payload(payload: dict) -> dict:
    operator = payload.get("operator_verdict")
    if isinstance(operator, dict):
        return operator
    return {
        "status": "UNKNOWN",
        "headline": "Hard-recommendation payload has no operator verdict.",
        "why": "Refresh hard recommendations with the current pilotBENCHY build.",
        "next_command": "apb hard-recommendations --runs-root _runs",
    }


def _performance_prediction_payload(payload: dict) -> dict:
    prediction = payload.get("performance_prediction")
    if isinstance(prediction, dict):
        return prediction
    return {
        "status": "UNKNOWN",
        "risk": "unknown",
        "deployment_expectation": "refresh_required",
        "expected_user_experience": "Refresh hard recommendations with the current build.",
    }


def _score_evidence_payload(payload: dict) -> dict:
    score_evidence = payload.get("score_evidence")
    if isinstance(score_evidence, dict):
        return score_evidence
    candidate_rankings = payload.get("candidate_rankings")
    candidates = candidate_rankings if isinstance(candidate_rankings, list) else []
    hard_recommendations = payload.get("hard_recommendations")
    recommendations = hard_recommendations if isinstance(hard_recommendations, list) else []
    return {
        "candidate_count": len(candidates),
        "scored_candidate_count": sum(
            1
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("agent_quality_score") is not None
        ),
        "proven_recommendation_count": len(recommendations),
    }


def _settings_candidate_lines(candidates: object) -> list[str]:
    if not isinstance(candidates, list) or not candidates:
        return ["Settings candidates: none"]
    lines = ["Settings candidates:"]
    for item in candidates[:5]:
        if not isinstance(item, dict):
            continue
        score = item.get("recommendation_score")
        score_label = f"{score:.4f}" if isinstance(score, int | float) else "unmeasured"
        lines.append(
            f"#{int(item.get('rank') or 0)} {item.get('profile_id') or 'unknown'} | "
            f"{item.get('status') or 'unknown'} | "
            f"{item.get('decision') or 'unknown'} | "
            f"ctx={int(item.get('context_size') or 0)} | "
            f"score={score_label}"
        )
    return lines


def _repeatability_cli_line(repeatability: dict) -> str:
    run_count = int(repeatability.get("run_count") or 0)
    run_label = "run" if run_count == 1 else "runs"
    return (
        f"Repeatability: {repeatability.get('confidence', 'unmeasured')} ({run_count} {run_label})"
    )


def _context_gate_cli_line(context_gate: object) -> str:
    if not isinstance(context_gate, dict):
        return "Context gate: unmeasured"
    return (
        f"Context gate: {context_gate.get('action') or 'unknown'} | "
        f"required={context_gate.get('required_context') or 'unknown'} | "
        f"proven={context_gate.get('proven_context') or 'none'} | "
        f"profile={context_gate.get('profile_id') or 'unknown'}"
    )


def _resource_gate_cli_line(resource_gate: object) -> str:
    if not isinstance(resource_gate, dict):
        return "Resource gate: unmeasured"
    return (
        f"Resource gate: {resource_gate.get('action') or 'unknown'} | "
        f"required={resource_gate.get('required') or 'unknown'}"
    )


def _is_mtp_model(model: Path) -> bool:
    return "mtp" in model.name.lower()
