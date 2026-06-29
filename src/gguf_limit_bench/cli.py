from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sqlite3
import time
from typing import Annotated, Callable, cast
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
    benchmark_suite_run_to_dict,
    run_benchmark_suite,
)
from gguf_limit_bench.config import (
    DEFAULT_DB_PATH,
    DEFAULT_LLAMA_BENCH,
    DEFAULT_LLAMA_CLI,
    DEFAULT_LLAMA_PERPLEXITY,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_RUNS_ROOT,
    load_config,
    with_cli_overrides,
)
from gguf_limit_bench.deployment import export_champion_profile
from gguf_limit_bench.gpu_profiles import detect_gpu_name, recommended_always_on
from gguf_limit_bench.discovery import discover_models
from gguf_limit_bench.doctor import DoctorReport, build_doctor_report
from gguf_limit_bench.evaluation_mode import (
    EvaluationMode,
    asks_questions,
    resolve_evaluation_mode,
)
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
from gguf_limit_bench.reports import write_leaderboard
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
from gguf_limit_bench.webui import serve_webui
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
    console.print(f"HF match decisions: {paths.matches}")


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
    console.print(f"HF match decisions: {paths.matches}")


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
    console.print(f"HF match decisions: {paths.matches}")


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
    run_config = _run_config_from_inputs(
        preset=preset, budget_minutes=budget_minutes, max_attempts=max_attempts
    )
    serve_webui(
        root=root,
        runs_root=runs_root,
        run_model=lambda model, options: (
            _run_one_autoresearch(
                model=model.path,
                llama_bench=llama_bench,
                llama_cli=llama_cli,
                llama_server=llama_server,
                llama_perplexity=llama_perplexity,
                runs_root=runs_root,
                budget_seconds=options.budget_minutes * 60,
                parallel_max=parallel_max,
                max_attempts=run_config.max_attempts,
                learning=learning,
                workflow_eval=workflow_eval,
                ttft_probe=ttft_probe,
                context_ladder=_context_ladder_or_none(context_ladder)
                or _context_ladder_or_none(mode_by_id(options.mode_id).context_ladder)
                or _context_ladder_or_none(run_config.context_ladder),
                benchmark_suite_plan=options.benchmark_suite_plan or benchmark_suite_plan,
                enable_mtp=model.has_mtp,
                evaluation=mode_by_id(options.mode_id).evaluation,
                forced_server_args=merge_flags(
                    merge_flags(
                        options.forced_server_args,
                        _sampler_flags_for_policy(model.path, options.sampler_policy),
                    ),
                    recommended_model_flags(model.path, search_roots=(root,)),
                ),
                champion_pack_ids=tuple(LIBRARIAN_PACK_IDS)
                if options.mode_id == "librarian_bench"
                else None,
                champion_sample_size=options.sample_size,
                champion_repeats=options.repeats,
                sampler_policy=options.sampler_policy,
            ).path
        ),
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
                        "uvx",
                        "--from",
                        "lm-eval",
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
    return receipt


def _safe_receipt_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)[:80]


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
        4096,
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
    simple_bench_max_tokens: int = 4096,
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
        benchmark_suite_plan=(
            BenchmarkSuitePlan.from_path(benchmark_suite_plan)
            if benchmark_suite_plan is not None
            else None
        ),
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


def _is_mtp_model(model: Path) -> bool:
    return "mtp" in model.name.lower()
