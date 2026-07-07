from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
import threading
from typing import Callable
from urllib.parse import quote, unquote, urlparse
import webbrowser

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

from gguf_limit_bench.discovery import ModelInfo, discover_models
from gguf_limit_bench.flight_plans import (
    DEFAULT_FLIGHT_PLAN_ID,
    default_flight_plan,
    flight_plan_by_id,
    flight_plan_payloads,
)
from gguf_limit_bench.flag_ladder import profile_descriptions, validate_extra_server_args
from gguf_limit_bench.gpu_profiles import (
    describe as describe_gpu_profile,
    detect_gpu_name,
    recommended_always_on,
    recommended_parallel,
)
from gguf_limit_bench.hard_recommendations import build_hard_recommendations
from gguf_limit_bench.hf_recommended_settings import recommended_sampler_presets
from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS
from gguf_limit_bench.modes import RUN_MODES
from gguf_limit_bench.programs import MIN_SERIOUS_CONTEXT_SIZE
from gguf_limit_bench.reports import (
    build_report_audit,
    build_verdict,
    score_summary_for_entry,
    write_leaderboard,
)
from gguf_limit_bench import run_dir as run_dir_io
from gguf_limit_bench.server_probe import kill_process_tree, process_group_kwargs
from gguf_limit_bench.telemetry import sample_telemetry
from gguf_limit_bench.tui import active_run_status

# Spawns the detached engine for a run directory and returns the process handle.
SpawnEngine = Callable[[Path], "subprocess.Popen"]
# A finished phase means a previous run dir is free to be replaced.
_DONE_PHASES = ("complete", "stopped", "failed", "aborted")

RECENT_RECEIPT_LIMIT = 8
GLOBAL_REPORTS = (
    ("Results dashboard", "results.html"),
    ("Leaderboard", "leaderboard.md"),
    ("Model comparison", "model-comparison.md"),
    ("Model comparison JSON", "model-comparison.json"),
    ("Hard recommendations", "hard-recommendations.md"),
    ("Hard recommendations JSON", "hard-recommendations.json"),
    ("QE format leaderboard", "qe-format-leaderboard.md"),
    ("QE format leaderboard JSON", "qe-format-leaderboard.json"),
    ("Flag recommendations", "flag-recommendations.md"),
    ("Flag recommendations JSON", "flag-recommendations.json"),
    ("Deployment readiness", "deployment-readiness.md"),
    ("Deployment readiness JSON", "deployment-readiness.json"),
)
RUN_ARTIFACTS = (
    ("Browser report", "report.html"),
    ("Itemized report", "itemized-report.md"),
    ("Summary", "summary.md"),
    ("Resolved plan", "resolved-plan.json"),
    ("Command", "command.txt"),
    ("Status", "status.json"),
    ("Best settings", "best-settings.json"),
    ("Machine report", "report.json"),
    ("Suite summary", "suite-summary.json"),
    ("Suite plan", "suite-plan.json"),
    ("Suite events", "events.jsonl"),
    ("Librarian summary", "librarian-suite-summary.json"),
    ("Librarian report", "librarian-suite.md"),
    ("QE format summary", "qe-format-summary.json"),
    ("QE format report", "qe-format-summary.md"),
    ("Preflight", "preflight.json"),
    ("Results", "results.md"),
    ("Results JSON", "results.json"),
)
OPTIONAL_FORCED_FLAGS = (
    ("--no-mmap", "Disable memory mapping when Windows paging makes loads unstable."),
    ("--mlock", "Ask the OS to keep model pages resident when supported."),
    ("--no-warmup", "Skip llama.cpp warmup when measuring cold-start behavior."),
)
WS_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class WebRunOptions:
    mode_id: str
    budget_minutes: int
    forced_server_args: tuple[str, ...]
    flight_plan_id: str | None = None
    show_thinking: bool = False
    stream_prompts: bool = True
    benchmark_suite_plan: Path | None = None
    repeats: int = 3
    sample_size: int = 15
    sampler_policy: str = "hf_recommended"


@dataclass
class WebRunEvent:
    at: str
    kind: str
    message: str


@dataclass
class WebRunState:
    phase: str = "idle"
    message: str = "Ready."
    selected_models: list[str] = field(default_factory=list)
    receipts: list[str] = field(default_factory=list)
    options: dict | None = None
    events: list[WebRunEvent] = field(default_factory=list)
    error: str | None = None
    stop_requested: bool = False


class WebUiState:
    def __init__(
        self,
        *,
        root: Path,
        runs_root: Path,
        spawn_engine: SpawnEngine | None = None,
        project_root: Path | None = None,
        llama_server: Path | None = None,
        llama_bench: Path | None = None,
        llama_cli: Path | None = None,
        llama_perplexity: Path | None = None,
        target_model: str | None = None,
        target_model_path: str | None = None,
        required_context: int | None = None,
    ) -> None:
        self.root = root
        self.runs_root = runs_root
        self.spawn_engine = spawn_engine or _default_spawn_engine
        self.project_root = project_root or Path.cwd()
        self.target_model = target_model
        self.target_model_path = target_model_path
        self.required_context = required_context
        # Resolved llama.cpp paths the engine should prefer over its own config.
        # Carried through run-spec.json so the detached engine can find the real
        # binaries (a cockpit launch knows them; the engine's config may not).
        self.llama_paths = {
            "llama_server": str(llama_server) if llama_server else None,
            "llama_bench": str(llama_bench) if llama_bench else None,
            "llama_cli": str(llama_cli) if llama_cli else None,
            "llama_perplexity": str(llama_perplexity) if llama_perplexity else None,
            "runs_root": str(self.runs_root),
        }
        self.active_run_dir: Path | None = None
        self.engine_process: subprocess.Popen | None = None
        self.run = WebRunState()
        self.run_configuration = run_configuration_payload()
        self._lock = threading.Lock()

    def models(self) -> list[ModelInfo]:
        return discover_models([self.root])

    def state_payload(
        self,
        *,
        target_model: str | None = None,
        target_model_path: str | None = None,
        required_context: int | None = None,
    ) -> dict:
        target_model = target_model or self.target_model
        target_model_path = target_model_path or self.target_model_path
        if required_context is None:
            required_context = self.required_context
        models = self.models()
        telemetry = sample_telemetry().to_dict()
        leaderboard = write_leaderboard(self.runs_root)
        hard_recommendations = build_hard_recommendations(
            self.runs_root,
            target_model=target_model,
            target_model_path=target_model_path,
            required_context=required_context,
        )
        target_scope = hard_recommendations.get("target_scope") or {}
        no_target_evidence = (
            bool(target_model) and target_scope.get("status") == "NO_TARGET_EVIDENCE"
        )
        verdict = build_verdict(leaderboard)
        report_audit = build_report_audit(leaderboard)
        champion = None
        if leaderboard.entries and not no_target_evidence:
            champion = {
                **score_summary_for_entry(leaderboard.champion),
                "model": leaderboard.champion.model_name,
                "score": leaderboard.champion.score,
            }
        verdict_payload = (
            hard_recommendations.get("model_gate") if no_target_evidence else asdict(verdict)
        )
        self.reattach()
        with self._lock:
            run_payload = asdict(self.run)
            active = self.active_run_dir
        if active is not None:
            status = run_dir_io.read_status(active)
            if status:
                run_payload["phase"] = status.get("phase") or run_payload["phase"]
                run_payload["message"] = _status_message(status)
            run_payload["events"] = run_payload["events"] + _tail_live_events(active)
            # Structured records the cockpit renders from (preserves type + data,
            # unlike the flattened `events` feed). Pure pass-through; no business logic.
            run_payload["live_events"] = _tail_live_records(active)
            run_payload["status"] = status or {}
        return {
            "models": [_model_payload(model) for model in models],
            "modes": [_mode_payload(mode) for mode in RUN_MODES],
            "flight_plans": flight_plan_payloads(self.project_root),
            "default_flight_plan": DEFAULT_FLIGHT_PLAN_ID,
            "default_mode": default_flight_plan().mode_id,
            "librarian_packs": list(LIBRARIAN_PACK_IDS),
            "run_configuration": self.run_configuration,
            "benchmark_suite_plans": benchmark_suite_plan_payloads(self.project_root),
            "telemetry": telemetry,
            "active_run": active_run_status(self.runs_root),
            "champion": champion,
            "verdict": verdict_payload,
            "report_audit": asdict(report_audit),
            "target_scope": target_scope,
            "operator_verdict": hard_recommendations.get("operator_verdict"),
            "performance_prediction": hard_recommendations.get("performance_prediction"),
            "score_evidence": hard_recommendations.get("score_evidence"),
            "candidate_assessment": hard_recommendations.get("candidate_assessment"),
            "candidate_rankings": hard_recommendations.get("candidate_rankings", []),
            "settings_candidates": hard_recommendations.get("settings_candidates", []),
            "repeatability": hard_recommendations.get("repeatability"),
            "context_gate": hard_recommendations.get("context_gate"),
            "resource_gate": hard_recommendations.get("resource_gate"),
            "stability_gate": hard_recommendations.get("stability_gate"),
            "proven_components": hard_recommendations.get("proven_components", []),
            "proof_runbook": hard_recommendations.get("proof_runbook", []),
            "proof_commands": hard_recommendations.get("proof_commands", []),
            "global_reports": global_report_payloads(self.runs_root),
            "receipts": recent_receipts(self.runs_root),
            "local_status": _local_status_payload(
                model_root=self.root,
                runs_root=self.runs_root,
                models=models,
                llama_paths=self.llama_paths,
                telemetry=telemetry,
                run_payload=run_payload,
            ),
            "run": run_payload,
        }

    def start_run(
        self, model_paths: list[str], mode_id: str, options_payload: dict | None = None
    ) -> tuple[bool, str]:
        models_by_path = {str(model.path): model for model in self.models()}
        unknown = [path for path in model_paths if path not in models_by_path]
        if unknown:
            return False, f"Unknown model path: {unknown[0]}"
        selected = [models_by_path[path] for path in model_paths if path in models_by_path]
        issue = validate_web_selection(selected, mode_id)
        if issue is not None:
            return False, issue
        try:
            options = build_run_options(
                mode_id, options_payload or {}, project_root=self.project_root
            )
        except ValueError as exc:
            return False, str(exc)
        with self._lock:
            if self._active_run_is_alive():
                return False, "A benchmark is already running."
            run_directory = self._new_run_dir()
            spec = _spec_payload(selected, mode_id, options, self.llama_paths)
            run_dir_io.write_spec(run_directory, spec)
            try:
                process = self.spawn_engine(run_directory)
            except Exception as exc:  # noqa: BLE001 - surface launch failure to UI
                return False, f"Could not start the engine: {exc}"
            self.active_run_dir = run_directory
            self.engine_process = process
            flight_plan = (
                flight_plan_by_id(options.flight_plan_id) if options.flight_plan_id else None
            )
            flight_plan_label = f"Flight plan: {flight_plan.label}; " if flight_plan else ""
            plan_label = (
                f"; suite plan: {options.benchmark_suite_plan.name}"
                if options.benchmark_suite_plan is not None
                else ""
            )
            events = [
                _event(
                    "configure",
                    (
                        f"{flight_plan_label}Mode: {options.mode_id}; "
                        f"budget: {options.budget_minutes} min/model{plan_label}"
                    ),
                ),
                _event(
                    "flags",
                    "Forced llama-server args: "
                    + (" ".join(options.forced_server_args) or "(none)"),
                ),
                _event("preflight", f"Sampler policy: {options.sampler_policy}"),
            ]
            self.run = WebRunState(
                phase="running",
                message=f"Engine launched for {len(selected)} model(s).",
                selected_models=[model.name for model in selected],
                options=_web_run_options_payload(options),
                events=events
                + [_event("engine", f"Detached engine started; run dir: {run_directory.name}")],
            )
        return True, "Benchmark started."

    def request_stop_after_current(self) -> tuple[bool, str]:
        with self._lock:
            if self.active_run_dir is None:
                return False, "No active benchmark run can be stopped."
            run_dir_io.write_control(self.active_run_dir, "stop")
            self.run.stop_requested = True
            self.run.events.append(
                _event("stop", "Stop requested. The current benchmark item will finish first.")
            )
            return True, "Stop requested after current item."

    def request_abort(self) -> tuple[bool, str]:
        with self._lock:
            if self.active_run_dir is None:
                return False, "No active benchmark run can be aborted."
            run_dir_io.write_control(self.active_run_dir, "abort")
            if self.engine_process is not None:
                kill_process_tree(self.engine_process)
            # A hard-killed engine can't update its own status, so stamp the final
            # phase here — otherwise the run dir stays "running" forever and the
            # cockpit shows a stale live run after an abort.
            run_dir_io.write_status(self.active_run_dir, phase="aborted")
            self.run.stop_requested = True
            self.run.events.append(_event("abort", "Abort requested. Killing the engine now."))
            return True, "Run aborted."

    def reattach(self) -> None:
        """Adopt a live engine run dir if we have none (survives browser refresh)."""
        with self._lock:
            if self.active_run_dir is not None:
                return
            found = _find_live_run_dir(self.runs_root)
            if found is not None:
                self.active_run_dir = found
                self.run = WebRunState(
                    phase="running",
                    message="Reattached to a running engine.",
                    events=[_event("engine", f"Reattached to run dir: {found.name}")],
                )

    def _active_run_is_alive(self) -> bool:
        if self.active_run_dir is None:
            return False
        status = run_dir_io.read_status(self.active_run_dir)
        if not status:
            return True  # just launched; the engine has not written a heartbeat yet
        if status.get("phase") in _DONE_PHASES:
            return False
        return run_dir_io.engine_is_alive(status, now=datetime.now())

    def _new_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_directory = self.runs_root / f"{stamp}-cockpit"
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory


def _default_spawn_engine(run_directory: Path) -> subprocess.Popen:
    """Launch the engine as a detached subprocess in its own process group."""
    return subprocess.Popen(
        [sys.executable, "-m", "gguf_limit_bench", "engine", "--run-dir", str(run_directory)],
        **process_group_kwargs(),
    )


def _spec_payload(
    selected: list[ModelInfo],
    mode_id: str,
    options: "WebRunOptions",
    llama_paths: dict[str, str | None] | None = None,
) -> dict:
    plan = options.benchmark_suite_plan
    return {
        "models": [{"path": str(model.path), "has_mtp": bool(model.has_mtp)} for model in selected],
        "mode": mode_id,
        "options": {
            "budget_minutes": options.budget_minutes,
            "forced_server_args": list(options.forced_server_args),
            "benchmark_suite_plan": str(plan) if plan is not None else None,
            "show_thinking": options.show_thinking,
            "stream_prompts": options.stream_prompts,
            "repeats": options.repeats,
            "sample_size": options.sample_size,
            "sampler_policy": options.sampler_policy,
        },
        "paths": _paths_block(llama_paths),
    }


def _paths_block(llama_paths: dict[str, str | None] | None) -> dict[str, str | None]:
    """Normalize the resolved llama path block written into run-spec.json.

    Every key is always present with a string path or null, so the engine can
    read it uniformly. A None mapping (no paths provided) yields all-null."""
    source = llama_paths or {}
    return {
        key: source.get(key)
        for key in ("llama_server", "llama_bench", "llama_cli", "llama_perplexity", "runs_root")
    }


def _status_row(
    row_id: str, label: str, value: str, state: str, detail: str = ""
) -> dict[str, str]:
    return {
        "id": row_id,
        "label": label,
        "value": value,
        "state": state,
        "detail": detail,
    }


def _local_status_payload(
    *,
    model_root: Path,
    runs_root: Path,
    models: list[ModelInfo],
    llama_paths: dict[str, str | None],
    telemetry: dict[str, int | float | None],
    run_payload: dict,
) -> dict[str, object]:
    rows: list[dict[str, str]] = [
        _status_row("api", "Local API", "online", "ok", "The browser reached /api/state."),
    ]

    phase = str(run_payload.get("phase") or "idle")
    if phase == "running":
        rows.append(
            _status_row(
                "engine",
                "Engine seam",
                "running",
                "ok",
                "Detached engine status is being read from the active run directory.",
            )
        )
    elif phase in {"failed", "aborted"}:
        rows.append(
            _status_row(
                "engine",
                "Engine seam",
                phase,
                "bad",
                "The last detached engine state needs attention.",
            )
        )
    else:
        rows.append(
            _status_row(
                "engine",
                "Engine seam",
                "detached idle",
                "info",
                "No engine is running until you start a benchmark.",
            )
        )

    if not model_root.exists():
        rows.append(
            _status_row(
                "models",
                "Model root",
                "missing",
                "bad",
                f"Model folder does not exist: {model_root}",
            )
        )
    elif not models:
        rows.append(
            _status_row(
                "models",
                "Model root",
                "0 models",
                "warn",
                f"No GGUF models were found under {model_root}.",
            )
        )
    else:
        rows.append(
            _status_row(
                "models",
                "Model root",
                f"{len(models)} model{'s' if len(models) != 1 else ''}",
                "ok",
                str(model_root),
            )
        )

    if runs_root.is_dir() and os.access(runs_root, os.W_OK):
        rows.append(_status_row("runs", "Run storage", "writable", "ok", str(runs_root)))
    elif runs_root.exists():
        rows.append(
            _status_row(
                "runs",
                "Run storage",
                "not writable",
                "bad",
                f"Run folder exists but is not writable: {runs_root}",
            )
        )
    else:
        rows.append(
            _status_row(
                "runs",
                "Run storage",
                "will create",
                "info",
                f"Run folder will be created when needed: {runs_root}",
            )
        )

    configured_paths = {
        key: value for key, value in llama_paths.items() if key != "runs_root" and value
    }
    missing_paths = [
        key for key, value in configured_paths.items() if not Path(str(value)).is_file()
    ]
    if missing_paths:
        rows.append(
            _status_row(
                "llama",
                "llama.cpp",
                f"{len(missing_paths)} missing",
                "warn",
                "Configured llama.cpp binary path(s) are missing: " + ", ".join(missing_paths),
            )
        )
    elif configured_paths:
        rows.append(
            _status_row(
                "llama",
                "llama.cpp",
                f"{len(configured_paths)} paths",
                "ok",
                "Configured llama.cpp binaries will be passed to the detached engine.",
            )
        )
    else:
        rows.append(
            _status_row(
                "llama",
                "llama.cpp",
                "engine resolves",
                "info",
                "No explicit binary paths were provided; the detached engine will use its config/PATH.",
            )
        )

    if telemetry.get("gpu_util_percent") is None:
        rows.append(
            _status_row(
                "telemetry",
                "Telemetry",
                "CPU/RAM only",
                "info",
                "nvidia-smi did not return GPU metrics for this sample.",
            )
        )
    else:
        rows.append(
            _status_row(
                "telemetry",
                "Telemetry",
                "GPU sampled",
                "ok",
                "CPU, RAM, and GPU telemetry are available.",
            )
        )

    if any(row["state"] == "bad" for row in rows):
        overall = {
            "state": "bad",
            "label": "Blocked",
            "detail": "One or more local prerequisites need attention.",
        }
    elif any(row["state"] == "warn" for row in rows):
        overall = {
            "state": "warn",
            "label": "Needs attention",
            "detail": "The cockpit can load, but one local check is weak.",
        }
    else:
        overall = {
            "state": "ok",
            "label": "Ready",
            "detail": "Local cockpit state refreshed.",
        }
    return {"overall": overall, "rows": rows}


def _tail_live_events(run_directory: Path, *, limit: int = 80) -> list[dict]:
    path = run_directory / run_dir_io.LIVE_FILE
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    return [_receipt_event_payload(line) for line in lines if line.strip()]


def _tail_live_records(run_directory: Path, *, limit: int = 600) -> list[dict]:
    """Structured ``{at, type, data}`` records the cockpit renders from.

    Unlike ``_tail_live_events`` (which flattens to ``{at, kind, message}`` for the
    legacy feed), this preserves the event ``type`` and full ``data`` so the client
    can fold per-question thinking/answer/score, running score, and the model queue.
    Tails the last ``limit`` lines; malformed lines are skipped."""
    path = run_directory / run_dir_io.LIVE_FILE
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    records: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        records.append(
            {
                "at": row.get("time"),
                "type": row.get("type", ""),
                "data": row.get("data", {}),
            }
        )
    return records


def _status_message(status: dict) -> str:
    phase = status.get("phase", "running")
    model = status.get("model")
    index = status.get("model_index")
    total = status.get("model_total")
    if model and index and total:
        return f"{phase}: {index}/{total} {model}"
    return str(phase)


def _find_live_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir() and (path / run_dir_io.STATUS_FILE).is_file()
    ]
    for candidate in sorted(candidates, key=_safe_mtime, reverse=True):
        status = run_dir_io.read_status(candidate)
        if status.get("phase") in _DONE_PHASES:
            continue
        if run_dir_io.engine_is_alive(status, now=datetime.now()):
            return candidate
    return None


def validate_web_selection(selected: list[ModelInfo], mode_id: str) -> str | None:
    if not selected:
        return "Select at least one model first."
    # pilotBENCHY benchmarks any GGUF model on agent workloads. There is no
    # hardcoded Gemma-vs-Qwen requirement; one model runs, two or more compare.
    return None


def build_run_options(
    mode_id: str, payload: dict, *, project_root: Path | None = None
) -> WebRunOptions:
    flight_plan_id = _flight_plan_id_from_payload(payload)
    flight_plan = flight_plan_by_id(flight_plan_id) if flight_plan_id is not None else None
    if flight_plan is not None:
        mode_id = flight_plan.mode_id
    mode = next((item for item in RUN_MODES if item.id == mode_id), None)
    if mode is None:
        raise ValueError(f"Unknown run mode: {mode_id}")
    budget_minutes = _int_option(
        payload,
        "budget_minutes",
        default=flight_plan.budget_minutes if flight_plan is not None else mode.budget_minutes,
        minimum=1,
        maximum=24 * 60,
        label="Budget",
    )
    gpu_name = detect_gpu_name()
    default_forced_args = recommended_always_on(gpu_name)
    forced_args = _string_tuple_option(payload, "forced_server_args", default=default_forced_args)
    validate_extra_server_args(forced_args)
    allowed = set(default_forced_args)
    allowed.update(flag for flag, _description in OPTIONAL_FORCED_FLAGS)
    unknown = [arg for arg in forced_args if arg.startswith("--") and arg not in allowed]
    if unknown:
        raise ValueError(f"Unsupported forced flag from Web UI: {unknown[0]}")
    raw_suite_plan = payload.get("benchmark_suite_plan")
    if raw_suite_plan in (None, "") and flight_plan is not None:
        raw_suite_plan = flight_plan.default_benchmark_suite_plan
    benchmark_suite_plan = resolve_benchmark_suite_plan(project_root or Path.cwd(), raw_suite_plan)
    sample_size = _int_option(
        payload,
        "sample_size",
        default=15,
        minimum=1,
        maximum=200,
        label="Sample size",
        suffix=" questions per pack",
    )
    repeats = _int_option(payload, "repeats", default=3, minimum=1, maximum=20, label="Repeats")
    sampler_policy = _sampler_policy_from_payload(payload)
    return WebRunOptions(
        mode_id=mode_id,
        budget_minutes=budget_minutes,
        forced_server_args=forced_args,
        flight_plan_id=flight_plan_id,
        show_thinking=_bool_option(payload, "show_thinking", default=False),
        stream_prompts=_bool_option(payload, "stream_prompts", default=True),
        benchmark_suite_plan=benchmark_suite_plan,
        repeats=repeats,
        sample_size=sample_size,
        sampler_policy=sampler_policy,
    )


def _int_option(
    payload: dict,
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
    label: str,
    suffix: str = "",
) -> int:
    raw_value = payload.get(name, default)
    if raw_value in (None, ""):
        raw_value = default
    if isinstance(raw_value, bool):
        raise ValueError(f"{label} must be a number.")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not minimum <= value <= maximum:
        if suffix:
            raise ValueError(f"{label} must be between {minimum} and {maximum}{suffix}.")
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _string_tuple_option(payload: dict, name: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = payload.get(name, default)
    if raw_value in (None, ""):
        return tuple(default)
    if not isinstance(raw_value, list | tuple):
        raise ValueError(f"{name} must be a list of strings.")
    if not all(isinstance(item, str) for item in raw_value):
        raise ValueError(f"{name} entries must be strings.")
    return tuple(raw_value)


def _sampler_policy_from_payload(payload: dict) -> str:
    raw_value = payload.get("sampler_policy")
    if raw_value in (None, ""):
        return "hf_recommended"
    if not isinstance(raw_value, str):
        raise ValueError("sampler_policy must be a string.")
    if raw_value in {"hf_recommended", "runtime_defaults"} or raw_value.startswith("hf:"):
        return raw_value
    raise ValueError(f"Unsupported sampler policy from Web UI: {raw_value}")


def _bool_option(payload: dict, name: str, *, default: bool) -> bool:
    raw_value = payload.get(name, default)
    if raw_value in (None, ""):
        return default
    if not isinstance(raw_value, bool):
        raise ValueError(f"{name} must be true or false.")
    return raw_value


def _flight_plan_id_from_payload(payload: dict) -> str | None:
    raw_value = payload.get("flight_plan_id")
    if raw_value in (None, ""):
        return None
    if not isinstance(raw_value, str):
        raise ValueError("flight_plan_id must be a string.")
    try:
        flight_plan_by_id(raw_value)
    except KeyError:
        raise ValueError(f"Unknown flight plan: {raw_value}") from None
    return raw_value


def _web_run_options_payload(options: WebRunOptions) -> dict:
    payload = asdict(options)
    if options.benchmark_suite_plan is not None:
        payload["benchmark_suite_plan"] = str(options.benchmark_suite_plan)
    return payload


def resolve_benchmark_suite_plan(project_root: Path, raw_path: object) -> Path | None:
    if raw_path in (None, ""):
        return None
    if not isinstance(raw_path, str):
        raise ValueError("benchmark_suite_plan must be a string path.")
    plans_root = (project_root / "benchmarks" / "plans").resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(plans_root)
    except (OSError, ValueError):
        raise ValueError("Benchmark suite plan must be under benchmarks/plans.") from None
    if resolved.suffixes[-2:] != [".plan", ".json"]:
        raise ValueError("Benchmark suite plan must be a .plan.json file.")
    if not resolved.is_file():
        raise ValueError(f"Benchmark suite plan not found: {raw_path}")
    return resolved


def run_configuration_payload() -> dict:
    gpu_name = detect_gpu_name()
    standard_args = recommended_always_on(gpu_name)
    parallel = recommended_parallel(gpu_name)
    ladder = profile_descriptions(
        context_size=MIN_SERIOUS_CONTEXT_SIZE,
        parallel_max=parallel,
        extra_server_args=standard_args,
        enable_mtp=True,
    )
    return {
        "gpu_name": gpu_name or "Unknown GPU",
        "gpu_profile": describe_gpu_profile(gpu_name or "Unknown GPU"),
        "recommended_parallel": parallel,
        "standard_forced_args": _flag_choices(standard_args, selected=True),
        "optional_forced_args": [
            {"flag": flag, "description": description, "selected": False}
            for flag, description in OPTIONAL_FORCED_FLAGS
        ],
        "flag_ladder": [
            {
                "name": profile.name,
                "hypothesis": profile.hypothesis,
                "flags": profile.settings.to_dict(),
            }
            for profile in ladder
        ],
    }


def websocket_message(message_type: str, payload: dict | None = None) -> dict:
    return {"type": message_type, "payload": payload or {}}


def websocket_error(message: str) -> dict:
    return {"type": "error", "message": message}


def create_web_app(state: WebUiState) -> FastAPI:
    app = FastAPI(title="pilotBENCHY local cockpit", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(state.state_payload())

    @app.post("/api/start")
    async def api_start(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"ok": False, "message": "Request body must be valid JSON."}, 400)
        ok, message = start_run_from_payload(state, payload)
        return JSONResponse({"ok": ok, "message": message}, 200 if ok else 400)

    @app.post("/api/stop-after-current")
    async def api_stop_after_current() -> JSONResponse:
        ok, message = state.request_stop_after_current()
        return JSONResponse({"ok": ok, "message": message}, 200 if ok else 400)

    @app.post("/api/abort")
    async def api_abort() -> JSONResponse:
        ok, message = state.request_abort()
        return JSONResponse({"ok": ok, "message": message}, 200 if ok else 400)

    @app.get("/runs/{encoded_relative_path:path}")
    async def run_artifact(encoded_relative_path: str) -> Response:
        artifact = resolve_run_artifact(state.runs_root, encoded_relative_path)
        if artifact is None:
            return Response(status_code=HTTPStatus.NOT_FOUND)
        data = artifact.read_bytes()
        return Response(content=data, media_type=_artifact_content_type(artifact))

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "hello", "protocol": WS_PROTOCOL_VERSION})
        await websocket.send_json(websocket_message("state", state.state_payload()))
        try:
            while True:
                try:
                    message = await websocket.receive_json()
                except json.JSONDecodeError:
                    await websocket.send_json(
                        websocket_error("WebSocket message must be valid JSON.")
                    )
                    continue
                response = await handle_websocket_command(state, message)
                if response is not None:
                    await websocket.send_json(response)
        except WebSocketDisconnect:
            return

    return app


def start_run_from_payload(state: WebUiState, payload: object) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Request body must be a JSON object."
    raw_model_paths = payload.get("model_paths", [])
    if not isinstance(raw_model_paths, list):
        return False, "model_paths must be a list."
    if not all(isinstance(path, str) for path in raw_model_paths):
        return False, "model_paths entries must be strings."
    raw_options = payload.get("options", {})
    if raw_options in (None, ""):
        raw_options = {}
    if not isinstance(raw_options, dict):
        return False, "options must be a JSON object."
    options = raw_options
    if "flight_plan_id" in payload and "flight_plan_id" not in options:
        options = {**options, "flight_plan_id": payload["flight_plan_id"]}
    default_mode = default_flight_plan().mode_id
    return state.start_run(
        model_paths=raw_model_paths,
        mode_id=str(payload.get("mode_id", default_mode)),
        options_payload=options,
    )


async def handle_websocket_command(state: WebUiState, message: object) -> dict | None:
    if not isinstance(message, dict):
        return websocket_error("WebSocket message must be a JSON object.")
    message_type = str(message.get("type") or "")
    if message_type in {"subscribe", "refresh"}:
        return websocket_message("state", state.state_payload())
    if message_type == "start_run":
        ok, response_message = start_run_from_payload(state, message)
        return {"type": "run_started", "ok": ok, "message": response_message}
    if message_type == "stop_after_current":
        ok, response_message = state.request_stop_after_current()
        return {"type": "stop_after_current", "ok": ok, "message": response_message}
    if message_type == "abort":
        ok, response_message = state.request_abort()
        return {"type": "abort", "ok": ok, "message": response_message}
    return websocket_error(f"Unknown WebSocket message type: {message_type}")


def serve_webui(
    *,
    root: Path,
    runs_root: Path,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    llama_server: Path | None = None,
    llama_bench: Path | None = None,
    llama_cli: Path | None = None,
    llama_perplexity: Path | None = None,
    target_model: str | None = None,
    target_model_path: str | None = None,
    required_context: int | None = None,
) -> str:
    resolved_port = port if port != 0 else _free_local_port(host)
    state = WebUiState(
        root=root,
        runs_root=runs_root,
        llama_server=llama_server,
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        llama_perplexity=llama_perplexity,
        target_model=target_model,
        target_model_path=target_model_path,
        required_context=required_context,
    )
    app = create_web_app(state)
    config = uvicorn.Config(app, host=host, port=resolved_port, log_level="warning")
    server = uvicorn.Server(config)
    url = f"http://{host}:{resolved_port}/"
    if open_browser:
        webbrowser.open(url)
    server.run()
    return url


def _free_local_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _handler_for(state: WebUiState):
    class PilotBenchHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlparse(self.path).path
            if path == "/":
                self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
                return
            if path == "/api/state":
                self._send_json(state.state_payload())
                return
            if path.startswith("/runs/"):
                artifact = resolve_run_artifact(state.runs_root, path.removeprefix("/runs/"))
                if artifact is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_file(artifact)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlparse(self.path).path
            if path != "/api/start":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self._read_json()
            except ValueError as exc:
                self._send_json({"ok": False, "message": str(exc)}, status=400)
                return
            ok, message = start_run_from_payload(state, payload)
            self._send_json({"ok": ok, "message": message}, status=200 if ok else 400)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return payload

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, text: str, *, content_type: str) -> None:
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path) -> None:
            data = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix.lower() in {".md", ".txt", ".tsv"}:
                content_type = "text/plain; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return PilotBenchHandler


def global_report_payloads(runs_root: Path) -> list[dict]:
    reports: list[dict] = []
    for label, name in GLOBAL_REPORTS:
        path = runs_root / name
        if path.is_file():
            reports.append({"label": label, "url": _runs_url(path, runs_root)})
    return reports


def benchmark_suite_plan_payloads(project_root: Path) -> list[dict]:
    plans_root = project_root / "benchmarks" / "plans"
    if not plans_root.exists():
        return []
    payloads: list[dict] = []
    for path in sorted(plans_root.glob("*.plan.json")):
        name = path.name
        description = ""
        warning = ""
        plan_kind = ""
        requires = ""
        score_contract = ""
        context = 0
        context_target = ""
        phases: list[str] = []
        harnesses: list[str] = []
        task_count = 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            name = str(data.get("name") or data.get("title") or path.name)
            description = str(data.get("description") or "")
            raw_settings = data.get("settings")
            settings = raw_settings if isinstance(raw_settings, dict) else {}
            plan_kind = str(settings.get("plan_kind") or data.get("plan_kind") or "")
            requires = str(settings.get("requires") or data.get("requires") or "")
            score_contract = str(settings.get("score_contract") or "")
            context = _int_option_value(data.get("context") or settings.get("context_size"))
            context_target = str(settings.get("context_target") or "")
            raw_tasks = data.get("tasks")
            tasks = raw_tasks if isinstance(raw_tasks, list) else []
            task_count = len(tasks)
            phases = sorted({str(task.get("phase")) for task in tasks if isinstance(task, dict)})
            harnesses = sorted(
                {str(task.get("harness")) for task in tasks if isinstance(task, dict)}
            )
            warning = _plan_warning(data, description, requires)
        payloads.append(
            {
                "path": str(path),
                "filename": path.name,
                "name": name,
                "description": description,
                "plan_kind": plan_kind,
                "requires": requires,
                "context": context,
                "context_target": context_target,
                "score_contract": score_contract,
                "task_count": task_count,
                "phases": phases,
                "harnesses": harnesses,
                "warning": warning,
            }
        )
    return payloads


def _int_option_value(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if not isinstance(value, int | float | str | bytes | bytearray):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _plan_warning(data: dict, description: str, requires: str = "") -> str:
    text = (
        json.dumps(data, ensure_ascii=True).lower()
        + " "
        + description.lower()
        + " "
        + requires.lower()
    )
    if "endpoint" in text:
        return requires or description
    if "external" in text or "uvx" in text:
        return "This plan may call an external benchmark tool."
    if "heavy" in text:
        return "This plan may take a long time."
    return ""


def _event(kind: str, message: str) -> WebRunEvent:
    return WebRunEvent(
        at=datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        kind=kind,
        message=message,
    )


def _flag_choices(args: tuple[str, ...], *, selected: bool) -> list[dict]:
    choices: list[dict] = []
    index = 0
    while index < len(args):
        flag = args[index]
        value = ""
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            value = args[index + 1]
            index += 1
        choices.append(
            {
                "flag": flag,
                "value": value,
                "selected": selected,
                "description": _flag_description(flag, value),
            }
        )
        index += 1
    return choices


def _flag_description(flag: str, value: str) -> str:
    descriptions = {
        "--flash-attn": "Flash attention for lower memory pressure and better throughput.",
        "--kv-unified": "Unified KV cache behavior used by the serious local-worker profile.",
        "--cache-type-k": "K-cache quantization for more usable context on fixed VRAM.",
        "--cache-type-v": "V-cache quantization for more usable context on fixed VRAM.",
        "--jinja": "Use llama.cpp Jinja chat-template handling.",
        "--gpu-layers": "Offload model layers to GPU.",
    }
    suffix = f" Value: {value}." if value else ""
    return descriptions.get(flag, "Standard llama.cpp server argument.") + suffix


def recent_receipts(runs_root: Path, *, limit: int = RECENT_RECEIPT_LIMIT) -> list[dict]:
    if not runs_root.exists():
        return []
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    ordered = sorted(candidates, key=_safe_mtime, reverse=True)[:limit]
    return [_receipt_payload(path, runs_root) for path in ordered]


def receipt_event_payloads(runs_root: Path, *, limit: int = 40) -> list[dict]:
    if not runs_root.exists():
        return []
    # Sort newest first by mtime, breaking ties on the (timestamped) directory
    # name so two receipts created in the same mtime tick still order correctly.
    receipts = sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        key=lambda path: (_safe_mtime(path), path.name),
        reverse=True,
    )
    for receipt in receipts:
        events_path = receipt / "events.jsonl"
        if not events_path.is_file():
            continue
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()[-limit:]
        except OSError:
            return []
        return [_receipt_event_payload(line) for line in lines if line.strip()]
    return []


def _receipt_event_payload(line: str) -> dict:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return {"at": "--", "kind": "receipt", "message": line[-240:]}
    return {
        "at": _event_time_label(payload.get("time")),
        "kind": str(payload.get("type") or "receipt"),
        "message": _receipt_event_message(payload.get("data")),
    }


def _event_time_label(value: object) -> str:
    text = str(value or "--")
    if "T" in text:
        return text.split("T", 1)[1].split(".", 1)[0]
    return text


def _receipt_event_message(data: object) -> str:
    if isinstance(data, dict):
        if "model" in data:
            return str(data["model"])
        if "error" in data:
            return str(data["error"])
        return json.dumps(data, ensure_ascii=True, sort_keys=True)[:320]
    if data is None:
        return ""
    return str(data)[:320]


def resolve_run_artifact(runs_root: Path, encoded_relative_path: str) -> Path | None:
    try:
        runs_root_resolved = runs_root.resolve()
        relative_path = PurePosixPath(unquote(encoded_relative_path))
        if relative_path.is_absolute():
            return None
        candidate = (runs_root_resolved / Path(*relative_path.parts)).resolve()
        candidate.relative_to(runs_root_resolved)
    except (OSError, ValueError):
        return None
    if not candidate.is_file():
        return None
    return candidate


def _model_payload(model: ModelInfo) -> dict:
    tiny_file = 0 < model.size_bytes < 1024 * 1024
    size_label = "tiny file" if tiny_file else _size_label(model.size_gb)
    size_display = f"{size_label} GB" if size_label not in {"unknown", "tiny file"} else size_label
    return {
        "path": str(model.path),
        "name": model.name,
        "family": model.family,
        "parameters": model.parameters,
        "quant": model.quant,
        "context_label": "unknown",
        "size_bytes": model.size_bytes,
        "size_gb": model.size_gb,
        "size_label": size_label,
        "size_display": size_display,
        "file_label": model.path.name,
        "size_warning": tiny_file,
        "is_moe": model.is_moe,
        "has_mtp": model.has_mtp,
        "has_vision": model.has_vision,
        "sampler_presets": recommended_sampler_presets(model.path),
    }


def _mode_payload(mode) -> dict:
    return {
        "id": mode.id,
        "label": mode.label,
        "description": mode.description,
        "budget_minutes": mode.budget_minutes,
    }


def _size_label(size_gb: float) -> str:
    if size_gb <= 0:
        return "unknown"
    if size_gb < 0.01:
        return "<0.01"
    return f"{size_gb:.2f}"


def _receipt_payload(path: Path, runs_root: Path) -> dict:
    artifacts = [
        {"label": label, "url": _runs_url(artifact, runs_root)}
        for label, name in RUN_ARTIFACTS
        if (artifact := path / name).is_file()
    ]
    best = _read_best_settings(path / "best-settings.json")
    return {
        "run_id": path.name,
        "model": best.get("model_name") or best.get("model") or path.name,
        "status": best.get("status") or "receipt",
        "score": best.get("score"),
        "modified": _mtime_label(path),
        "path": str(path),
        "artifacts": artifacts,
    }


def _read_best_settings(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    model = Path(str(payload.get("model", ""))).name
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    score = result.get("agent_bench_score") or result.get("score")
    return {
        "model": str(payload.get("model", "")),
        "model_name": model,
        "status": str(payload.get("status") or result.get("failure") or "recorded"),
        "score": score,
    }


def _runs_url(path: Path, runs_root: Path) -> str:
    relative = path.resolve().relative_to(runs_root.resolve()).as_posix()
    return f"/runs/{quote(relative)}"


def _artifact_content_type(path: Path) -> str:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if path.suffix.lower() in {".md", ".txt", ".tsv"}:
        return "text/plain; charset=utf-8"
    return content_type


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _mtime_label(path: Path) -> str:
    return datetime.fromtimestamp(_safe_mtime(path), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230e141c'/%3E%3Cpath d='M7 21h18M7 16h18M7 11h18' stroke='%2354d2bd' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E" />
  <title>pilotBENCHY</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1015;
      --rail: #090e13;
      --panel: #141a21;
      --panel-2: #10161d;
      --panel-3: #0c1117;
      --line: #2a343f;
      --line-soft: rgba(255,255,255,.06);
      --text: #e8edf3;
      --muted: #9aa8b7;
      --faint: #657282;
      --teal: #20c4cf;
      --teal-dim: #2c6c63;
      --amber: #f4b860;
      --amber-dim: #7a5a2c;
      --bad: #ff7373;
      --good: #79d18a;
      --shadow: 0 24px 70px rgba(0,0,0,.28);
    }
    body.sepia {
      --bg: #16120f;
      --panel: #211b16;
      --panel-2: #18130f;
      --panel-3: #120f0c;
      --line: #4a4037;
      --text: #eee2d1;
      --muted: #b8aa98;
      --faint: #8f8170;
      --teal: #d0b06f;
      --teal-dim: #6b5935;
      --amber: #c9925b;
      --amber-dim: #735033;
      --bad: #e98570;
      --good: #a8c47a;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 42% 0%, rgba(32,196,207,.10), transparent 34%),
        linear-gradient(180deg, rgba(255,255,255,.025), transparent 260px),
        var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, Segoe UI, Arial, sans-serif;
    }
    .shell { display: grid; grid-template-columns: 252px minmax(0, 1fr); min-height: 100vh; }
    aside {
      border-right: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,.035), transparent 180px),
        var(--rail);
      padding: 22px 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .brand { font-size: 26px; font-weight: 850; letter-spacing: 0; margin-bottom: 2px; display:flex; align-items:center; gap:8px; }
    .brand-mark {
      width: 27px; height: 27px; border-radius: 50%;
      border: 2px solid var(--teal); color: var(--teal);
      display: inline-grid; place-items: center; font-size: 15px; box-shadow: 0 0 24px rgba(32,196,207,.22);
    }
    .brand-sub { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    .version-row { display:flex; gap:10px; align-items:center; margin-top:12px; }
    .pill {
      display:inline-flex; align-items:center; gap:6px; border-radius:999px;
      padding:3px 9px; border:1px solid var(--line); color:var(--muted);
      background:rgba(255,255,255,.035); font-size:12px; font-weight:700;
    }
    .pill.local { color:var(--good); border-color:rgba(121,209,138,.22); background:rgba(121,209,138,.08); }
    .nav { display:grid; gap:4px; }
    .navlink {
      display:flex; align-items:center; gap:11px;
      padding:10px 12px; border-radius:6px; color:var(--muted);
      text-decoration:none; font-weight:700;
    }
    .navlink.active { color:var(--teal); background:rgba(255,255,255,.11); }
    .navico { width:18px; text-align:center; color:inherit; }
    .navitem {
      display: flex; justify-content: space-between; gap: 12px;
      padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,.05); color: var(--muted);
    }
    .navitem b { color: var(--text); font-weight: 700; }
    .status-block { border-top:1px solid var(--line); padding-top:12px; margin-top:2px; }
    .status-title { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; margin-bottom:8px; }
    .status-row { display:grid; grid-template-columns: 1fr auto 10px; gap:8px; align-items:center; color:var(--muted); font-size:12px; padding:4px 0; }
    .status-row strong { color:var(--text); font-weight:600; text-align:right; }
    .status-row[data-state="info"] strong { color:var(--muted); }
    .status-row[data-state="warn"] strong { color:var(--amber); }
    .status-row[data-state="bad"] strong { color:var(--bad); }
    .ok-dot { width:7px; height:7px; border-radius:50%; background:var(--good); box-shadow:0 0 12px rgba(121,209,138,.45); }
    .ok-dot.state-info { background:var(--teal); box-shadow:0 0 12px rgba(32,196,207,.42); }
    .ok-dot.state-warn { background:var(--amber); box-shadow:0 0 12px rgba(244,184,96,.42); }
    .ok-dot.state-bad { background:var(--bad); box-shadow:0 0 12px rgba(255,115,115,.42); }
    .ok-dot.state-muted { background:var(--faint); box-shadow:none; }
    .system-check { margin:10px 0 0; width:100%; background:var(--panel-2); color:var(--text); border:1px solid var(--line); }
    .rail-foot { margin-top:auto; border-top:1px solid var(--line); padding-top:12px; display:flex; justify-content:space-between; color:var(--muted); font-size:12px; }
    .rail-note {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      color: var(--muted);
      background: var(--panel-2);
      font-size: 12px;
    }
    .rail-note strong { display: block; color: var(--text); margin-bottom: 4px; }
    main { width: min(100%, 1920px); margin: 0 auto; padding: 4px 8px 10px; overflow-x:hidden; }
    header { display: none; }
    h1 { margin: 0; font-size: clamp(32px, 2.4vw, 50px); line-height: 0.96; letter-spacing: 0; }
    .sub { margin-top: 7px; color: var(--muted); max-width: 840px; overflow-wrap: anywhere; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.8fr) minmax(360px, .96fr); gap: 8px; align-items: stretch; }
    .grid > *, .side, .panel { min-width: 0; }
    .grid > .panel:first-child { align-self: stretch; }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,.026), transparent 96px), var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      box-shadow: 0 16px 60px rgba(0,0,0,.20);
    }
    .panel h2 { margin: 0; padding: 12px 16px; font-size: 15px; border-bottom: 1px solid var(--line); }
    .library-head { display:grid; grid-template-columns: auto auto 1fr auto auto; gap:10px; align-items:center; padding:10px 14px; border-bottom:1px solid var(--line); }
    .library-title { font-size:16px; font-weight:800; }
    .library-count { color:var(--muted); font-size:12px; }
    .searchbox {
      justify-self:end; width:min(360px,100%); display:flex; align-items:center; gap:8px;
      border:1px solid var(--line); border-radius:5px; background:var(--panel-3); padding:6px 10px; color:var(--muted);
    }
    .searchbox input { width:100%; border:0; outline:0; background:transparent; color:var(--text); font:inherit; padding:0; }
    .section-title { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .count-chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 9px;
      color: var(--muted);
      background: var(--panel-2);
      font-size: 12px;
      white-space: nowrap;
    }
    .model-table-wrap { overflow:auto; height:390px; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td { text-align: left; padding: 7px 12px; border-bottom: 1px solid rgba(255,255,255,.075); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    th { color: var(--muted); font-size: 12px; font-weight: 700; }
    td { vertical-align: middle; }
    th:nth-child(1), td:nth-child(1) { width: 50px; }
    th:nth-child(3), td:nth-child(3) { width: 86px; text-align:right; }
    th:nth-child(4), td:nth-child(4) { width: 92px; }
    th:nth-child(5), td:nth-child(5) { width: 86px; }
    th:nth-child(6), td:nth-child(6) { width: 92px; text-align:right; }
    th:nth-child(7), td:nth-child(7) { width: 142px; }
    tr { cursor: pointer; }
    tr:hover td { background: rgba(84,210,189,.06); }
    tr.selected td {
      background: linear-gradient(90deg, rgba(32,196,207,.18), rgba(32,196,207,.07));
      box-shadow: inset 3px 0 0 var(--teal);
    }
    input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--teal); }
    .chip { display: inline-block; border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px; color: var(--muted); }
    .chip.warn { color: var(--amber); border-color: var(--amber-dim); background: rgba(244,184,96,.06); }
    .size-compact { display:none; }
    .qwen { color: var(--teal); }
    .gemma { color: var(--amber); }
    .side { display: grid; gap: 8px; }
    .body { padding: 14px 16px; }
    select, button {
      width: 100%; border-radius: 6px; border: 1px solid var(--line);
      background: var(--panel-2); color: var(--text); padding: 8px 10px;
      font: inherit;
    }
    button {
      margin-top: 12px; background: var(--teal); color: #07100e; font-weight: 800;
      cursor: pointer; border-color: transparent;
      transition: transform .14s ease, background-color .14s ease, border-color .14s ease, box-shadow .14s ease, color .14s ease;
    }
    button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(0,0,0,.18); }
    button:active:not(:disabled) { transform: translateY(0); box-shadow: inset 0 1px 0 rgba(0,0,0,.28); }
    button:focus-visible, a:focus-visible, summary:focus-visible, input:focus-visible, select:focus-visible {
      outline: 2px solid rgba(32,196,207,.72);
      outline-offset: 2px;
    }
    #start {
      min-height: 50px;
      margin: 0;
      border-radius: 0;
      font-size: 20px;
      box-shadow: 0 16px 38px rgba(32,196,207,.18);
    }
    #start::before { content:"▶"; margin-right:12px; }
    .launch-zone {
      order: 3;
      margin-top: 0;
      padding: 0;
      border: 1px solid var(--teal-dim);
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(32,196,207,.10), rgba(32,196,207,.035));
      overflow:hidden;
    }
    .launch-readiness {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px 12px;
      align-items: start;
      color: var(--muted);
      font-size: 12px;
      padding: 8px 12px;
      border-bottom: 1px solid rgba(255,255,255,.08);
    }
    .launch-readiness strong { color: var(--text); font-size: 13px; }
    .launch-readiness .ready-pill {
      border: 1px solid var(--line);
      background: var(--panel-3);
      border-radius: 999px;
      padding: 3px 9px;
      color: var(--teal);
      font-weight: 800;
      white-space: nowrap;
    }
    #guard {
      min-height: 0;
      margin: 0;
      padding: 7px 12px 9px;
      border-top: 1px solid rgba(255,255,255,.07);
      max-width: none;
      text-align: center;
      font-size: 11px;
    }
    .launch-dock {
      position: fixed;
      left: 8px;
      right: 8px;
      bottom: 8px;
      z-index: 40;
      display: none;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 9px;
      border: 1px solid rgba(32,196,207,.34);
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.025)),
        rgba(13,18,24,.96);
      box-shadow: 0 18px 44px rgba(0,0,0,.38);
      backdrop-filter: blur(12px);
    }
    .launch-dock[aria-disabled="true"] {
      border-color: var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.02)),
        rgba(13,18,24,.94);
    }
    .launch-dock-meta {
      min-width:0;
      display:grid;
      gap:2px;
    }
    .launch-dock-meta strong {
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      font-size:13px;
    }
    .launch-dock-meta span {
      color:var(--muted);
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      font-size:11px;
    }
    .launch-dock button {
      width:auto;
      min-width:138px;
      margin:0;
      padding:10px 13px;
      border-radius:6px;
      box-shadow:none;
    }
    .launch-dock button::before { content:"▶"; margin-right:8px; }
    .ghost-button {
      width: auto; margin: 0; padding: 8px 10px; background: var(--panel-2);
      border-color: var(--line); color: var(--text); font-weight: 700;
    }
    .ghost-button:hover:not(:disabled) { border-color: var(--teal-dim); background: rgba(32,196,207,.08); }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .field { margin-top: 12px; }
    .field label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .field input[type="number"] {
      width: 100%; border-radius: 6px; border: 1px solid var(--line);
      background: var(--panel-2); color: var(--text); padding: 10px 12px; font: inherit;
    }
    .checkline {
      display: grid; grid-template-columns: 22px 1fr; gap: 8px; align-items: start;
      padding: 8px 0; color: var(--muted); border-bottom: 1px solid rgba(255,255,255,.05);
    }
    .checkline strong { display: block; color: var(--text); font-size: 13px; overflow-wrap: anywhere; }
    .checkline small { display: block; margin-top: 2px; }
    .plan-cards { display: grid; gap: 6px; margin-top: 4px; position:relative; }
    .plan-toolbar { display:flex; justify-content:flex-end; margin-bottom:6px; }
    .plan-toolbar .ghost-button { padding:6px 10px; font-size:12px; }
    .plan-card {
      width: 100%;
      margin: 0;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
      padding: 7px 9px;
      box-shadow: none;
      transition: transform .14s ease, background-color .14s ease, border-color .14s ease, box-shadow .14s ease;
    }
    .plan-card:hover { border-color: var(--teal-dim); background: rgba(84,210,189,.05); transform: translateY(-1px); }
    .plan-card:active { transform: translateY(0); }
    .plan-card.selected {
      border-color: var(--teal);
      background: rgba(84,210,189,.10);
      box-shadow: inset 3px 0 0 var(--teal);
    }
    .plan-current { cursor: default; }
    .plan-current:hover { transform:none; }
    .plan-card strong { display: block; font-size: 13px; margin-bottom: 3px; }
    .plan-card span {
      display: -webkit-box;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
      overflow: hidden;
      overflow-wrap: anywhere;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }
    .plan-card small { display: block; color: var(--amber); margin-top: 5px; }
    .plan-menu {
      display:none;
      position:absolute;
      left:0;
      right:0;
      top:calc(100% + 8px);
      z-index:20;
      gap:8px;
      padding:8px;
      border:1px solid var(--line);
      border-radius:8px;
      background:var(--panel);
      box-shadow:0 18px 48px rgba(0,0,0,.36);
      max-height:360px;
      overflow:auto;
    }
    .plan-cards.open .plan-menu { display:grid; }
    .builder-stack { display:grid; gap:6px; padding:8px; }
    .builder-card {
      border:1px solid var(--line);
      border-radius:8px;
      background:linear-gradient(180deg, rgba(255,255,255,.025), rgba(255,255,255,.01));
      overflow:hidden;
    }
    .builder-card-head {
      display:flex; align-items:center; justify-content:space-between; gap:12px;
      padding:7px 10px; border-bottom:1px solid var(--line-soft); font-weight:800; font-size:13px;
    }
    .builder-card-body { padding:7px 10px; color:var(--muted); }
    .builder-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .builder-grid b { display:block; color:var(--text); margin-top:3px; }
    .selected-preview {
      display:grid;
      gap:5px;
      margin-top:8px;
    }
    .selected-preview span {
      min-width:0;
      border:1px solid var(--line-soft);
      border-radius:5px;
      background:rgba(255,255,255,.025);
      color:var(--text);
      padding:5px 7px;
      font-size:11px;
      line-height:1.25;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }
    .selected-preview .empty { color:var(--muted); }
    .selected-preview .more { color:var(--teal); border-color:rgba(32,196,207,.22); }
    .builder-badge {
      min-width:24px; height:24px; border-radius:50%; display:inline-grid; place-items:center;
      background:rgba(32,196,207,.18); color:var(--teal); font-weight:900;
    }
    .engine-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px 18px; font-size:12px; }
    .engine-grid span { color:var(--muted); }
    .engine-grid b { color:var(--text); font-weight:600; }
    #engine-card { order:4; }
    #engine-card .builder-card-head { border-bottom:0; }
    #engine-card .builder-card-body { display:none; }
    .builder-proof {
      display:none;
      order:5;
      grid-template-columns:repeat(4,minmax(0,1fr));
      gap:6px;
    }
    .builder-proof span {
      min-width:0;
      border:1px solid var(--line);
      border-radius:6px;
      background:rgba(255,255,255,.025);
      padding:7px 8px;
      color:var(--muted);
      font-size:11px;
      line-height:1.25;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }
    .builder-proof b { color:var(--teal); margin-right:5px; }
    #run-builder .run-summary { display:none; }
    #run-builder .builder-stack > details.controls,
    #run-builder #stop-after-current { display:none; }
    .side > details.panel.controls,
    .side > .panel:not(#run-builder) { display:none; }
    details.controls {
      margin-top: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      overflow: hidden;
    }
    details.controls > summary {
      cursor: pointer;
      padding: 9px 12px;
      font-weight: 800;
      list-style: none;
      display: flex;
      justify-content: space-between;
      gap: 16px;
    }
    details.controls > summary::after { content: "open"; color: var(--muted); font-size: 12px; }
    details.controls[open] > summary::after { content: "close"; }
    details.controls .inside { padding: 0 12px 12px; }
    .run-summary {
      margin: 14px 0;
      padding: 14px;
      border-radius: 8px;
      background: var(--panel-2);
      border: 1px solid var(--line);
    }
    .summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0; }
    .summary-grid span { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: var(--bg); }
    .summary-grid b { display: block; font-size: 24px; line-height: 1; }
    .summary-grid small { color: var(--muted); }
    .flow-diagram {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .flow-step {
      position: relative;
      min-height: 78px;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(84,210,189,.08), rgba(84,210,189,.02));
    }
    .flow-step:not(:last-child)::after {
      content: "";
      position: absolute;
      top: 50%;
      right: -10px;
      width: 10px;
      height: 1px;
      background: var(--teal-dim);
    }
    .flow-step.done { border-color: var(--teal-dim); }
    .flow-step.active { border-color: var(--teal); box-shadow: 0 0 0 1px rgba(84,210,189,.14); }
    .flow-step b { display: block; margin-bottom: 4px; }
    .flow-step small { color: var(--muted); }
    .selection-strip {
      margin-top:8px;
      border:1px solid var(--line);
      border-radius:6px;
      background:var(--panel);
      overflow:hidden;
    }
    .selected-row {
      min-height:44px;
      display:grid;
      grid-template-columns:auto auto minmax(130px,auto) minmax(160px,1fr) auto;
      gap:14px;
      align-items:center;
      padding:8px 16px;
      border-bottom:1px solid var(--line);
      color:var(--muted);
    }
    #strip-clear { justify-self:start; min-width:74px; }
    .selected-row b { color:var(--teal); font-size:18px; margin-right:4px; }
    .sizebar { height:12px; border-radius:999px; border:1px solid var(--line-soft); background:var(--panel-3); overflow:hidden; }
    .sizebar i { display:block; height:100%; width:0%; background:linear-gradient(90deg, var(--teal), #168a94); }
    .run-flow {
      display:grid;
      grid-template-columns: minmax(120px,.8fr) repeat(4,minmax(0,1fr));
      gap:18px;
      align-items:center;
      padding:14px 22px;
    }
    .flow-label { font-weight:800; letter-spacing:.04em; }
    .flow-mini { display:grid; grid-template-columns:34px 1fr; gap:10px; align-items:center; min-width:0; }
    .flow-mini .num { width:28px; height:28px; border-radius:50%; display:grid; place-items:center; background:var(--teal); color:#061012; font-weight:900; box-shadow:0 0 22px rgba(32,196,207,.22); }
    .flow-mini b { display:block; font-size:12px; }
    .flow-mini small { color:var(--muted); display:block; font-size:11px; }
    .analytics-grid {
      display:grid;
      grid-template-columns:minmax(280px,.9fr) minmax(360px,1.15fr) minmax(300px,.95fr);
      gap:8px;
      margin-top:8px;
    }
    .analytics-panel { padding:12px; }
    .analytics-title { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px; font-weight:800; }
    .analytics-title span { color:var(--muted); font-size:12px; font-weight:700; }
    .scope-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .scope-cell {
      border:1px solid var(--line);
      border-radius:6px;
      background:rgba(255,255,255,.025);
      padding:10px;
      min-width:0;
    }
    .scope-cell span { display:block; color:var(--muted); font-size:11px; }
    .scope-cell b { display:block; margin-top:4px; font-size:20px; color:var(--text); }
    .heatmap { display:grid; gap:6px; overflow:auto; }
    .heat-row { display:grid; grid-template-columns: minmax(92px,.8fr) repeat(var(--model-cols, 2), minmax(56px,1fr)); gap:5px; align-items:center; min-width:0; }
    .heat-label { color:var(--muted); font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .heat-cell {
      height:24px;
      border:1px solid rgba(32,196,207,.18);
      border-radius:4px;
      background:rgba(32,196,207,.08);
      color:var(--teal);
      display:grid;
      place-items:center;
      font-size:10px;
      font-weight:800;
    }
    .heat-cell.dim { background:rgba(255,255,255,.025); color:var(--muted); border-color:var(--line-soft); }
    .timeline { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; }
    .timeline-stage {
      border:1px solid var(--line);
      border-radius:6px;
      background:rgba(255,255,255,.025);
      padding:9px;
      min-height:76px;
    }
    .timeline-stage b { display:block; font-size:12px; }
    .timeline-stage span { display:block; color:var(--muted); font-size:11px; margin-top:4px; }
    .timeline-stage.active { border-color:var(--teal); background:rgba(32,196,207,.10); box-shadow:inset 3px 0 0 var(--teal); }
    .timeline-stage.done { border-color:rgba(121,209,138,.35); }
    .seam-diagram {
      display: grid;
      grid-template-columns: 1fr auto 1fr auto 1fr;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-3);
    }
    .seam-node {
      min-height: 62px;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 9px 10px;
      background: rgba(255,255,255,.025);
    }
    .seam-node b { display: block; font-size: 13px; }
    .seam-node small { display: block; margin-top: 3px; color: var(--muted); font-size: 11px; }
    .seam-arrow { color: var(--teal); font-weight: 900; }
    .status-legend {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .legend-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: var(--panel-2);
    }
    .legend-item b { display: block; font-size: 12px; color: var(--text); }
    .legend-item small { color: var(--muted); font-size: 11px; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    .pack { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; color: var(--muted); }
    .pack small { display: block; color: var(--muted); }
    .bottom-grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(360px,.95fr); gap:8px; margin-top:8px; }
    .telemetry { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 0; margin-top: 0; }
    .metric { background: transparent; border-right: 1px solid var(--line); padding: 14px; min-height:105px; }
    .metric:last-child { border-right:0; }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 18px; font-weight: 800; margin-top: 4px; }
    .metric::after {
      content:""; display:block; height:26px; margin-top:10px; opacity:.8;
      background:
        linear-gradient(90deg, transparent 0 7%, rgba(32,196,207,.7) 7% 9%, transparent 9% 17%, rgba(32,196,207,.45) 17% 19%, transparent 19% 28%, rgba(32,196,207,.8) 28% 30%, transparent 30% 39%, rgba(32,196,207,.35) 39% 41%, transparent 41% 50%, rgba(32,196,207,.65) 50% 52%, transparent 52% 61%, rgba(32,196,207,.25) 61% 63%, transparent 63% 71%, rgba(32,196,207,.55) 71% 73%, transparent 73% 82%, rgba(32,196,207,.38) 82% 84%, transparent 84% 100%);
      border-bottom:1px solid rgba(32,196,207,.42);
    }
    .status { margin-top: 8px; padding: 14px 16px; }
    .reports { margin-top: 0; }
    .reports-head {
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      min-width:0;
    }
    .report-count {
      color:var(--muted);
      font-size:11px;
      font-weight:800;
      white-space:nowrap;
    }
    .links {
      display:flex;
      flex-wrap:wrap;
      gap:0;
      margin-top:10px;
      overflow:hidden;
      border:1px solid var(--line);
      border-radius:6px;
      background:rgba(0,0,0,.12);
      width:max-content;
      max-width:100%;
    }
    .links a, .receipt-actions a, .receipt-more a, .link-count, .artifact-toggle {
      color: var(--text); text-decoration: none; border: 1px solid var(--line);
      border-radius: 6px; padding: 6px 8px; background: var(--panel-2);
      transition: transform .14s ease, background-color .14s ease, border-color .14s ease;
    }
    .links a {
      border:0;
      border-right:1px solid var(--line);
      border-radius:0;
      background:transparent;
      font-size:12px;
      font-weight:800;
      white-space:nowrap;
    }
    .links a:last-of-type { border-right:0; }
    .links a:hover { color:var(--teal); background:rgba(32,196,207,.08); }
    .receipt-actions a:hover, .receipt-more a:hover, .artifact-toggle:hover { transform: translateY(-1px); border-color:var(--teal-dim); background:rgba(32,196,207,.08); }
    .link-count {
      color:var(--muted);
      border:0;
      border-left:1px solid var(--line);
      border-radius:0;
      background:rgba(255,255,255,.025);
      font-size:12px;
      font-weight:800;
      white-space:nowrap;
    }
    .receipt-list { display: grid; gap: 0; margin-top: 12px; }
    .receipt-metrics {
      display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr));
      gap:8px;
      margin:10px 0 8px;
    }
    .receipt-metric {
      border:1px solid var(--line);
      border-radius:6px;
      background:rgba(255,255,255,.025);
      padding:8px 10px;
      min-width:0;
    }
    .receipt-metric span { display:block; color:var(--muted); font-size:11px; }
    .receipt-metric b { display:block; color:var(--text); font-size:18px; margin-top:2px; }
    .receipt-table { border:1px solid var(--line); border-radius:6px; overflow:hidden; background:rgba(0,0,0,.10); }
    .receipt-row {
      display:grid;
      grid-template-columns:minmax(0,1fr) 86px 64px;
      gap:8px;
      align-items:center;
      min-width:0;
      border-bottom:1px solid var(--line-soft);
      padding:8px 10px;
    }
    .receipt-row:last-child { border-bottom:0; }
    .receipt-head {
      color:var(--muted);
      font-size:11px;
      text-transform:uppercase;
      letter-spacing:.04em;
      background:rgba(255,255,255,.025);
    }
    .receipt-head span:nth-child(4) { display:none; }
    .receipt-main { min-width:0; }
    .receipt-main strong {
      display:block;
      min-width:0;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }
    .receipt-meta { color: var(--muted); font-size: 11px; margin-top: 3px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .receipt-status { color:var(--good); font-weight:800; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .receipt-status.failed, .receipt-status.error { color:var(--bad); }
    .receipt-status.partial { color:var(--amber); }
    .receipt-score { font-weight:900; color:var(--teal); }
    .receipt-actions {
      display:flex;
      flex-wrap:wrap;
      grid-column:1 / -1;
      justify-content:flex-start;
      gap:6px;
      min-width:0;
      padding-top:2px;
    }
    .receipt-actions a, .receipt-more a {
      max-width:124px;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      font-size:11px;
      line-height:1.2;
    }
    .artifact-toggle {
      width:auto;
      margin:0;
      border:1px solid var(--line);
      padding:6px 8px;
      color:var(--muted);
      background:rgba(255,255,255,.025);
      font-size:11px;
      line-height:1.2;
      cursor:pointer;
      font-weight:800;
    }
    .artifact-toggle[aria-expanded="true"] { color:var(--teal); border-color:var(--teal-dim); background:rgba(32,196,207,.08); }
    .receipt-more {
      grid-column:1 / -1;
      display:none;
      flex-wrap:wrap;
      gap:6px;
      padding-top:2px;
    }
    .receipt-row.expanded .receipt-more { display:flex; }
    .event-feed {
      display: grid; gap: 6px; margin-top: 10px; max-height: 240px; overflow: auto;
      border: 1px solid var(--line); border-radius: 6px; background: var(--panel-2); padding: 10px;
    }
    .event {
      display: grid; grid-template-columns: minmax(82px, 0.65fr) minmax(126px, 0.9fr) minmax(0, 2fr);
      gap: 10px; color: var(--muted); align-items: start;
    }
    .event > * { min-width: 0; overflow-wrap: anywhere; }
    .event strong { color: var(--text); font-size: 12px; }
    .winner {
      margin-top: 10px; border: 1px solid var(--line); border-radius: 6px;
      background: var(--panel-2); padding: 10px;
    }
    .winner strong { color: var(--good); }
    .ok { color: var(--good); }
    .warn { color: var(--amber); }
    .bad { color: var(--bad); }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      aside { display: none; }
      main { padding-bottom:86px; }
      header {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        margin:0 0 8px;
        padding:8px 10px;
        border:1px solid var(--line);
        border-radius:6px;
        background:linear-gradient(180deg, rgba(255,255,255,.035), transparent 120px), var(--panel);
      }
      header h1 { font-size:21px; line-height:1; }
      header .sub { display:none; }
      header .ghost-button { width:auto; min-width:92px; margin:0; }
      .grid { grid-template-columns: 1fr; }
      .side { order:2; }
      .grid > .panel:first-child { order:1; }
      .launch-dock { display:grid; }
      .library-head { grid-template-columns:1fr auto; }
      .searchbox { grid-column:1/-1; justify-self:stretch; width:100%; }
      .model-table-wrap table { min-width: 820px; }
      .builder-proof { display:grid; }
      .telemetry { grid-template-columns: repeat(2, 1fr); }
      .flow-diagram { grid-template-columns: 1fr 1fr; }
      .flow-step::after { display: none; }
      .seam-diagram { grid-template-columns: 1fr; }
      .seam-arrow { text-align: center; transform: rotate(90deg); }
      .status-legend { grid-template-columns: 1fr 1fr; }
      .summary-grid { grid-template-columns: 1fr; }
      .selected-row { grid-template-columns:1fr; gap:8px; }
      #strip-clear { justify-self:stretch; width:100%; }
      .plan-menu { position:static; margin-top:8px; }
      .run-flow { grid-template-columns:1fr; }
      .analytics-grid { grid-template-columns:1fr; }
      .bottom-grid { grid-template-columns:1fr; }
      .receipt-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .receipt-row { grid-template-columns:minmax(0,1fr) 86px 64px; gap:6px; }
      .receipt-head { display:none; }
      .receipt-actions { justify-content:flex-start; }
    }
    @media (min-width: 981px) and (max-width: 1366px) {
      .shell { grid-template-columns: 220px minmax(0, 1fr); }
      aside { padding: 18px 14px; gap: 13px; }
      .brand { font-size: 22px; }
      .brand-mark { width: 24px; height: 24px; font-size: 13px; }
      .brand-sub { font-size: 12px; }
      .navlink { padding: 9px 10px; gap: 9px; }
      .status-row { grid-template-columns: 1fr auto 8px; font-size: 11px; gap: 6px; }
      th:nth-child(1), td:nth-child(1) { width: 42px; }
      th:nth-child(3), td:nth-child(3) { width: 76px; }
      th:nth-child(4), td:nth-child(4) { width: 82px; }
      th:nth-child(5), td:nth-child(5) { width: 92px; }
      th:nth-child(6), td:nth-child(6) { width: 104px; }
      th:nth-child(7), td:nth-child(7) { display:none; }
      .selection-strip { margin-top:3px; }
      .selected-row {
        min-height:32px;
        padding:4px 14px;
        gap:10px;
      }
      .selected-row b { font-size:16px; }
      .sizebar { height:10px; }
      .run-flow {
        gap:10px;
        padding:5px 18px;
      }
      .flow-mini {
        grid-template-columns:26px 1fr;
        gap:7px;
      }
      .flow-mini .num {
        width:22px;
        height:22px;
        font-size:12px;
      }
      .flow-mini b { font-size:11px; }
      .flow-mini small { display:none; }
      .bottom-grid { grid-template-columns:1fr; }
      .reports { order:-1; }
      .receipt-row { grid-template-columns:minmax(0,1fr) 118px 76px; }
    }
    @media (max-width: 640px) {
      main { padding:4px 8px 10px; }
      .library-head { grid-template-columns:1fr auto; }
      .library-count { justify-self:end; }
      .library-head .ghost-button { min-width:0; }
      .reports-head { align-items:flex-start; flex-direction:column; gap:4px; }
      .links { width:100%; }
      .links a, .link-count { flex:1 1 auto; text-align:center; }
      .model-table-wrap { height:auto; max-height:420px; }
      .model-table-wrap table { min-width:0; table-layout:fixed; }
      th, td { padding:8px 8px; }
      th:nth-child(1), td:nth-child(1) { width:38px; }
      th:nth-child(2), td:nth-child(2) { width:auto; }
      th:nth-child(3), td:nth-child(3),
      th:nth-child(5), td:nth-child(5),
      th:nth-child(7), td:nth-child(7) { display:none; }
      th:nth-child(4), td:nth-child(4) { width:76px; }
      th:nth-child(6), td:nth-child(6) { width:74px; }
      .chip { max-width:66px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .chip.warn { max-width:54px; }
      .size-full { display:none; }
      .size-compact { display:inline; }
      #start { font-size:18px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
      button:hover:not(:disabled), .plan-card:hover { transform:none; }
    }
    @media (min-width: 1700px) {
      .shell { grid-template-columns: 252px minmax(0, 1fr); }
      .grid { grid-template-columns: minmax(920px, 1.75fr) minmax(520px, .95fr); }
      .bottom-grid { grid-template-columns:minmax(520px,1fr) minmax(520px,.95fr); }
      body { font-size: 15px; }
      .panel h2 { font-size: 16px; }
    }
    @media (min-width: 2200px) {
      main { max-width: 2060px; }
      .grid { grid-template-columns: minmax(980px, 1.4fr) minmax(560px, .72fr); }
    }

    /* ===== in-flight cockpit (mission-control) ===== */
    :root{
      --panel-3:#0a0e14; --line-soft:rgba(255,255,255,.06); --faint:#5b6675;
      --teal-dim:#2c6c63; --amber-dim:#7a5a2c; --violet:#9aa5ff;
      --glow-teal:0 0 18px rgba(84,210,189,.28); --glow-amber:0 0 16px rgba(244,184,96,.18);
    }
    .mono{font-family:"JetBrains Mono", ui-monospace, "Cascadia Code", Consolas, monospace;}
    .cmdbar{display:flex; align-items:center; gap:18px; min-height:60px; padding:10px 4px 16px; border-bottom:1px solid var(--line); margin-bottom:16px; flex-wrap:wrap;}
    .cmdbar .ck-model{font-weight:700; font-size:15px; max-width:380px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
    .cmdbar .ck-sub{color:var(--muted); font-size:11.5px;}
    .live-pill{display:inline-flex; align-items:center; gap:7px; padding:4px 10px; border-radius:999px; border:1px solid var(--teal-dim); color:var(--teal); font-weight:700; font-size:11px; letter-spacing:.12em; background:rgba(84,210,189,.06);}
    .live-pill .beat,.dotbeat{width:7px;height:7px;border-radius:50%;background:var(--teal);box-shadow:var(--glow-teal);animation:beat 1.4s ease-in-out infinite;}
    @keyframes beat{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(.65)}}
    .phasepill{display:inline-flex; gap:7px; align-items:center; padding:5px 12px; border-radius:8px; border:1px solid var(--line); background:var(--panel-2); font-weight:700; font-size:12px;}
    .phasepill .ph{color:var(--faint);} .phasepill .ph.active{color:var(--teal); text-shadow:var(--glow-teal);} .phasepill .ph.done{color:var(--muted);}
    .ck-spacer{flex:1;}
    .progress-read{text-align:right; font-variant-numeric:tabular-nums;} .progress-read b{font-size:15px;} .progress-read small{display:block; color:var(--muted); font-size:11px;}
    .ck-btns{display:flex; gap:8px;}
    #cockpit button{width:auto; margin:0; font:inherit; cursor:pointer; border-radius:8px; border:1px solid var(--line); background:var(--panel-2); color:var(--text); padding:8px 13px; font-weight:700;}
    #cockpit button.stop{border-color:var(--amber-dim); color:var(--amber);}
    #cockpit button.abort{border-color:#5a2630; color:var(--bad);}
    #cockpit button:disabled{opacity:.4; cursor:not-allowed;}
    .stage{display:grid; grid-template-columns:minmax(0,1.38fr) minmax(360px,.9fr); gap:16px; align-items:start;}
    .col{display:grid; gap:16px;}
    .card{background:linear-gradient(180deg, var(--panel), var(--panel-2)); border:1px solid var(--line); border-radius:12px; overflow:hidden;}
    .card > h3{margin:0; padding:11px 15px; font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--line-soft); display:flex; align-items:center; justify-content:space-between; gap:10px;}
    .card .pad{padding:14px 15px;}
    .launch-overview{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px;}
    .overview-tile{min-height:94px; border:1px solid var(--line-soft); border-radius:10px; padding:12px; background:var(--panel-3);}
    .overview-tile strong{display:block; margin-bottom:8px; font-size:20px; line-height:1; color:var(--text); overflow-wrap:anywhere;}
    .overview-tile span{display:block; color:var(--muted); font-size:11.5px;}
    .overview-tile.live strong{color:var(--teal); text-shadow:var(--glow-teal);}
    .overview-note{grid-column:1/-1; border:1px solid var(--teal-dim); border-radius:10px; padding:11px 12px; color:var(--muted); background:rgba(84,210,189,.05);}
    .overview-note b{color:var(--text);}
    .qhead{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px;}
    .badge{font-size:11px; font-weight:800; letter-spacing:.06em; padding:3px 9px; border-radius:6px; border:1px solid var(--line); color:var(--muted); background:var(--panel-3);}
    .badge.pack{color:var(--violet); border-color:#33397a;}
    .qnum{color:var(--muted); font-variant-numeric:tabular-nums;}
    .prompt{color:var(--text); background:var(--panel-3); border:1px solid var(--line-soft); border-left:3px solid var(--teal-dim); border-radius:8px; padding:11px 13px; margin-bottom:14px; white-space:pre-wrap;}
    .streamlabel{display:flex; align-items:center; gap:8px; font-size:11px; letter-spacing:.16em; text-transform:uppercase; margin:0 0 6px;}
    .streamlabel.think{color:var(--amber);} .streamlabel.ans{color:var(--teal);}
    .streamlabel .tick{width:6px;height:6px;border-radius:50%;}
    .think .tick{background:var(--amber); box-shadow:var(--glow-amber);} .ans .tick{background:var(--teal); box-shadow:var(--glow-teal);}
    .stream{font-family:"JetBrains Mono", ui-monospace, Consolas, monospace; font-size:12.5px; line-height:1.65; white-space:pre-wrap; word-break:break-word; border-radius:9px; padding:12px 14px; min-height:34px;}
    .stream.think{color:#d9c39c; background:repeating-linear-gradient(0deg, rgba(244,184,96,.025) 0 2px, transparent 2px 4px), var(--panel-3); border:1px solid var(--amber-dim); box-shadow:inset 0 0 24px rgba(244,184,96,.05); margin-bottom:14px; max-height:280px; overflow:auto;}
    .stream.ans{color:#cdeee7; background:var(--panel-3); border:1px solid var(--teal-dim); box-shadow:inset 0 0 24px rgba(84,210,189,.06);}
    .cursor{display:inline-block; width:8px; height:15px; vertical-align:-2px; background:var(--teal); margin-left:2px; animation:blink 1s steps(2,end) infinite; box-shadow:var(--glow-teal);}
    .cursor.amber{background:var(--amber); box-shadow:var(--glow-amber);}
    @keyframes blink{50%{opacity:0}}
    .scorebar{display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-top:14px; padding-top:13px; border-top:1px solid var(--line-soft);}
    .scorechip{font-weight:800; padding:6px 14px; border-radius:8px; font-size:14px;}
    .scorechip.grading{color:var(--amber); border:1px dashed var(--amber-dim); background:rgba(244,184,96,.05);}
    .scorechip.pass{color:#07140f; background:var(--good); box-shadow:0 0 16px rgba(121,209,138,.3);}
    .scorechip.fail{color:#1a0a0a; background:var(--bad); box-shadow:0 0 16px rgba(255,115,115,.25);}
    .scoremeta{color:var(--muted); font-variant-numeric:tabular-nums; font-size:12px; display:flex; gap:16px; flex-wrap:wrap;} .scoremeta b{color:var(--text);}
    .hist{display:grid; gap:0;}
    .hrow{display:grid; grid-template-columns:16px 1fr auto auto auto; gap:12px; align-items:center; padding:9px 4px; border-bottom:1px solid var(--line-soft); font-size:12.5px;} .hrow:last-child{border-bottom:0;}
    .hrow .pf{width:9px;height:9px;border-radius:50%;} .pf.ok{background:var(--good);} .pf.no{background:var(--bad);}
    .hrow .hid{color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
    .hrow .hsc{font-weight:800; font-variant-numeric:tabular-nums;} .hrow .hmeta{color:var(--faint); font-variant-numeric:tabular-nums; font-size:11.5px;}
    .score-hero{display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;}
    .score-hero .big{font-size:44px; font-weight:800; line-height:1; font-variant-numeric:tabular-nums; color:var(--teal); text-shadow:var(--glow-teal);}
    .score-hero .of{color:var(--faint); font-size:18px; font-weight:700;}
    .partial{display:inline-flex; gap:6px; align-items:center; font-size:10.5px; letter-spacing:.13em; text-transform:uppercase; color:var(--amber); border:1px solid var(--amber-dim); border-radius:999px; padding:2px 8px; margin-left:auto;}
    .partial .beat{background:var(--amber); box-shadow:var(--glow-amber);}
    .scorebar2{height:8px; border-radius:999px; background:var(--panel-3); border:1px solid var(--line-soft); overflow:hidden; margin-top:12px;}
    .scorebar2 > i{display:block; height:100%; background:linear-gradient(90deg, var(--teal-dim), var(--teal)); box-shadow:var(--glow-teal);}
    .score-sub{display:flex; gap:18px; margin-top:10px; color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums;} .score-sub b{color:var(--text);}
    .gauges{display:grid; grid-template-columns:1fr 1fr; gap:10px;}
    .gauge{background:var(--panel-3); border:1px solid var(--line-soft); border-radius:10px; padding:10px 12px; position:relative; overflow:hidden;}
    .gauge .glab{color:var(--muted); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;}
    .gauge .gval{font-size:20px; font-weight:800; font-variant-numeric:tabular-nums; margin-top:3px;} .gauge .gunit{color:var(--faint); font-size:12px; font-weight:600;}
    .gauge svg{position:absolute; right:8px; bottom:6px; opacity:.85;} .gauge.full{grid-column:1/3;}
    .vrambar{height:6px; border-radius:999px; background:#10151d; overflow:hidden; margin-top:8px; border:1px solid var(--line-soft);}
    .vrambar > i{display:block;height:100%; background:linear-gradient(90deg,#3f7d72,var(--teal));} .vrambar > i.warn{background:linear-gradient(90deg,#9a7a3a,var(--amber));}
    .pipe{display:flex; flex-direction:column; gap:2px;}
    .pstep{display:flex; align-items:center; gap:11px; padding:6px 2px; color:var(--muted);}
    .pstep .node{width:13px;height:13px;border-radius:50%; border:2px solid var(--line); flex:none;}
    .pstep.done .node{background:var(--teal-dim); border-color:var(--teal-dim);}
    .pstep.active .node{border-color:var(--teal); background:var(--teal); box-shadow:var(--glow-teal); animation:beat 1.6s infinite;}
    .pstep.active{color:var(--text); font-weight:700;} .pstep .pconn{width:2px; height:9px; background:var(--line); margin-left:5px;}
    .queue{display:grid; gap:8px;}
    .qitem{display:flex; align-items:center; gap:11px; padding:9px 11px; border:1px solid var(--line-soft); border-radius:9px; background:var(--panel-3);}
    .qitem .qdot{width:9px;height:9px;border-radius:50%; flex:none; background:var(--faint);}
    .qitem.run .qdot{background:var(--teal); box-shadow:var(--glow-teal); animation:beat 1.5s infinite;} .qitem.done .qdot{background:var(--good);}
    .qitem .qn{flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12.5px;} .qitem.run{border-color:var(--teal-dim);}
    .qitem .qsc{font-weight:800; font-variant-numeric:tabular-nums; color:var(--good);} .qitem .qstate{color:var(--muted); font-size:11px; letter-spacing:.08em; text-transform:uppercase;}
    .bts{max-height:150px; overflow:auto; display:grid; gap:5px; font-size:11.5px;}
    .btsrow{display:grid; grid-template-columns:60px 130px 1fr; gap:10px; color:var(--muted);} .btsrow .bt{color:var(--faint); font-variant-numeric:tabular-nums;} .btsrow .bk{color:var(--violet); font-weight:700;}
    .ck-empty{color:var(--muted); padding:22px; text-align:center;}
    .ck-empty.waiting{min-height:180px; display:grid; place-items:center; border:1px dashed var(--line); border-radius:10px; background:var(--panel-3);}
    .ck-empty.waiting strong{display:block; color:var(--text); font-size:18px; margin-bottom:6px;}
    .ck-empty.waiting span{display:block; max-width:460px;}
    @media (max-width: 980px){ .stage{grid-template-columns:1fr;} }
    @media (max-width: 760px){ .launch-overview{grid-template-columns:1fr;} }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div>
        <div class="brand"><span class="brand-mark">PB</span><span><span style="color:var(--teal)">pilot</span>BENCHY</span></div>
        <div class="brand-sub">Local-first GGUF/llama.cpp benchmark cockpit</div>
        <div class="version-row"><span class="pill">v0.9.4</span><span class="pill local">Local</span></div>
      </div>
      <nav class="nav" aria-label="Cockpit navigation">
        <a class="navlink active" href="#"><span class="navico">◉</span>Cockpit</a>
        <a class="navlink" href="#recent-receipts"><span class="navico">☷</span>Runs</a>
        <a class="navlink" href="#recent-receipts"><span class="navico">▣</span>Receipts</a>
        <a class="navlink" href="#model-library"><span class="navico">⬡</span>Models</a>
        <a class="navlink" href="#run-builder"><span class="navico">□</span>Plans</a>
        <a class="navlink" href="#engine-card"><span class="navico">✧</span>Engines</a>
        <a class="navlink" href="#advanced-controls"><span class="navico">⚙</span>Settings</a>
        <a class="navlink" href="#run-flow"><span class="navico">ⓘ</span>About</a>
      </nav>
      <div class="status-block">
        <div class="status-title">Local state</div>
        <div class="status-row" data-status-row="api" data-state="info"><span>Local API</span><strong id="rail-api">checking</strong><i class="ok-dot state-info"></i></div>
        <div class="status-row" data-status-row="engine" data-state="info"><span>Engine seam</span><strong id="rail-engine">checking</strong><i class="ok-dot state-info"></i></div>
        <div class="status-row" data-status-row="models" data-state="info"><span>Model root</span><strong id="rail-model-root">checking</strong><i class="ok-dot state-info"></i></div>
        <div class="status-row" data-status-row="runs" data-state="info"><span>Run storage</span><strong id="rail-runs">checking</strong><i class="ok-dot state-info"></i></div>
        <div class="status-row" data-status-row="llama" data-state="info"><span>llama.cpp</span><strong id="rail-llama">checking</strong><i class="ok-dot state-info"></i></div>
        <div class="status-row" data-status-row="telemetry" data-state="info"><span>Telemetry</span><strong id="rail-telemetry">checking</strong><i class="ok-dot state-info"></i></div>
        <div class="status-row" data-status-row="ram" data-state="info"><span>RAM</span><strong id="rail-ram">-</strong><i class="ok-dot state-info"></i></div>
        <div class="status-row" data-status-row="cpu" data-state="info"><span>CPU</span><strong id="rail-cpu">-</strong><i class="ok-dot state-info"></i></div>
        <div class="status-row" data-status-row="gpu" data-state="info"><span>GPU</span><strong id="rail-gpu">-</strong><i class="ok-dot state-info"></i></div>
        <button class="system-check ghost-button" type="button">↻ Refresh state</button>
      </div>
      <div class="rail-foot"><span id="rail-local-state" class="ok">● Local state</span><span id="rail-clock">local</span></div>
    </aside>
    <main>
      <header>
        <div>
          <h1 id="app-title">pilotBENCHY</h1>
          <div class="sub" id="preflight-sub">Pick any local GGUF models, choose an agent-workload test, and launch repeatable local receipts from the browser.</div>
        </div>
        <button id="theme" class="ghost-button" type="button">Sepia dark</button>
      </header>
      <div id="cockpit" hidden></div>
      <div id="preflight">
      <section class="grid">
        <div class="panel" id="model-library">
          <div class="library-head">
            <div class="library-title">Model library</div>
            <div id="library-count" class="library-count">0 models</div>
            <label class="searchbox"><span>⌕</span><input id="model-search" type="search" placeholder="Search models..." /></label>
            <button id="select-all" class="ghost-button" type="button">Select visible</button>
            <button id="sort-models" class="ghost-button" type="button" aria-label="Cycle model sort">Sort: family</button>
          </div>
          <div class="model-table-wrap">
            <table>
              <thead><tr><th></th><th>Model ↓</th><th>Params</th><th>Quant</th><th>Context</th><th>Size</th><th>File</th></tr></thead>
              <tbody id="models"></tbody>
            </table>
          </div>
          <div class="body toolbar" style="justify-content:space-between;border-top:1px solid var(--line);">
            <span class="sub" id="library-scope">Showing local GGUF models</span>
            <span class="sub" id="library-page">0 of 0</span>
            <button id="clear-selection" class="ghost-button" type="button">Clear</button>
          </div>
        </div>
        <div class="side">
          <div class="panel" id="run-builder">
          <h2 class="section-title"><span>Run builder</span><span class="count-chip">Local draft</span></h2>
              <div class="builder-stack">
                <div class="builder-card">
                  <div class="builder-card-head"><span>Selected models</span><span id="selected-count" class="builder-badge">0</span></div>
                  <div class="builder-card-body">
                    <div class="builder-grid">
                      <span>Total size<b id="selected-size">-</b></span>
                      <span>Est. VRAM<b id="selected-vram">-</b></span>
                    </div>
                    <div id="selected-model-preview" class="selected-preview"></div>
                  </div>
                </div>
                <div class="builder-card">
                  <div class="builder-card-head"><span>Benchmark plan</span><span id="plan-count" class="count-chip">cards</span></div>
                  <div class="builder-card-body">
                    <select id="flight-plan" hidden aria-hidden="true"></select>
                    <div id="plan-cards" class="plan-cards"></div>
                  </div>
                </div>
                <div id="run-summary" class="run-summary"></div>
                <div class="builder-card" id="engine-card">
                  <div class="builder-card-head"><span>Engine (detached)</span><span id="engine-status-chip" class="ok">● Detached idle</span></div>
                  <div class="builder-card-body engine-grid">
                    <span>Backend</span><b>llama.cpp</b>
                    <span>Context</span><b>Auto</b>
                    <span>Threads</span><b>Auto</b>
                    <span>Batch size</span><b>Auto</b>
                    <span>GPU offload</span><b>Auto</b>
                    <span>Mode</span><b>Thin client</b>
                  </div>
                </div>
                <div class="builder-proof" aria-label="Run proof checkpoints">
                  <span><b>1</b>Models</span>
                  <span><b>2</b>Plan</span>
                  <span><b>3</b>Engine</span>
                  <span><b>4</b>Receipts</span>
                </div>
                <div class="flow-diagram" aria-label="Run pipeline" hidden>
                  <div class="flow-step"><b>1. Select</b><small>Choose one or more GGUF models.</small></div>
                  <div class="flow-step"><b>2. Plan</b><small>Pick a benchmark contract.</small></div>
                  <div class="flow-step"><b>3. Engine</b><small>Detached runner writes receipts.</small></div>
                  <div class="flow-step"><b>4. Report</b><small>Review evidence, scores, and artifacts.</small></div>
                </div>
                <div class="seam-diagram" aria-label="Detached engine architecture" hidden>
                  <div class="seam-node"><b>Browser</b><small>Chooses models and sends the run order.</small></div>
                  <div class="seam-arrow">→</div>
                  <div class="seam-node"><b>Run folder</b><small>run-spec, status, live events, receipts.</small></div>
                  <div class="seam-arrow">→</div>
                  <div class="seam-node"><b>Engine</b><small>Runs llama.cpp outside the web server.</small></div>
                </div>
                <div class="launch-zone">
                  <div class="launch-readiness">
                    <div>
                      <strong id="launch-title">Ready to configure</strong>
                      <span id="launch-detail">Select models and a benchmark plan.</span>
                    </div>
                    <span class="ready-pill" id="launch-pill">idle</span>
                  </div>
                  <button id="start" type="button">Start benchmark</button>
                  <p id="guard" class="sub"></p>
                </div>

                <details class="controls" id="advanced-controls">
                  <summary>Advanced controls</summary>
                  <div class="inside">
                    <div class="field" id="advanced-mode-field">
                      <label for="mode">Mode</label>
                      <select id="mode"></select>
                    </div>
                    <div class="field">
                      <label for="benchmark-suite-plan">Benchmark suite plan</label>
                      <select id="benchmark-suite-plan"></select>
                    </div>
                    <div class="field">
                      <label for="budget">Budget minutes per model</label>
                      <input id="budget" type="number" min="1" max="1440" value="30" />
                    </div>
                    <div class="field">
                      <label for="sample-size">Questions per pack</label>
                      <input id="sample-size" type="number" min="1" max="200" value="15" />
                    </div>
                    <div class="field">
                      <label for="repeats">Repeats per question</label>
                      <input id="repeats" type="number" min="1" max="20" value="3" />
                    </div>
                    <div class="field">
                      <label for="sampler-policy">Sampler policy</label>
                      <select id="sampler-policy">
                        <option value="hf_recommended">HF recommended</option>
                        <option value="runtime_defaults">Runtime defaults</option>
                      </select>
                    </div>
                  </div>
                </details>

                <details class="controls">
                  <summary>Server flags</summary>
                  <div class="inside">
                    <div class="field">
                      <label>Standard forced flags</label>
                      <div id="standard-flags"></div>
                    </div>
                    <div class="field">
                      <label>Optional forced flags</label>
                      <div id="optional-flags"></div>
                    </div>
                  </div>
                </details>

                <details class="controls">
                  <summary>Live display</summary>
                  <div class="inside">
                    <label class="checkline">
                      <input id="stream-prompts" type="checkbox" checked />
                      <span><strong>Show live prompt/test activity</strong><small>Lightweight event feed while the backend runs.</small></span>
                    </label>
                    <label class="checkline">
                      <input id="show-thinking" type="checkbox" />
                      <span><strong>Show model thinking if a run records it</strong><small>Only appears when the backend receipt exposes safe thinking text.</small></span>
                    </label>
                  </div>
                </details>

                <button id="stop-after-current" class="ghost-button" type="button">Stop after current</button>
            </div>
          </div>
          <details class="panel controls">
            <summary>Librarian bot jobs</summary>
            <div class="body" id="packs"></div>
          </details>
          <div class="panel">
            <h2>Fast evidence</h2>
            <div class="body">
              <div id="winner" class="winner"></div>
              <div id="events" class="event-feed"></div>
            </div>
          </div>
        </div>
      </section>
      <section class="selection-strip" id="run-flow">
        <div class="selected-row">
          <span><b id="strip-selected">0</b>models selected</span>
          <button id="strip-clear" class="ghost-button" type="button">Clear</button>
          <span>Selected size: <b id="strip-size">-</b></span>
          <div class="sizebar"><i id="strip-sizebar"></i></div>
          <span>Free disk: local</span>
        </div>
        <div class="run-flow" aria-label="Run flow">
          <div class="flow-label">RUN FLOW</div>
          <div class="flow-mini"><span class="num">1</span><span><b>Select models</b><small>Pick GGUF models from your library</small></span></div>
          <div class="flow-mini"><span class="num">2</span><span><b>Choose plan</b><small>Select benchmark cards and settings</small></span></div>
          <div class="flow-mini"><span class="num">3</span><span><b>Detached engine</b><small>pilotBENCHY runs in the background</small></span></div>
          <div class="flow-mini"><span class="num">4</span><span><b>Receipts</b><small>Results, metrics, and artifacts saved</small></span></div>
        </div>
      </section>
      <section class="analytics-grid" id="benchmark-analytics" aria-label="Benchmark analytics">
        <section class="panel analytics-panel">
          <div class="analytics-title">Plan scope <span id="scope-plan">preset</span></div>
          <div class="scope-grid" id="scope-metrics"></div>
        </section>
        <section class="panel analytics-panel">
          <div class="analytics-title">Task heatmap <span>packs x models</span></div>
          <div class="heatmap" id="task-heatmap"></div>
        </section>
        <section class="panel analytics-panel">
          <div class="analytics-title">Run timeline <span id="timeline-phase">ready</span></div>
          <div class="timeline" id="stage-timeline"></div>
        </section>
      </section>
      <section class="bottom-grid">
        <section class="panel">
          <h2 class="section-title"><span>Live telemetry</span><span class="count-chip">All good</span></h2>
          <div class="telemetry">
            <div class="metric"><div class="label">CPU</div><div class="value" id="cpu">-</div></div>
            <div class="metric"><div class="label">RAM</div><div class="value" id="ram">-</div></div>
            <div class="metric"><div class="label">GPU</div><div class="value" id="gpu">-</div></div>
            <div class="metric"><div class="label">VRAM</div><div class="value" id="vram">-</div></div>
          </div>
        </section>
        <section class="panel status reports" id="recent-receipts">
          <div class="reports-head">
            <strong>Recent receipts</strong>
            <span id="receipt-count" class="report-count">0 receipts</span>
          </div>
          <div id="global-reports" class="links"></div>
          <div id="receipts" class="receipt-list"></div>
        </section>
      </section>
      <section class="panel status">
        <strong>Run status</strong>
        <div id="run-status" class="sub">Loading...</div>
      </section>
      </div><!-- /preflight -->
    </main>
  </div>
  <div class="launch-dock" id="launch-dock" aria-disabled="true">
    <div class="launch-dock-meta">
      <strong id="dock-title">Select model first</strong>
      <span id="dock-detail">Choose GGUF models from the library.</span>
    </div>
    <button id="dock-start" type="button" disabled>Select</button>
  </div>
  <script>
    const selected = new Set();
    const sortModes = ["family", "size", "name"];
    let sortIndex = 0;
    let appState = null;
    let socket = null;
    let fallbackNotice = "";
    let selectionInitialized = false;
    let startPending = false;
    let modelSearch = "";

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function familyClass(family) {
      return family === "gemma" ? "gemma" : family === "qwen" ? "qwen" : "";
    }

    function sortedModels(models) {
      const mode = sortModes[sortIndex];
      const query = modelSearch.trim().toLowerCase();
      return [...models].filter(model => {
        if (!query) return true;
        return [model.name, model.family, model.parameters, model.quant, model.path]
          .join(" ")
          .toLowerCase()
          .includes(query);
      }).sort((a, b) => {
        if (mode === "size") return modelSizeGb(b) - modelSizeGb(a);
        return String(a[mode] || a.name).localeCompare(String(b[mode] || b.name));
      });
    }

    function statusClass(state) {
      if (state === "bad") return "bad";
      if (state === "warn") return "warn";
      if (state === "ok") return "ok";
      return "info";
    }

    function setRailStatus(rowId, value, state, detail) {
      const row = document.querySelector(`[data-status-row="${rowId}"]`);
      if (!row) return;
      const status = statusClass(state);
      row.dataset.state = status;
      row.title = detail || "";
      const strong = row.querySelector("strong");
      const dot = row.querySelector(".ok-dot");
      if (strong) strong.textContent = value || "-";
      if (dot) dot.className = `ok-dot state-${status}`;
    }

    function renderLocalStatus(status) {
      const rows = Array.isArray(status?.rows) ? status.rows : [];
      for (const row of rows) {
        setRailStatus(row.id, row.value, row.state, row.detail);
      }
      const overall = status?.overall || {};
      const foot = document.querySelector("#rail-local-state");
      if (foot) {
        const state = statusClass(overall.state);
        foot.className = state === "bad" ? "bad" : state === "warn" ? "warn" : "ok";
        foot.textContent = `● Local state: ${overall.label || "Unknown"}`;
        foot.title = overall.detail || "";
      }
      const engine = rows.find(row => row.id === "engine") || {};
      const engineChip = document.querySelector("#engine-status-chip");
      if (engineChip) {
        const state = statusClass(engine.state);
        engineChip.className = state === "bad" ? "bad" : state === "warn" ? "warn" : "ok";
        engineChip.textContent = `● ${engine.value || "Detached idle"}`;
        engineChip.title = engine.detail || "";
      }
    }

    function renderTelemetryRail(t) {
      const cpu = Math.round(t.cpu_used_percent || 0);
      const ram = Math.round(t.ram_used_percent || 0);
      const gpuText = t.gpu_util_percent == null ? "n/a" : `${t.gpu_util_percent}%`;
      setRailStatus("cpu", `${cpu}%`, cpu >= 90 ? "warn" : "ok", "Current CPU utilization sample.");
      setRailStatus("ram", `${ram}%`, ram >= 90 ? "warn" : "ok", "Current RAM utilization sample.");
      setRailStatus(
        "gpu",
        gpuText,
        t.gpu_util_percent == null ? "info" : t.gpu_util_percent >= 95 ? "warn" : "ok",
        t.gpu_util_percent == null ? "GPU metrics unavailable for this sample." : "Current GPU utilization sample.",
      );
    }

    function render(state) {
      appState = state;
      if (!selectionInitialized && state.models.length) {
        selectionInitialized = true;
      }
      const visibleModels = sortedModels(state.models);
      const libraryCount = document.querySelector("#library-count");
      const libraryPage = document.querySelector("#library-page");
      const libraryScope = document.querySelector("#library-scope");
      if (libraryCount) libraryCount.textContent = `${state.models.length} model${state.models.length === 1 ? "" : "s"}`;
      if (libraryPage) libraryPage.textContent = state.models.length
        ? `${visibleModels.length} visible of ${state.models.length}`
        : "0 of 0";
      if (libraryScope) libraryScope.textContent = modelSearch.trim()
        ? `Filter: ${modelSearch.trim()}`
        : "Showing local GGUF models";
      const tbody = document.querySelector("#models");
      tbody.innerHTML = "";
      if (!visibleModels.length) {
        tbody.innerHTML = `<tr><td colspan="7" class="sub" style="padding:28px;">No GGUF models found here. Scan or point pilotBENCHY at a model folder.</td></tr>`;
      }
      for (const model of visibleModels) {
        const tr = document.createElement("tr");
        const checked = selected.has(model.path) ? "checked" : "";
        tr.dataset.path = model.path;
        if (selected.has(model.path)) tr.classList.add("selected");
        tr.innerHTML = `
          <td><input type="checkbox" data-path="${escapeHtml(model.path)}" ${checked}></td>
          <td title="${escapeHtml(model.path)}">${escapeHtml(model.name)} <span class="${familyClass(model.family)}">●</span></td>
          <td>${escapeHtml(model.parameters)}</td>
          <td><span class="chip">${escapeHtml(model.quant)}</span></td>
          <td>${escapeHtml(model.context_label || "unknown")}</td>
          <td><span class="chip ${model.size_warning ? "warn" : ""}" title="${escapeHtml(model.size_warning ? "Suspiciously small GGUF file; not a real model payload." : model.size_bytes + " bytes")}">${modelSizeDisplay(model)}</span></td>
          <td title="${escapeHtml(model.path)}">${escapeHtml(model.file_label || model.name)}</td>`;
        tbody.appendChild(tr);
      }
      tbody.querySelectorAll("input").forEach(input => {
        input.addEventListener("change", event => {
          if (event.target.checked) selected.add(event.target.dataset.path);
          else selected.delete(event.target.dataset.path);
          render(appState);
        });
      });
      tbody.querySelectorAll("tr").forEach(row => {
        row.addEventListener("click", event => {
          if (event.target.tagName === "INPUT") return;
          if (selected.has(row.dataset.path)) selected.delete(row.dataset.path);
          else selected.add(row.dataset.path);
          render(appState);
        });
      });

      const mode = document.querySelector("#mode");
      if (!mode.children.length) {
        for (const item of state.modes) {
          const option = document.createElement("option");
          option.value = item.id;
          option.textContent = `${item.label} (${item.budget_minutes} min/model)`;
          if (item.id === state.default_mode) option.selected = true;
          mode.appendChild(option);
        }
      }
      renderFlightPlans(state.flight_plans || [], state.default_flight_plan);
      const planCount = document.querySelector("#plan-count");
      if (planCount) planCount.textContent = `${(state.flight_plans || []).length} cards`;
      const selectedMode = state.modes.find(item => item.id === mode.value);
      if (selectedMode && mode.dataset.defaultedFor !== mode.value) {
        document.querySelector("#budget").value = selectedMode.budget_minutes;
        mode.dataset.defaultedFor = mode.value;
      }
      renderBenchmarkSuitePlans(state.benchmark_suite_plans || []);
      renderConfiguration(state.run_configuration);
      document.querySelector("#packs").innerHTML = state.librarian_packs
        .map((pack, index) => `<div class="pack"><span>${escapeHtml(pack)}<small>${packDescription(pack)}</small></span><span>${index + 1}</span></div>`)
        .join("");
      const t = state.telemetry;
      document.querySelector("#cpu").textContent = `${Math.round(t.cpu_used_percent)}%`;
      document.querySelector("#ram").textContent = `${Math.round(t.ram_used_percent)}%`;
      document.querySelector("#gpu").textContent = t.gpu_util_percent == null ? "n/a" : `${t.gpu_util_percent}%`;
      document.querySelector("#vram").textContent = t.gpu_used_mb == null ? "n/a" : `${t.gpu_used_mb}/${t.gpu_total_mb} MB`;
      renderLocalStatus(state.local_status || {});
      renderTelemetryRail(t);
      document.querySelector("#rail-clock").textContent = new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
      const run = state.run;
      const active = state.active_run ? ` | ${state.active_run}` : "";
      document.querySelector("#run-status").innerHTML =
        `<span class="${run.phase === "failed" ? "bad" : "ok"}">${escapeHtml(run.phase)}</span>: ${escapeHtml(run.message)}${escapeHtml(active)}`;
      renderWinner(state);
      renderEvents(run.events || []);
      renderReceipts(state);
      renderBenchmarkGraphics(state);
      updateGuard();

      // pre-flight -> in-flight transform: once the engine is running (or a run dir
      // with a live stream is attached) the cockpit takes the stage.
      const liveEvents = run.live_events || [];
      const inflight = run.phase === "running" || liveEvents.length > 0;
      document.getElementById("cockpit").hidden = !inflight;
      document.getElementById("preflight").hidden = inflight;
      document.getElementById("preflight-sub").style.display = inflight ? "none" : "";
      document.getElementById("app-title").textContent = inflight ? "In-flight" : "pilotBENCHY";
      if (inflight) renderCockpit(state);
    }

    // ===== in-flight cockpit (renders purely from run.live_events + status + telemetry) =====
    const CK_PHASES = [["gate","Gate"],["reasoning","Reasoning"],["librarian","Librarian"],["complete","Complete"]];
    const CK_DONE = ["complete","stopped","failed","aborted"];
    function ckRound(n,d){const f=10**(d||0);return Math.round((Number(n)||0)*f)/f;}
    window.ckStop = () => sendSocket({type:"stop_after_current"});
    window.ckAbort = () => { if (confirm("Abort the run now? This kills the engine and llama-server.")) sendAbort(); };

    function ckBuild(events){
      const q=new Map(); let curQ=null, running=null, model=null, mi=null, mt=null;
      const models=[]; const bts=[]; let finished=null; const packTotals=new Map();
      for(const ev of events){
        const d=ev.data||{}; const t=ev.type||"";
        if(t==="model_started"){model=d.model;mi=d.index;mt=d.total; models.push({name:d.model,index:d.index,state:"run"}); packTotals.clear();}
        else if(t==="model_finished"){const m=models.find(x=>x.index===d.index); if(m)m.state="done";}
        else if(t==="question_started"){q.set(d.q_id,{q_id:d.q_id,index:d.index,total:d.total,pack:d.pack,prompt:d.prompt,thinking:"",answer:"",scored:null,modelIndex:mi}); curQ=d.q_id; if(d.pack)packTotals.set(d.pack,d.total||packTotals.get(d.pack)||0);}
        else if(t==="question_progress"){const it=q.get(d.q_id); if(it){if(d.thinking!=null)it.thinking=d.thinking; if(d.answer!=null)it.answer=d.answer;}}
        else if(t==="question_scored"){const it=q.get(d.q_id); if(it)it.scored=d;}
        else if(t==="running_score"){running=d;}
        else if(t==="run_finished"||t==="run_stopped"){finished=d;}
        else if(t==="receipt_ready"){bts.push({at:ev.at,k:"receipt",d:d});}
        else if(t.startsWith("autoresearch")||t.startsWith("command")||t.startsWith("champion")||t.startsWith("benchmark_suite")||t==="model_failed"||t==="task_started"||t==="task_finished"){
          bts.push({at:ev.at,k:t,d:d});
        }
      }
      let planned=0; for(const v of packTotals.values()) planned+=v;
      return {questions:[...q.values()],curQ,running,model,mi,mt,models,bts,finished,planned};
    }
    function ckTime(at){return String(at||"").slice(11,19);}
    function ckBase(p){return String(p||"").split(/[\\/]/).pop();}
    function ckMsg(k,d){
      d=d||{};
      const prof=(d.settings||{}).profile_name;
      if(k==="receipt") return "receipt · "+ckBase(d.path||d.model||"");
      if(k==="autoresearch_started") return "flag ladder · "+((d.candidate_sequence||[]).length)+" profiles · budget "+(d.budget_seconds!=null?d.budget_seconds+"s":"?");
      if(k==="autoresearch_attempt_started") return "attempt "+(d.attempt!=null?d.attempt:"?")+(prof?(" · "+prof):"");
      if(k==="autoresearch_attempt_finished") return "attempt "+(d.attempt!=null?d.attempt:"?")+" done"+(prof?(" · "+prof):"");
      if(k==="autoresearch_finished") return "best profile · "+(prof||"selected");
      if(k==="champion_pack_eval_started") return "champion eval · "+((d.pack_ids||[]).length)+" packs";
      if(k==="champion_pack_eval_finished") return "champion eval complete";
      if(k==="champion_pack_eval_failed") return "champion eval failed · "+(d.error||"");
      if(k==="model_failed") return "model failed · "+(d.error||"");
      if(k.indexOf("command")===0) return String(d.command||d.cmd||d.name||"command").slice(0,80);
      if(k.indexOf("task")===0) return k.replace("_"," ")+(d.name||d.task?(" · "+(d.name||d.task)):"");
      const parts=[]; for(const key in d){const v=d[key]; if(v==null||typeof v==="object")continue; parts.push(key+"="+v); if(parts.length>=3)break;}
      return parts.join(" · ")||k;
    }
    function ckSpark(values,w,h){ w=w||70; h=h||22; if(!values.length) return "";
      const max=Math.max.apply(null,values.concat([1])), min=Math.min.apply(null,values.concat([0])); const span=(max-min)||1;
      const pts=values.map((v,i)=>{const x=values.length===1?w:(i/(values.length-1))*w; const y=h-2-((v-min)/span)*(h-4); return ckRound(x,1)+","+ckRound(y,1);}).join(" ");
      return '<svg width="'+w+'" height="'+h+'"><polyline fill="none" stroke="var(--teal)" stroke-width="1.5" points="'+pts+'"/></svg>';
    }
    function renderCockpit(state){
      const run=state.run||{}; const status=run.status||{}; const tel=state.telemetry||{};
      const m=ckBuild(run.live_events||[]);
      const phase=run.phase||status.phase||"running"; const done=CK_DONE.includes(phase);
      // Scope score / history / progress to the CURRENT model (each model earns its
      // own Agent Index); the model queue still shows every model.
      const modelQs=m.questions.filter(x=>x.modelIndex===m.mi);
      const scoredQs=modelQs.filter(x=>x.scored);
      const cur=m.questions.find(x=>x.q_id===m.curQ);
      const answered=scoredQs.length;
      const correct=scoredQs.filter(x=>x.scored.correct).length;
      const quality=answered?Math.round(1000*correct/answered)/10:0;
      // run-wide total from status if the engine exposes it, else discovered-plan fallback
      const total=Number(status.question_total)||m.planned||answered||0;
      const progFrac=total?Math.min(1,answered/total):0;
      const selectedRunModels=Array.isArray(run.selected_models)?run.selected_models:[];
      const modelName=(m.model||status.model||selectedRunModels[0]||"").replace(/\.gguf$/i,"");
      const modelTotal=m.mt||status.model_total||selectedRunModels.length||1;
      const modelIndex=m.mi||status.model_index||1;
      const activePhase=done?"complete":(CK_PHASES.some(p=>p[0]===phase)?phase:"librarian");
      const idx=CK_PHASES.findIndex(z=>z[0]===activePhase);

      // command bar
      const phaseHtml=CK_PHASES.map((p,i)=>{const cls=i<idx?"done":i===idx?"active":""; return '<span class="ph '+cls+'">'+p[1]+'</span>'+(i<CK_PHASES.length-1?'<span style="color:var(--faint)">›</span>':'');}).join(" ");
      const remain=Math.max(0,total-answered);
      const eta=done?"finished":(remain?("ETA ~"+(remain*4)+"s"):"ETA —");
      const totalLabel=total||"?";
      const livePill=done?'<span class="phasepill"><span class="ph done">'+escapeHtml(phase)+'</span></span>':'<span class="live-pill"><span class="beat"></span> LIVE</span>';
      const cmd=
        '<div><div class="ck-model mono">'+escapeHtml(modelName||"Preparing model")+'</div><div class="ck-sub">model '+modelIndex+'/'+modelTotal+'</div></div>'
        +livePill+'<div class="phasepill">'+phaseHtml+'</div><div class="ck-spacer"></div>'
        +'<div class="progress-read"><b>'+answered+'/'+totalLabel+'</b><small>'+eta+'</small></div>'
        +'<div class="ck-btns"><button class="stop" '+(done?"disabled":"")+' onclick="ckStop()">■ Stop after current</button>'
        +'<button class="abort" '+(done?"disabled":"")+' onclick="ckAbort()">✕ Abort</button></div>';

      // current question / reasoning terminal
      let curHtml;
      if(!cur){ curHtml='<div class="ck-empty waiting"><div><strong>Engine is warming up</strong><span>Waiting for the first question from the detached runner. The run folder remains the source of truth.</span></div></div>'; }
      else{
        const s=cur.scored; const streaming=!s;
        const chip=!s?'<span class="scorechip grading">grading…</span>':'<span class="scorechip '+(s.correct?'pass':'fail')+'">'+(s.correct?'PASS':'FAIL')+' · '+ckRound(s.score,2).toFixed(2)+'</span>';
        const meta=s?'<div class="scoremeta"><span>expected <b>'+escapeHtml(s.expected)+'</b></span><span>got <b>'+escapeHtml(String(s.predicted||"").slice(0,30))+'</b></span><span>ttft <b>'+ckRound(s.ttft_ms)+'ms</b></span><span><b>'+ckRound(s.tok_s,1)+'</b> tok/s</span></div>':'<div class="scoremeta"><span>streaming…</span></div>';
        const thinkCursor=streaming&&!cur.answer?'<span class="cursor amber"></span>':'';
        const ansCursor=streaming&&cur.answer?'<span class="cursor"></span>':'';
        curHtml=
          '<div class="qhead"><span class="badge pack">'+escapeHtml(cur.pack||"pack")+'</span><span class="qnum mono">Q'+cur.index+'/'+cur.total+'</span></div>'
          +'<div class="prompt mono">'+escapeHtml(cur.prompt)+'</div>'
          +'<div class="streamlabel think"><span class="tick"></span> Thinking</div>'
          +'<div class="stream think">'+(escapeHtml(cur.thinking)||'<span style="color:var(--faint)">—</span>')+thinkCursor+'</div>'
          +'<div class="streamlabel ans"><span class="tick"></span> Answer</div>'
          +'<div class="stream ans">'+(escapeHtml(cur.answer)||'<span style="color:var(--faint)">—</span>')+ansCursor+'</div>'
          +'<div class="scorebar">'+chip+meta+'</div>';
      }
      // history (current model's completed questions)
      const hist=scoredQs.filter(x=>x.q_id!==m.curQ);
      const histHtml=hist.length?hist.map(h=>{const s=h.scored; return '<div class="hrow"><span class="pf '+(s.correct?'ok':'no')+'"></span><span class="hid mono">'+escapeHtml(h.q_id)+'</span><span class="hsc" style="color:'+(s.correct?'var(--good)':'var(--bad)')+'">'+ckRound(s.score,2).toFixed(2)+'</span><span class="hmeta">'+ckRound(s.ttft_ms)+'ms</span><span class="hmeta">'+ckRound(s.tok_s,1)+' t/s</span></div>';}).join(""):'<div class="ck-empty" style="padding:14px">No completed questions yet.</div>';

      // telemetry (real from sampler) + tok/s sparkline from this model's scored questions
      const tok=scoredQs.map(x=>ckRound(x.scored.tok_s,1));
      const lastTok=tok.length?tok[tok.length-1]:0;
      const lastTtft=scoredQs.length?ckRound(scoredQs[scoredQs.length-1].scored.ttft_ms):0;
      const vu=tel.gpu_used_mb, vt=tel.gpu_total_mb; const vpct=(vu!=null&&vt)?ckRound(vu/vt*100):0;
      const gpu=tel.gpu_util_percent==null?"n/a":tel.gpu_util_percent; const pw=tel.gpu_power_watts==null?"n/a":ckRound(tel.gpu_power_watts);
      const gaugesHtml=
        '<div class="gauge"><div class="glab">Throughput</div><div class="gval">'+lastTok+'<span class="gunit"> tok/s</span></div>'+ckSpark(tok)+'</div>'
        +'<div class="gauge"><div class="glab">TTFT</div><div class="gval">'+lastTtft+'<span class="gunit"> ms</span></div></div>'
        +'<div class="gauge"><div class="glab">GPU util</div><div class="gval">'+gpu+'<span class="gunit"> %</span></div></div>'
        +'<div class="gauge"><div class="glab">GPU power</div><div class="gval">'+pw+'<span class="gunit"> W</span></div></div>'
        +'<div class="gauge full"><div class="glab">VRAM</div><div class="gval">'+(vu==null?"n/a":vu)+' <span class="gunit">/ '+(vt||"—")+' MB</span></div><div class="vrambar"><i class="'+(vpct>90?'warn':'')+'" style="width:'+vpct+'%"></i></div></div>'
        +'<div class="gauge"><div class="glab">CPU</div><div class="gval">'+ckRound(tel.cpu_used_percent)+'<span class="gunit"> %</span></div></div>'
        +'<div class="gauge"><div class="glab">RAM</div><div class="gval">'+ckRound(tel.ram_used_percent)+'<span class="gunit"> %</span></div></div>';

      const pipeHtml=CK_PHASES.map((p,i)=>{const cls=i<idx?"done":i===idx?"active":""; return '<div class="pstep '+cls+'"><span class="node"></span><span>'+p[1]+'</span></div>'+(i<CK_PHASES.length-1?'<div class="pconn"></div>':'');}).join("");
      const queueSrc=m.models.length
        ? m.models
        : (selectedRunModels.length
          ? selectedRunModels.map((name,i)=>({name,index:i+1,state:done?"done":(i===0?"run":"queued")}))
          : [{name:modelName||"Preparing model",index:1,state:done?"done":"run"}]);
      const queueHtml=queueSrc.map(mm=>{const cls=mm.state==="done"?"done":mm.state==="run"?"run":""; const right=mm.state==="done"?'<span class="qsc">✓</span>':mm.state==="run"?'<span class="qstate">running</span>':'<span class="qstate">queued</span>'; return '<div class="qitem '+cls+'"><span class="qdot"></span><span class="qn mono">'+escapeHtml(String(mm.name||"").replace(/\.gguf$/i,""))+'</span>'+right+'</div>';}).join("");
      const bts=m.bts.slice(-30);
      const btsHtml=bts.length?bts.map(b=>'<div class="btsrow"><span class="bt">'+ckTime(b.at)+'</span><span class="bk">'+escapeHtml(b.k.replace(/_/g," "))+'</span><span>'+escapeHtml(ckMsg(b.k,b.d))+'</span></div>').join(""):'<div class="ck-empty" style="padding:10px">Engine attempts, flag ladder, and champion eval appear here.</div>';
      const overviewHtml=
        '<div class="card"><h3>Run overview</h3><div class="pad"><div class="launch-overview">'
        +'<div class="overview-tile live"><strong>'+escapeHtml(phase)+'</strong><span>detached engine phase</span></div>'
        +'<div class="overview-tile"><strong>'+answered+'/'+totalLabel+'</strong><span>questions scored</span></div>'
        +'<div class="overview-tile"><strong>'+escapeHtml(String(modelTotal))+'</strong><span>models in queue</span></div>'
        +'<div class="overview-note"><b>'+escapeHtml(modelName||"Preparing model")+'</b><br>Live events, scores, telemetry, and receipts update here as the run folder changes.</div>'
        +'</div></div></div>';

      document.getElementById("cockpit").innerHTML=
        '<div class="cmdbar">'+cmd+'</div>'
        +'<div class="stage"><div class="col">'
        +overviewHtml
        +'<div class="card"><h3>Reasoning terminal <span class="mono" style="text-transform:none;letter-spacing:0;color:var(--faint)">'+escapeHtml(cur?cur.q_id:"")+'</span></h3><div class="pad">'+curHtml+'</div></div>'
        +'<div class="card"><h3>Completed this run <span style="color:var(--faint)">'+(hist.length||"")+'</span></h3><div class="pad"><div class="hist">'+histHtml+'</div></div></div>'
        +'</div><div class="col">'
        +'<div class="card"><h3>Live score</h3><div class="pad"><div class="score-hero"><span class="big">'+ckRound(quality)+'</span><span class="of">/ 100</span><span class="partial"><span class="beat"></span> live · partial</span></div><div class="scorebar2"><i style="width:'+ckRound(quality)+'%"></i></div><div class="score-sub"><span>correct <b>'+correct+'</b>/<b>'+answered+'</b></span><span>coverage <b>'+ckRound(progFrac*100)+'%</b></span></div></div></div>'
        +'<div class="card"><h3>Telemetry</h3><div class="pad gauges">'+gaugesHtml+'</div></div>'
        +'<div class="card"><h3>Phase pipeline</h3><div class="pad"><div class="pipe">'+pipeHtml+'</div></div></div>'
        +'<div class="card"><h3>Model queue <span style="color:var(--faint)">sequential</span></h3><div class="pad"><div class="queue">'+queueHtml+'</div></div></div>'
        +'<div class="card"><h3>Behind the scenes</h3><div class="pad"><div class="bts">'+btsHtml+'</div></div></div>'
        +'</div></div>';
    }

    function setPlanDrawerOpen(open) {
      const cards = document.querySelector("#plan-cards");
      if (!cards) return;
      cards.classList.toggle("open", Boolean(open));
      const toggle = cards.querySelector("#change-plan");
      if (toggle) toggle.setAttribute("aria-expanded", String(Boolean(open)));
    }

    function renderFlightPlans(plans, defaultFlightPlanId) {
      const select = document.querySelector("#flight-plan");
      if (!select.children.length) {
        const advanced = document.createElement("option");
        advanced.value = "";
        advanced.textContent = "Advanced / choose mode directly";
        select.appendChild(advanced);
        for (const plan of plans) {
          const option = document.createElement("option");
          option.value = plan.id;
          option.textContent = `${plan.label} (${plan.budget_minutes} min/model)`;
          option.title = plan.description || plan.evidence_goal || "";
          if (plan.id === defaultFlightPlanId) option.selected = true;
          select.appendChild(option);
        }
      }
      const cards = document.querySelector("#plan-cards");
      if (cards) {
        const selectedId = select.value;
        const planItems = [
          {
            id: "",
            label: "Advanced controls",
            description: "Choose mode, budget, repeats, sampler policy, and optional suite manually.",
            budget: "For experiments"
          },
          ...plans.map(plan => ({
            id: plan.id,
            label: plan.label,
            description: plan.description || plan.evidence_goal || "Ready-made benchmark contract.",
            budget: `${plan.budget_minutes} min/model`
          }))
        ];
        const currentPlan = planItems.find(item => item.id === selectedId) || planItems[0];
        const currentPlanCard = `
          <div class="plan-card selected plan-current" aria-live="polite">
            <strong>${escapeHtml(currentPlan.label)}</strong>
            <span>${escapeHtml(currentPlan.description)}</span>
            <small>${escapeHtml(currentPlan.budget)}</small>
          </div>`;
        const planButton = (item, extraClass = "") => `
          <button type="button" class="plan-card ${item.id === selectedId ? "selected" : ""} ${extraClass}" data-plan-id="${escapeHtml(item.id)}">
            <strong>${escapeHtml(item.label)}</strong>
            <span>${escapeHtml(item.description)}</span>
            <small>${escapeHtml(item.budget)}</small>
          </button>`;
        cards.innerHTML = [
          `<div class="plan-toolbar"><button id="change-plan" class="ghost-button" type="button" aria-expanded="false" aria-controls="plan-menu">Change plan</button></div>`,
          currentPlanCard,
          `<div id="plan-menu" class="plan-menu" role="listbox" aria-label="Benchmark plans">
            ${planItems.map(item => planButton(item, "plan-option")).join("")}
          </div>`
        ].join("");
        cards.querySelectorAll(".plan-menu .plan-card").forEach(card => {
          card.addEventListener("pointerdown", event => {
            event.stopPropagation();
          });
          card.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();
            select.value = card.dataset.planId || "";
            const plan = selectedFlightPlan();
            if (plan) applyFlightPlan(plan);
            else document.querySelector("#start").textContent = "Start benchmark";
            renderFlightPlans(plans, defaultFlightPlanId);
            setPlanDrawerOpen(false);
            updateGuard();
          });
        });
        const change = cards.querySelector("#change-plan");
        if (change) {
          change.addEventListener("pointerdown", event => event.stopPropagation());
          change.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();
            setPlanDrawerOpen(!cards.classList.contains("open"));
          });
        }
      }
      const selectedPlan = selectedFlightPlan();
      if (selectedPlan && !select.dataset.appliedDefault) {
        applyFlightPlan(selectedPlan);
        select.dataset.appliedDefault = "true";
      }
    }

    function selectedFlightPlan() {
      const select = document.querySelector("#flight-plan");
      if (!appState || !select.value) return null;
      return (appState.flight_plans || []).find(plan => plan.id === select.value) || null;
    }

    function applyFlightPlan(plan) {
      const mode = document.querySelector("#mode");
      if ([...mode.options].some(option => option.value === plan.mode_id)) {
        mode.value = plan.mode_id;
        mode.dataset.defaultedFor = plan.mode_id;
      }
      document.querySelector("#budget").value = plan.budget_minutes;
      document.querySelector("#start").textContent = plan.start_label || "Start benchmark";
    }

    function clearFlightPlanForAdvancedMode() {
      const select = document.querySelector("#flight-plan");
      if (select.value) {
        select.value = "";
        document.querySelector("#start").textContent = "Start benchmark";
      }
    }

    function renderConfiguration(config) {
      if (!config || document.querySelector("#standard-flags").children.length) return;
      const standard = document.querySelector("#standard-flags");
      standard.innerHTML = config.standard_forced_args.map(item => flagChoice(item, true)).join("");
      const optional = document.querySelector("#optional-flags");
      optional.innerHTML = config.optional_forced_args.map(item => flagChoice(item, false)).join("");
      standard.querySelectorAll("input").forEach(input => input.checked = true);
      document.querySelectorAll(".forced-flag").forEach(input => input.addEventListener("change", updateGuard));
    }

    function renderBenchmarkSuitePlans(plans) {
      const select = document.querySelector("#benchmark-suite-plan");
      const current = select.value;
      select.innerHTML = `<option value="">Autoresearch only</option>` + plans
        .map(plan => {
          const label = plan.name && plan.name !== plan.filename ? `${plan.name} (${plan.filename})` : plan.filename;
          return `<option value="${escapeHtml(plan.path)}" title="${escapeHtml(plan.warning || plan.description || "")}">${escapeHtml(label)}</option>`;
        })
        .join("");
      if ([...select.options].some(option => option.value === current)) select.value = current;
    }

    function flagChoice(item, checked) {
      const value = item.value ? ` ${item.value}` : "";
      return `<label class="checkline">
        <input class="forced-flag" type="checkbox" data-flag="${escapeHtml(item.flag)}" data-value="${escapeHtml(item.value || "")}" ${checked ? "checked" : ""} />
        <span><strong>${escapeHtml(item.flag + value)}</strong><small>${escapeHtml(item.description)}</small></span>
      </label>`;
    }

    function selectedForcedArgs() {
      const args = [];
      document.querySelectorAll(".forced-flag:checked").forEach(input => {
        args.push(input.dataset.flag);
        if (input.dataset.value) args.push(input.dataset.value);
      });
      return args;
    }

    function renderWinner(state) {
      const winner = document.querySelector("#winner");
      if (!state.champion) {
        winner.innerHTML = "<span class=\"sub\">No winner yet. Run Gemma and Qwen, then this panel will show the current champion.</span>";
        return;
      }
      winner.innerHTML = `<strong>Current best for this machine:</strong> ${escapeHtml(state.champion.model)} (${Number(state.champion.score).toFixed(2)})`;
    }

    function renderEvents(events) {
      const feed = document.querySelector("#events");
      feed.innerHTML = events.length
        ? events.slice(-80).map(event => `<div class="event"><span>${escapeHtml(event.at)}</span><strong>${escapeHtml(event.kind)}</strong><span>${escapeHtml(event.message)}</span></div>`).join("")
        : `<div class="sub">Live benchmark activity will appear here.</div>`;
      feed.scrollTop = feed.scrollHeight;
    }

    function renderReceipts(state) {
      const receiptCount = document.querySelector("#receipt-count");
      if (receiptCount) {
        receiptCount.textContent = `${state.receipts.length} receipt${state.receipts.length === 1 ? "" : "s"}`;
      }
      const globalReports = document.querySelector("#global-reports");
      if (state.global_reports.length) {
        const primaryReports = primaryGlobalReports(state.global_reports);
        const remainingReports = Math.max(0, state.global_reports.length - primaryReports.length);
        globalReports.innerHTML = primaryReports
          .map(report => `<a href="${escapeHtml(report.url)}" target="_blank" rel="noreferrer" title="${escapeHtml(report.label)}">${escapeHtml(report.label)}</a>`)
          .join("") + (remainingReports ? `<span class="link-count" title="${escapeHtml(state.global_reports.map(report => report.label).join(", "))}">+${remainingReports} reports</span>` : "");
      } else {
        globalReports.innerHTML = `<span class="sub">Reports appear here after the first run.</span>`;
      }

      const receipts = document.querySelector("#receipts");
      if (!state.receipts.length) {
        receipts.innerHTML = `<div class="sub">No receipt folders found yet.</div>`;
        return;
      }
      const scores = state.receipts.map(receipt => Number(receipt.score)).filter(value => Number.isFinite(value));
      const statusCounts = state.receipts.reduce((counts, receipt) => {
        const key = String(receipt.status || "receipt").toLowerCase();
        if (key.includes("fail") || key.includes("error")) counts.failed += 1;
        else if (key.includes("partial") || key.includes("stop")) counts.partial += 1;
        else counts.completed += 1;
        return counts;
      }, {completed: 0, partial: 0, failed: 0});
      const bestScore = scores.length ? Math.max(...scores).toFixed(3) : "-";
      const metricsHtml = `
        <div class="receipt-metrics" aria-label="Receipt metrics">
          <div class="receipt-metric"><span>Receipts</span><b>${state.receipts.length}</b></div>
          <div class="receipt-metric"><span>Completed</span><b>${statusCounts.completed}</b></div>
          <div class="receipt-metric"><span>Partial / failed</span><b>${statusCounts.partial + statusCounts.failed}</b></div>
          <div class="receipt-metric"><span>Best score</span><b>${escapeHtml(bestScore)}</b></div>
        </div>`;
      const tableHtml = `
        <div class="receipt-table">
          <div class="receipt-row receipt-head"><span>Run</span><span>Status</span><span>Score</span><span>Artifacts</span></div>
          ${state.receipts.map((receipt, index) => {
            const status = String(receipt.status || "receipt");
            const statusClass = status.toLowerCase().includes("fail") || status.toLowerCase().includes("error")
              ? "failed"
              : status.toLowerCase().includes("partial") || status.toLowerCase().includes("stop")
                ? "partial"
                : "";
            const score = Number(receipt.score);
            const scoreLabel = Number.isFinite(score) ? score.toFixed(3) : "-";
            const displayStatus = receiptStatusLabel(status);
            const primaryArtifacts = primaryReceiptArtifacts(receipt.artifacts);
            const overflowArtifacts = receipt.artifacts.filter(artifact => !primaryArtifacts.includes(artifact));
            const remainingArtifacts = overflowArtifacts.length;
            return `
              <div class="receipt-row" data-receipt-row="${index}">
                <div class="receipt-main">
                  <strong title="${escapeHtml(receipt.model)}">${escapeHtml(receipt.model)}</strong>
                  <div class="receipt-meta" title="${escapeHtml(receipt.run_id)}">${escapeHtml(receipt.run_id)} | ${escapeHtml(receipt.modified)}</div>
                </div>
                <div class="receipt-status ${statusClass}" title="${escapeHtml(status)}">${escapeHtml(displayStatus)}</div>
                <div class="receipt-score">${escapeHtml(scoreLabel)}</div>
                <div class="receipt-actions">
                  ${primaryArtifacts.map(artifact => `<a href="${escapeHtml(artifact.url)}" target="_blank" rel="noreferrer" title="${escapeHtml(artifact.label)}">${escapeHtml(shortArtifactLabel(artifact.label))}</a>`).join("")}
                  ${remainingArtifacts ? `<button type="button" class="artifact-toggle" aria-expanded="false" data-artifact-toggle title="${escapeHtml(overflowArtifacts.map(artifact => artifact.label).join(", "))}">+${remainingArtifacts}</button>` : ""}
                </div>
                ${remainingArtifacts ? `<div class="receipt-more">${overflowArtifacts.map(artifact => `<a href="${escapeHtml(artifact.url)}" target="_blank" rel="noreferrer" title="${escapeHtml(artifact.label)}">${escapeHtml(shortArtifactLabel(artifact.label))}</a>`).join("")}</div>` : ""}
              </div>`;
          }).join("")}
        </div>`;
      receipts.innerHTML = metricsHtml + tableHtml;
      receipts.querySelectorAll("[data-artifact-toggle]").forEach(button => {
        button.addEventListener("click", event => {
          event.preventDefault();
          const row = button.closest(".receipt-row");
          if (!row) return;
          const expanded = !row.classList.contains("expanded");
          row.classList.toggle("expanded", expanded);
          button.setAttribute("aria-expanded", String(expanded));
          button.textContent = expanded ? "Hide" : button.dataset.count || button.textContent;
        });
        button.dataset.count = button.textContent || "";
      });
    }

    function primaryReceiptArtifacts(artifacts) {
      const priority = ["Browser report", "Summary", "Best settings"];
      const selectedArtifacts = [];
      for (const label of priority) {
        const match = artifacts.find(artifact => artifact.label === label);
        if (match) selectedArtifacts.push(match);
      }
      for (const artifact of artifacts) {
        if (selectedArtifacts.length >= 3) break;
        if (!selectedArtifacts.includes(artifact)) selectedArtifacts.push(artifact);
      }
      return selectedArtifacts;
    }

    function primaryGlobalReports(reports) {
      const priority = ["Results dashboard", "Leaderboard", "Model comparison"];
      const selectedReports = [];
      for (const label of priority) {
        const match = reports.find(report => report.label === label);
        if (match) selectedReports.push(match);
      }
      for (const report of reports) {
        if (selectedReports.length >= 3) break;
        if (!selectedReports.includes(report)) selectedReports.push(report);
      }
      return selectedReports;
    }

    function receiptStatusLabel(status) {
      const text = String(status || "receipt");
      const key = text.toLowerCase();
      if (key.includes("fail") || key.includes("error")) return "failed";
      if (key.includes("partial") || key.includes("stop")) return "partial";
      if (key.includes("context")) return "context";
      if (key.includes("complete") || key.includes("pass") || key.includes("score")) return "complete";
      return text.length > 12 ? text.slice(0, 11) + "…" : text;
    }

    function renderBenchmarkGraphics(state) {
      const selectedModels = state.models.filter(model => selected.has(model.path));
      const shownModels = (selectedModels.length ? selectedModels : state.models).slice(0, 3);
      const packs = (state.librarian_packs || []).slice(0, 5);
      const sampleSize = Number(document.querySelector("#sample-size")?.value || 0);
      const repeats = Number(document.querySelector("#repeats")?.value || 0);
      const plan = selectedFlightPlan();
      const run = state.run || {};
      const phase = run.phase || "ready";
      const receiptCount = (state.receipts || []).length;
      const questionCount = packs.length && sampleSize && repeats ? packs.length * sampleSize * repeats : 0;

      const scopePlan = document.querySelector("#scope-plan");
      if (scopePlan) scopePlan.textContent = plan ? plan.label : "manual";
      const scopeMetrics = document.querySelector("#scope-metrics");
      if (scopeMetrics) {
        scopeMetrics.innerHTML = [
          ["Models", selectedModels.length || state.models.length || 0],
          ["Questions / model", questionCount || "-"],
          ["Repeats", repeats || "-"],
          ["Receipts", receiptCount],
        ].map(([label, value]) => `<div class="scope-cell"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`).join("");
      }

      const heatmap = document.querySelector("#task-heatmap");
      if (heatmap) {
        const modelHeaders = shownModels.map(model => `<span class="heat-label" title="${escapeHtml(model.name)}">${escapeHtml(model.name.replace(/\.gguf$/i, ""))}</span>`).join("");
        const rows = packs.length
          ? packs.map(pack => {
              const cells = shownModels.length
                ? shownModels.map(() => `<span class="heat-cell">ready</span>`).join("")
                : `<span class="heat-cell dim">no model</span>`;
              return `<div class="heat-row"><span class="heat-label" title="${escapeHtml(pack)}">${escapeHtml(pack.replace(/^librarian-/, ""))}</span>${cells}</div>`;
            }).join("")
          : `<div class="sub">Benchmark packs appear after configuration loads.</div>`;
        heatmap.style.setProperty("--model-cols", String(Math.max(1, shownModels.length)));
        heatmap.innerHTML = shownModels.length
          ? `<div class="heat-row"><span class="heat-label">pack</span>${modelHeaders}</div>${rows}`
          : rows;
      }

      const stages = [
        ["ready", "Preflight", "models, disk, backend"],
        ["running", "Evaluate", "questions and scorers"],
        ["recorded", "Receipts", "artifacts saved"],
        ["complete", "Promote", "best for machine"],
      ];
      const activeIndex = phase === "running" ? 1 : phase === "complete" ? 3 : phase === "failed" ? 2 : 0;
      const timelinePhase = document.querySelector("#timeline-phase");
      if (timelinePhase) timelinePhase.textContent = phase;
      const stageTimeline = document.querySelector("#stage-timeline");
      if (stageTimeline) {
        stageTimeline.innerHTML = stages.map((stage, index) => {
          const cls = index < activeIndex ? "done" : index === activeIndex ? "active" : "";
          return `<div class="timeline-stage ${cls}"><b>${escapeHtml(stage[1])}</b><span>${escapeHtml(stage[2])}</span></div>`;
        }).join("");
      }
    }

    function shortArtifactLabel(label) {
      const text = String(label || "");
      const map = {
        "Browser report": "Browser",
        "Itemized report": "Items",
        "Resolved plan": "Plan",
        "Best settings": "Best",
        "Machine report": "Machine",
        "Suite events": "Events",
        "Summary": "Summary",
        "Command": "Cmd",
        "Status": "Status"
      };
      return map[text] || text.replace(/\breport\b/i, "").trim() || text;
    }

    function packDescription(pack) {
      if (pack.includes("gate")) return "Must-pass checks before trusting the model.";
      if (pack.includes("memory")) return "Checks local memory and retrieval behavior.";
      if (pack.includes("synthesis")) return "Combines scattered facts into useful answers.";
      if (pack.includes("citation")) return "Rewards grounded answers with proof.";
      if (pack.includes("maintenance")) return "Tests update/cleanup worker behavior.";
      return "Local worker benchmark job.";
    }

    function samplerPolicyText() {
      const sampler = document.querySelector("#sampler-policy").value;
      return sampler === "runtime_defaults"
        ? "Sampler: llama.cpp defaults."
        : "Sampler: HF recommended.";
    }

    function runSnapshotText() {
      if (!appState) return "Run state: loading local state.";
      const run = appState.run || {};
      const phase = run.phase || "idle";
      const message = run.message ? ` - ${run.message}` : "";
      const active = appState.active_run ? ` (${appState.active_run})` : "";
      if (phase === "running") return `Run state: running${message}${active}.`;
      if (["failed", "aborted", "stopped", "complete"].includes(phase)) {
        return `Last run: ${phase}${message}${active}.`;
      }
      return "Run state: detached idle.";
    }

    function updateGuard() {
      if (!appState) return;
      const flightPlan = selectedFlightPlan();
      const mode = document.querySelector("#mode").value;
      const plan = document.querySelector("#benchmark-suite-plan").value;
      const models = appState.models.filter(model => selected.has(model.path));
      const guard = document.querySelector("#guard");
      if (models.length === 0) {
        if (!appState.models.length) {
          guard.textContent = "No GGUF models found in the configured model folder.";
        } else if (appState.models.length === 1) {
          guard.textContent = "One model found. Start will use it automatically.";
        } else {
          guard.textContent = "Click Select visible, or choose one or more models before starting.";
        }
      } else {
        const flightPlanText = flightPlan ? ` Flight plan: ${flightPlan.label}.` : "";
        const planText = plan ? ` Benchmark suite plan: ${plan.split(/[\\\\/]/).pop()}.` : "";
        const compareHint = (mode === "librarian_bench" && models.length === 1)
          ? " Add a second model to compare them head-to-head."
          : "";
        guard.textContent = `${models.length} model(s) ready.${flightPlanText}${planText} ${samplerPolicyText()} ${runSnapshotText()}${compareHint}`;
      }
      updateSelectedCount(models.length);
      updateSelectedModelStats(models);
      updateRunSummary(models);
      updateFlowState(models.length, Boolean(flightPlan || plan || mode));
      updateLaunchState(models, flightPlan);
    }

    function updateSelectedCount(count) {
      const chip = document.querySelector("#selected-count");
      if (!chip) return;
      chip.textContent = String(count);
    }

    function modelSizeGb(model) {
      if (model.size_warning) return 0;
      const value = Number(model.size_gb);
      return Number.isFinite(value) ? value : 0;
    }

    function modelSizeDisplay(model) {
      const label = escapeHtml(model.size_display || "unknown");
      if (!model.size_warning) return label;
      return `<span class="size-full">${label}</span><span class="size-compact">tiny</span>`;
    }

    function conciseModelName(name) {
      return String(name || "")
        .replace(/\.gguf$/i, "")
        .replace(/-Instruct/gi, "")
        .replace(/-Chat/gi, "");
    }

    function updateSelectedModelStats(models) {
      const totalSize = models.reduce((sum, model) => sum + modelSizeGb(model), 0);
      const hasTinyFiles = models.some(model => model.size_warning);
      const vram = totalSize ? totalSize * 1.17 : 0;
      const sizeLabel = totalSize ? `${totalSize.toFixed(2)} GB` : (hasTinyFiles ? "tiny file" : "-");
      const vramLabel = vram ? `~${vram.toFixed(1)} GB` : (hasTinyFiles ? "not estimated" : "-");
      const fill = Math.max(4, Math.min(100, totalSize ? totalSize * 1.8 : 0));
      const setText = (selector, text) => {
        const el = document.querySelector(selector);
        if (el) el.textContent = text;
      };
      setText("#selected-size", sizeLabel);
      setText("#selected-vram", vramLabel);
      setText("#strip-selected", String(models.length));
      setText("#strip-size", sizeLabel);
      const bar = document.querySelector("#strip-sizebar");
      if (bar) bar.style.width = `${fill}%`;
      const preview = document.querySelector("#selected-model-preview");
      if (preview) {
        const chips = models.slice(0, 2).map(model =>
          `<span title="${escapeHtml(model.name)}">${escapeHtml(conciseModelName(model.name))}</span>`
        );
        if (models.length > 2) chips.push(`<span class="more">+${models.length - 2} more selected</span>`);
        preview.innerHTML = chips.length ? chips.join("") : `<span class="empty">Select models from the library.</span>`;
      }
    }

    function updateFlowState(modelCount, hasPlan) {
      const steps = document.querySelectorAll(".flow-step");
      if (!steps.length) return;
      steps.forEach(step => step.classList.remove("done", "active"));
      if (modelCount > 0) steps[0].classList.add("done");
      else steps[0].classList.add("active");
      if (modelCount > 0 && hasPlan) steps[1].classList.add("done");
      else if (modelCount > 0) steps[1].classList.add("active");
      if (appState?.run?.phase === "running") steps[2].classList.add("active");
      if (["complete", "stopped"].includes(appState?.run?.phase)) steps[3].classList.add("done");
    }

    function updateRunSummary(models) {
      const summary = document.querySelector("#run-summary");
      if (!summary || !appState) return;
      const flightPlan = selectedFlightPlan();
      const modeId = document.querySelector("#mode").value;
      const mode = appState.modes.find(item => item.id === modeId);
      const budget = Number(document.querySelector("#budget").value || flightPlan?.budget_minutes || mode?.budget_minutes || 0);
      const sampleSize = Number(document.querySelector("#sample-size").value || 0);
      const repeats = Number(document.querySelector("#repeats").value || 0);
      const packCount = modeId === "librarian_bench" ? appState.librarian_packs.length : 1;
      const scoredAttempts = modeId === "quick" ? 0 : models.length * packCount * sampleSize * repeats;
      const totalMinutes = models.length * budget;
      const evidence = flightPlan?.evidence_goal || (modeId === "quick"
        ? "load receipt"
        : "weighted score + bias checks");
      const workflow = flightPlan?.workflow?.length ? ` Plan: ${flightPlan.workflow.join(" -> ")}.` : "";
      summary.innerHTML = `
        <strong>Run summary</strong>
        <div class="summary-grid">
          <span><b>${models.length || "-"}</b><small>model(s)</small></span>
          <span><b>${totalMinutes || "-"}</b><small>max minutes</small></span>
          <span><b>${scoredAttempts || "-"}</b><small>scored attempts</small></span>
        </div>
        <div class="sub">${escapeHtml(evidence)}; ${escapeHtml(samplerPolicyText())} Receipts saved under _runs.${escapeHtml(workflow)}</div>`;
    }

    function updateLaunchState(models, flightPlan) {
      const start = document.querySelector("#start");
      const title = document.querySelector("#launch-title");
      const detail = document.querySelector("#launch-detail");
      const pill = document.querySelector("#launch-pill");
      const dock = document.querySelector("#launch-dock");
      const dockTitle = document.querySelector("#dock-title");
      const dockDetail = document.querySelector("#dock-detail");
      const dockStart = document.querySelector("#dock-start");
      if (!start || !title || !detail || !pill || !appState) return;
      const phase = appState.run?.phase || "idle";
      const runSnapshot = runSnapshotText();
      const running = phase === "running";
      const hasModels = appState.models.length > 0;
      const selectedCount = models.length;
      const planName = flightPlan?.label || "advanced settings";
      start.disabled = startPending || running || !hasModels || selectedCount === 0;
      let dockLabel = "Start run";
      if (!hasModels) {
        title.textContent = "No models found";
        detail.textContent = "Point pilotBENCHY at a folder containing GGUF files.";
        pill.textContent = "blocked";
        start.textContent = "No models found";
        dockLabel = "Blocked";
      } else if (running) {
        title.textContent = "Run in progress";
        detail.textContent = runSnapshot;
        pill.textContent = "running";
        start.textContent = "Engine running";
        dockLabel = "Running";
      } else if (startPending) {
        title.textContent = "Starting detached engine";
        detail.textContent = "Writing run-spec.json and launching the engine process.";
        pill.textContent = "launching";
        start.textContent = "Starting...";
        dockLabel = "Starting";
      } else if (selectedCount === 0) {
        title.textContent = "Ready after model selection";
        detail.textContent = `Choose the exact GGUF model(s) to benchmark. ${runSnapshot}`;
        pill.textContent = "needs model";
        start.textContent = "Select model first";
        dockLabel = "Select";
      } else {
        title.textContent = `${selectedCount} model${selectedCount === 1 ? "" : "s"} ready`;
        detail.textContent = `${planName}; ${runSnapshot} Receipts will be saved under _runs.`;
        pill.textContent = "ready";
        start.textContent = flightPlan?.start_label || "Start benchmark";
        dockLabel = "Start run";
      }
      if (dock && dockTitle && dockDetail && dockStart) {
        dockTitle.textContent = title.textContent;
        dockDetail.textContent = detail.textContent;
        dockStart.textContent = dockLabel;
        dockStart.disabled = start.disabled;
        dock.setAttribute("aria-disabled", String(start.disabled));
      }
    }

    function modelPathsForStart() {
      const paths = Array.from(selected);
      const guard = document.querySelector("#guard");
      if (paths.length) return paths;
      if (!appState || appState.models.length === 0) {
        guard.textContent = "No GGUF models found in the configured model folder.";
        return null;
      }
      guard.textContent = "Select one or more GGUF models before starting a run.";
      return null;
    }

    function connectSocket() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
      socket.addEventListener("message", event => {
        const message = JSON.parse(event.data);
        if (message.type === "state") render(message.payload);
        if (message.type === "run_started") {
          startPending = false;
          document.querySelector("#guard").textContent = message.message;
          sendSocket({type: "refresh"});
        }
        if (message.type === "stop_after_current") {
          document.querySelector("#guard").textContent = message.message;
          sendSocket({type: "refresh"});
        }
        if (message.type === "error") {
          startPending = false;
          document.querySelector("#guard").textContent = message.message;
          updateGuard();
        }
      });
      socket.addEventListener("open", () => {
        fallbackNotice = "";
        sendSocket({type: "refresh"});
      });
      socket.addEventListener("close", () => {
        fallbackNotice = "Live connection lost. Reconnecting with HTTP refresh...";
        document.querySelector("#guard").textContent = fallbackNotice;
        loadStateViaHttp(fallbackNotice);
        setTimeout(connectSocket, 2000);
      });
    }

    async function loadStateViaHttp(statusText = "") {
      try {
        if (statusText) document.querySelector("#guard").textContent = statusText;
        const response = await fetch("/api/state", {cache: "no-store"});
        if (!response.ok) throw new Error(`state ${response.status}`);
        render(await response.json());
      } catch (error) {
        document.querySelector("#guard").textContent = `Could not load local state: ${error.message}`;
      }
    }

    async function sendHttpCommand(message) {
      const url = message.type === "stop_after_current" ? "/api/stop-after-current" : "/api/start";
      const response = await fetch(url, {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify(message)
      });
      const payload = await response.json();
      if (message.type === "start_run") startPending = false;
      document.querySelector("#guard").textContent = payload.message;
      await loadStateViaHttp();
    }

    function sendSocket(message) {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        sendHttpCommand(message).catch(error => {
          if (message.type === "start_run") startPending = false;
          document.querySelector("#guard").textContent = `Local command failed: ${error.message}`;
          updateGuard();
        });
        return;
      }
      socket.send(JSON.stringify(message));
    }

    async function sendAbort() {
      try {
        const response = await fetch("/api/abort", {method: "POST", headers: {"content-type": "application/json"}, body: "{}"});
        const payload = await response.json();
        document.querySelector("#guard").textContent = payload.message || "Abort requested.";
      } catch (error) {
        document.querySelector("#guard").textContent = `Abort failed: ${error.message}`;
      }
      await loadStateViaHttp();
    }

    // Poll faster while a run is in flight so the thinking/answer stream feels live.
    function ckInflight() {
      return !!(appState && appState.run && (appState.run.phase === "running"
        || (appState.run.live_events && appState.run.live_events.length
            && !["complete","stopped","failed","aborted"].includes(appState.run.phase))));
    }

    document.querySelector("#theme").addEventListener("click", () => {
      document.body.classList.toggle("sepia");
      document.querySelector("#theme").textContent = document.body.classList.contains("sepia") ? "Codex dark" : "Sepia dark";
    });
    document.querySelectorAll(".system-check").forEach(button => {
      button.addEventListener("click", async () => {
        const original = button.textContent;
        button.disabled = true;
        button.textContent = "Checking...";
        await loadStateViaHttp();
        const now = new Date();
        const clock = document.querySelector("#rail-clock");
        if (clock) clock.textContent = now.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
        button.textContent = "Checked";
        window.setTimeout(() => {
          button.textContent = original || "↻ Refresh state";
          button.disabled = false;
        }, 900);
      });
    });
    document.querySelector("#select-all").addEventListener("click", () => {
      if (!appState) return;
      sortedModels(appState.models).forEach(model => selected.add(model.path));
      render(appState);
    });
    document.querySelector("#clear-selection").addEventListener("click", () => {
      selected.clear();
      render(appState);
    });
    document.querySelector("#strip-clear").addEventListener("click", () => {
      selected.clear();
      render(appState);
    });
    document.querySelector("#model-search").addEventListener("input", event => {
      modelSearch = event.target.value || "";
      if (appState) render(appState);
    });
    document.querySelector("#sort-models").addEventListener("click", () => {
      sortIndex = (sortIndex + 1) % sortModes.length;
      document.querySelector("#sort-models").textContent = `Sort: ${sortModes[sortIndex]}`;
      render(appState);
    });
    document.addEventListener("pointerdown", event => {
      const cards = document.querySelector("#plan-cards");
      if (!cards || cards.contains(event.target)) return;
      setPlanDrawerOpen(false);
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") setPlanDrawerOpen(false);
    });
    document.querySelector("#mode").addEventListener("change", () => {
      clearFlightPlanForAdvancedMode();
      const mode = appState?.modes.find(item => item.id === document.querySelector("#mode").value);
      if (mode) {
        document.querySelector("#budget").value = mode.budget_minutes;
        document.querySelector("#mode").dataset.defaultedFor = mode.id;
      }
      updateGuard();
    });
    document.querySelector("#flight-plan").addEventListener("change", () => {
      const plan = selectedFlightPlan();
      if (plan) applyFlightPlan(plan);
      updateGuard();
    });
    document.querySelector("#start").addEventListener("click", () => {
      if (startPending) return;
      const flightPlan = selectedFlightPlan();
      const modelPaths = modelPathsForStart();
      if (!modelPaths) return;
      startPending = true;
      document.querySelector("#guard").textContent = "Starting detached engine...";
      updateGuard();
      sendSocket({
        type: "start_run",
        flight_plan_id: flightPlan?.id || "",
        model_paths: modelPaths,
        mode_id: document.querySelector("#mode").value,
        options: {
          flight_plan_id: flightPlan?.id || "",
          budget_minutes: Number(document.querySelector("#budget").value),
          repeats: Number(document.querySelector("#repeats").value),
          sample_size: Number(document.querySelector("#sample-size").value),
          sampler_policy: document.querySelector("#sampler-policy").value,
          benchmark_suite_plan: document.querySelector("#benchmark-suite-plan").value,
          forced_server_args: selectedForcedArgs(),
          stream_prompts: document.querySelector("#stream-prompts").checked,
          show_thinking: document.querySelector("#show-thinking").checked
        }
      });
    });
    document.querySelector("#dock-start").addEventListener("click", () => {
      document.querySelector("#start").click();
    });
    document.querySelector("#benchmark-suite-plan").addEventListener("change", updateGuard);
    document.querySelector("#budget").addEventListener("input", updateGuard);
    document.querySelector("#sample-size").addEventListener("input", updateGuard);
    document.querySelector("#repeats").addEventListener("input", updateGuard);
    document.querySelector("#sampler-policy").addEventListener("change", updateGuard);
    document.querySelectorAll(".forced-flag").forEach(input => input.addEventListener("change", updateGuard));
    document.querySelector("#stop-after-current").addEventListener("click", () => {
      sendSocket({type: "stop_after_current"});
    });

    loadStateViaHttp("Loading local state...");
    connectSocket();
    // Adaptive cadence: ~800ms while a run streams, 2.5s when idle.
    function pollOnce() {
      if (socket && socket.readyState === WebSocket.OPEN) {
        sendSocket({type: "refresh"});
      } else {
        loadStateViaHttp(fallbackNotice);
      }
      setTimeout(pollOnce, ckInflight() ? 800 : 2500);
    }
    setTimeout(pollOnce, 2500);
  </script>
</body>
</html>
"""
