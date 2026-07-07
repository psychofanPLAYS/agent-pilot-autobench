from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
import json
import mimetypes
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
        "models": [
            {"path": str(model.path), "has_mtp": bool(model.has_mtp)} for model in selected
        ],
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
    forced_args = _string_tuple_option(
        payload, "forced_server_args", default=default_forced_args
    )
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
    repeats = _int_option(
        payload, "repeats", default=3, minimum=1, maximum=20, label="Repeats"
    )
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
            settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
            plan_kind = str(settings.get("plan_kind") or data.get("plan_kind") or "")
            requires = str(settings.get("requires") or data.get("requires") or "")
            score_contract = str(settings.get("score_contract") or "")
            context = _int_option_value(data.get("context") or settings.get("context_size"))
            context_target = str(settings.get("context_target") or "")
            tasks = data.get("tasks") if isinstance(data.get("tasks"), list) else []
            task_count = len(tasks)
            phases = sorted(
                {str(task.get("phase")) for task in tasks if isinstance(task, dict)}
            )
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
    return {
        "path": str(model.path),
        "name": model.name,
        "family": model.family,
        "parameters": model.parameters,
        "quant": model.quant,
        "size_label": _size_label(model.size_gb),
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
        return "0"
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
      --bg: #0b0f14;
      --rail: #090d12;
      --panel: #121820;
      --panel-2: #0e141b;
      --panel-3: #0a0e14;
      --line: #26313d;
      --line-soft: rgba(255,255,255,.06);
      --text: #e8edf3;
      --muted: #9aa8b7;
      --faint: #657282;
      --teal: #54d2bd;
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
    body {
      margin: 0;
      background:
        linear-gradient(180deg, rgba(84,210,189,.035), transparent 280px),
        var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, Segoe UI, Arial, sans-serif;
    }
    .shell { display: grid; grid-template-columns: 236px minmax(0, 1fr); min-height: 100vh; }
    aside {
      border-right: 1px solid var(--line);
      background: var(--rail);
      padding: 24px 18px;
      position: sticky;
      top: 0;
      height: 100vh;
    }
    .brand { font-size: 22px; font-weight: 800; letter-spacing: 0; margin-bottom: 2px; }
    .brand-sub { color: var(--muted); font-size: 12px; margin-bottom: 24px; }
    .navitem {
      display: flex; justify-content: space-between; gap: 12px;
      padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,.05); color: var(--muted);
    }
    .navitem b { color: var(--text); font-weight: 700; }
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
    main { width: min(100%, 1920px); margin: 0 auto; padding: clamp(18px, 2vw, 44px); }
    header { display: flex; align-items: center; justify-content: space-between; gap: 24px; margin-bottom: 24px; }
    h1 { margin: 0; font-size: clamp(32px, 2.4vw, 50px); line-height: 0.96; letter-spacing: 0; }
    .sub { margin-top: 7px; color: var(--muted); max-width: 840px; overflow-wrap: anywhere; }
    .grid { display: grid; grid-template-columns: minmax(500px, 1.18fr) minmax(360px, .82fr); gap: clamp(16px, 1.4vw, 30px); align-items: start; }
    .grid > *, .side, .panel { min-width: 0; }
    .grid > .panel:first-child { align-self: start; position: sticky; top: 16px; }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,.018), transparent 90px), var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }
    .panel h2 { margin: 0; padding: 14px 16px; font-size: 15px; border-bottom: 1px solid var(--line); }
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
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td { text-align: left; padding: 11px 12px; border-bottom: 1px solid rgba(255,255,255,.06); overflow-wrap: anywhere; }
    th { color: var(--muted); font-size: 12px; font-weight: 700; }
    td { vertical-align: middle; }
    td:nth-child(2) { max-width: 360px; overflow-wrap: anywhere; }
    th:nth-child(1), td:nth-child(1) { width: 50px; }
    th:nth-child(3), td:nth-child(3) { width: 84px; }
    th:nth-child(4), td:nth-child(4) { width: 78px; }
    th:nth-child(5), td:nth-child(5) { width: 88px; }
    th:nth-child(6), td:nth-child(6) { width: 68px; }
    th:nth-child(7), td:nth-child(7) { width: 70px; }
    tr { cursor: pointer; }
    tr:hover td { background: rgba(84,210,189,.06); }
    tr.selected td {
      background: rgba(84,210,189,.10);
      box-shadow: inset 3px 0 0 var(--teal);
    }
    input[type="checkbox"] { width: 18px; height: 18px; accent-color: var(--teal); }
    .chip { display: inline-block; border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px; color: var(--muted); }
    .qwen { color: var(--teal); }
    .gemma { color: var(--amber); }
    .side { display: grid; gap: 16px; }
    .body { padding: 14px 16px; }
    select, button {
      width: 100%; border-radius: 6px; border: 1px solid var(--line);
      background: var(--panel-2); color: var(--text); padding: 10px 12px;
      font: inherit;
    }
    button {
      margin-top: 12px; background: var(--teal); color: #07100e; font-weight: 800;
      cursor: pointer; border-color: transparent;
    }
    #start {
      min-height: 52px;
      font-size: 16px;
      box-shadow: 0 14px 34px rgba(84, 210, 189, .18);
    }
    .launch-zone {
      margin-top: 14px;
      padding: 14px;
      border: 1px solid var(--teal-dim);
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(84,210,189,.09), rgba(84,210,189,.025));
    }
    .launch-readiness {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px 12px;
      align-items: start;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
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
      min-height: 22px;
      margin: 10px 0 0;
      padding-top: 10px;
      border-top: 1px solid rgba(255,255,255,.07);
    }
    .ghost-button {
      width: auto; margin: 0; padding: 8px 10px; background: var(--panel-2);
      border-color: var(--line); color: var(--text); font-weight: 700;
    }
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
    .plan-cards { display: grid; gap: 10px; margin-top: 8px; }
    .plan-card {
      width: 100%;
      margin: 0;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
      padding: 12px;
      box-shadow: none;
    }
    .plan-card:hover { border-color: var(--teal-dim); background: rgba(84,210,189,.05); }
    .plan-card.selected {
      border-color: var(--teal);
      background: rgba(84,210,189,.10);
      box-shadow: inset 3px 0 0 var(--teal);
    }
    .plan-card strong { display: block; font-size: 14px; margin-bottom: 4px; }
    .plan-card span { display: block; color: var(--muted); font-size: 12px; line-height: 1.35; overflow-wrap: anywhere; }
    .plan-card small { display: block; color: var(--amber); margin-top: 7px; }
    details.controls {
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      overflow: hidden;
    }
    details.controls > summary {
      cursor: pointer;
      padding: 12px 14px;
      font-weight: 800;
      list-style: none;
      display: flex;
      justify-content: space-between;
      gap: 16px;
    }
    details.controls > summary::after { content: "open"; color: var(--muted); font-size: 12px; }
    details.controls[open] > summary::after { content: "close"; }
    details.controls .inside { padding: 0 14px 14px; }
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
    .telemetry { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 10px; margin-top: 18px; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px; }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 18px; font-weight: 800; margin-top: 4px; }
    .status { margin-top: 16px; padding: 14px 16px; }
    .reports { margin-top: 16px; }
    .links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .links a, .receipt-card a {
      color: var(--text); text-decoration: none; border: 1px solid var(--line);
      border-radius: 6px; padding: 7px 9px; background: var(--panel-2);
    }
    .receipt-list { display: grid; gap: 10px; margin-top: 12px; }
    .receipt-card {
      display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px;
      border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: var(--panel-2);
    }
    .receipt-card strong { display: block; overflow-wrap: anywhere; }
    .receipt-meta { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .receipt-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; align-content: start; }
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
    .bad { color: var(--bad); }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      aside { display: none; }
      .grid { grid-template-columns: 1fr; }
      .grid > .panel:first-child { position: static; }
      .telemetry { grid-template-columns: repeat(2, 1fr); }
      .flow-diagram { grid-template-columns: 1fr 1fr; }
      .flow-step::after { display: none; }
      .seam-diagram { grid-template-columns: 1fr; }
      .seam-arrow { text-align: center; transform: rotate(90deg); }
      .status-legend { grid-template-columns: 1fr 1fr; }
      .summary-grid { grid-template-columns: 1fr; }
    }
    @media (min-width: 1700px) {
      .shell { grid-template-columns: 268px minmax(0, 1fr); }
      .grid { grid-template-columns: minmax(820px, 1.35fr) minmax(520px, .75fr); }
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
      <div class="brand">pilotBENCHY</div>
      <div class="brand-sub">local benchmark cockpit</div>
      <div class="navitem"><span>Control</span><b>local</b></div>
      <div class="navitem"><span>Models</span><b id="nav-models">0</b></div>
      <div class="navitem"><span>Receipts</span><b>_runs</b></div>
      <div class="navitem"><span>Server</span><b>127.0.0.1</b></div>
      <div class="rail-note">
        <strong>Thin cockpit</strong>
        The browser starts a detached engine, then reads the run folder. Refreshes do not own the benchmark.
      </div>
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
        <div class="panel">
          <h2 class="section-title"><span>Model selection</span><span id="selected-count" class="count-chip">0 selected</span></h2>
          <div class="body toolbar">
            <button id="select-all" class="ghost-button" type="button">Select all</button>
            <button id="clear-selection" class="ghost-button" type="button">Clear</button>
            <button id="sort-models" class="ghost-button" type="button">Sort: family</button>
          </div>
          <table>
            <thead><tr><th></th><th>Model</th><th>Family</th><th>Params</th><th>Quant</th><th>GB</th><th>Flags</th></tr></thead>
            <tbody id="models"></tbody>
          </table>
        </div>
        <div class="side">
          <div class="panel">
          <h2>Run builder</h2>
              <div class="body">
                <div class="field">
                  <label>Benchmark plan</label>
                  <select id="flight-plan" hidden aria-hidden="true"></select>
                  <div id="plan-cards" class="plan-cards"></div>
                </div>
                <div id="run-summary" class="run-summary"></div>
                <div class="flow-diagram" aria-label="Run pipeline">
                  <div class="flow-step"><b>1. Select</b><small>Choose one or more GGUF models.</small></div>
                  <div class="flow-step"><b>2. Plan</b><small>Pick a benchmark contract.</small></div>
                  <div class="flow-step"><b>3. Engine</b><small>Detached runner writes receipts.</small></div>
                  <div class="flow-step"><b>4. Report</b><small>Review evidence, scores, and artifacts.</small></div>
                </div>
                <div class="seam-diagram" aria-label="Detached engine architecture">
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
      <section class="telemetry">
        <div class="metric"><div class="label">CPU</div><div class="value" id="cpu">-</div></div>
        <div class="metric"><div class="label">RAM</div><div class="value" id="ram">-</div></div>
        <div class="metric"><div class="label">GPU</div><div class="value" id="gpu">-</div></div>
        <div class="metric"><div class="label">VRAM</div><div class="value" id="vram">-</div></div>
      </section>
      <section class="panel status">
        <strong>Run status</strong>
        <div id="run-status" class="sub">Loading...</div>
        <div class="status-legend">
          <div class="legend-item"><b>Ready</b><small>Pick models and plan.</small></div>
          <div class="legend-item"><b>Running</b><small>Detached engine owns eval.</small></div>
          <div class="legend-item"><b>Stopped</b><small>Finishes current item first.</small></div>
          <div class="legend-item"><b>Receipt</b><small>Evidence saved under _runs.</small></div>
        </div>
      </section>
      <section class="panel status reports">
        <strong>Receipts and reports</strong>
        <div id="global-reports" class="links"></div>
        <div id="receipts" class="receipt-list"></div>
      </section>
      </div><!-- /preflight -->
    </main>
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
      return [...models].sort((a, b) => {
        if (mode === "size") return Number(b.size_label || 0) - Number(a.size_label || 0);
        return String(a[mode] || a.name).localeCompare(String(b[mode] || b.name));
      });
    }

    function render(state) {
      appState = state;
      if (!selectionInitialized && state.models.length) {
        state.models.slice(0, Math.min(2, state.models.length)).forEach(model => selected.add(model.path));
        selectionInitialized = true;
      }
      document.querySelector("#nav-models").textContent = state.models.length;
      const tbody = document.querySelector("#models");
      tbody.innerHTML = "";
      for (const model of sortedModels(state.models)) {
        const tr = document.createElement("tr");
        const checked = selected.has(model.path) ? "checked" : "";
        tr.dataset.path = model.path;
        if (selected.has(model.path)) tr.classList.add("selected");
        tr.innerHTML = `
          <td><input type="checkbox" data-path="${escapeHtml(model.path)}" ${checked}></td>
          <td>${escapeHtml(model.name)}</td>
          <td class="${familyClass(model.family)}">${escapeHtml(model.family)}</td>
          <td>${escapeHtml(model.parameters)}</td>
          <td><span class="chip">${escapeHtml(model.quant)}</span></td>
          <td>${escapeHtml(model.size_label)}</td>
          <td>${model.has_mtp ? "MTP " : ""}${model.has_vision ? "vision" : ""}</td>`;
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
      const run = state.run;
      const active = state.active_run ? ` | ${state.active_run}` : "";
      document.querySelector("#run-status").innerHTML =
        `<span class="${run.phase === "failed" ? "bad" : "ok"}">${escapeHtml(run.phase)}</span>: ${escapeHtml(run.message)}${escapeHtml(active)}`;
      renderWinner(state);
      renderEvents(run.events || []);
      renderReceipts(state);
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
        cards.innerHTML = [
          `<button type="button" class="plan-card ${selectedId ? "" : "selected"}" data-plan-id="">
            <strong>Advanced controls</strong>
            <span>Choose mode, budget, repeats, sampler policy, and optional suite manually.</span>
            <small>For experiments</small>
          </button>`,
          ...plans.map(plan => `
            <button type="button" class="plan-card ${selectedId === plan.id ? "selected" : ""}" data-plan-id="${escapeHtml(plan.id)}">
              <strong>${escapeHtml(plan.label)}</strong>
              <span>${escapeHtml(plan.description || plan.evidence_goal || "Ready-made benchmark contract.")}</span>
              <small>${escapeHtml(plan.budget_minutes)} min/model</small>
            </button>`)
        ].join("");
        cards.querySelectorAll(".plan-card").forEach(card => {
          card.addEventListener("click", () => {
            select.value = card.dataset.planId || "";
            const plan = selectedFlightPlan();
            if (plan) applyFlightPlan(plan);
            else document.querySelector("#start").textContent = "Start benchmark";
            renderFlightPlans(plans, defaultFlightPlanId);
            updateGuard();
          });
        });
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
      const globalReports = document.querySelector("#global-reports");
      globalReports.innerHTML = state.global_reports.length
        ? state.global_reports.map(report => `<a href="${escapeHtml(report.url)}" target="_blank" rel="noreferrer">${escapeHtml(report.label)}</a>`).join("")
        : `<span class="sub">Reports appear here after the first run.</span>`;

      const receipts = document.querySelector("#receipts");
      receipts.innerHTML = state.receipts.length
        ? state.receipts.map(receipt => `
            <div class="receipt-card">
              <div>
                <strong>${escapeHtml(receipt.model)}</strong>
                <div class="receipt-meta">${escapeHtml(receipt.run_id)} | ${escapeHtml(receipt.status)} | ${escapeHtml(receipt.modified)}</div>
              </div>
              <div class="receipt-actions">
                ${receipt.artifacts.map(artifact => `<a href="${escapeHtml(artifact.url)}" target="_blank" rel="noreferrer">${escapeHtml(artifact.label)}</a>`).join("")}
              </div>
            </div>`).join("")
        : `<div class="sub">No receipt folders found yet.</div>`;
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
          guard.textContent = "Click Select all, or choose one or more models before starting.";
        }
      } else {
        const flightPlanText = flightPlan ? ` Flight plan: ${flightPlan.label}.` : "";
        const planText = plan ? ` Benchmark suite plan: ${plan.split(/[\\\\/]/).pop()}.` : "";
        const compareHint = (mode === "librarian_bench" && models.length === 1)
          ? " Add a second model to compare them head-to-head."
          : "";
        guard.textContent = `${models.length} model(s) ready.${flightPlanText}${planText} ${samplerPolicyText()}${compareHint}`;
      }
      updateSelectedCount(models.length);
      updateRunSummary(models);
      updateFlowState(models.length, Boolean(flightPlan || plan || mode));
      updateLaunchState(models, flightPlan);
    }

    function updateSelectedCount(count) {
      const chip = document.querySelector("#selected-count");
      if (!chip) return;
      chip.textContent = `${count} selected`;
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
      if (!start || !title || !detail || !pill || !appState) return;
      const phase = appState.run?.phase || "idle";
      const running = phase === "running";
      const hasModels = appState.models.length > 0;
      const selectedCount = models.length;
      const planName = flightPlan?.label || "advanced settings";
      start.disabled = startPending || running || !hasModels;
      if (!hasModels) {
        title.textContent = "No models found";
        detail.textContent = "Point pilotBENCHY at a folder containing GGUF files.";
        pill.textContent = "blocked";
        start.textContent = "No models found";
      } else if (running) {
        title.textContent = "Run in progress";
        detail.textContent = "The detached engine is writing live status and receipts.";
        pill.textContent = "running";
        start.textContent = "Engine running";
      } else if (startPending) {
        title.textContent = "Starting detached engine";
        detail.textContent = "Writing run-spec.json and launching the engine process.";
        pill.textContent = "launching";
        start.textContent = "Starting...";
      } else if (selectedCount === 0) {
        title.textContent = "Ready after model selection";
        detail.textContent = "Choose one or more models, or click Start to use the first detected models.";
        pill.textContent = "needs model";
        start.textContent = flightPlan?.start_label || "Start benchmark";
      } else {
        title.textContent = `${selectedCount} model${selectedCount === 1 ? "" : "s"} ready`;
        detail.textContent = `${planName}; receipts will be saved under _runs.`;
        pill.textContent = "ready";
        start.textContent = flightPlan?.start_label || "Start benchmark";
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
      if (appState.models.length === 1) {
        selected.add(appState.models[0].path);
        render(appState);
        return [appState.models[0].path];
      }
      const defaultModels = appState.models.slice(0, Math.min(2, appState.models.length));
      defaultModels.forEach(model => selected.add(model.path));
      render(appState);
      guard.textContent = `No manual selection was set, so Start will use the first ${defaultModels.length} detected model(s).`;
      return defaultModels.map(model => model.path);
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
        document.querySelector("#guard").textContent = "Live connection ready.";
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
        const response = await fetch("/api/state", {cache: "no-store"});
        if (!response.ok) throw new Error(`state ${response.status}`);
        render(await response.json());
        if (statusText) document.querySelector("#guard").textContent = statusText;
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
    document.querySelector("#select-all").addEventListener("click", () => {
      if (!appState) return;
      appState.models.forEach(model => selected.add(model.path));
      render(appState);
    });
    document.querySelector("#clear-selection").addEventListener("click", () => {
      selected.clear();
      render(appState);
    });
    document.querySelector("#sort-models").addEventListener("click", () => {
      sortIndex = (sortIndex + 1) % sortModes.length;
      document.querySelector("#sort-models").textContent = `Sort: ${sortModes[sortIndex]}`;
      render(appState);
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
