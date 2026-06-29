from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
import json
import mimetypes
from pathlib import Path, PurePosixPath
import socket
import threading
from typing import Callable
from urllib.parse import quote, unquote, urlparse
import webbrowser

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

from gguf_limit_bench.discovery import ModelInfo, discover_models
from gguf_limit_bench.flag_ladder import profile_descriptions, validate_extra_server_args
from gguf_limit_bench.gpu_profiles import (
    describe as describe_gpu_profile,
    detect_gpu_name,
    recommended_always_on,
    recommended_parallel,
)
from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS
from gguf_limit_bench.modes import RUN_MODES
from gguf_limit_bench.programs import MIN_SERIOUS_CONTEXT_SIZE
from gguf_limit_bench.reports import write_leaderboard
from gguf_limit_bench.telemetry import sample_telemetry
from gguf_limit_bench.tui import active_run_status

WebRunModelCallback = Callable[[ModelInfo, "WebRunOptions"], Path]

RECENT_RECEIPT_LIMIT = 8
GLOBAL_REPORTS = (
    ("Results dashboard", "results.html"),
    ("Leaderboard", "leaderboard.md"),
    ("Model comparison", "model-comparison.md"),
    ("Model comparison JSON", "model-comparison.json"),
)
RUN_ARTIFACTS = (
    ("Browser report", "report.html"),
    ("Itemized report", "itemized-report.md"),
    ("Summary", "summary.md"),
    ("Best settings", "best-settings.json"),
    ("Machine report", "report.json"),
    ("Suite summary", "suite-summary.json"),
    ("Suite plan", "suite-plan.json"),
    ("Suite events", "events.jsonl"),
    ("Librarian summary", "librarian-suite-summary.json"),
    ("Librarian report", "librarian-suite.md"),
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
    show_thinking: bool = False
    stream_prompts: bool = True
    benchmark_suite_plan: Path | None = None


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
        run_model: WebRunModelCallback | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.root = root
        self.runs_root = runs_root
        self.run_model = run_model
        self.project_root = project_root or Path.cwd()
        self.run = WebRunState()
        self.run_configuration = run_configuration_payload()
        self._lock = threading.Lock()

    def models(self) -> list[ModelInfo]:
        return discover_models([self.root])

    def state_payload(self) -> dict:
        models = self.models()
        telemetry = sample_telemetry().to_dict()
        leaderboard = write_leaderboard(self.runs_root)
        champion = None
        if leaderboard.entries:
            champion = {
                "model": leaderboard.champion.model_name,
                "score": leaderboard.champion.score,
            }
        with self._lock:
            run_payload = asdict(self.run)
        if run_payload.get("phase") == "running" and (
            run_payload.get("options") or {}
        ).get("stream_prompts", True):
            run_payload["events"] = run_payload["events"] + receipt_event_payloads(self.runs_root)
        return {
            "models": [_model_payload(model) for model in models],
            "modes": [_mode_payload(mode) for mode in RUN_MODES],
            "default_mode": "librarian_bench",
            "librarian_packs": list(LIBRARIAN_PACK_IDS),
            "run_configuration": self.run_configuration,
            "benchmark_suite_plans": benchmark_suite_plan_payloads(self.project_root),
            "telemetry": telemetry,
            "active_run": active_run_status(self.runs_root),
            "champion": champion,
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
            if self.run.phase == "running":
                return False, "A benchmark is already running."
            plan_label = (
                f"; suite plan: {options.benchmark_suite_plan.name}"
                if options.benchmark_suite_plan is not None
                else ""
            )
            events = [
                _event(
                    "configure",
                    f"Mode: {mode_id}; budget: {options.budget_minutes} min/model{plan_label}",
                ),
                _event(
                    "flags",
                    "Forced llama-server args: "
                    + (" ".join(options.forced_server_args) or "(none)"),
                ),
            ]
            self.run = WebRunState(
                phase="running",
                message=f"Queued {len(selected)} model(s).",
                selected_models=[model.name for model in selected],
                options=asdict(options),
                events=events,
            )
        thread = threading.Thread(
            target=self._run_models,
            args=(selected, options),
            name="AgentPilot-WebUI-runner",
            daemon=True,
        )
        thread.start()
        return True, "Benchmark started."

    def request_stop_after_current(self) -> tuple[bool, str]:
        with self._lock:
            if self.run.phase != "running":
                return False, "No active benchmark run can be stopped."
            self.run.stop_requested = True
            self.run.events.append(
                _event("stop", "Stop requested. The current benchmark item will finish first.")
            )
            return True, "Stop requested after current item."

    def _run_models(self, selected: list[ModelInfo], options: WebRunOptions) -> None:
        receipts: list[str] = []
        try:
            for index, model in enumerate(selected, start=1):
                with self._lock:
                    self.run.message = f"Running {index}/{len(selected)}: {model.name}"
                    self.run.events.append(
                        _event(
                            "model",
                            f"Starting {model.name}; prompt streaming is "
                            f"{'on' if options.stream_prompts else 'off'}.",
                        )
                    )
                if self.run_model is None:
                    receipt = self.runs_root / "webui-preview"
                    receipt.mkdir(parents=True, exist_ok=True)
                else:
                    receipt = self.run_model(model, options)
                receipts.append(str(receipt))
                with self._lock:
                    self.run.receipts = receipts[:]
                    self.run.events.append(
                        _event("receipt", f"Finished {model.name}; receipt: {receipt}")
                    )
                    stop_requested = self.run.stop_requested
                if stop_requested:
                    with self._lock:
                        self.run.phase = "stopping"
                        self.run.message = "Stopped after the current benchmark item."
                        self.run.events.append(_event("stop", "Run queue stopped by request."))
                    break
            with self._lock:
                if self.run.phase != "stopping":
                    self.run.phase = "complete"
                    self.run.message = "Benchmark complete."
                    self.run.events.append(_event("complete", "Benchmark queue finished."))
                self.run.receipts = receipts
        except Exception as exc:  # noqa: BLE001 - surface background failures to UI
            with self._lock:
                self.run.phase = "failed"
                self.run.error = str(exc)
                self.run.message = "Benchmark failed."
                self.run.events.append(_event("error", str(exc)))


def validate_web_selection(selected: list[ModelInfo], mode_id: str) -> str | None:
    if not selected:
        return "Select at least one model first."
    # Agent Pilot benchmarks any GGUF model on agent workloads. There is no
    # hardcoded Gemma-vs-Qwen requirement; one model runs, two or more compare.
    return None


def build_run_options(
    mode_id: str, payload: dict, *, project_root: Path | None = None
) -> WebRunOptions:
    mode = next((item for item in RUN_MODES if item.id == mode_id), None)
    if mode is None:
        raise ValueError(f"Unknown run mode: {mode_id}")
    budget_minutes = int(payload.get("budget_minutes") or mode.budget_minutes)
    if not 1 <= budget_minutes <= 24 * 60:
        raise ValueError("Budget must be between 1 minute and 24 hours.")
    gpu_name = detect_gpu_name()
    default_forced_args = recommended_always_on(gpu_name)
    raw_forced_args = payload.get("forced_server_args", default_forced_args)
    forced_args = tuple(str(arg) for arg in raw_forced_args)
    validate_extra_server_args(forced_args)
    allowed = set(default_forced_args)
    allowed.update(flag for flag, _description in OPTIONAL_FORCED_FLAGS)
    unknown = [arg for arg in forced_args if arg.startswith("--") and arg not in allowed]
    if unknown:
        raise ValueError(f"Unsupported forced flag from Web UI: {unknown[0]}")
    benchmark_suite_plan = resolve_benchmark_suite_plan(
        project_root or Path.cwd(), payload.get("benchmark_suite_plan")
    )
    return WebRunOptions(
        mode_id=mode_id,
        budget_minutes=budget_minutes,
        forced_server_args=forced_args,
        show_thinking=bool(payload.get("show_thinking", False)),
        stream_prompts=bool(payload.get("stream_prompts", True)),
        benchmark_suite_plan=benchmark_suite_plan,
    )


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
    app = FastAPI(title="Agent Pilot local cockpit", docs_url=None, redoc_url=None)

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
                    await websocket.send_json(websocket_error("WebSocket message must be valid JSON."))
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
    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    return state.start_run(
        model_paths=raw_model_paths,
        mode_id=str(payload.get("mode_id", "librarian_bench")),
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
    return websocket_error(f"Unknown WebSocket message type: {message_type}")


def serve_webui(
    *,
    root: Path,
    runs_root: Path,
    run_model: WebRunModelCallback | None,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> str:
    resolved_port = port if port != 0 else _free_local_port(host)
    state = WebUiState(root=root, runs_root=runs_root, run_model=run_model)
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
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            name = str(data.get("name") or data.get("title") or path.name)
            description = str(data.get("description") or "")
            warning = _plan_warning(data, description)
        payloads.append(
            {
                "path": str(path),
                "filename": path.name,
                "name": name,
                "description": description,
                "warning": warning,
            }
        )
    return payloads


def _plan_warning(data: dict, description: str) -> str:
    text = json.dumps(data, ensure_ascii=True).lower() + " " + description.lower()
    if "endpoint" in text:
        return description
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
  <title>Agent Pilot</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0e1117;
      --panel: #151a22;
      --panel-2: #10151d;
      --line: #2b3441;
      --text: #e8edf3;
      --muted: #9aa8b7;
      --teal: #54d2bd;
      --amber: #f4b860;
      --bad: #ff7373;
      --good: #79d18a;
    }
    body.sepia {
      --bg: #16120f;
      --panel: #211b16;
      --panel-2: #18130f;
      --line: #4a4037;
      --text: #eee2d1;
      --muted: #b8aa98;
      --teal: #d0b06f;
      --amber: #c9925b;
      --bad: #e98570;
      --good: #a8c47a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, Segoe UI, Arial, sans-serif;
    }
    .shell { display: grid; grid-template-columns: 250px 1fr; min-height: 100vh; }
    aside {
      border-right: 1px solid var(--line);
      background: #0b0f15;
      padding: 24px 18px;
    }
    .brand { font-size: 22px; font-weight: 800; letter-spacing: 0; margin-bottom: 22px; }
    .navitem {
      display: flex; justify-content: space-between; gap: 12px;
      padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,.05); color: var(--muted);
    }
    main { padding: 24px; }
    header { display: flex; align-items: start; justify-content: space-between; gap: 20px; margin-bottom: 20px; }
    h1 { margin: 0; font-size: 30px; line-height: 1.1; letter-spacing: 0; }
    .sub { margin-top: 7px; color: var(--muted); max-width: 760px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.5fr) 360px; gap: 16px; }
    .grid > .panel:first-child { align-self: start; position: sticky; top: 16px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    .panel h2 { margin: 0; padding: 14px 16px; font-size: 15px; border-bottom: 1px solid var(--line); }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 11px 12px; border-bottom: 1px solid rgba(255,255,255,.06); }
    th { color: var(--muted); font-size: 12px; font-weight: 700; }
    td { vertical-align: middle; }
    tr:hover td { background: rgba(84,210,189,.06); }
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
    button:disabled { opacity: .5; cursor: not-allowed; }
    .pack { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; color: var(--muted); }
    .pack small { display: block; color: var(--muted); }
    .telemetry { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 16px; }
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
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">Agent Pilot</div>
      <div class="navitem"><span>Control</span><span>local</span></div>
      <div class="navitem"><span>Models</span><span id="nav-models">0</span></div>
      <div class="navitem"><span>Receipts</span><span>_runs</span></div>
      <div class="navitem"><span>Server</span><span>127.0.0.1</span></div>
    </aside>
    <main>
      <header>
        <div>
          <h1>Agent Pilot benchmark cockpit</h1>
          <div class="sub">Pick any local GGUF models, choose an agent-workload test, and launch repeatable local receipts from the browser.</div>
        </div>
        <button id="theme" class="ghost-button" type="button">Sepia dark</button>
      </header>
      <section class="grid">
        <div class="panel">
          <h2>Model selection</h2>
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
            <h2>Run menu</h2>
              <div class="body">
                <select id="mode"></select>
                <div class="field">
                  <label for="benchmark-suite-plan">Benchmark suite plan</label>
                  <select id="benchmark-suite-plan"></select>
                </div>
                <div class="field">
                  <label for="budget">Budget minutes per model</label>
                  <input id="budget" type="number" min="1" max="1440" value="30" />
                </div>
              <div class="field">
                <label>Standard forced flags</label>
                <div id="standard-flags"></div>
              </div>
              <div class="field">
                <label>Optional forced flags</label>
                <div id="optional-flags"></div>
              </div>
              <label class="checkline">
                <input id="stream-prompts" type="checkbox" checked />
                <span><strong>Show live prompt/test activity</strong><small>Lightweight event feed while the backend runs.</small></span>
              </label>
              <label class="checkline">
                <input id="show-thinking" type="checkbox" />
                <span><strong>Show model thinking if a run records it</strong><small>Only appears when the backend receipt exposes safe thinking text.</small></span>
              </label>
              <button id="start">Start benchmark</button>
              <button id="stop-after-current" class="ghost-button" type="button">Stop after current</button>
              <p id="guard" class="sub"></p>
            </div>
          </div>
          <div class="panel">
            <h2>Librarian bot jobs</h2>
            <div class="body" id="packs"></div>
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
        <div id="winner" class="winner"></div>
        <div id="events" class="event-feed"></div>
      </section>
      <section class="panel status reports">
        <strong>Receipts and reports</strong>
        <div id="global-reports" class="links"></div>
        <div id="receipts" class="receipt-list"></div>
      </section>
    </main>
  </div>
  <script>
    const selected = new Set();
    const sortModes = ["family", "size", "name"];
    let sortIndex = 0;
    let appState = null;
    let socket = null;
    let fallbackNotice = "";

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
      document.querySelector("#nav-models").textContent = state.models.length;
      const tbody = document.querySelector("#models");
      tbody.innerHTML = "";
      for (const model of sortedModels(state.models)) {
        const tr = document.createElement("tr");
        const checked = selected.has(model.path) ? "checked" : "";
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
          updateGuard();
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
    }

    function renderConfiguration(config) {
      if (!config || document.querySelector("#standard-flags").children.length) return;
      const standard = document.querySelector("#standard-flags");
      standard.innerHTML = config.standard_forced_args.map(item => flagChoice(item, true)).join("");
      const optional = document.querySelector("#optional-flags");
      optional.innerHTML = config.optional_forced_args.map(item => flagChoice(item, false)).join("");
      standard.querySelectorAll("input").forEach(input => input.checked = true);
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

    function updateGuard() {
      if (!appState) return;
      const mode = document.querySelector("#mode").value;
      const plan = document.querySelector("#benchmark-suite-plan").value;
      const models = appState.models.filter(model => selected.has(model.path));
      const guard = document.querySelector("#guard");
      if (models.length === 0) {
        guard.textContent = "Select one or more models to benchmark.";
      } else {
        const planText = plan ? ` Benchmark suite plan: ${plan.split(/[\\\\/]/).pop()}.` : "";
        const compareHint = (mode === "librarian_bench" && models.length === 1)
          ? " Add a second model to compare them head-to-head."
          : "";
        guard.textContent = `${models.length} model(s) ready.${planText}${compareHint}`;
      }
    }

    function connectSocket() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
      socket.addEventListener("message", event => {
        const message = JSON.parse(event.data);
        if (message.type === "state") render(message.payload);
        if (message.type === "run_started") {
          document.querySelector("#guard").textContent = message.message;
          sendSocket({type: "refresh"});
        }
        if (message.type === "stop_after_current") {
          document.querySelector("#guard").textContent = message.message;
          sendSocket({type: "refresh"});
        }
        if (message.type === "error") document.querySelector("#guard").textContent = message.message;
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
      document.querySelector("#guard").textContent = payload.message;
      await loadStateViaHttp();
    }

    function sendSocket(message) {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        sendHttpCommand(message).catch(error => {
          document.querySelector("#guard").textContent = `Local command failed: ${error.message}`;
        });
        return;
      }
      socket.send(JSON.stringify(message));
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
      const mode = appState?.modes.find(item => item.id === document.querySelector("#mode").value);
      if (mode) {
        document.querySelector("#budget").value = mode.budget_minutes;
        document.querySelector("#mode").dataset.defaultedFor = mode.id;
      }
      updateGuard();
    });
    document.querySelector("#start").addEventListener("click", () => {
      sendSocket({
        type: "start_run",
        model_paths: Array.from(selected),
        mode_id: document.querySelector("#mode").value,
        options: {
          budget_minutes: Number(document.querySelector("#budget").value),
          benchmark_suite_plan: document.querySelector("#benchmark-suite-plan").value,
          forced_server_args: selectedForcedArgs(),
          stream_prompts: document.querySelector("#stream-prompts").checked,
          show_thinking: document.querySelector("#show-thinking").checked
        }
      });
    });
    document.querySelector("#benchmark-suite-plan").addEventListener("change", updateGuard);
    document.querySelector("#stop-after-current").addEventListener("click", () => {
      sendSocket({type: "stop_after_current"});
    });

    loadStateViaHttp("Loading local state...");
    connectSocket();
    setInterval(() => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        sendSocket({type: "refresh"});
      } else {
        loadStateViaHttp(fallbackNotice);
      }
    }, 2500);
  </script>
</body>
</html>
"""
