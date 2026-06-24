from __future__ import annotations

from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Callable
from urllib.parse import urlparse
import webbrowser

from gguf_limit_bench.discovery import ModelInfo, discover_models
from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS
from gguf_limit_bench.modes import RUN_MODES
from gguf_limit_bench.reports import write_leaderboard
from gguf_limit_bench.telemetry import sample_telemetry
from gguf_limit_bench.tui import active_run_status

WebRunModelCallback = Callable[[ModelInfo, str], Path]


@dataclass
class WebRunState:
    phase: str = "idle"
    message: str = "Ready."
    selected_models: list[str] = field(default_factory=list)
    receipts: list[str] = field(default_factory=list)
    error: str | None = None


class WebUiState:
    def __init__(
        self,
        *,
        root: Path,
        runs_root: Path,
        run_model: WebRunModelCallback | None = None,
    ) -> None:
        self.root = root
        self.runs_root = runs_root
        self.run_model = run_model
        self.run = WebRunState()
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
        return {
            "models": [_model_payload(model) for model in models],
            "modes": [_mode_payload(mode) for mode in RUN_MODES],
            "default_mode": "librarian_bench",
            "librarian_packs": list(LIBRARIAN_PACK_IDS),
            "telemetry": telemetry,
            "active_run": active_run_status(self.runs_root),
            "champion": champion,
            "run": run_payload,
        }

    def start_run(self, model_paths: list[str], mode_id: str) -> tuple[bool, str]:
        models_by_path = {str(model.path): model for model in self.models()}
        selected = [models_by_path[path] for path in model_paths if path in models_by_path]
        issue = validate_web_selection(selected, mode_id)
        if issue is not None:
            return False, issue
        with self._lock:
            if self.run.phase == "running":
                return False, "A benchmark is already running."
            self.run = WebRunState(
                phase="running",
                message=f"Queued {len(selected)} model(s).",
                selected_models=[model.name for model in selected],
            )
        thread = threading.Thread(
            target=self._run_models,
            args=(selected, mode_id),
            name="pilotBENCHY-WebUI-runner",
            daemon=True,
        )
        thread.start()
        return True, "Benchmark started."

    def _run_models(self, selected: list[ModelInfo], mode_id: str) -> None:
        receipts: list[str] = []
        try:
            for index, model in enumerate(selected, start=1):
                with self._lock:
                    self.run.message = f"Running {index}/{len(selected)}: {model.name}"
                if self.run_model is None:
                    receipt = self.runs_root / "webui-preview"
                    receipt.mkdir(parents=True, exist_ok=True)
                else:
                    receipt = self.run_model(model, mode_id)
                receipts.append(str(receipt))
                with self._lock:
                    self.run.receipts = receipts[:]
            with self._lock:
                self.run.phase = "complete"
                self.run.message = "Benchmark complete."
                self.run.receipts = receipts
        except Exception as exc:  # noqa: BLE001 - surface background failures to UI
            with self._lock:
                self.run.phase = "failed"
                self.run.error = str(exc)
                self.run.message = "Benchmark failed."


def validate_web_selection(selected: list[ModelInfo], mode_id: str) -> str | None:
    if not selected:
        return "Select at least one model first."
    if mode_id != "librarian_bench":
        return None
    has_gemma = any(_looks_like_family(model, "gemma") for model in selected)
    has_qwen = any(_looks_like_family(model, "qwen") for model in selected)
    if has_gemma and has_qwen:
        return None
    return "Librarian bot test needs at least one Gemma model and one Qwen model."


def serve_webui(
    *,
    root: Path,
    runs_root: Path,
    run_model: WebRunModelCallback | None,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> str:
    state = WebUiState(root=root, runs_root=runs_root, run_model=run_model)
    handler = _handler_for(state)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}/"
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return url


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
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlparse(self.path).path
            if path != "/api/start":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = self._read_json()
            ok, message = state.start_run(
                model_paths=[str(path) for path in payload.get("model_paths", [])],
                mode_id=str(payload.get("mode_id", "librarian_bench")),
            )
            self._send_json({"ok": ok, "message": message}, status=200 if ok else 400)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

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

    return PilotBenchHandler


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


def _looks_like_family(model: ModelInfo, family: str) -> bool:
    target = family.lower()
    return model.family.lower() == target or target in model.name.lower()


def _size_label(size_gb: float) -> str:
    if size_gb <= 0:
        return "0"
    if size_gb < 0.01:
        return "<0.01"
    return f"{size_gb:.2f}"


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>pilotBENCHY</title>
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
    button:disabled { opacity: .5; cursor: not-allowed; }
    .pack { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; color: var(--muted); }
    .telemetry { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 16px; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px; }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 18px; font-weight: 800; margin-top: 4px; }
    .status { margin-top: 16px; padding: 14px 16px; }
    .ok { color: var(--good); }
    .bad { color: var(--bad); }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      aside { display: none; }
      .grid { grid-template-columns: 1fr; }
      .telemetry { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">pilotBENCHY</div>
      <div class="navitem"><span>Control</span><span>local</span></div>
      <div class="navitem"><span>Models</span><span id="nav-models">0</span></div>
      <div class="navitem"><span>Receipts</span><span>_runs</span></div>
      <div class="navitem"><span>Server</span><span>127.0.0.1</span></div>
    </aside>
    <main>
      <header>
        <div>
          <h1>Local bot benchmark cockpit</h1>
          <div class="sub">Pick Gemma and Qwen models, choose the librarian worker test, and launch repeatable local receipts from the browser.</div>
        </div>
      </header>
      <section class="grid">
        <div class="panel">
          <h2>Model selection</h2>
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
              <button id="start">Start benchmark</button>
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
      </section>
    </main>
  </div>
  <script>
    const selected = new Set();
    let appState = null;

    function familyClass(family) {
      return family === "gemma" ? "gemma" : family === "qwen" ? "qwen" : "";
    }

    function render(state) {
      appState = state;
      document.querySelector("#nav-models").textContent = state.models.length;
      const tbody = document.querySelector("#models");
      tbody.innerHTML = "";
      for (const model of state.models) {
        const tr = document.createElement("tr");
        const checked = selected.has(model.path) ? "checked" : "";
        tr.innerHTML = `
          <td><input type="checkbox" data-path="${model.path}" ${checked}></td>
          <td>${model.name}</td>
          <td class="${familyClass(model.family)}">${model.family}</td>
          <td>${model.parameters}</td>
          <td><span class="chip">${model.quant}</span></td>
          <td>${model.size_label}</td>
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
      document.querySelector("#packs").innerHTML = state.librarian_packs
        .map((pack, index) => `<div class="pack"><span>${pack}</span><span>${index + 1}</span></div>`)
        .join("");
      const t = state.telemetry;
      document.querySelector("#cpu").textContent = `${Math.round(t.cpu_used_percent)}%`;
      document.querySelector("#ram").textContent = `${Math.round(t.ram_used_percent)}%`;
      document.querySelector("#gpu").textContent = t.gpu_util_percent == null ? "n/a" : `${t.gpu_util_percent}%`;
      document.querySelector("#vram").textContent = t.gpu_used_mb == null ? "n/a" : `${t.gpu_used_mb}/${t.gpu_total_mb} MB`;
      const run = state.run;
      const active = state.active_run ? ` | ${state.active_run}` : "";
      document.querySelector("#run-status").innerHTML =
        `<span class="${run.phase === "failed" ? "bad" : "ok"}">${run.phase}</span>: ${run.message}${active}`;
      updateGuard();
    }

    function updateGuard() {
      if (!appState) return;
      const mode = document.querySelector("#mode").value;
      const models = appState.models.filter(model => selected.has(model.path));
      const hasGemma = models.some(model => model.family === "gemma" || model.name.toLowerCase().includes("gemma"));
      const hasQwen = models.some(model => model.family === "qwen" || model.name.toLowerCase().includes("qwen"));
      const guard = document.querySelector("#guard");
      if (mode === "librarian_bench" && (!hasGemma || !hasQwen)) {
        guard.textContent = "Select at least one Gemma and one Qwen model for a direct worker comparison.";
      } else if (models.length === 0) {
        guard.textContent = "Select one or more models.";
      } else {
        guard.textContent = `${models.length} model(s) ready.`;
      }
    }

    async function refresh() {
      const response = await fetch("/api/state");
      render(await response.json());
    }

    document.querySelector("#mode").addEventListener("change", updateGuard);
    document.querySelector("#start").addEventListener("click", async () => {
      const response = await fetch("/api/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({model_paths: Array.from(selected), mode_id: document.querySelector("#mode").value})
      });
      const payload = await response.json();
      document.querySelector("#guard").textContent = payload.message;
      await refresh();
    });

    refresh();
    setInterval(refresh, 2500);
  </script>
</body>
</html>
"""
