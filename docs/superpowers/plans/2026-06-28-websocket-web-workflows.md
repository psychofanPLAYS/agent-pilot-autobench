# WebSocket Web Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first WebSocket-backed browser workflow slice while keeping `apb tui` available as a fallback.

**Architecture:** Keep the existing Typer CLI and benchmark core as the contract. Add a FastAPI/uvicorn local service behind the existing `serve_webui` entrypoint, with WebSocket JSON messages for state, run start, stop-after-current, telemetry, and receipt updates. Preserve the current stdlib HTTP behavior only as a rollback reference, not as the browser's normal live path.

**Tech Stack:** Python 3.11-3.13, Typer, FastAPI, uvicorn, pytest, existing `gguf_limit_bench` benchmark modules.

---

## File Structure

- Modify `pyproject.toml`: add `fastapi` and `uvicorn` runtime dependencies.
- Modify `uv.lock`: refresh via `uv lock` after dependency edits.
- Modify `src/gguf_limit_bench/webui.py`: keep public API names but add WebSocket state, stop-after-current state, benchmark-suite plan payloads, and FastAPI app factory.
- Modify `src/gguf_limit_bench/cli.py`: keep calling `serve_webui`; pass benchmark-suite plan and callback data through the new service unchanged.
- Modify `tests/test_webui.py`: add FastAPI `TestClient` WebSocket tests, benchmark plan listing tests, and stop-after-current tests.
- Modify `tests/test_cli.py`: keep existing `serve_webui` monkeypatch tests passing; add assertions only if new `serve_webui` arguments are introduced.
- Modify `README.md`, `docs/ARCHITECTURE.md`, `docs/START-FOR-NORMAL-PEOPLE.md`, and `docs/COMMAND-BOARD.md`: update wording after code passes.

## Task 1: Add Web Runtime Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add dependency entries**

In `pyproject.toml`, update the runtime dependency list to include:

```toml
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
```

Keep existing dependencies such as `textual` because `apb tui` remains supported.

- [ ] **Step 2: Refresh the lockfile**

Run:

```powershell
uv lock
```

Expected: `uv.lock` updates cleanly and does not install large model or benchmark artifacts.

- [ ] **Step 3: Verify import availability**

Run:

```powershell
uv run --extra dev python -c "import fastapi, uvicorn; print(fastapi.__version__)"
```

Expected: exits `0` and prints a FastAPI version.

## Task 2: Test WebSocket State Contract

**Files:**
- Modify: `tests/test_webui.py`

- [ ] **Step 1: Write failing WebSocket bootstrap test**

Add imports:

```python
from fastapi.testclient import TestClient
```

Add this test:

```python
def test_webui_websocket_sends_hello_and_state(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "Gemma-3-27B-Q4_K_M.gguf").write_bytes(b"1" * 20)
    state = WebUiState(root=model_root, runs_root=tmp_path / "_runs")
    client = TestClient(create_web_app(state))

    with client.websocket_connect("/ws") as websocket:
        hello = websocket.receive_json()
        state_message = websocket.receive_json()

    assert hello["type"] == "hello"
    assert hello["protocol"] == 1
    assert state_message["type"] == "state"
    assert state_message["payload"]["models"][0]["name"] == "Gemma-3-27B-Q4_K_M.gguf"
```

Also update the import from `gguf_limit_bench.webui` to include `create_web_app`.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_webui_websocket_sends_hello_and_state -q
```

Expected: fails because `create_web_app` is not defined.

## Task 3: Implement FastAPI App Factory And WebSocket Bootstrap

**Files:**
- Modify: `src/gguf_limit_bench/webui.py`

- [ ] **Step 1: Add FastAPI imports**

Add near the existing imports:

```python
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
```

- [ ] **Step 2: Add message helpers**

Add functions near the web state helpers:

```python
WS_PROTOCOL_VERSION = 1


def websocket_message(message_type: str, payload: dict | None = None) -> dict:
    return {"type": message_type, "payload": payload or {}}


def websocket_error(message: str) -> dict:
    return {"type": "error", "message": message}
```

- [ ] **Step 3: Add app factory**

Add this function before `serve_webui`:

```python
def create_web_app(state: WebUiState) -> FastAPI:
    app = FastAPI(title="pilotBENCHY local cockpit", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(state.state_payload())

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "hello", "protocol": WS_PROTOCOL_VERSION})
        await websocket.send_json(websocket_message("state", state.state_payload()))
        try:
            while True:
                message = await websocket.receive_json()
                response = await handle_websocket_command(state, message)
                if response is not None:
                    await websocket.send_json(response)
        except WebSocketDisconnect:
            return

    return app
```

- [ ] **Step 4: Add initial command handler**

Add this function after `create_web_app`:

```python
async def handle_websocket_command(state: WebUiState, message: object) -> dict | None:
    if not isinstance(message, dict):
        return websocket_error("WebSocket message must be a JSON object.")
    message_type = str(message.get("type") or "")
    if message_type in {"subscribe", "refresh"}:
        return websocket_message("state", state.state_payload())
    return websocket_error(f"Unknown WebSocket message type: {message_type}")
```

- [ ] **Step 5: Run the bootstrap test and verify GREEN**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_webui_websocket_sends_hello_and_state -q
```

Expected: passes.

## Task 4: Test WebSocket Start Run Command

**Files:**
- Modify: `tests/test_webui.py`

- [ ] **Step 1: Write failing WebSocket start test**

Add:

```python
def test_webui_websocket_start_run_dispatches_backend(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    gemma_path = model_root / "Gemma-3-27B-Q4_K_M.gguf"
    qwen_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    gemma_path.write_bytes(b"1" * 20)
    qwen_path.write_bytes(b"1" * 30)
    calls: list[tuple[str, WebRunOptions]] = []

    def fake_run_model(model: ModelInfo, options: WebRunOptions):
        calls.append((model.name, options))
        receipt = tmp_path / "_runs" / model.name
        receipt.mkdir(parents=True)
        return receipt

    state = WebUiState(root=model_root, runs_root=tmp_path / "_runs", run_model=fake_run_model)
    client = TestClient(create_web_app(state))

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "start_run",
                "model_paths": [str(gemma_path), str(qwen_path)],
                "mode_id": "librarian_bench",
                "options": {"budget_minutes": 3, "forced_server_args": ["--jinja"]},
            }
        )
        reply = websocket.receive_json()

    assert reply["type"] == "run_started"
    assert reply["ok"] is True
    deadline = time.time() + 2
    while time.time() < deadline and len(calls) < 2:
        time.sleep(0.02)
    assert [call[1].budget_minutes for call in calls] == [3, 3]
```

- [ ] **Step 2: Run the new start test and verify RED**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_webui_websocket_start_run_dispatches_backend -q
```

Expected: fails because `start_run` WebSocket messages are unknown.

## Task 5: Implement WebSocket Start Command

**Files:**
- Modify: `src/gguf_limit_bench/webui.py`

- [ ] **Step 1: Extend command handler**

Change `handle_websocket_command`:

```python
async def handle_websocket_command(state: WebUiState, message: object) -> dict | None:
    if not isinstance(message, dict):
        return websocket_error("WebSocket message must be a JSON object.")
    message_type = str(message.get("type") or "")
    if message_type in {"subscribe", "refresh"}:
        return websocket_message("state", state.state_payload())
    if message_type == "start_run":
        model_paths = [str(path) for path in message.get("model_paths", [])]
        options = message.get("options") if isinstance(message.get("options"), dict) else {}
        ok, response_message = state.start_run(
            model_paths=model_paths,
            mode_id=str(message.get("mode_id", "librarian_bench")),
            options_payload=options,
        )
        return {"type": "run_started", "ok": ok, "message": response_message}
    return websocket_error(f"Unknown WebSocket message type: {message_type}")
```

- [ ] **Step 2: Run the start test and verify GREEN**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_webui_websocket_start_run_dispatches_backend -q
```

Expected: passes.

## Task 6: Add Stop-After-Current State

**Files:**
- Modify: `tests/test_webui.py`
- Modify: `src/gguf_limit_bench/webui.py`

- [ ] **Step 1: Write failing state test**

Add:

```python
def test_webui_websocket_stop_after_current_marks_run_state(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    client = TestClient(create_web_app(state))

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json({"type": "stop_after_current"})
        reply = websocket.receive_json()

    assert reply["type"] == "stop_after_current"
    assert reply["ok"] is True
    assert state.run.stop_requested is True
    assert any(event.kind == "stop" for event in state.run.events)
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_webui_websocket_stop_after_current_marks_run_state -q
```

Expected: fails because `stop_requested` does not exist.

- [ ] **Step 3: Add state field and method**

Update `WebRunState`:

```python
    stop_requested: bool = False
```

Add to `WebUiState`:

```python
    def request_stop_after_current(self) -> tuple[bool, str]:
        with self._lock:
            if self.run.phase not in {"running", "idle"}:
                return False, "No active benchmark run can be stopped."
            self.run.stop_requested = True
            self.run.events.append(
                _event("stop", "Stop requested. The current benchmark item will finish first.")
            )
            return True, "Stop requested after current item."
```

- [ ] **Step 4: Honor stop state in the run loop**

Inside `_run_models`, after appending each receipt and event, add:

```python
                with self._lock:
                    stop_requested = self.run.stop_requested
                if stop_requested:
                    with self._lock:
                        self.run.phase = "stopping"
                        self.run.message = "Stopped after the current benchmark item."
                        self.run.events.append(_event("stop", "Run queue stopped by request."))
                    break
```

Then change the final completion block to preserve the stopped phase:

```python
            with self._lock:
                if self.run.phase != "stopping":
                    self.run.phase = "complete"
                    self.run.message = "Benchmark complete."
                    self.run.events.append(_event("complete", "Benchmark queue finished."))
                self.run.receipts = receipts
```

- [ ] **Step 5: Add WebSocket command**

Extend `handle_websocket_command`:

```python
    if message_type == "stop_after_current":
        ok, response_message = state.request_stop_after_current()
        return {"type": "stop_after_current", "ok": ok, "message": response_message}
```

- [ ] **Step 6: Run and verify GREEN**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_webui_websocket_stop_after_current_marks_run_state -q
```

Expected: passes.

## Task 7: List Benchmark-Suite Plans In Web State

**Files:**
- Modify: `tests/test_webui.py`
- Modify: `src/gguf_limit_bench/webui.py`

- [ ] **Step 1: Write failing plan payload test**

Add:

```python
def test_webui_state_lists_benchmark_suite_plans(tmp_path):
    root = tmp_path
    plans = root / "benchmarks" / "plans"
    plans.mkdir(parents=True)
    (plans / "local-openai-smoke.plan.json").write_text(
        '{"name": "Local OpenAI smoke", "description": "Requires a local endpoint."}',
        encoding="utf-8",
    )
    state = WebUiState(root=root / "models", runs_root=root / "_runs", project_root=root)

    payload = state.state_payload()

    assert payload["benchmark_suite_plans"] == [
        {
            "path": str(plans / "local-openai-smoke.plan.json"),
            "filename": "local-openai-smoke.plan.json",
            "name": "Local OpenAI smoke",
            "description": "Requires a local endpoint.",
            "warning": "Requires a local endpoint.",
        }
    ]
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_webui_state_lists_benchmark_suite_plans -q
```

Expected: fails because `project_root` or `benchmark_suite_plans` is missing.

- [ ] **Step 3: Add project root and plan payload code**

Update `WebUiState.__init__`:

```python
        project_root: Path | None = None,
```

Set:

```python
        self.project_root = project_root or Path.cwd()
```

Add to `state_payload`:

```python
            "benchmark_suite_plans": benchmark_suite_plan_payloads(self.project_root),
```

Add helper:

```python
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
    if "already" in text and "endpoint" in text:
        return description
    if "external" in text or "uvx" in text:
        return "This plan may call an external benchmark tool."
    if "heavy" in text:
        return "This plan may take a long time."
    return ""
```

- [ ] **Step 4: Run and verify GREEN**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_webui_state_lists_benchmark_suite_plans -q
```

Expected: passes.

## Task 8: Serve With Uvicorn Through Existing Entry Point

**Files:**
- Modify: `src/gguf_limit_bench/webui.py`
- Modify: `tests/test_webui.py`

- [ ] **Step 1: Write serving construction test**

Add:

```python
def test_serve_webui_builds_fastapi_app_without_starting_benchmark(tmp_path, monkeypatch):
    captured = {}

    class FakeServer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("gguf_limit_bench.webui.uvicorn.Server", FakeServer)
    monkeypatch.setattr("gguf_limit_bench.webui.webbrowser.open", lambda url: captured.setdefault("url", url))

    url = serve_webui(
        root=tmp_path / "models",
        runs_root=tmp_path / "_runs",
        run_model=None,
        host="127.0.0.1",
        port=8765,
        open_browser=True,
    )

    assert url == "http://127.0.0.1:8765/"
    assert captured["url"] == "http://127.0.0.1:8765/"
    assert captured["ran"] is True
    assert captured["config"].host == "127.0.0.1"
    assert captured["config"].port == 8765
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_serve_webui_builds_fastapi_app_without_starting_benchmark -q
```

Expected: fails until `serve_webui` uses `uvicorn.Server`.

- [ ] **Step 3: Replace stdlib server startup**

Add import:

```python
import uvicorn
```

Change `serve_webui` to:

```python
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
    app = create_web_app(state)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    resolved_port = port
    url = f"http://{host}:{resolved_port}/"
    if open_browser and port != 0:
        webbrowser.open(url)
    server.run()
    return url
```

If dynamic port `0` must remain supported, follow up with a socket pre-bind helper
that chooses a free localhost port before constructing the uvicorn config.

- [ ] **Step 4: Run serving test and verify GREEN**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py::test_serve_webui_builds_fastapi_app_without_starting_benchmark -q
```

Expected: passes.

## Task 9: Move Browser JavaScript To WebSocket Path

**Files:**
- Modify: `src/gguf_limit_bench/webui.py`

- [ ] **Step 1: Replace polling `refresh` behavior**

Inside `INDEX_HTML`, add a WebSocket connection:

```javascript
    let socket = null;

    function connectSocket() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
      socket.addEventListener("message", event => {
        const message = JSON.parse(event.data);
        if (message.type === "state") render(message.payload);
        if (message.type === "run_started") {
          document.querySelector("#guard").textContent = message.message;
          socket.send(JSON.stringify({type: "refresh"}));
        }
        if (message.type === "stop_after_current") {
          document.querySelector("#guard").textContent = message.message;
          socket.send(JSON.stringify({type: "refresh"}));
        }
        if (message.type === "error") document.querySelector("#guard").textContent = message.message;
      });
      socket.addEventListener("close", () => setTimeout(connectSocket, 2000));
    }

    function sendSocket(message) {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        document.querySelector("#guard").textContent = "Live connection is not ready yet.";
        return;
      }
      socket.send(JSON.stringify(message));
    }
```

- [ ] **Step 2: Change Start button to send WebSocket command**

Replace the `fetch("/api/start"...` block with:

```javascript
      sendSocket({
        type: "start_run",
        model_paths: Array.from(selected),
        mode_id: document.querySelector("#mode").value,
        options: {
          budget_minutes: Number(document.querySelector("#budget").value),
          forced_server_args: selectedForcedArgs(),
          stream_prompts: document.querySelector("#stream-prompts").checked,
          show_thinking: document.querySelector("#show-thinking").checked
        }
      });
```

- [ ] **Step 3: Add stop-after-current button**

Add a button near `#start`:

```html
              <button id="stop-after-current" class="ghost-button" type="button">Stop after current</button>
```

Add listener:

```javascript
    document.querySelector("#stop-after-current").addEventListener("click", () => {
      sendSocket({type: "stop_after_current"});
    });
```

- [ ] **Step 4: Start WebSocket on load**

Replace:

```javascript
    refresh();
    setInterval(refresh, 2500);
```

with:

```javascript
    connectSocket();
    setInterval(() => sendSocket({type: "refresh"}), 2500);
```

## Task 10: Update CLI And Existing Tests If Needed

**Files:**
- Modify: `src/gguf_limit_bench/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Run web and CLI tests**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py tests/test_cli.py -q
```

Expected: failures only where signatures changed.

- [ ] **Step 2: Preserve `serve_webui` monkeypatch compatibility**

If CLI tests fail because `serve_webui` gained new arguments, keep the public
signature stable or update tests to capture the new argument explicitly:

```python
def fake_serve_webui(**kwargs) -> str:
    captured.update(kwargs)
    return "http://127.0.0.1:8765/"
```

- [ ] **Step 3: Re-run tests**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py tests/test_cli.py -q
```

Expected: passes.

## Task 11: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/START-FOR-NORMAL-PEOPLE.md`
- Modify: `docs/COMMAND-BOARD.md`

- [ ] **Step 1: Update primary cockpit wording**

Use this wording in the relevant docs:

```markdown
Plain `apb` opens the local browser cockpit. The browser is the primary workflow
for model selection, benchmark-suite plan selection, live run progress, telemetry,
and receipt links. `apb tui` remains available as a fallback terminal cockpit.
```

- [ ] **Step 2: Update architecture feature map**

In `docs/ARCHITECTURE.md`, update the Browser cockpit row to say:

```markdown
| Browser cockpit | `apb` / `agent-autobench start` | `webui.py`, `cli.py`, `discovery.py`, FastAPI/WebSocket service | selected model paths, WebSocket run events, telemetry, receipts | `test_webui.py`, `test_cli.py` |
```

- [ ] **Step 3: Run public identity/docs tests**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_public_identity.py tests/test_windows_scripts.py -q
```

Expected: passes, or fails only on wording that needs the docs adjusted.

## Task 12: Final Verification

**Files:**
- No new edits unless verification finds a bug.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
uv run --extra dev python -m pytest tests/test_webui.py tests/test_cli.py -q
```

Expected: passes.

- [ ] **Step 2: Compile source and tests**

Run:

```powershell
uv run --extra dev python -m compileall src tests
```

Expected: exits `0`.

- [ ] **Step 3: Run lightweight CLI smoke**

Run:

```powershell
uv run --extra dev agent-autobench results
```

Expected: command exits `0` and does not start model servers.

- [ ] **Step 4: Inspect git scope**

Run:

```powershell
git status --short --branch
git diff --stat
```

Expected: changes are limited to the web workflow files, dependency files, docs, and tests. Pre-existing unrelated dirty files remain untouched.

## Self-Review Notes

- Spec coverage: this plan covers WebSocket live state, run start, stop-after-current, benchmark-suite plan listing, docs, TUI fallback preservation, and narrow tests.
- Scope held back: charts, cloud dashboards, account systems, hard process kill controls, and live llama.cpp benchmark runs are intentionally excluded.
- Type consistency: WebSocket messages use `type`; run options keep `WebRunOptions`; public CLI integration remains `serve_webui`.
