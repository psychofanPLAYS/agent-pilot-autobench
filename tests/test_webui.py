from __future__ import annotations

import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

from fastapi.testclient import TestClient

from gguf_limit_bench import run_dir
from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS
from gguf_limit_bench import webui
from gguf_limit_bench.webui import (
    WebUiState,
    _handler_for,
    build_run_options,
    create_web_app,
    recent_receipts,
    receipt_event_payloads,
    resolve_run_artifact,
    serve_webui,
    validate_web_selection,
)


class _FakeProc:
    """Stand-in for a detached engine subprocess in tests (never really runs)."""

    pid = 4242

    def poll(self):
        return None


def _fake_spawn_factory(recorder=None):
    def _spawn(run_dir_path):
        if recorder is not None:
            recorder.append(run_dir_path)
        return _FakeProc()

    return _spawn


def test_webui_state_lists_models_modes_and_librarian_packs(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "Gemma-3-27B-Q4_K_M.gguf").write_bytes(b"1" * 20)
    (model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf").write_bytes(b"1" * 30)

    state = WebUiState(root=model_root, runs_root=tmp_path / "_runs")
    payload = state.state_payload()

    assert [model["family"] for model in payload["models"]] == ["qwen", "gemma"]
    assert payload["default_mode"] == "librarian_bench"
    assert payload["librarian_packs"] == list(LIBRARIAN_PACK_IDS)
    assert any(mode["id"] == "librarian_bench" for mode in payload["modes"])
    assert payload["run_configuration"]["standard_forced_args"]
    assert "receipts" in payload


def test_librarian_web_selection_accepts_any_models():
    gemma = ModelInfo(path=Path("Gemma-3-27B.gguf"), name="Gemma-3-27B.gguf", family="gemma")
    qwen = ModelInfo(path=Path("Qwen3.6-35B-A3B.gguf"), name="Qwen3.6-35B-A3B.gguf", family="qwen")

    # Agent Pilot benchmarks any GGUF model — no hardcoded Gemma-vs-Qwen requirement.
    assert validate_web_selection([qwen], "librarian_bench") is None
    assert validate_web_selection([gemma], "librarian_bench") is None
    assert validate_web_selection([gemma, qwen], "librarian_bench") is None
    assert validate_web_selection([qwen], "quick") is None
    # An empty selection is still rejected for every mode.
    assert validate_web_selection([], "librarian_bench") is not None


def test_webui_start_run_writes_spec_and_spawns_engine(tmp_path):
    plans_root = tmp_path / "benchmarks" / "plans"
    plans_root.mkdir(parents=True)
    plan_path = plans_root / "wiki-librarian-gemma3-27b-direct.plan.json"
    plan_path.write_text('{"model": "gemma-3-27b-it", "tasks": []}', encoding="utf-8")
    model_root = tmp_path / "models"
    model_root.mkdir()
    gemma_path = model_root / "Gemma-3-27B-Q4_K_M.gguf"
    qwen_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    gemma_path.write_bytes(b"1" * 20)
    qwen_path.write_bytes(b"1" * 30)

    spawned: list = []
    state = WebUiState(
        root=model_root,
        runs_root=tmp_path / "_runs",
        spawn_engine=_fake_spawn_factory(spawned),
        project_root=tmp_path,
    )

    ok, message = state.start_run(
        [str(gemma_path), str(qwen_path)],
        "librarian_bench",
        {
            "budget_minutes": 7,
            "benchmark_suite_plan": str(plan_path),
            "forced_server_args": ["--flash-attn", "on", "--jinja"],
            "stream_prompts": True,
        },
    )

    assert ok is True, message
    # the engine is launched exactly once and owns the sequential queue itself
    assert len(spawned) == 1
    assert state.active_run_dir == spawned[0]
    spec = run_dir.read_spec(spawned[0])
    assert spec["mode"] == "librarian_bench"
    assert spec["options"]["budget_minutes"] == 7
    assert spec["options"]["forced_server_args"] == ["--flash-attn", "on", "--jinja"]
    assert spec["options"]["benchmark_suite_plan"] == str(plan_path.resolve())
    assert [m["path"] for m in spec["models"]] == [str(gemma_path), str(qwen_path)]
    assert state.run.phase == "running"


def test_webui_start_run_writes_resolved_llama_paths(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.write_bytes(b"1" * 30)
    runs_root = tmp_path / "_runs"
    server = tmp_path / "llama" / "llama-server.exe"
    bench = tmp_path / "llama" / "llama-bench.exe"

    spawned: list = []
    state = WebUiState(
        root=model_root,
        runs_root=runs_root,
        spawn_engine=_fake_spawn_factory(spawned),
        project_root=tmp_path,
        llama_server=server,
        llama_bench=bench,
    )

    ok, message = state.start_run([str(model_path)], "librarian_bench", {"budget_minutes": 1})
    assert ok is True, message

    spec = run_dir.read_spec(spawned[0])
    assert spec["paths"]["llama_server"] == str(server)
    assert spec["paths"]["llama_bench"] == str(bench)
    # Paths not provided are null; runs_root always reflects the configured root.
    assert spec["paths"]["llama_cli"] is None
    assert spec["paths"]["llama_perplexity"] is None
    assert spec["paths"]["runs_root"] == str(runs_root)


def test_webui_start_run_writes_null_llama_paths_when_unset(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.write_bytes(b"1" * 30)
    runs_root = tmp_path / "_runs"

    spawned: list = []
    state = WebUiState(
        root=model_root,
        runs_root=runs_root,
        spawn_engine=_fake_spawn_factory(spawned),
        project_root=tmp_path,
    )

    ok, message = state.start_run([str(model_path)], "librarian_bench", {"budget_minutes": 1})
    assert ok is True, message

    spec = run_dir.read_spec(spawned[0])
    assert spec["paths"] == {
        "llama_server": None,
        "llama_bench": None,
        "llama_cli": None,
        "llama_perplexity": None,
        "runs_root": str(runs_root),
    }


def test_webui_start_run_rejects_when_engine_already_running(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.write_bytes(b"1" * 30)
    state = WebUiState(
        root=model_root,
        runs_root=tmp_path / "_runs",
        spawn_engine=_fake_spawn_factory(),
        project_root=tmp_path,
    )

    ok, _ = state.start_run([str(model_path)], "librarian_bench", {"budget_minutes": 1})
    assert ok is True

    ok2, message = state.start_run([str(model_path)], "librarian_bench", {"budget_minutes": 1})
    assert ok2 is False
    assert "already running" in message


def test_webui_start_run_allows_new_run_after_previous_completed(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.write_bytes(b"1" * 30)
    state = WebUiState(
        root=model_root,
        runs_root=tmp_path / "_runs",
        spawn_engine=_fake_spawn_factory(),
        project_root=tmp_path,
    )
    state.start_run([str(model_path)], "librarian_bench", {"budget_minutes": 1})
    run_dir.write_status(state.active_run_dir, phase="complete")

    ok, message = state.start_run([str(model_path)], "librarian_bench", {"budget_minutes": 1})
    assert ok is True, message


def test_webui_state_payload_reflects_engine_status_and_live_events(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.write_bytes(b"1" * 30)
    state = WebUiState(
        root=model_root,
        runs_root=tmp_path / "_runs",
        spawn_engine=_fake_spawn_factory(),
        project_root=tmp_path,
    )
    state.start_run([str(model_path)], "librarian_bench", {"budget_minutes": 1})
    rd = state.active_run_dir
    run_dir.write_status(rd, phase="running", model="Qwen", model_index=1, model_total=1, pid=7)
    run_dir.append_event(rd, "question_scored", {"q_id": "q1", "score": 1.0})

    payload = state.state_payload()

    assert payload["run"]["phase"] == "running"
    assert "question_scored" in [event["kind"] for event in payload["run"]["events"]]


def test_webui_reattaches_to_live_engine_run(tmp_path):
    runs_root = tmp_path / "_runs"
    rd = runs_root / "20260630-010101-cockpit"
    rd.mkdir(parents=True)
    run_dir.write_status(rd, phase="running", pid=7)
    model_root = tmp_path / "models"
    model_root.mkdir()
    state = WebUiState(
        root=model_root,
        runs_root=runs_root,
        spawn_engine=_fake_spawn_factory(),
        project_root=tmp_path,
    )

    state.reattach()

    assert state.active_run_dir == rd


def test_webui_abort_writes_control_and_kills_engine(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.write_bytes(b"1" * 30)
    killed: list = []
    monkeypatch.setattr(webui, "kill_process_tree", lambda proc: killed.append(proc))
    state = WebUiState(
        root=model_root,
        runs_root=tmp_path / "_runs",
        spawn_engine=_fake_spawn_factory(),
        project_root=tmp_path,
    )
    state.start_run([str(model_path)], "librarian_bench", {"budget_minutes": 1})

    ok, _ = state.request_abort()

    assert ok is True
    assert run_dir.read_control(state.active_run_dir)["action"] == "abort"
    assert len(killed) == 1


def test_webui_rejects_unknown_model_path(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    qwen_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    qwen_path.write_bytes(b"1" * 30)
    state = WebUiState(root=model_root, runs_root=tmp_path / "_runs")

    ok, message = state.start_run([str(qwen_path), str(model_root / "typo.gguf")], "quick")

    assert ok is False
    assert "Unknown model path" in message


def test_build_run_options_rejects_unsafe_web_flags():
    try:
        build_run_options("quick", {"forced_server_args": ["--host", "0.0.0.0"]})
    except ValueError as exc:
        assert "--host is managed" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected unsafe flag to be rejected")


def test_build_run_options_rejects_outside_benchmark_suite_plan(tmp_path):
    outside = tmp_path / "outside.plan.json"
    outside.write_text("{}", encoding="utf-8")

    try:
        build_run_options(
            "quick",
            {"benchmark_suite_plan": str(outside)},
            project_root=tmp_path,
        )
    except ValueError as exc:
        assert "under benchmarks/plans" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected outside benchmark-suite plan to be rejected")


def test_recent_receipts_and_run_artifact_links_stay_under_runs_root(tmp_path):
    runs_root = tmp_path / "_runs"
    receipt = runs_root / "2026-06-24-qwen"
    receipt.mkdir(parents=True)
    (receipt / "report.html").write_text("<h1>ok</h1>", encoding="utf-8")
    (receipt / "best-settings.json").write_text(
        '{"model": "G:/models/Qwen.gguf", "status": "complete", "result": {"score": 12.5}}',
        encoding="utf-8",
    )
    (receipt / "suite-summary.json").write_text('{"ok": true}', encoding="utf-8")
    (receipt / "librarian-suite.md").write_text("# ok", encoding="utf-8")

    receipts = recent_receipts(runs_root)

    assert receipts[0]["model"] == "Qwen.gguf"
    assert receipts[0]["artifacts"][0]["url"] == "/runs/2026-06-24-qwen/report.html"
    artifact_labels = {artifact["label"] for artifact in receipts[0]["artifacts"]}
    assert {"Suite summary", "Librarian report"} <= artifact_labels
    assert resolve_run_artifact(runs_root, "2026-06-24-qwen/report.html") == receipt / "report.html"
    assert resolve_run_artifact(runs_root, "../outside.txt") is None


def test_webui_start_endpoint_rejects_malformed_json(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request(
            "POST",
            "/api/start",
            body="{not-json",
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status == 400
    assert "valid JSON" in payload


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


def test_webui_websocket_start_run_spawns_engine(tmp_path):
    plans_root = tmp_path / "benchmarks" / "plans"
    plans_root.mkdir(parents=True)
    plan_path = plans_root / "wiki-librarian-qwen3-moe-thinking.plan.json"
    plan_path.write_text('{"model": "qwen3.6-35b-a3b", "tasks": []}', encoding="utf-8")
    model_root = tmp_path / "models"
    model_root.mkdir()
    gemma_path = model_root / "Gemma-3-27B-Q4_K_M.gguf"
    qwen_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    gemma_path.write_bytes(b"1" * 20)
    qwen_path.write_bytes(b"1" * 30)
    spawned: list = []

    state = WebUiState(
        root=model_root,
        runs_root=tmp_path / "_runs",
        spawn_engine=_fake_spawn_factory(spawned),
        project_root=tmp_path,
    )
    client = TestClient(create_web_app(state))

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "start_run",
                "model_paths": [str(gemma_path), str(qwen_path)],
                "mode_id": "librarian_bench",
                "options": {
                    "budget_minutes": 3,
                    "benchmark_suite_plan": str(plan_path),
                    "forced_server_args": ["--jinja"],
                },
            }
        )
        reply = websocket.receive_json()

    assert reply["type"] == "run_started"
    assert reply["ok"] is True
    assert len(spawned) == 1
    spec = run_dir.read_spec(spawned[0])
    assert spec["options"]["budget_minutes"] == 3
    assert [m["path"] for m in spec["models"]] == [str(gemma_path), str(qwen_path)]


def _running_state(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.write_bytes(b"1" * 30)
    state = WebUiState(
        root=model_root,
        runs_root=tmp_path / "_runs",
        spawn_engine=_fake_spawn_factory(),
        project_root=tmp_path,
    )
    state.start_run([str(model_path)], "librarian_bench", {"budget_minutes": 1})
    return state


def test_webui_websocket_stop_after_current_writes_control(tmp_path):
    state = _running_state(tmp_path)
    client = TestClient(create_web_app(state))

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json({"type": "stop_after_current"})
        reply = websocket.receive_json()

    assert reply["type"] == "stop_after_current"
    assert reply["ok"] is True
    assert run_dir.read_control(state.active_run_dir)["action"] == "stop"


def test_webui_http_stop_after_current_writes_control(tmp_path):
    state = _running_state(tmp_path)
    client = TestClient(create_web_app(state))

    response = client.post("/api/stop-after-current")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert run_dir.read_control(state.active_run_dir)["action"] == "stop"


def test_webui_websocket_stop_after_current_rejects_idle_run(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    client = TestClient(create_web_app(state))

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json({"type": "stop_after_current"})
        reply = websocket.receive_json()

    assert reply["type"] == "stop_after_current"
    assert reply["ok"] is False
    assert state.run.stop_requested is False
    assert "No active benchmark run" in reply["message"]


def test_webui_websocket_rejects_invalid_json_without_crashing(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    client = TestClient(create_web_app(state))

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_text("{not-json")
        reply = websocket.receive_json()

    assert reply["type"] == "error"
    assert "valid JSON" in reply["message"]


def test_webui_websocket_rejects_non_list_model_paths(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    client = TestClient(create_web_app(state))

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "start_run",
                "model_paths": "not-a-list",
                "mode_id": "quick",
            }
        )
        reply = websocket.receive_json()

    assert reply["type"] == "run_started"
    assert reply["ok"] is False
    assert "model_paths must be a list" in reply["message"]


def test_webui_api_start_rejects_non_list_model_paths(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    client = TestClient(create_web_app(state))

    response = client.post(
        "/api/start",
        json={"model_paths": "not-a-list", "mode_id": "quick"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert "model_paths must be a list" in payload["message"]


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


def test_receipt_event_payloads_tails_latest_receipt(tmp_path):
    runs_root = tmp_path / "_runs"
    older = runs_root / "20260101-old"
    latest = runs_root / "20260102-new"
    older.mkdir(parents=True)
    latest.mkdir()
    (older / "events.jsonl").write_text(
        '{"time":"2026-01-01T00:00:00","type":"old","data":{"model":"old.gguf"}}\n',
        encoding="utf-8",
    )
    (latest / "events.jsonl").write_text(
        '{"time":"2026-01-02T00:00:00","type":"attempt","data":{"context":8192,"score":1.23}}\n',
        encoding="utf-8",
    )

    assert receipt_event_payloads(runs_root) == [
        {
            "at": "00:00:00",
            "kind": "attempt",
            "message": '{"context": 8192, "score": 1.23}',
        }
    ]


def test_serve_webui_builds_fastapi_app_without_starting_benchmark(tmp_path, monkeypatch):
    captured = {}

    class FakeServer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("gguf_limit_bench.webui.uvicorn.Server", FakeServer)
    monkeypatch.setattr(
        "gguf_limit_bench.webui.webbrowser.open", lambda url: captured.setdefault("url", url)
    )

    url = serve_webui(
        root=tmp_path / "models",
        runs_root=tmp_path / "_runs",
        host="127.0.0.1",
        port=8765,
        open_browser=True,
    )

    assert url == "http://127.0.0.1:8765/"
    assert captured["url"] == "http://127.0.0.1:8765/"
    assert captured["ran"] is True
    assert captured["config"].host == "127.0.0.1"
    assert captured["config"].port == 8765
