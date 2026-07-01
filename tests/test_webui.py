from __future__ import annotations

from pathlib import Path
import time

from fastapi.testclient import TestClient

from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS
from gguf_limit_bench.webui import (
    DEFAULT_WEBUI_PORT,
    INDEX_HTML,
    WebRunOptions,
    WebUiState,
    build_run_options,
    create_web_app,
    recent_receipts,
    receipt_event_payloads,
    resolve_run_artifact,
    serve_webui,
    validate_web_selection,
)


def test_webui_state_lists_models_modes_and_librarian_packs(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "Gemma-3-27B-Q4_K_M.gguf").write_bytes(b"1" * 20)
    (model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf").write_bytes(b"1" * 30)

    state = WebUiState(root=model_root, runs_root=tmp_path / "_runs")
    payload = state.state_payload()

    assert [model["family"] for model in payload["models"]] == ["qwen", "gemma"]
    assert payload["default_mode"] == "librarian_bench"
    assert payload["default_flight_plan"] == "librarian_benchmark"
    assert any(plan["id"] == "librarian_benchmark" for plan in payload["flight_plans"])
    assert payload["librarian_packs"] == list(LIBRARIAN_PACK_IDS)
    assert any(mode["id"] == "librarian_bench" for mode in payload["modes"])
    assert payload["run_configuration"]["standard_forced_args"]
    assert "receipts" in payload
    qwen = next(model for model in payload["models"] if model["family"] == "qwen")
    assert any(preset["id"] == "hf:thinking_general" for preset in qwen["sampler_presets"])


def test_webui_shell_explains_run_cost_and_evidence():
    assert "<title>pilotBENCHY</title>" in INDEX_HTML
    assert '<div class="brand">pilotBENCHY</div>' in INDEX_HTML
    assert "<h1>pilotBENCHY</h1>" in INDEX_HTML
    assert "Agent Pilot benchmark cockpit" not in INDEX_HTML
    assert "Previous results" in INDEX_HTML
    assert 'id="quick-reports"' in INDEX_HTML
    assert 'id="quick-receipts"' in INDEX_HTML
    assert 'id="flight-plan"' in INDEX_HTML
    assert "Advanced / choose mode directly" in INDEX_HTML
    assert 'id="run-summary"' in INDEX_HTML
    assert "127.0.0.1:36939" in INDEX_HTML
    assert "window.location.host" in INDEX_HTML
    assert "function updateRunSummary" in INDEX_HTML
    assert "function modelPathsForStart" in INDEX_HTML
    assert "scored attempts" in INDEX_HTML
    assert "weighted score + bias checks" in INDEX_HTML
    assert "Click Select all, or choose one or more models before starting." in INDEX_HTML
    assert "One model found. Start will use it automatically." in INDEX_HTML


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


def test_webui_start_run_calls_backend_for_selected_models(tmp_path):
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
    calls: list[tuple[str, WebRunOptions]] = []

    def fake_run_model(model: ModelInfo, options: WebRunOptions):
        calls.append((model.name, options))
        receipt = tmp_path / "_runs" / model.name
        receipt.mkdir(parents=True)
        return receipt

    state = WebUiState(
        root=model_root,
        runs_root=tmp_path / "_runs",
        run_model=fake_run_model,
        project_root=tmp_path,
    )

    ok, message = state.start_run(
        [str(gemma_path), str(qwen_path)],
        "librarian_bench",
        {
            "budget_minutes": 7,
            "sample_size": 15,
            "repeats": 3,
            "benchmark_suite_plan": str(plan_path),
            "forced_server_args": ["--flash-attn", "on", "--jinja"],
            "stream_prompts": True,
        },
    )

    assert ok is True, message
    deadline = time.time() + 2
    while time.time() < deadline and len(calls) < 2:
        time.sleep(0.02)
    assert sorted((name, options.mode_id, options.budget_minutes) for name, options in calls) == [
        ("Gemma-3-27B-Q4_K_M.gguf", "librarian_bench", 7),
        ("Qwen3.6-35B-A3B-Q4_K_M.gguf", "librarian_bench", 7),
    ]
    assert calls[0][1].forced_server_args == ("--flash-attn", "on", "--jinja")
    assert calls[0][1].flight_plan_id is None
    assert calls[0][1].sample_size == 15
    assert calls[0][1].repeats == 3
    assert calls[0][1].sampler_policy == "hf_recommended"
    assert {options.benchmark_suite_plan for _name, options in calls} == {plan_path.resolve()}
    assert state.run.phase == "complete"
    assert any(event.kind == "receipt" for event in state.run.events)


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


def test_build_run_options_accepts_flight_plan_as_beginner_contract():
    options = build_run_options(
        "quick",
        {
            "flight_plan_id": "librarian_benchmark",
            "forced_server_args": ["--jinja"],
        },
    )

    assert options.flight_plan_id == "librarian_benchmark"
    assert options.mode_id == "librarian_bench"
    assert options.budget_minutes == 30


def test_build_run_options_preserves_mode_without_flight_plan():
    options = build_run_options(
        "quick",
        {
            "flight_plan_id": "",
            "forced_server_args": ["--jinja"],
        },
    )

    assert options.flight_plan_id is None
    assert options.mode_id == "quick"
    assert options.budget_minutes == 5


def test_build_run_options_rejects_unknown_flight_plan():
    try:
        build_run_options("quick", {"flight_plan_id": "typo"})
    except ValueError as exc:
        assert "Unknown flight plan" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected unknown flight plan to be rejected")


def test_build_run_options_rejects_malformed_numeric_payloads():
    for payload, expected in (
        ({"budget_minutes": "soon"}, "Budget must be a number"),
        ({"sample_size": True}, "Sample size must be a number"),
        ({"repeats": 99}, "Repeats must be between 1 and 20"),
    ):
        try:
            build_run_options("quick", payload)
        except ValueError as exc:
            assert expected in str(exc)
        else:  # pragma: no cover - assertion clarity
            raise AssertionError(f"expected malformed payload to be rejected: {payload}")


def test_build_run_options_rejects_malformed_forced_args_and_sampler_policy():
    for payload, expected in (
        ({"forced_server_args": "--jinja"}, "forced_server_args must be a list"),
        ({"forced_server_args": ["--jinja", 123]}, "forced_server_args entries"),
        ({"stream_prompts": "false"}, "stream_prompts must be true or false"),
        ({"show_thinking": 1}, "show_thinking must be true or false"),
        ({"sampler_policy": 0}, "sampler_policy must be a string"),
        ({"sampler_policy": 123}, "sampler_policy must be a string"),
        ({"sampler_policy": "surprise-me"}, "Unsupported sampler policy"),
    ):
        try:
            build_run_options("quick", payload)
        except ValueError as exc:
            assert expected in str(exc)
        else:  # pragma: no cover - assertion clarity
            raise AssertionError(f"expected malformed payload to be rejected: {payload}")


def test_recent_receipts_and_run_artifact_links_stay_under_runs_root(tmp_path):
    runs_root = tmp_path / "_runs"
    receipt = runs_root / "2026-06-24-qwen"
    receipt.mkdir(parents=True)
    (receipt / "report.html").write_text("<h1>ok</h1>", encoding="utf-8")
    (receipt / "resolved-plan.json").write_text('{"schema_version": 1}', encoding="utf-8")
    (receipt / "command.txt").write_text("agent-autobench autoresearch\n", encoding="utf-8")
    (receipt / "status.json").write_text('{"status": "finished"}', encoding="utf-8")
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
    assert {"Resolved plan", "Command", "Status", "Suite summary", "Librarian report"} <= artifact_labels
    assert resolve_run_artifact(runs_root, "2026-06-24-qwen/report.html") == receipt / "report.html"
    assert resolve_run_artifact(runs_root, "../outside.txt") is None


def test_webui_start_endpoint_rejects_malformed_json(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    client = TestClient(create_web_app(state))

    response = client.post(
        "/api/start",
        content="{not-json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert "valid JSON" in response.text


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


def test_webui_websocket_start_run_dispatches_backend(tmp_path):
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
    calls: list[tuple[str, WebRunOptions]] = []

    def fake_run_model(model: ModelInfo, options: WebRunOptions):
        calls.append((model.name, options))
        receipt = tmp_path / "_runs" / model.name
        receipt.mkdir(parents=True)
        return receipt

    state = WebUiState(
        root=model_root,
        runs_root=tmp_path / "_runs",
        run_model=fake_run_model,
        project_root=tmp_path,
    )
    client = TestClient(create_web_app(state))

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "start_run",
                "flight_plan_id": "librarian_benchmark",
                "model_paths": [str(gemma_path), str(qwen_path)],
                "mode_id": "quick",
                "options": {
                    "sample_size": 17,
                    "repeats": 4,
                    "sampler_policy": "runtime_defaults",
                    "flight_plan_id": "librarian_benchmark",
                    "benchmark_suite_plan": str(plan_path),
                    "forced_server_args": ["--jinja"],
                },
            }
        )
        reply = websocket.receive_json()

    assert reply["type"] == "run_started"
    assert reply["ok"] is True
    deadline = time.time() + 2
    while time.time() < deadline and len(calls) < 2:
        time.sleep(0.02)
    assert [call[1].flight_plan_id for call in calls] == [
        "librarian_benchmark",
        "librarian_benchmark",
    ]
    assert [call[1].mode_id for call in calls] == ["librarian_bench", "librarian_bench"]
    assert [call[1].budget_minutes for call in calls] == [30, 30]
    assert [call[1].sample_size for call in calls] == [17, 17]
    assert [call[1].repeats for call in calls] == [4, 4]
    assert [call[1].sampler_policy for call in calls] == [
        "runtime_defaults",
        "runtime_defaults",
    ]
    assert [call[1].benchmark_suite_plan for call in calls] == [
        plan_path.resolve(),
        plan_path.resolve(),
    ]


def test_webui_websocket_mode_only_start_is_not_overridden_by_flight_plan(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    qwen_path = model_root / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
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
                "flight_plan_id": "",
                "model_paths": [str(qwen_path)],
                "mode_id": "quick",
                "options": {
                    "flight_plan_id": "",
                    "forced_server_args": ["--jinja"],
                },
            }
        )
        reply = websocket.receive_json()

    assert reply["type"] == "run_started"
    assert reply["ok"] is True
    deadline = time.time() + 2
    while time.time() < deadline and not calls:
        time.sleep(0.02)
    assert calls[0][1].flight_plan_id is None
    assert calls[0][1].mode_id == "quick"
    assert calls[0][1].budget_minutes == 5


def test_webui_websocket_stop_after_current_marks_run_state(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    state.run.phase = "running"
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


def test_webui_http_stop_after_current_marks_run_state(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    state.run.phase = "running"
    client = TestClient(create_web_app(state))

    response = client.post("/api/stop-after-current")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert state.run.stop_requested is True
    assert any(event.kind == "stop" for event in state.run.events)


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


def test_webui_api_start_rejects_non_object_options(tmp_path):
    state = WebUiState(root=tmp_path / "models", runs_root=tmp_path / "_runs")
    client = TestClient(create_web_app(state))

    response = client.post(
        "/api/start",
        json={"model_paths": [], "mode_id": "quick", "options": "not-an-object"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert "options must be a JSON object" in payload["message"]


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
            "plan_kind": "",
            "requires": "",
            "score_contract": "",
            "task_count": 0,
            "phases": [],
            "harnesses": [],
            "warning": "Requires a local endpoint.",
        }
    ]


def test_webui_state_tails_receipt_events_while_running(tmp_path):
    runs_root = tmp_path / "_runs"
    latest = runs_root / "20260102-new"
    latest.mkdir(parents=True)
    (latest / "events.jsonl").write_text(
        '{"time":"2026-01-02T00:00:00","type":"attempt","data":{"context":8192,"score":1.23}}\n',
        encoding="utf-8",
    )
    state = WebUiState(root=tmp_path / "models", runs_root=runs_root)
    state.run.phase = "running"
    state.run.options = {"stream_prompts": True}

    payload = state.state_payload()

    assert payload["run"]["events"][-1] == {
        "at": "00:00:00",
        "kind": "attempt",
        "message": '{"context": 8192, "score": 1.23}',
    }


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
        run_model=None,
        host="127.0.0.1",
        open_browser=True,
    )

    assert DEFAULT_WEBUI_PORT == 36939
    assert url == "http://127.0.0.1:36939/"
    assert captured["url"] == "http://127.0.0.1:36939/"
    assert captured["ran"] is True
    assert captured["config"].host == "127.0.0.1"
    assert captured["config"].port == 36939
