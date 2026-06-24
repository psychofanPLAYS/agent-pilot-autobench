from __future__ import annotations

import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
import time

from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS
from gguf_limit_bench.webui import (
    WebRunOptions,
    WebUiState,
    _handler_for,
    build_run_options,
    recent_receipts,
    resolve_run_artifact,
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
    assert payload["librarian_packs"] == list(LIBRARIAN_PACK_IDS)
    assert any(mode["id"] == "librarian_bench" for mode in payload["modes"])
    assert payload["run_configuration"]["standard_forced_args"]
    assert "receipts" in payload


def test_librarian_web_selection_requires_gemma_and_qwen():
    gemma = ModelInfo(path=Path("Gemma-3-27B.gguf"), name="Gemma-3-27B.gguf", family="gemma")
    qwen = ModelInfo(path=Path("Qwen3.6-35B-A3B.gguf"), name="Qwen3.6-35B-A3B.gguf", family="qwen")

    assert validate_web_selection([qwen], "librarian_bench") is not None
    assert validate_web_selection([gemma], "librarian_bench") is not None
    assert validate_web_selection([gemma, qwen], "librarian_bench") is None
    assert validate_web_selection([qwen], "quick") is None


def test_webui_start_run_calls_backend_for_selected_models(tmp_path):
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

    ok, message = state.start_run(
        [str(gemma_path), str(qwen_path)],
        "librarian_bench",
        {
            "budget_minutes": 7,
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


def test_recent_receipts_and_run_artifact_links_stay_under_runs_root(tmp_path):
    runs_root = tmp_path / "_runs"
    receipt = runs_root / "2026-06-24-qwen"
    receipt.mkdir(parents=True)
    (receipt / "report.html").write_text("<h1>ok</h1>", encoding="utf-8")
    (receipt / "best-settings.json").write_text(
        '{"model": "G:/models/Qwen.gguf", "status": "complete", "result": {"score": 12.5}}',
        encoding="utf-8",
    )

    receipts = recent_receipts(runs_root)

    assert receipts[0]["model"] == "Qwen.gguf"
    assert receipts[0]["artifacts"][0]["url"] == "/runs/2026-06-24-qwen/report.html"
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
