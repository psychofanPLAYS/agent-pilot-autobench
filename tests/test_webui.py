from __future__ import annotations

from pathlib import Path
import time

from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS
from gguf_limit_bench.webui import WebUiState, validate_web_selection


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
    calls: list[tuple[str, str]] = []

    def fake_run_model(model: ModelInfo, mode_id: str):
        calls.append((model.name, mode_id))
        receipt = tmp_path / "_runs" / model.name
        receipt.mkdir(parents=True)
        return receipt

    state = WebUiState(root=model_root, runs_root=tmp_path / "_runs", run_model=fake_run_model)

    ok, message = state.start_run([str(gemma_path), str(qwen_path)], "librarian_bench")

    assert ok is True, message
    deadline = time.time() + 2
    while time.time() < deadline and len(calls) < 2:
        time.sleep(0.02)
    assert sorted(calls) == [
        ("Gemma-3-27B-Q4_K_M.gguf", "librarian_bench"),
        ("Qwen3.6-35B-A3B-Q4_K_M.gguf", "librarian_bench"),
    ]
    assert state.run.phase == "complete"
