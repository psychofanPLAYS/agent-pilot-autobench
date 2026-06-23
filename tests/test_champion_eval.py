"""Tests for champion_eval.evaluate_champion_packs (Task 6).

All tests use monkeypatching — no real llama-server is started.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Generator

from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.packs import AnswerType, PackQuestion, QuestionPack
from gguf_limit_bench.simple_bench import SimpleBenchBatchResult, SimpleBenchQuestionResult
from gguf_limit_bench.state_db import _init_connection, get_selection_cursor


# ---------------------------------------------------------------------------
# Shared test fixtures and helpers
# ---------------------------------------------------------------------------

_FAKE_SETTINGS = AutoresearchSettings(
    profile_name="test-profile",
    context_size=4096,
    parallel=1,
    gpu_layers=99,
    batch_size=2048,
    ubatch_size=512,
    flash_attention=True,
    kv_unified=True,
)

_FAKE_PACK = QuestionPack(
    pack_id="easy-mc",
    title="Easy MC",
    tier="easy",
    answer_type=AnswerType.MULTIPLE_CHOICE,
    system_prompt="You are a test assistant.",
    questions=tuple(
        PackQuestion(
            question_id=f"q{i}",
            prompt=f"Question {i}?",
            answer="A",
            answer_source="test",
            choices=("A", "B", "C", "D"),
        )
        for i in range(1, 6)
    ),
)

_FAKE_QUESTION_RESULT = SimpleBenchQuestionResult(
    question_id="q1",
    expected_answer="A",
    predicted_answer="A",
    correct=True,
    ttft_ms=50.0,
    tokens_per_second=30.0,
    generated_tokens=10,
    output_chars=20,
    prompt_chars=100,
    response="Final Answer: A",
    outcome="correct",
)

_FAKE_BATCH = SimpleBenchBatchResult(
    ok=True,
    score=1000.0,
    accuracy=1.0,
    correct=1,
    total=1,
    median_tps=30.0,
    min_tps=30.0,
    median_ttft_ms=50.0,
    results=[_FAKE_QUESTION_RESULT],
    incomplete=0,
    completion_rate=1.0,
)


@contextmanager
def _fake_server_session(**_kwargs) -> Generator[str, None, None]:
    """Fake llama_server_session that yields a dummy base_url."""
    yield "http://127.0.0.1:19999"


def _make_fake_run_pack_questions(**_kwargs) -> SimpleBenchBatchResult:
    return _FAKE_BATCH


def _make_fake_load_pack(pack_id: str) -> QuestionPack:
    if pack_id == "easy-mc":
        return _FAKE_PACK
    raise KeyError(f"Unknown pack: {pack_id}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_evaluate_champion_packs_writes_results_json(tmp_path, monkeypatch):
    """evaluate_champion_packs writes results.json into run_dir."""
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.llama_server_session", _fake_server_session
    )
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.run_pack_questions",
        lambda **kwargs: _make_fake_run_pack_questions(**kwargs),
    )
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.load_pack",
        lambda pack_id: _make_fake_load_pack(pack_id),
    )

    from gguf_limit_bench.champion_eval import evaluate_champion_packs

    model = tmp_path / "test-model.gguf"
    model.touch()
    llama_server = tmp_path / "llama-server.exe"
    llama_server.touch()
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()

    evaluate_champion_packs(
        model=model,
        llama_server=llama_server,
        best_settings=_FAKE_SETTINGS,
        run_dir=run_dir,
        pack_ids=("easy-mc",),
        sample_size=3,
        selection="sequential",
        seed=None,
        state_db_path=None,
        gpu_name="RTX 4090",
    )

    results_json = run_dir / "results.json"
    assert results_json.exists(), "results.json must be written into run_dir"


def test_evaluate_champion_packs_results_references_packs(tmp_path, monkeypatch):
    """The written results.json references the requested pack IDs."""
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.llama_server_session", _fake_server_session
    )
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.run_pack_questions",
        lambda **kwargs: _make_fake_run_pack_questions(**kwargs),
    )
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.load_pack",
        lambda pack_id: _make_fake_load_pack(pack_id),
    )

    from gguf_limit_bench.champion_eval import evaluate_champion_packs

    model = tmp_path / "model.gguf"
    model.touch()
    llama_server = tmp_path / "llama-server.exe"
    llama_server.touch()
    run_dir = tmp_path / "run-002"
    run_dir.mkdir()

    evaluate_champion_packs(
        model=model,
        llama_server=llama_server,
        best_settings=_FAKE_SETTINGS,
        run_dir=run_dir,
        pack_ids=("easy-mc",),
        sample_size=3,
        selection="sequential",
        seed=None,
        state_db_path=None,
        gpu_name="",
    )

    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    pack_ids_in_result = [p["pack_id"] for p in payload["packs"]]
    assert "easy-mc" in pack_ids_in_result


def test_evaluate_champion_packs_advances_state_db_cursor(tmp_path, monkeypatch):
    """After the call the selection cursor in the state DB must have advanced."""
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.llama_server_session", _fake_server_session
    )
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.run_pack_questions",
        lambda **kwargs: _make_fake_run_pack_questions(**kwargs),
    )
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.load_pack",
        lambda pack_id: _make_fake_load_pack(pack_id),
    )

    from gguf_limit_bench.champion_eval import evaluate_champion_packs

    model = tmp_path / "model.gguf"
    model.touch()
    llama_server = tmp_path / "llama-server.exe"
    llama_server.touch()
    run_dir = tmp_path / "run-003"
    run_dir.mkdir()
    db_path = tmp_path / "state.sqlite"

    evaluate_champion_packs(
        model=model,
        llama_server=llama_server,
        best_settings=_FAKE_SETTINGS,
        run_dir=run_dir,
        pack_ids=("easy-mc",),
        sample_size=3,
        selection="sequential",
        seed=None,
        state_db_path=db_path,
        gpu_name="",
    )

    # After running 3 questions sequentially from a 5-question pack, cursor must be 3.
    conn = sqlite3.connect(db_path)
    _init_connection(conn)
    cursor_after = get_selection_cursor(conn, model_key=model.name, pack_id="easy-mc")
    conn.close()
    assert cursor_after == 3, (
        f"Expected cursor=3 after selecting 3 of 5 questions sequentially, got {cursor_after}"
    )


def test_evaluate_champion_packs_results_json_contains_model_name(tmp_path, monkeypatch):
    """results.json must record the model path."""
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.llama_server_session", _fake_server_session
    )
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.run_pack_questions",
        lambda **kwargs: _make_fake_run_pack_questions(**kwargs),
    )
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.load_pack",
        lambda pack_id: _make_fake_load_pack(pack_id),
    )

    from gguf_limit_bench.champion_eval import evaluate_champion_packs

    model = tmp_path / "my-cool-model.gguf"
    model.touch()
    llama_server = tmp_path / "llama-server.exe"
    llama_server.touch()
    run_dir = tmp_path / "run-004"
    run_dir.mkdir()

    evaluate_champion_packs(
        model=model,
        llama_server=llama_server,
        best_settings=_FAKE_SETTINGS,
        run_dir=run_dir,
        pack_ids=("easy-mc",),
        sample_size=2,
        selection="sequential",
        seed=None,
        state_db_path=None,
        gpu_name="RTX 3090",
    )

    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert "my-cool-model.gguf" in payload["model"]
    assert payload["gpu"] == "RTX 3090"


def test_evaluate_champion_packs_unknown_pack_skipped(tmp_path, monkeypatch):
    """An unknown pack ID is skipped; results.json is still written."""
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.llama_server_session", _fake_server_session
    )
    monkeypatch.setattr(
        "gguf_limit_bench.champion_eval.run_pack_questions",
        lambda **kwargs: _make_fake_run_pack_questions(**kwargs),
    )
    # load_pack raises KeyError for unknown packs — use real behaviour
    from gguf_limit_bench.champion_eval import evaluate_champion_packs

    model = tmp_path / "model.gguf"
    model.touch()
    llama_server = tmp_path / "llama-server.exe"
    llama_server.touch()
    run_dir = tmp_path / "run-005"
    run_dir.mkdir()

    # "definitely-not-a-pack" does not exist in the data dir
    evaluate_champion_packs(
        model=model,
        llama_server=llama_server,
        best_settings=_FAKE_SETTINGS,
        run_dir=run_dir,
        pack_ids=("definitely-not-a-pack",),
        sample_size=2,
        selection="sequential",
        seed=None,
        state_db_path=None,
        gpu_name="",
    )

    results_json = run_dir / "results.json"
    assert results_json.exists(), "results.json must be written even when all packs are unknown"
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    assert payload["packs"][0]["pack_id"] == "definitely-not-a-pack"
    assert payload["packs"][0]["asked"] == 0
