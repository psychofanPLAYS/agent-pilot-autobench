"""Tests for results_report.py (Task 9) — written before implementation (TDD)."""

from __future__ import annotations

import json

from gguf_limit_bench.results_report import (
    build_results_payload,
    render_results_markdown,
    write_results,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_question(
    question_id: str | int = "q1",
    prompt: str = "What is 2+2?",
    expected: str = "4",
    predicted: str | None = "4",
    outcome: str = "correct",
) -> dict:
    return {
        "question_id": question_id,
        "prompt": prompt,
        "expected": expected,
        "predicted": predicted,
        "outcome": outcome,
    }


def _make_pack(
    pack_id: str = "easy-gotcha",
    tier: str = "easy",
    asked: int = 2,
    correct: int = 1,
    wrong: int = 0,
    incomplete: int = 1,
    accuracy: float = 0.5,
    median_tps: float = 42.0,
    median_ttft_ms: float | None = 150.0,
    questions: list[dict] | None = None,
) -> dict:
    if questions is None:
        questions = [
            _make_question("q1", "What is 2+2?", "4", "4", "correct"),
            _make_question(
                "q2",
                "A man has no brothers or sisters...",
                "his son",
                None,
                "incomplete",
            ),
        ]
    return {
        "pack_id": pack_id,
        "tier": tier,
        "asked": asked,
        "correct": correct,
        "wrong": wrong,
        "incomplete": incomplete,
        "accuracy": accuracy,
        "median_tps": median_tps,
        "median_ttft_ms": median_ttft_ms,
        "questions": questions,
    }


def _make_payload(packs: list[dict] | None = None) -> dict:
    if packs is None:
        packs = [
            _make_pack("easy-gotcha", incomplete=1),
            _make_pack(
                "easy-mc",
                tier="easy",
                asked=3,
                correct=3,
                wrong=0,
                incomplete=0,
                accuracy=1.0,
                median_tps=38.0,
                median_ttft_ms=None,
                questions=[
                    _make_question("mc1", "Which is larger: 9.11 or 9.9?", "9.9", "9.9", "correct"),
                    _make_question("mc2", "Q2", "A", "B", "wrong"),
                    _make_question("mc3", "Q3", "C", "C", "correct"),
                ],
            ),
        ]
    return build_results_payload(
        model="Qwen3-8B-Q4.gguf",
        selection_mode="random",
        selection_seed=42,
        sample_size=5,
        gpu="RTX 4090",
        recommended_flags=["--flash-attn", "--ctx-size 8192"],
        packs=packs,
    )


# ---------------------------------------------------------------------------
# build_results_payload tests
# ---------------------------------------------------------------------------


def test_build_results_payload_is_json_serialisable():
    payload = _make_payload()

    serialised = json.dumps(payload)
    round_tripped = json.loads(serialised)

    assert round_tripped["model"] == "Qwen3-8B-Q4.gguf"
    assert round_tripped["gpu"] == "RTX 4090"
    assert round_tripped["selection_mode"] == "random"
    assert round_tripped["selection_seed"] == 42
    assert round_tripped["sample_size"] == 5
    assert round_tripped["recommended_flags"] == ["--flash-attn", "--ctx-size 8192"]


def test_build_results_payload_contains_both_packs():
    payload = _make_payload()

    pack_ids = [p["pack_id"] for p in payload["packs"]]
    assert "easy-gotcha" in pack_ids
    assert "easy-mc" in pack_ids


def test_build_results_payload_none_seed_is_preserved():
    payload = build_results_payload(
        model="m.gguf",
        selection_mode="all",
        selection_seed=None,
        sample_size=10,
        gpu="RTX 3080",
        recommended_flags=[],
        packs=[],
    )

    assert payload["selection_seed"] is None
    assert json.dumps(payload)  # must still serialise


# ---------------------------------------------------------------------------
# render_results_markdown tests
# ---------------------------------------------------------------------------


def test_render_results_markdown_contains_both_pack_ids():
    payload = _make_payload()

    md = render_results_markdown(payload)

    assert "easy-gotcha" in md
    assert "easy-mc" in md


def test_render_results_markdown_contains_score_fraction():
    payload = _make_payload()

    md = render_results_markdown(payload)

    # At least one pack must show a "correct/asked" pattern, e.g. "1/2" or "3/3"
    assert "/2" in md or "/3" in md


def test_render_results_markdown_contains_incomplete_when_pack_has_incomplete():
    payload = _make_payload()

    md = render_results_markdown(payload)

    assert "incomplete" in md.lower()


def test_render_results_markdown_contains_a_predicted_answer():
    payload = _make_payload()

    md = render_results_markdown(payload)

    # The predicted answer "4" must appear somewhere
    assert "4" in md


def test_render_results_markdown_contains_model_and_gpu():
    payload = _make_payload()

    md = render_results_markdown(payload)

    assert "Qwen3-8B-Q4.gguf" in md
    assert "RTX 4090" in md


# ---------------------------------------------------------------------------
# write_results tests
# ---------------------------------------------------------------------------


def test_write_results_creates_json_and_md_files(tmp_path):
    payload = _make_payload()

    json_path, md_path = write_results(tmp_path, payload)

    assert json_path.exists()
    assert md_path.exists()
    assert json_path.name == "results.json"
    assert md_path.name == "results.md"


def test_write_results_json_is_valid_and_matches_payload(tmp_path):
    payload = _make_payload()

    json_path, _ = write_results(tmp_path, payload)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["model"] == payload["model"]
    assert len(loaded["packs"]) == len(payload["packs"])


def test_write_results_returns_paths_inside_run_dir(tmp_path):
    payload = _make_payload()
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()

    json_path, md_path = write_results(run_dir, payload)

    assert json_path.parent == run_dir
    assert md_path.parent == run_dir
