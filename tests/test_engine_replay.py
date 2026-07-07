"""Replay engine: re-stream a recorded live.jsonl into a run dir so the cockpit
can be driven and visually verified with no GPU."""

from __future__ import annotations

import json

from gguf_limit_bench import engine_replay, run_dir


def _write_source(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_replay_copies_events_and_completes(tmp_path):
    src = tmp_path / "src.jsonl"
    _write_source(
        src,
        [
            {"time": "t", "type": "model_started", "data": {"model": "m", "index": 1, "total": 1}},
            {
                "time": "t",
                "type": "question_started",
                "data": {"q_id": "q1", "index": 1, "total": 2, "prompt": "P"},
            },
            {"time": "t", "type": "question_scored", "data": {"q_id": "q1", "correct": True}},
        ],
    )
    rd = tmp_path / "run"
    rd.mkdir()

    engine_replay.replay(rd, src, delay=0.0)

    live = (rd / run_dir.LIVE_FILE).read_text(encoding="utf-8")
    assert "question_started" in live and "question_scored" in live
    assert run_dir.read_status(rd)["phase"] == "complete"


def test_replay_updates_status_from_events(tmp_path):
    src = tmp_path / "src.jsonl"
    _write_source(
        src,
        [
            {
                "time": "t",
                "type": "model_started",
                "data": {"model": "Qwen", "index": 1, "total": 3},
            },
            {
                "time": "t",
                "type": "question_started",
                "data": {"q_id": "q1", "index": 2, "total": 5, "prompt": "P"},
            },
        ],
    )
    rd = tmp_path / "run"
    rd.mkdir()
    captured = []
    engine_replay.replay(
        rd, src, delay=0.0, on_step=lambda: captured.append(run_dir.read_status(rd))
    )

    # after model_started the status reflects the model; after question_started the index
    assert any(s.get("model") == "Qwen" for s in captured)
    assert any(s.get("question_index") == 2 and s.get("question_total") == 5 for s in captured)


def test_replay_paces_with_injected_sleep(tmp_path):
    src = tmp_path / "src.jsonl"
    _write_source(src, [{"time": "t", "type": "model_started", "data": {}}])
    rd = tmp_path / "run"
    rd.mkdir()
    slept: list = []
    engine_replay.replay(rd, src, delay=0.25, sleep=lambda s: slept.append(s))
    assert slept == [0.25]
