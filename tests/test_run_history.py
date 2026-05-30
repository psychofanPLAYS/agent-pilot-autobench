import json

from gguf_limit_bench.run_history import truncated_previous_runs_text


def test_truncated_previous_runs_text_summarizes_champion(tmp_path):
    run = tmp_path / "20260527-test"
    run.mkdir()
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "models/Winner.gguf",
                "settings": {"context_size": 4096},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "unknown",
                },
                "score": 51.0,
            }
        ),
        encoding="utf-8",
    )

    text = truncated_previous_runs_text(tmp_path)

    assert "Previous runs" in text
    assert "Winner.gguf" in text
    assert "Champion" in text


def test_truncated_previous_runs_text_handles_empty_folder(tmp_path):
    assert "No receipts yet" in truncated_previous_runs_text(tmp_path)


def test_truncated_previous_runs_text_reads_legacy_runs_when_new_folder_is_empty(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "runs" / "20260527-test"
    legacy.mkdir(parents=True)
    (legacy / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "models/LegacyWinner.gguf",
                "settings": {"context_size": 4096},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 30.0,
                    "prompt_tokens_per_second": 700.0,
                    "failure": "unknown",
                },
                "score": 40.0,
            }
        ),
        encoding="utf-8",
    )

    assert "LegacyWinner.gguf" in truncated_previous_runs_text(tmp_path / "_runs")
