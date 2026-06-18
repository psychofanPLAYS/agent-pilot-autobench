import json
from io import BytesIO
from pathlib import Path

import pytest

from gguf_limit_bench import simple_bench_runner
from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchLoop, AutoresearchSettings
from gguf_limit_bench.flag_ladder import (
    build_core_flag_ladder,
    build_flag_ladder_plan,
    validate_extra_server_args,
)
from gguf_limit_bench.simple_bench import (
    SimpleBenchQuestionResult,
    SimpleBenchQuestion,
    combine_simple_bench_results,
    extract_final_answer,
    load_simple_bench_questions,
)
from gguf_limit_bench.simple_bench_runner import _write_short_logs, measure_simple_bench_completion


def test_load_simple_bench_public_shape(tmp_path):
    path = tmp_path / "simple_bench_public.json"
    path.write_text(
        json.dumps({"eval_data": [{"question_id": 1, "prompt": "Q\nA. x\n", "answer": "B"}]}),
        encoding="utf-8",
    )

    questions = load_simple_bench_questions(path)

    assert len(questions) == 1
    assert questions[0].question_id == 1
    assert questions[0].answer == "B"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"eval_data": "not-a-list"}, "eval_data must be a list"),
        (
            {"eval_data": [{"question_id": 1, "prompt": "", "answer": "A"}]},
            "non-empty prompt",
        ),
        (
            {"eval_data": [{"question_id": 1, "prompt": "Question", "answer": "Z"}]},
            "answer A-F",
        ),
    ],
)
def test_load_simple_bench_rejects_invalid_shapes_and_rows(tmp_path, payload, message):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_simple_bench_questions(path)


def test_load_simple_bench_rejects_duplicate_question_ids(tmp_path):
    row = {"question_id": 1, "prompt": "Question", "answer": "A"}
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps({"eval_data": [row, row]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate question_id 1"):
        load_simple_bench_questions(path)


def test_extract_final_answer_prefers_final_answer_marker():
    assert extract_final_answer("Reasoning mentions A and B. Final Answer: C") == "C"
    assert extract_final_answer("Reasoning mentions A, B, and C but never finishes") is None


def test_extract_final_answer_uses_latest_explicit_revision():
    response = "Final Answer: A\nCorrection after checking: Final Answer: B"

    assert extract_final_answer(response) == "B"


def test_simple_bench_completion_uses_openai_chat_messages(monkeypatch):
    captured: dict = {}

    class FakeResponse(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            b'data: {"choices":[{"delta":{"reasoning_content":"Brief thought. "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"Final "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"Answer: B"}}],'
            b'"usage":{"completion_tokens":3},'
            b'"timings":{"predicted_per_second":42.5}}\n\n'
            b"data: [DONE]\n\n"
        )

    monkeypatch.setattr("gguf_limit_bench.simple_bench_runner.urlopen", fake_urlopen)

    result = measure_simple_bench_completion(
        base_url="http://127.0.0.1:8080",
        question=SimpleBenchQuestion(question_id=1, prompt="Pick one", answer="B"),
        system_prompt="Use Final Answer: X",
        max_tokens=32,
        timeout_seconds=10,
    )

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "Use Final Answer: X"},
        {"role": "user", "content": "Pick one"},
    ]
    assert result.response == "Brief thought. Final Answer: B"
    assert result.generated_tokens == 3
    assert result.tokens_per_second == 42.5


def test_simple_bench_score_is_accuracy_first_speed_second():
    results = [
        SimpleBenchQuestionResult(
            question_id=1,
            expected_answer="A",
            predicted_answer="A",
            correct=True,
            ttft_ms=100.0,
            tokens_per_second=40.0,
            generated_tokens=32,
            output_chars=200,
            prompt_chars=1000,
            response="Final Answer: A",
        ),
        SimpleBenchQuestionResult(
            question_id=2,
            expected_answer="B",
            predicted_answer="C",
            correct=False,
            ttft_ms=120.0,
            tokens_per_second=60.0,
            generated_tokens=32,
            output_chars=200,
            prompt_chars=1000,
            response="Final Answer: C",
        ),
    ]

    batch = combine_simple_bench_results(results)

    assert batch.ok is True
    assert batch.accuracy == 0.5
    assert batch.median_tps == 50.0
    assert batch.score == 550.0


def test_core_flag_ladder_builds_ordered_profiles_and_extra_args():
    ladder = build_core_flag_ladder(
        context_size=8192,
        parallel_max=8,
        extra_server_args=("--dry-run",),
    )

    assert [settings.profile_name for settings in ladder[:7]] == [
        "L0-baseline",
        "L1-parallel",
        "L2-kv-unified",
        "L3-ram-cache",
        "L4-cache-reuse",
        "L5-checkpoints",
        "L6-q8-kv",
    ]
    assert ladder[0].context_size == 8192
    assert ladder[1].parallel == 6
    assert ladder[-1].threads == 32
    assert ladder[0].extra_server_args == ("--dry-run",)


def test_extra_server_args_cannot_override_managed_bindings():
    with pytest.raises(ValueError, match="--host is managed by Agent Pilot Autobench"):
        validate_extra_server_args(("--host=0.0.0.0",))


def test_core_flag_ladder_adds_native_mtp_draft_profiles_only_when_detected():
    plain = build_core_flag_ladder(enable_mtp=False)
    mtp = build_core_flag_ladder(enable_mtp=True)

    assert not any(settings.profile_name.startswith("MTP-") for settings in plain)
    assert [settings.profile_name for settings in mtp[-3:]] == [
        "MTP-draft-8",
        "MTP-draft-16",
        "MTP-draft-32",
    ]
    assert mtp[-2].draft_max == 16
    assert mtp[-2].draft_p_min == 0.75


def test_flag_ladder_plan_contains_llama_server_commands():
    plan = build_flag_ladder_plan(
        llama_server=Path("llama-server.exe"),
        model=Path("model.gguf"),
        host="127.0.0.1",
        port=6939,
        context_size=4096,
        parallel_max=4,
        extra_server_args=("--dry-run",),
    )

    assert plan[0]["name"] == "L0-baseline"
    assert plan[0]["command"][:3] == ["llama-server.exe", "--model", "model.gguf"]
    assert "--dry-run" in plan[0]["command"]
    assert "--cache-type-k" in plan[6]["command"]
    assert "--cache-idle-slots" not in [part for row in plan for part in row["command"]]


def test_flag_ladder_plan_adds_mtp_commands_when_heads_are_detected():
    plan = build_flag_ladder_plan(
        llama_server=Path("llama-server.exe"),
        model=Path("Qwen-MTP.gguf"),
        host="127.0.0.1",
        port=6939,
        context_size=4096,
        parallel_max=4,
        enable_mtp=True,
    )

    mtp_rows = [row for row in plan if row["name"].startswith("MTP-")]
    assert [row["name"] for row in mtp_rows] == [
        "MTP-draft-8",
        "MTP-draft-16",
        "MTP-draft-32",
    ]
    assert [row["command"][row["command"].index("--draft-max") + 1] for row in mtp_rows] == [
        "8",
        "16",
        "32",
    ]


def test_autoresearch_settings_can_hold_flag_specific_fields():
    settings = AutoresearchSettings(
        profile_name="custom",
        cache_type_k="q8_0",
        threads=16,
        extra_server_args=("--dry-run",),
    )

    payload = settings.to_dict()

    assert payload["profile_name"] == "custom"
    assert payload["cache_type_k"] == "q8_0"
    assert payload["threads"] == 16


def test_short_logs_keep_warning_lines_and_bounded_tail(tmp_path):
    (tmp_path / "server.stdout.log").write_text("server ready\nnormal info\n", encoding="utf-8")
    (tmp_path / "server.stderr.log").write_text(
        "warning: cache fallback\nerror: draft disabled\n",
        encoding="utf-8",
    )

    warning_count = _write_short_logs(
        attempt_dir=tmp_path,
        settings=AutoresearchSettings(profile_name="MTP-draft-16", draft_max=16),
        returncode=0,
    )

    assert warning_count == 2
    assert "cache fallback" in (tmp_path / "warnings.log").read_text(encoding="utf-8")
    tail = (tmp_path / "server-tail.log").read_text(encoding="utf-8")
    assert "profile=MTP-draft-16" in tail
    assert "warning_count=2" in tail


def test_remaining_attempt_timeout_is_bounded_and_fails_when_exhausted(monkeypatch):
    monkeypatch.setattr(simple_bench_runner.time, "monotonic", lambda: 100.0)

    assert simple_bench_runner._remaining_timeout_seconds(105.2) == 6
    with pytest.raises(TimeoutError, match="attempt budget exhausted"):
        simple_bench_runner._remaining_timeout_seconds(100.0)


def test_candidate_sequence_writes_flag_slowdown_comparison(tmp_path):
    candidates = (
        AutoresearchSettings(profile_name="L0-baseline"),
        AutoresearchSettings(profile_name="L1-parallel", parallel=4),
    )

    def fake_runner(settings):
        tps = 100.0 if settings.profile_name == "L0-baseline" else 80.0
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=tps,
            prompt_tokens_per_second=0.0,
            ttft_ms=50.0,
            context_size=settings.context_size,
            failure="none",
            stdout="",
            stderr="",
            returncode=0,
            flag_profile=settings.profile_name,
            simple_bench_score=500.0 + tps,
            simple_bench_accuracy=0.5,
        )

    receipt = AutoresearchLoop(
        model=Path("model.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        candidate_sequence=candidates,
    ).run()

    payload = json.loads((receipt.path / "flag-ladder-results.json").read_text(encoding="utf-8"))
    assert payload["champion_profile"] == "L0-baseline"
    assert payload["rows"][1]["slowdown_vs_baseline_percent"] == 20.0
    assert "Slowdown vs L0" in (receipt.path / "flag-ladder-results.md").read_text(encoding="utf-8")
