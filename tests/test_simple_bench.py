import json
from io import BytesIO
from pathlib import Path

import pytest

from gguf_limit_bench import simple_bench_runner
from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchLoop, AutoresearchSettings
from gguf_limit_bench.flag_ladder import (
    build_core_flag_ladder,
    build_flag_ladder_plan,
    llama_server_args_for_settings,
    validate_extra_server_args,
)
from gguf_limit_bench.simple_bench import (
    SimpleBenchBatchResult,
    SimpleBenchQuestionResult,
    SimpleBenchQuestion,
    combine_simple_bench_results,
    extract_final_answer,
    load_simple_bench_questions,
    load_simple_bench_system_prompt,
)
from gguf_limit_bench.simple_bench_runner import (
    _write_launch_receipt,
    _write_short_logs,
    measure_simple_bench_completion,
)


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


def test_extract_final_answer_handles_varied_real_world_formats():
    assert extract_final_answer("...lots of reasoning...\nThe answer is B.") == "B"
    assert extract_final_answer("Therefore the answer is (D).") == "D"
    assert extract_final_answer("**Final Answer:** A") == "A"
    assert extract_final_answer("After the work, \\boxed{C}") == "C"
    assert extract_final_answer("Answer: E") == "E"
    assert extract_final_answer("The option F fits best") == "F"


def test_extract_final_answer_reads_letter_alone_on_last_line():
    assert extract_final_answer("long reasoning here\n\nC") == "C"
    assert extract_final_answer("long reasoning here\n\n**E**") == "E"
    assert extract_final_answer("long reasoning here\n\n(D).") == "D"


def test_extract_final_answer_does_not_match_letters_inside_words():
    assert extract_final_answer("Apples and bananas are Fine choices overall") is None
    assert extract_final_answer("Reasoning mentions C but the model stops mid sentence") is None


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
    assert 500.0 < batch.score < 1000.0


def test_one_more_correct_answer_always_beats_unbounded_speed():
    def batch(*, correct: bool, tps: float):
        return combine_simple_bench_results(
            [
                SimpleBenchQuestionResult(
                    question_id=1,
                    expected_answer="A",
                    predicted_answer="A" if correct else "B",
                    correct=correct,
                    ttft_ms=10.0,
                    tokens_per_second=tps,
                    generated_tokens=1,
                    output_chars=1,
                    prompt_chars=1,
                    response="Final Answer: A" if correct else "Final Answer: B",
                )
            ]
        )

    assert batch(correct=True, tps=1.0).score > batch(correct=False, tps=1_000_000.0).score


def test_explicit_missing_system_prompt_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="system prompt"):
        load_simple_bench_system_prompt(tmp_path / "missing.txt")


def test_core_flag_ladder_builds_ordered_profiles_and_extra_args():
    ladder = build_core_flag_ladder(
        context_size=8192,
        parallel_max=8,
        extra_server_args=("--dry-run",),
    )

    names = [settings.profile_name for settings in ladder]
    # Speed flags first, all single stream; the stripped rung leads so we can
    # measure whether adding the standard flags helps or hurts ("fewer = faster?").
    assert names[:7] == [
        "Lmin-stripped",
        "L0-baseline",
        "L2-kv-unified",
        "L3-ram-cache",
        "L4-cache-reuse",
        "L5-checkpoints",
        "L6-q8-kv",
    ]
    assert ladder[0].profile_name == "Lmin-stripped"
    assert ladder[0].flash_attention is False
    assert ladder[0].cont_batching is False
    assert ladder[0].context_size == 8192
    assert ladder[0].extra_server_args == ("--dry-run",)
    # Every speed/thread profile runs single stream (parallel=1).
    speed_and_threads = [s for s in ladder if not s.profile_name.startswith("Lpar-")]
    assert all(s.parallel == 1 for s in speed_and_threads)
    assert any(s.profile_name == "T32-threads" and s.threads == 32 for s in ladder)
    # Parallel capability is tested LAST, matching a 1-heavy-plus-2-light server.
    assert names[-2:] == ["Lpar-2", "Lpar-3"]
    assert ladder[-2].parallel == 2
    assert ladder[-1].parallel == 3
    # q8 KV profiles carry the q8 cache types.
    q8 = [s for s in ladder if s.profile_name == "L6-q8-kv" or s.profile_name.startswith("T")]
    assert all(s.cache_type_k == "q8_0" and s.cache_type_v == "q8_0" for s in q8)


def test_extra_server_args_cannot_override_managed_bindings():
    with pytest.raises(ValueError, match="--host is managed by Agent Pilot Autobench"):
        validate_extra_server_args(("--host=0.0.0.0",))


def test_core_flag_ladder_adds_native_mtp_draft_profiles_only_when_detected():
    plain = build_core_flag_ladder(enable_mtp=False)
    mtp = build_core_flag_ladder(enable_mtp=True)

    assert not any(settings.profile_name.startswith("MTP-") for settings in plain)
    assert [
        settings.profile_name for settings in mtp if settings.profile_name.startswith("MTP-")
    ] == [
        "MTP-draft-3",
    ]
    mtp_profile = next(s for s in mtp if s.profile_name == "MTP-draft-3")
    assert mtp_profile.spec_type == "draft-mtp"
    assert mtp_profile.spec_draft_n_max == 3
    assert mtp_profile.spec_draft_n_max <= 4


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

    assert plan[0]["name"] == "Lmin-stripped"
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
    assert [row["name"] for row in mtp_rows] == ["MTP-draft-3"]
    command = mtp_rows[0]["command"]
    assert command[-4:] == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    assert "--draft-max" not in command
    assert "--draft-min" not in command


def test_server_args_reject_mtp_draft_max_above_four():
    settings = AutoresearchSettings(
        spec_type="draft-mtp",
        spec_draft_n_max=5,
    )

    with pytest.raises(ValueError, match="between 1 and 4"):
        llama_server_args_for_settings(settings)


def test_deprecated_draft_settings_translate_to_native_spec_flags():
    with pytest.warns(DeprecationWarning, match="draft_max"):
        settings = AutoresearchSettings(draft_max=3, draft_min=1, draft_p_min=0.75)

    command = llama_server_args_for_settings(settings)

    assert command[-8:] == [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "3",
        "--spec-draft-n-min",
        "1",
        "--spec-draft-p-min",
        "0.75",
    ]
    assert "--draft-max" not in command


@pytest.mark.parametrize(
    "settings",
    [
        AutoresearchSettings(spec_type="draft-mtp", spec_draft_n_max=0),
        AutoresearchSettings(spec_type="draft-mtp", spec_draft_n_max=5),
        AutoresearchSettings(spec_type="draft-mtp", spec_draft_n_max=3, spec_draft_n_min=-1),
        AutoresearchSettings(spec_type="draft-mtp", spec_draft_n_max=3, spec_draft_n_min=4),
        AutoresearchSettings(spec_type="draft-mtp", spec_draft_p_min=-0.1),
        AutoresearchSettings(spec_type="draft-mtp", spec_draft_p_min=1.1),
    ],
)
def test_server_args_reject_incoherent_native_mtp_settings(settings):
    with pytest.raises(ValueError):
        llama_server_args_for_settings(settings)


def test_david_mtp_cap_does_not_limit_other_speculation_types():
    command = llama_server_args_for_settings(
        AutoresearchSettings(spec_type="draft-simple", spec_draft_n_max=8)
    )

    assert command[-4:] == ["--spec-type", "draft-simple", "--spec-draft-n-max", "8"]


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
        settings=AutoresearchSettings(
            profile_name="MTP-draft-3", spec_type="draft-mtp", spec_draft_n_max=3
        ),
        returncode=0,
    )

    assert warning_count == 2
    assert "cache fallback" in (tmp_path / "warnings.log").read_text(encoding="utf-8")
    tail = (tmp_path / "server-tail.log").read_text(encoding="utf-8")
    assert "profile=MTP-draft-3" in tail
    assert "warning_count=2" in tail


def test_remaining_attempt_timeout_is_bounded_and_fails_when_exhausted(monkeypatch):
    monkeypatch.setattr(simple_bench_runner.time, "monotonic", lambda: 100.0)

    assert simple_bench_runner._remaining_timeout_seconds(105.2) == 6
    with pytest.raises(TimeoutError, match="attempt budget exhausted"):
        simple_bench_runner._remaining_timeout_seconds(100.0)


def test_launch_receipt_stores_exact_argv_without_executable_cmd(tmp_path):
    command = ["llama-server.exe", "--model", "G:\\models\\A&B 50%.gguf", "--flag=^value"]

    _write_launch_receipt(tmp_path, command)

    assert json.loads((tmp_path / "launch-command.json").read_text(encoding="utf-8")) == command
    assert not (tmp_path / "launch.cmd").exists()


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


def test_partial_candidate_sequence_is_labeled_and_has_no_champion(tmp_path):
    candidates = (
        AutoresearchSettings(profile_name="L0-baseline"),
        AutoresearchSettings(profile_name="L1-parallel", parallel=4),
    )

    class BoundedRunner:
        def __init__(self):
            self.timeouts: list[int] = []

        def set_timeout_seconds(self, timeout_seconds: int) -> None:
            self.timeouts.append(timeout_seconds)

        def __call__(self, settings):
            return AttemptResult(
                ok=True,
                generation_tokens_per_second=100.0,
                prompt_tokens_per_second=0.0,
                ttft_ms=50.0,
                context_size=settings.context_size,
                failure="none",
                stdout="",
                stderr="",
                returncode=0,
                flag_profile=settings.profile_name,
                simple_bench_score=500.0,
                simple_bench_accuracy=0.5,
            )

    runner = BoundedRunner()
    receipt = AutoresearchLoop(
        model=Path("model.gguf"),
        runs_root=tmp_path,
        attempt_runner=runner,
        budget_seconds=7,
        max_attempts=1,
        candidate_sequence=candidates,
    ).run()

    payload = json.loads((receipt.path / "flag-ladder-results.json").read_text(encoding="utf-8"))
    assert runner.timeouts and 1 <= runner.timeouts[0] <= 7
    assert payload["status"] == "partial"
    assert payload["planned_profiles"] == 2
    assert payload["completed_profiles"] == 1
    assert payload["champion_profile"] is None
    assert payload["provisional_best_profile"] == "L0-baseline"
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))
    assert best["status"] == "partial"
    assert best["promotion_eligible"] is False


def _question_result(tps, ttft, prompt_tps):
    from gguf_limit_bench.simple_bench import SimpleBenchQuestionResult

    return SimpleBenchQuestionResult(
        question_id=1,
        expected_answer="A",
        predicted_answer="A",
        correct=True,
        ttft_ms=ttft,
        tokens_per_second=tps,
        generated_tokens=10,
        output_chars=20,
        prompt_chars=50,
        response="ok",
        prompt_tokens_per_second=prompt_tps,
    )


def test_combine_reports_prefill_variance_and_tail_latency():
    from gguf_limit_bench.simple_bench import combine_simple_bench_results

    batch = combine_simple_bench_results(
        [
            _question_result(100.0, 50.0, 500.0),
            _question_result(120.0, 80.0, 600.0),
            _question_result(140.0, 200.0, 700.0),
        ]
    )
    assert batch.median_prompt_tps == 600.0
    assert batch.gen_tps_stddev > 0
    assert batch.ttft_p90_ms is not None and batch.ttft_p99_ms is not None
    assert batch.ttft_p99_ms >= batch.ttft_p90_ms >= (batch.median_ttft_ms or 0)


def test_measure_captures_server_prompt_per_second(monkeypatch):
    from io import BytesIO

    class FakeResponse(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_urlopen(request, timeout):
        return FakeResponse(
            b'data: {"choices":[{"delta":{"content":"Final Answer: A"}}],'
            b'"timings":{"predicted_per_second":42.5,"prompt_per_second":350.0}}\n\n'
            b"data: [DONE]\n\n"
        )

    monkeypatch.setattr("gguf_limit_bench.simple_bench_runner.urlopen", fake_urlopen)

    result = measure_simple_bench_completion(
        base_url="http://127.0.0.1:8080",
        question=SimpleBenchQuestion(question_id=1, prompt="Pick one", answer="A"),
        system_prompt="Use Final Answer: X",
        max_tokens=32,
        timeout_seconds=10,
    )
    assert result.prompt_tokens_per_second == 350.0


# --- Task 4: outcome taxonomy fields ---


def test_question_result_outcome_field_default_and_custom():
    result = SimpleBenchQuestionResult(
        question_id=1,
        expected_answer="A",
        predicted_answer=None,
        correct=False,
        ttft_ms=None,
        tokens_per_second=0.0,
        generated_tokens=0,
        output_chars=0,
        prompt_chars=0,
        response="",
        outcome="incomplete",
    )
    assert result.outcome == "incomplete"
    d = result.to_dict()
    assert "outcome" in d
    assert d["outcome"] == "incomplete"


def test_question_result_outcome_defaults_to_wrong():
    result = SimpleBenchQuestionResult(
        question_id=2,
        expected_answer="B",
        predicted_answer="B",
        correct=True,
        ttft_ms=50.0,
        tokens_per_second=30.0,
        generated_tokens=10,
        output_chars=20,
        prompt_chars=100,
        response="Final Answer: B",
    )
    assert result.outcome == "wrong"
    assert result.to_dict()["outcome"] == "wrong"


def test_batch_result_incomplete_and_completion_rate_fields():
    batch = SimpleBenchBatchResult(
        ok=True,
        score=800.0,
        accuracy=0.8,
        correct=4,
        total=5,
        median_tps=40.0,
        min_tps=30.0,
        median_ttft_ms=100.0,
        results=[],
        incomplete=1,
        completion_rate=0.8,
    )
    assert batch.incomplete == 1
    assert batch.completion_rate == 0.8
    d = batch.to_dict()
    assert "incomplete" in d
    assert d["incomplete"] == 1
    assert "completion_rate" in d
    assert d["completion_rate"] == 0.8


def test_batch_result_new_fields_default_to_zero():
    batch = SimpleBenchBatchResult(
        ok=True,
        score=500.0,
        accuracy=0.5,
        correct=1,
        total=2,
        median_tps=50.0,
        min_tps=40.0,
        median_ttft_ms=None,
        results=[],
    )
    assert batch.incomplete == 0
    assert batch.completion_rate == 0.0
