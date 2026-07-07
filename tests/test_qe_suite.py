import json

from gguf_limit_bench.qe_suite import QeCase, run_qe_format_suite


def test_qe_suite_runs_each_case_ten_times_in_fresh_sessions(tmp_path):
    calls: list[dict] = []

    def fake_chat(*, base_url, system_prompt, user_content, max_tokens, timeout_seconds, sampling):
        calls.append(
            {
                "base_url": base_url,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
                "sampling": sampling,
            }
        )
        return (
            "LEX: qwen template, reasoning_content\n"
            "HYDE: A note about Qwen reasoning extraction after template changes.",
            11.0,
            123.0,
            456.0,
            17,
        )

    summary = run_qe_format_suite(
        model="qwen3.5-qe-2b",
        base_url="http://127.0.0.1:8080",
        out_dir=tmp_path,
        cases=(QeCase("template", "Why did Qwen reasoning disappear?"),),
        timeout_seconds=55,
        chat=fake_chat,
    )

    assert len(calls) == 10
    assert len({call["user_content"] for call in calls}) == 1
    assert all("previous answer" not in call["user_content"].lower() for call in calls)
    assert all(call["base_url"] == "http://127.0.0.1:8080" for call in calls)
    assert all(call["timeout_seconds"] == 55 for call in calls)
    assert summary["attempts"] == 10
    assert summary["format_rate"] == 1.0
    assert summary["direct_answer_rate"] == 0.0
    assert summary["score"] == 1.0


def test_qe_suite_writes_receipts_with_issue_counts(tmp_path):
    responses = iter(
        [
            (
                "LEX: qwen\nHYDE: A relevant note.",
                10.0,
                50.0,
                100.0,
                8,
            ),
            (
                "LEX: qwen\nANSWER: Use the old template.",
                12.0,
                40.0,
                90.0,
                7,
            ),
        ]
    )

    def fake_chat(**_kwargs):
        return next(responses)

    summary = run_qe_format_suite(
        model="qwen3.5-qe-2b",
        base_url="http://127.0.0.1:8080",
        out_dir=tmp_path,
        cases=(QeCase("template", "Why did Qwen reasoning disappear?"),),
        repeats=2,
        chat=fake_chat,
    )

    saved = json.loads((tmp_path / "qe-format-summary.json").read_text(encoding="utf-8"))
    attempts = json.loads((tmp_path / "qe-format-attempts.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "qe-format-summary.md").read_text(encoding="utf-8")

    assert summary["attempts"] == 2
    assert summary["valid"] == 1
    assert summary["direct_answer_count"] == 1
    assert summary["issue_counts"]["missing_hyde"] == 1
    assert summary["issue_counts"]["direct_answer"] == 1
    assert saved["score_contract"] == "qe_format_score = mean deterministic LEX/HYDE format score"
    assert attempts[1]["issues"] == ["missing_hyde", "direct_answer"]
    assert "Direct-answer rate: `0.500000`" in markdown


def test_qe_suite_records_sampling_and_resource_snapshots(tmp_path):
    class FakeTelemetry:
        def __init__(self, used):
            self.used = used

        def to_dict(self):
            return {
                "gpu_used_mb": self.used,
                "gpu_total_mb": 24564,
                "gpu_util_percent": 12,
                "gpu_power_watts": 55.0,
            }

    samples = iter([FakeTelemetry(3000), FakeTelemetry(3400)])
    calls = []

    def fake_chat(*, sampling, **_kwargs):
        calls.append(sampling)
        return (
            "LEX: qwen\nHYDE: A relevant note.",
            10.0,
            50.0,
            100.0,
            8,
        )

    summary = run_qe_format_suite(
        model="qwen3.5-qe-2b",
        base_url="http://127.0.0.1:8080",
        out_dir=tmp_path,
        cases=(QeCase("template", "Why did Qwen reasoning disappear?"),),
        repeats=1,
        sampling={"temperature": 0.1, "top_p": 0.8, "dry_multiplier": 0.6},
        chat=fake_chat,
        telemetry_sampler=lambda: next(samples),
    )

    assert calls == [{"temperature": 0.1, "top_p": 0.8, "dry_multiplier": 0.6}]
    assert summary["sampling"] == {"temperature": 0.1, "top_p": 0.8, "dry_multiplier": 0.6}
    assert summary["resources"]["start"]["gpu_used_mb"] == 3000
    assert summary["resources"]["end"]["gpu_used_mb"] == 3400
    assert summary["resources"]["delta"]["gpu_used_mb"] == 400


def test_qe_suite_caps_answer_tokens_and_records_cap(tmp_path):
    calls = []

    def fake_chat(*, max_tokens, **_kwargs):
        calls.append(max_tokens)
        return (
            "LEX: qwen\nHYDE: A relevant note.",
            10.0,
            50.0,
            100.0,
            8,
        )

    summary = run_qe_format_suite(
        model="qwen3.5-qe-2b",
        base_url="http://127.0.0.1:8080",
        out_dir=tmp_path,
        cases=(QeCase("template", "Why did Qwen reasoning disappear?"),),
        repeats=2,
        answer_max_tokens=96,
        chat=fake_chat,
    )

    saved = json.loads((tmp_path / "qe-format-summary.json").read_text(encoding="utf-8"))
    assert calls == [96, 96]
    assert summary["answer_max_tokens"] == 96
    assert saved["answer_max_tokens"] == 96
