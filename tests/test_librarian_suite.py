import json

from gguf_limit_bench.librarian.preflight import LibrarianPreflightReceipt, PreflightGateReceipt
from gguf_limit_bench.librarian_suite import run_librarian_suite
from gguf_limit_bench.simple_bench import SimpleBenchBatchResult, SimpleBenchQuestionResult


def test_librarian_suite_writes_score_and_receipts(tmp_path, monkeypatch):
    def fake_preflight(**kwargs):
        return LibrarianPreflightReceipt(
            ok=True,
            failure_class="none",
            model=str(kwargs["model"]),
            family="gemma",
            quant="Q4_K_M",
            settings={},
            gates=(PreflightGateReceipt("identity", "pass"),),
        )

    def fake_run_pack_questions(*, pack, questions, base_url, timeout_seconds, sampling):
        assert base_url == "http://127.0.0.1:8080"
        assert timeout_seconds == 123
        assert sampling["temperature"] == 0.6
        results = [
            SimpleBenchQuestionResult(
                question_id=questions[0].question_id,
                expected_answer=questions[0].answer,
                predicted_answer=questions[0].answer,
                correct=True,
                ttft_ms=12.0,
                tokens_per_second=42.0,
                generated_tokens=8,
                output_chars=20,
                prompt_chars=100,
                response=f"Final Answer: {questions[0].answer}",
                prompt_tokens_per_second=1000.0,
                outcome="correct",
            )
        ]
        return SimpleBenchBatchResult(
            ok=True,
            score=1000.0,
            accuracy=1.0,
            correct=1,
            total=1,
            median_tps=42.0,
            min_tps=42.0,
            median_ttft_ms=12.0,
            results=results,
            median_prompt_tps=1000.0,
            incomplete=0,
            completion_rate=1.0,
        )

    monkeypatch.setattr("gguf_limit_bench.librarian_suite.run_librarian_preflight", fake_preflight)
    monkeypatch.setattr(
        "gguf_limit_bench.librarian_suite.run_pack_questions", fake_run_pack_questions
    )

    summary = run_librarian_suite(
        model="gemma-3-27b-it",
        base_url="http://127.0.0.1:8080",
        out_dir=tmp_path,
        pack_ids=("librarian-gate", "librarian-rerank"),
        sample_size=1,
        repeats=1,
        timeout_seconds=123,
        settings={"reasoning_mode": "direct", "temperature": 0.6},
    )

    assert summary["librarian_bench_score"] == 1.0
    assert summary["agent_bench_score"] == 1.0
    assert summary["asked"] == 2
    assert summary["packs"][0]["repeats"] == 1
    expected_letter = summary["packs"][0]["questions"][0]["expected"]
    assert summary["packs"][0]["letter_accuracy"][expected_letter]["accuracy"] == 1.0
    assert summary["packs"][0]["predicted_letter_counts"] == {expected_letter: 1}
    assert json.loads((tmp_path / "librarian-suite-summary.json").read_text())["settings"] == {
        "reasoning_mode": "direct",
        "temperature": 0.6,
    }
    assert (tmp_path / "librarian-suite.tsv").exists()
    assert (tmp_path / "librarian-suite.md").exists()
    assert (tmp_path / "librarian-gate.json").exists()


def test_librarian_suite_preflight_failure_writes_blocked_receipts(tmp_path, monkeypatch):
    def fake_preflight(**kwargs):
        return LibrarianPreflightReceipt(
            ok=False,
            failure_class="preflight_fail",
            model=str(kwargs["model"]),
            family="qwen",
            quant="Q4_K_M",
            settings={},
            gates=(PreflightGateReceipt("template_load", "fail", "missing --jinja"),),
        )

    def fail_if_scored(**_kwargs):
        raise AssertionError("run_pack_questions should not run after preflight failure")

    monkeypatch.setattr("gguf_limit_bench.librarian_suite.run_librarian_preflight", fake_preflight)
    monkeypatch.setattr("gguf_limit_bench.librarian_suite.run_pack_questions", fail_if_scored)

    summary = run_librarian_suite(
        model="qwen3.6-35b-a3b",
        base_url="http://127.0.0.1:8080",
        out_dir=tmp_path,
        pack_ids=("librarian-gate",),
        sample_size=1,
        repeats=1,
        timeout_seconds=123,
        settings={"template": "froggeric-v19"},
    )

    saved = json.loads((tmp_path / "librarian-suite-summary.json").read_text())
    tsv = (tmp_path / "librarian-suite.tsv").read_text()
    markdown = (tmp_path / "librarian-suite.md").read_text()

    assert summary["status"] == "preflight_fail"
    assert summary["failure_class"] == "preflight_fail"
    assert summary["asked"] == 0
    assert saved["packs"][0]["status"] == "preflight_fail"
    assert "preflight_fail" in tsv
    assert "preflight_fail" in markdown


def test_librarian_suite_repeats_and_uses_weighted_score(tmp_path, monkeypatch):
    def fake_preflight(**kwargs):
        return LibrarianPreflightReceipt(
            ok=True,
            failure_class="none",
            model=str(kwargs["model"]),
            family="qwen",
            quant="Q4_K_M",
            settings={},
            gates=(PreflightGateReceipt("identity", "pass"),),
        )

    calls = {"n": 0}

    def fake_run_pack_questions(*, pack, questions, base_url, timeout_seconds, sampling):
        calls["n"] += 1
        correct = calls["n"] in {1, 3}
        answer = questions[0].answer
        predicted = answer if correct else ("A" if answer != "A" else "B")
        result = SimpleBenchQuestionResult(
            question_id=questions[0].question_id,
            expected_answer=answer,
            predicted_answer=predicted,
            correct=correct,
            ttft_ms=12.0,
            tokens_per_second=42.0,
            generated_tokens=8,
            output_chars=20,
            prompt_chars=100,
            response=f"Final Answer: {predicted}",
            prompt_tokens_per_second=1000.0,
            outcome="correct" if correct else "wrong",
        )
        return SimpleBenchBatchResult(
            ok=True,
            score=1000.0 if correct else 0.0,
            accuracy=1.0 if correct else 0.0,
            correct=1 if correct else 0,
            total=1,
            median_tps=42.0,
            min_tps=42.0,
            median_ttft_ms=12.0,
            results=[result],
            median_prompt_tps=1000.0,
            incomplete=0,
            completion_rate=1.0,
        )

    monkeypatch.setattr("gguf_limit_bench.librarian_suite.run_librarian_preflight", fake_preflight)
    monkeypatch.setattr(
        "gguf_limit_bench.librarian_suite.run_pack_questions", fake_run_pack_questions
    )

    summary = run_librarian_suite(
        model="qwen3.6-35b-a3b",
        base_url="http://127.0.0.1:8080",
        out_dir=tmp_path,
        pack_ids=("librarian-gate",),
        sample_size=1,
        repeats=3,
        timeout_seconds=123,
        settings={"temperature": 0.6},
    )

    assert calls["n"] == 3
    assert summary["asked"] == 3
    assert summary["correct"] == 2
    assert summary["librarian_bench_score"] == 2 / 3
    assert summary["packs"][0]["repeats"] == 3
    assert sum(summary["packs"][0]["predicted_letter_counts"].values()) == 3
