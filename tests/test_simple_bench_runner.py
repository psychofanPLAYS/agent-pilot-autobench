"""Tests for pack_runner with forced-final follow-up and outcome taxonomy.

Uses the same fake-client pattern as test_simple_bench.py — no real llama-server
is started.  Each test patches urlopen so we control exactly what the server
"returns".
"""

from __future__ import annotations

from io import BytesIO

from gguf_limit_bench.packs import AnswerType, PackQuestion, QuestionPack
from gguf_limit_bench.pack_runner import run_pack_questions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pack(
    *,
    question_id: str = "q1",
    prompt: str = "Which option?",
    answer: str = "B",
    answer_type: AnswerType = AnswerType.MULTIPLE_CHOICE,
    accept: tuple[str, ...] = (),
    choices: tuple[str, ...] | None = ("Alpha", "Beta", "Gamma", "Delta"),
) -> QuestionPack:
    return QuestionPack(
        pack_id="test-pack",
        title="Test Pack",
        tier="easy",
        answer_type=answer_type,
        system_prompt="You are a helpful assistant. End with Final Answer: X.",
        questions=(
            PackQuestion(
                question_id=question_id,
                prompt=prompt,
                answer=answer,
                answer_source="test",
                choices=choices,
                tags=(),
                accept=accept,
            ),
        ),
    )


class FakeResponse(BytesIO):
    """Minimal context manager wrapping BytesIO to stand in for urlopen response."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _sse(content: str) -> bytes:
    """Wrap content in a minimal SSE frame that the stream parser can consume."""
    import json

    event = json.dumps(
        {
            "choices": [{"delta": {"content": content}}],
            "usage": {"completion_tokens": len(content.split())},
            "timings": {"predicted_per_second": 30.0, "prompt_per_second": 200.0},
        }
    )
    return (f"data: {event}\n\ndata: [DONE]\n\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Test 1: MC question, model returns "Final Answer: B", expected B → correct
# ---------------------------------------------------------------------------


def test_pack_runner_mc_direct_correct(monkeypatch):
    """Model immediately returns 'Final Answer: B'; expected B → outcome 'correct'."""

    def fake_urlopen(request, timeout):
        return FakeResponse(_sse("...reasoning... Final Answer: B"))

    monkeypatch.setattr("gguf_limit_bench.pack_runner.urlopen", fake_urlopen)

    pack = _make_pack(answer="B", answer_type=AnswerType.MULTIPLE_CHOICE)
    batch = run_pack_questions(
        pack=pack,
        questions=list(pack.questions),
        answer_max_tokens=256,
        base_url="http://127.0.0.1:8080",
        timeout_seconds=10,
    )

    assert batch.total == 1
    assert batch.correct == 1
    assert batch.results[0].outcome == "correct"
    assert batch.results[0].correct is True
    assert batch.accuracy == 1.0
    assert batch.incomplete == 0
    assert batch.completion_rate == 1.0


def test_pack_runner_sends_model_sampling_options(monkeypatch):
    """Sampling is explicit so Qwen-style plans are not forced into greedy decode."""
    seen_payloads = []

    def fake_urlopen(request, timeout):
        import json

        seen_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(_sse("Final Answer: B"))

    monkeypatch.setattr("gguf_limit_bench.pack_runner.urlopen", fake_urlopen)

    pack = _make_pack(answer="B", answer_type=AnswerType.MULTIPLE_CHOICE)
    run_pack_questions(
        pack=pack,
        questions=list(pack.questions),
        answer_max_tokens=256,
        base_url="http://127.0.0.1:8080",
        timeout_seconds=10,
        sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "ignored": "nope"},
    )

    assert seen_payloads[0]["temperature"] == 0.6
    assert seen_payloads[0]["top_p"] == 0.95
    assert seen_payloads[0]["top_k"] == 20
    assert "ignored" not in seen_payloads[0]


def test_pack_runner_default_sampling_is_not_greedy(monkeypatch):
    seen_payloads = []

    def fake_urlopen(request, timeout):
        import json

        seen_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(_sse("Final Answer: B"))

    monkeypatch.setattr("gguf_limit_bench.pack_runner.urlopen", fake_urlopen)

    pack = _make_pack(answer="B", answer_type=AnswerType.MULTIPLE_CHOICE)
    run_pack_questions(
        pack=pack,
        questions=list(pack.questions),
        answer_max_tokens=256,
        base_url="http://127.0.0.1:8080",
        timeout_seconds=10,
    )

    assert seen_payloads[0]["temperature"] == 1.0


# ---------------------------------------------------------------------------
# Test 2: No "Final Answer:" in first response; follow-up returns it → correct,
#         and exactly one follow-up call was made.
# ---------------------------------------------------------------------------


def test_pack_runner_mc_forced_final_followup_correct(monkeypatch):
    """First turn has no Final Answer marker; follow-up supplies it → correct,
    and exactly one follow-up call is made."""

    call_count = {"n": 0}
    # Deliberately crafted to contain NO extractable MC letter:
    # - no "Final Answer:" marker
    # - no single letter alone on its own line
    # - no "answer is X", "option X", or "boxed{X}" patterns
    long_reasoning = (
        "Let me think about this carefully. I need to evaluate the candidates. "
        "After extensive analysis, I believe the second candidate is best because it "
        "aligns with the theoretical framework described in the question. "
        "The remaining candidates have fundamental flaws that disqualify them from "
        "consideration. My reasoning involves multiple steps of logical deduction."
    )

    def fake_urlopen(request, timeout):
        call_count["n"] += 1
        import json

        payload = json.loads(request.data.decode("utf-8"))
        messages = payload["messages"]

        # Second call is the forced-final follow-up
        if call_count["n"] == 2:
            # Should be stateless: contains the prior assistant text + instruction
            last_user = messages[-1]["content"]
            assert "Final Answer" in last_user
            return FakeResponse(_sse("Final Answer: B"))

        # First call returns long reasoning with NO Final Answer marker
        return FakeResponse(_sse(long_reasoning))

    monkeypatch.setattr("gguf_limit_bench.pack_runner.urlopen", fake_urlopen)

    pack = _make_pack(answer="B", answer_type=AnswerType.MULTIPLE_CHOICE)
    batch = run_pack_questions(
        pack=pack,
        questions=list(pack.questions),
        answer_max_tokens=256,
        base_url="http://127.0.0.1:8080",
        timeout_seconds=10,
    )

    assert call_count["n"] == 2, "Exactly one follow-up call must be made"
    assert batch.results[0].outcome == "correct"
    assert batch.results[0].correct is True
    assert batch.incomplete == 0


# ---------------------------------------------------------------------------
# Test 3: Both turns return no answer → outcome "incomplete"
# ---------------------------------------------------------------------------


def test_pack_runner_mc_both_turns_no_answer_incomplete(monkeypatch):
    """First turn AND follow-up both return text with no extractable answer
    → outcome 'incomplete', batch.incomplete == 1, correct is False."""

    def fake_urlopen(request, timeout):
        # Both calls return generic text with no letter answer
        return FakeResponse(_sse("I am not able to determine the answer."))

    monkeypatch.setattr("gguf_limit_bench.pack_runner.urlopen", fake_urlopen)

    pack = _make_pack(answer="B", answer_type=AnswerType.MULTIPLE_CHOICE)
    batch = run_pack_questions(
        pack=pack,
        questions=list(pack.questions),
        answer_max_tokens=256,
        base_url="http://127.0.0.1:8080",
        timeout_seconds=10,
    )

    assert batch.results[0].outcome == "incomplete"
    assert batch.results[0].correct is False
    assert batch.incomplete == 1
    assert batch.completion_rate == 0.0


# ---------------------------------------------------------------------------
# Test 4: EXACT question, model returns "Final Answer: 3" → correct
# ---------------------------------------------------------------------------


def test_pack_runner_exact_direct_correct(monkeypatch):
    """EXACT answer type: model returns 'Final Answer: 3', expected '3' → correct."""

    def fake_urlopen(request, timeout):
        return FakeResponse(_sse("After calculating, Final Answer: 3"))

    monkeypatch.setattr("gguf_limit_bench.pack_runner.urlopen", fake_urlopen)

    pack = _make_pack(
        answer="3",
        answer_type=AnswerType.EXACT,
        choices=None,
        prompt="How many sides does a triangle have?",
    )
    batch = run_pack_questions(
        pack=pack,
        questions=list(pack.questions),
        answer_max_tokens=256,
        base_url="http://127.0.0.1:8080",
        timeout_seconds=10,
    )

    assert batch.results[0].outcome == "correct"
    assert batch.results[0].correct is True
    assert batch.accuracy == 1.0
    assert batch.incomplete == 0
