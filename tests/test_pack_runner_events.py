"""Phase 2: pack_runner emits live question_started / question_progress /
question_scored events through the context-local sink."""

from __future__ import annotations

from gguf_limit_bench import events, pack_runner
from gguf_limit_bench.packs import AnswerType, PackQuestion, QuestionPack


def _pack():
    return QuestionPack(
        pack_id="demo-pack",
        title="Demo",
        tier="gate",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        system_prompt="You are precise.",
        questions=(),
    )


def _question():
    return PackQuestion(
        question_id="q1",
        prompt="Which letter?",
        answer="B",
        answer_source="test",
    )


def test_run_pack_questions_emits_started_and_scored(monkeypatch):
    monkeypatch.setattr(
        pack_runner,
        "_chat",
        lambda **kwargs: ("Final Answer: B", 12.0, 30.0, 100.0, 5),
    )
    seen: list = []
    with events.set_event_sink(lambda t, d: seen.append((t, d))):
        pack_runner.run_pack_questions(
            pack=_pack(), questions=[_question()], base_url="http://x"
        )

    started = [d for t, d in seen if t == "question_started"]
    scored = [d for t, d in seen if t == "question_scored"]
    assert len(started) == 1 and len(scored) == 1
    assert started[0] == {
        "q_id": "q1",
        "index": 1,
        "total": 1,
        "pack": "demo-pack",
        "prompt": "Which letter?",
    }
    assert scored[0]["q_id"] == "q1"
    assert scored[0]["expected"] == "B"
    assert scored[0]["predicted"] == "B"
    assert scored[0]["correct"] is True
    assert scored[0]["outcome"] == "correct"
    assert scored[0]["score"] == 1.0
    assert scored[0]["index"] == 1 and scored[0]["total"] == 1


def test_chat_emits_throttled_question_progress(monkeypatch):
    stream = [
        {"choices": [{"delta": {"reasoning_content": "Let me think"}}]},
        {"choices": [{"delta": {"content": "Final"}}]},
        {"choices": [{"delta": {"content": " Answer: B"}}]},
    ]

    class _DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(pack_runner, "urlopen", lambda *a, **k: _DummyResp())
    monkeypatch.setattr(
        pack_runner, "iter_llama_completion_stream_events", lambda resp: iter(stream)
    )
    monkeypatch.setattr(pack_runner, "_PROGRESS_THROTTLE_SECONDS", 0.0)

    seen: list = []
    with events.set_event_sink(lambda t, d: seen.append((t, d))):
        text, *_ = pack_runner._chat(
            base_url="http://x",
            system_prompt="s",
            user_content="u",
            max_tokens=0,
            timeout_seconds=5,
            q_id="q1",
        )

    assert text == "Let me thinkFinal Answer: B"  # combined response preserved
    progress = [d for t, d in seen if t == "question_progress"]
    assert progress, "expected at least one question_progress event"
    last = progress[-1]
    assert last["q_id"] == "q1"
    assert last["thinking"] == "Let me think"
    assert last["answer"] == "Final Answer: B"


def test_chat_without_q_id_emits_no_progress(monkeypatch):
    stream = [{"choices": [{"delta": {"content": "hi"}}]}]

    class _DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(pack_runner, "urlopen", lambda *a, **k: _DummyResp())
    monkeypatch.setattr(
        pack_runner, "iter_llama_completion_stream_events", lambda resp: iter(stream)
    )
    seen: list = []
    with events.set_event_sink(lambda t, d: seen.append((t, d))):
        pack_runner._chat(
            base_url="http://x",
            system_prompt="s",
            user_content="u",
            max_tokens=0,
            timeout_seconds=5,
        )
    assert [t for t, _ in seen] == []


def _exact_pack():
    return QuestionPack(
        pack_id="exact-demo",
        title="Exact Demo",
        tier="gate",
        answer_type=AnswerType.EXACT,
        system_prompt="Be precise.",
        questions=(),
    )


def _exact_question():
    return PackQuestion(
        question_id="q1",
        prompt="What slug?",
        answer="ops-runbook-location",
        answer_source="test",
    )


def test_exact_plain_correct_answer_without_marker_scores_correct(monkeypatch):
    # Regression (#2): a correct EXACT answer lacking the "Final Answer:" marker must
    # score correct, not "incomplete". The model answers plainly with a leading newline.
    monkeypatch.setattr(
        pack_runner, "_chat", lambda **kw: ("\nops-runbook-location", 10.0, 30.0, 100.0, 5)
    )
    seen: list = []
    with events.set_event_sink(lambda t, d: seen.append((t, d))):
        pack_runner.run_pack_questions(
            pack=_exact_pack(), questions=[_exact_question()], base_url="http://x"
        )
    scored = [d for t, d in seen if t == "question_scored"][0]
    assert scored["outcome"] == "correct"
    assert scored["correct"] is True
    assert scored["predicted"] == "ops-runbook-location"


def test_exact_wrong_plain_answer_is_not_falsely_correct(monkeypatch):
    # The lenient path must not turn a genuinely wrong answer into a pass.
    monkeypatch.setattr(
        pack_runner, "_chat", lambda **kw: ("\nsomething-else-entirely", 10.0, 30.0, 100.0, 5)
    )
    seen: list = []
    with events.set_event_sink(lambda t, d: seen.append((t, d))):
        pack_runner.run_pack_questions(
            pack=_exact_pack(), questions=[_exact_question()], base_url="http://x"
        )
    scored = [d for t, d in seen if t == "question_scored"][0]
    assert scored["correct"] is False
    assert scored["outcome"] == "incomplete"
