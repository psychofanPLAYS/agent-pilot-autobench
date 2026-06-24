"""Tests for the librarian *triage* (salience) job module.

Triage asks the local model to decide what is worth remembering: a per-snippet
keep/drop judgement, plus a count of durable facts planted in filler. These
tests pin determinism, the EXACT scorer round-trip, pack shape, and gold sanity.
"""

from __future__ import annotations

from gguf_limit_bench.answer_scoring import score_answer
from gguf_limit_bench.librarian.triage import PACK_ID, build
from gguf_limit_bench.packs import AnswerType, QuestionPack


def test_build_is_deterministic_across_two_calls():
    a = build(0)
    b = build(0)
    assert [(q.question_id, q.prompt, q.answer, q.tags, q.accept) for q in a.questions] == [
        (q.question_id, q.prompt, q.answer, q.tags, q.accept) for q in b.questions
    ]


def test_pack_shape():
    pack = build(0)
    assert isinstance(pack, QuestionPack)
    assert pack.pack_id == PACK_ID
    assert pack.answer_type is AnswerType.EXACT
    assert 10 <= len(pack.questions) <= 16
    ids = [q.question_id for q in pack.questions]
    assert len(ids) == len(set(ids))  # unique ids
    for q in pack.questions:
        assert q.question_id.startswith(f"{PACK_ID}-s0-")
        assert "librarian" in q.tags
        assert "triage" in q.tags
        assert q.choices is None
        assert q.answer_source == "librarian:triage"


def test_scorer_round_trip_correct_scores_true():
    pack = build(0)
    for q in pack.questions:
        good = f"Reasoning about the snippet.\nFinal Answer: {q.answer}"
        assert score_answer(good, q.answer, AnswerType.EXACT, q.accept) is True


def test_scorer_round_trip_wrong_scores_false():
    pack = build(0)
    for q in pack.questions:
        # Pick a clearly wrong, non-colliding value for this question.
        if q.answer in {"keep", "drop"}:
            wrong_value = "drop" if q.answer == "keep" else "keep"
        else:
            # numeric count: use a number outside the 0..5 band so it cannot
            # collide with this question's gold or its number-word accept form.
            wrong_value = "99"
        wrong = f"Final Answer: {wrong_value}"
        assert score_answer(wrong, q.answer, AnswerType.EXACT, q.accept) is False


def test_keep_drop_golds_are_keep_or_drop():
    pack = build(0)
    kd = [q for q in pack.questions if "keep_drop" in q.tags]
    assert len(kd) >= 12
    for q in kd:
        assert q.answer in {"keep", "drop"}
    keeps = [q for q in kd if q.answer == "keep"]
    drops = [q for q in kd if q.answer == "drop"]
    assert len(keeps) >= 6
    assert len(drops) >= 6


def test_count_golds_match_planted_durable_facts():
    pack = build(0)
    counts = [q for q in pack.questions if "count" in q.tags]
    assert counts, "expected at least one count question"
    for q in counts:
        value = int(q.answer)
        assert 0 <= value <= 5
        assert q.answer == str(value)
        # The tag records how many durable facts were planted; gold must match.
        fact_tag = next(t for t in q.tags if t.startswith("facts="))
        planted = int(fact_tag.split("=", 1)[1])
        assert value == planted


def test_build_is_deterministic_for_nonzero_seed():
    a = build(7)
    b = build(7)
    assert [q.answer for q in a.questions] == [q.answer for q in b.questions]
    assert [q.prompt for q in a.questions] == [q.prompt for q in b.questions]
