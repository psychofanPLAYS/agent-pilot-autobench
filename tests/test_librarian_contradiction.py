"""Tests for the librarian *contradiction* (staleness/conflict) job module.

Contradiction asks the local model to classify a NEW statement against an
EXISTING memory: confirms (A), contradicts (B), or unrelated (C). These tests pin
determinism, the MULTIPLE_CHOICE scorer round-trip, pack shape, and gold sanity
(all three classes present, with at least four contradiction cases).
"""

from __future__ import annotations

from gguf_limit_bench.answer_scoring import score_answer
from gguf_limit_bench.librarian import contradiction
from gguf_limit_bench.librarian.contradiction import PACK_ID, build
from gguf_limit_bench.packs import AnswerType, QuestionPack

_LETTERS = "ABC"


def test_build_is_deterministic_across_two_calls():
    a = build(0)
    b = build(0)
    assert [(q.question_id, q.prompt, q.answer, q.choices, q.tags) for q in a.questions] == [
        (q.question_id, q.prompt, q.answer, q.choices, q.tags) for q in b.questions
    ]


def test_build_is_deterministic_for_nonzero_seed():
    a = build(7)
    b = build(7)
    assert [q.answer for q in a.questions] == [q.answer for q in b.questions]
    assert [q.prompt for q in a.questions] == [q.prompt for q in b.questions]


def test_pack_shape():
    pack = build(0)
    assert isinstance(pack, QuestionPack)
    assert pack.pack_id == PACK_ID
    assert pack.tier == "librarian"
    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert 10 <= len(pack.questions) <= 16
    ids = [q.question_id for q in pack.questions]
    assert len(ids) == len(set(ids))  # unique ids
    for q in pack.questions:
        assert q.question_id.startswith(f"{PACK_ID}-s0-")
        assert "librarian" in q.tags
        assert "contradiction" in q.tags
        assert q.answer_source == "librarian:contradiction"
        # MC: choices present and answer is a single letter within range.
        assert q.choices is not None and len(q.choices) == 3
        assert len(q.answer) == 1 and q.answer in _LETTERS
        assert ord(q.answer) - ord("A") < len(q.choices)


def test_scorer_round_trip_correct_scores_true():
    pack = build(0)
    for q in pack.questions:
        good = f"Comparing the new statement to the memory.\nFinal Answer: {q.answer}"
        assert score_answer(good, q.answer, AnswerType.MULTIPLE_CHOICE) is True


def test_scorer_round_trip_wrong_scores_false():
    pack = build(0)
    for q in pack.questions:
        # Pick any letter that is not the gold answer (still within range).
        wrong_letter = next(letter for letter in _LETTERS if letter != q.answer)
        wrong = f"Final Answer: {wrong_letter}"
        assert score_answer(wrong, q.answer, AnswerType.MULTIPLE_CHOICE) is False


def test_gold_sanity_all_three_classes_present():
    pack = build(0)
    # Letters are randomized; assert class coverage via the subtype tags.
    subtypes = {
        t for q in pack.questions for t in q.tags if t in {"confirms", "contradicts", "unrelated"}
    }
    assert subtypes == {"confirms", "contradicts", "unrelated"}


def test_gold_sanity_at_least_four_contradicts():
    pack = build(0)
    contradicts = [q for q in pack.questions if "contradicts" in q.tags]
    assert len(contradicts) >= 4
    # The randomized gold letter must still point at the "contradicts" label.
    contradicts_label = contradiction._CHOICES[1]
    for q in contradicts:
        assert q.choices is not None
        assert q.choices[ord(q.answer) - ord("A")] == contradicts_label
