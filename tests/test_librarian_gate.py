"""Tests for the librarian inject/skip gate job module."""

from __future__ import annotations

from dataclasses import replace

from gguf_limit_bench.answer_scoring import score_answer
from gguf_limit_bench.librarian import gate
from gguf_limit_bench.librarian._common import LIBRARIAN_SYSTEM_PROMPT
from gguf_limit_bench.librarian.gate import PACK_ID, build
from gguf_limit_bench.packs import AnswerType


def _selected_choice(q):
    return q.choices[ord(q.answer) - ord("A")]


def _wrong_letter(answer: str) -> str:
    return "B" if answer == "A" else "A"


def test_build_is_deterministic_for_a_seed():
    a = build(0)
    b = build(0)
    assert a == b


def test_build_varies_across_seeds_but_stays_valid():
    # Different seeds may differ; this only asserts both are well-formed packs.
    p0 = build(0)
    p1 = build(1)
    assert p0.pack_id == p1.pack_id == PACK_ID
    assert all(q.answer in ("A", "B") for q in p0.questions)
    assert all(q.answer in ("A", "B") for q in p1.questions)


def test_scorer_round_trip_correct_letter_passes():
    pack = build(0)
    for q in pack.questions:
        good = f"Reasoning about relevance.\nFinal Answer: {q.answer}"
        assert score_answer(good, q.answer, AnswerType.MULTIPLE_CHOICE, q.accept) is True


def test_scorer_round_trip_wrong_letter_fails():
    pack = build(0)
    for q in pack.questions:
        bad = f"Final Answer: {_wrong_letter(q.answer)}"
        assert score_answer(bad, q.answer, AnswerType.MULTIPLE_CHOICE, q.accept) is False


def test_pack_shape():
    pack = build(0)
    assert pack.pack_id == PACK_ID
    assert pack.tier == "librarian"
    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert pack.system_prompt == LIBRARIAN_SYSTEM_PROMPT
    assert 10 <= len(pack.questions) <= 16


def test_question_ids_are_unique_and_seed_stamped():
    seed = 3
    pack = build(seed)
    ids = [q.question_id for q in pack.questions]
    assert len(ids) == len(set(ids))
    for i, q in enumerate(pack.questions):
        assert q.question_id == f"{PACK_ID}-s{seed}-{i}"


def test_questions_are_valid_mc_items():
    pack = build(0)
    for q in pack.questions:
        assert q.choices is not None
        assert len(q.choices) == 2
        assert q.answer in ("A", "B")
        # answer letter is in range for the choices tuple
        assert ord(q.answer) - ord("A") < len(q.choices)
        assert q.answer_source == "librarian:gate"
        assert "librarian" in q.tags
        assert "gate" in q.tags


def test_gold_sanity_distractor_and_stale_are_skip():
    pack = build(0)
    skip_label = gate._CHOICES[1]
    distractors = [q for q in pack.questions if "distractor" in q.tags]
    stales = [q for q in pack.questions if "stale" in q.tags]
    assert len(distractors) >= 2
    assert len(stales) >= 1
    # Letter is randomized; the selected choice must be the "skip" label.
    for q in distractors + stales:
        assert _selected_choice(q) == skip_label


def test_relevant_items_are_inject_when_present():
    pack = build(0)
    inject_label = gate._CHOICES[0]
    relevants = [q for q in pack.questions if "relevant" in q.tags]
    for q in relevants:
        assert _selected_choice(q) == inject_label


def test_choices_label_set_is_fixed_across_questions():
    # Order is randomized per question, but the SET of labels is constant.
    pack = build(0)
    label_sets = {frozenset(q.choices) for q in pack.questions if q.choices is not None}
    assert label_sets == {frozenset(gate._CHOICES)}


def test_frozen_question_is_immutable_via_replace():
    # PackQuestion is frozen; replace returns a copy without mutating the source.
    pack = build(0)
    q = pack.questions[0]
    other = replace(q, answer=_wrong_letter(q.answer))
    assert other.answer != q.answer
