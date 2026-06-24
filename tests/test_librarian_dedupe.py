"""Tests for the librarian de-duplication job module."""

from __future__ import annotations

from collections import Counter

import pytest

from gguf_limit_bench.answer_scoring import score_answer
from gguf_limit_bench.librarian import dedupe
from gguf_limit_bench.librarian._common import LIBRARIAN_SYSTEM_PROMPT
from gguf_limit_bench.packs import AnswerType


def test_determinism_same_seed_identical() -> None:
    first = dedupe.build(0)
    second = dedupe.build(0)
    assert first == second


def test_determinism_across_seeds_varies() -> None:
    # Sanity: different seeds should not be forced to coincide.
    assert dedupe.build(0) != dedupe.build(1)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_shape(seed: int) -> None:
    pack = dedupe.build(seed)

    assert pack.pack_id == dedupe.PACK_ID == "librarian-dedupe"
    assert pack.tier == "librarian"
    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert pack.system_prompt == LIBRARIAN_SYSTEM_PROMPT

    questions = pack.questions
    assert 10 <= len(questions) <= 16

    ids = [q.question_id for q in questions]
    assert len(ids) == len(set(ids)), "question ids must be unique"

    for i, q in enumerate(questions):
        assert q.question_id == f"librarian-dedupe-s{seed}-{i}"
        assert q.answer_source == "librarian:dedupe"
        assert q.choices is not None
        # Tags include the suite markers plus a subtype.
        assert "librarian" in q.tags
        assert "dedupe" in q.tags
        # Answer is a single letter A-C, in range of choices.
        assert len(q.answer) == 1
        assert q.answer in "ABC"
        idx = ord(q.answer) - ord("A")
        assert 0 <= idx < len(q.choices)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_scorer_round_trip(seed: int) -> None:
    pack = dedupe.build(seed)
    for q in pack.questions:
        good = f"Reasoning here.\nFinal Answer: {q.answer}"
        assert score_answer(good, q.answer, AnswerType.MULTIPLE_CHOICE, q.accept) is True

        # A different letter must score False.
        wrong_letter = "A" if q.answer != "A" else "B"
        bad = f"Reasoning here.\nFinal Answer: {wrong_letter}"
        assert score_answer(bad, q.answer, AnswerType.MULTIPLE_CHOICE, q.accept) is False


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_gold_sanity_all_classes_appear_twice(seed: int) -> None:
    pack = dedupe.build(seed)
    # After letter randomization the answer letter no longer encodes the class, so
    # class balance is asserted via the subtype tag, not the letter.
    counts = Counter(
        next(t for t in q.tags if t in {"duplicate", "related", "new"}) for q in pack.questions
    )
    for sub in ("duplicate", "related", "new"):
        assert counts[sub] >= 2, f"class {sub} appears {counts[sub]} times for seed {seed}"


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_gold_letter_matches_class_label(seed: int) -> None:
    # The randomized gold letter must still point at the correct class label.
    labels = {
        "duplicate": dedupe._CHOICES[0],
        "related": dedupe._CHOICES[1],
        "new": dedupe._CHOICES[2],
    }
    pack = dedupe.build(seed)
    for q in pack.questions:
        sub = next(t for t in q.tags if t in labels)
        assert q.choices is not None
        idx = ord(q.answer) - ord("A")
        assert q.choices[idx] == labels[sub]


def test_item_bank_size() -> None:
    # Curated bank must be large enough to honor the spec floor.
    assert len(dedupe._ITEM_BANK) >= 12
