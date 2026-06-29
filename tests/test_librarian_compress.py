"""Tests for the librarian ``compress`` job: faithful compression of a memory
note, MULTIPLE_CHOICE-scored and seed-deterministic.
"""

from __future__ import annotations

from gguf_limit_bench.answer_scoring import score_answer
from gguf_limit_bench.librarian.compress import (
    _SCENARIOS,
    PACK_ID,
    build,
)
from gguf_limit_bench.packs import AnswerType, QuestionPack

# Flat (true, false) pairs across every scenario, for clause/token lookups.
_ALL_FACTS = tuple(fact for _name, facts in _SCENARIOS for fact in facts)
_TRUE_CLAUSES = tuple(true for true, _false in _ALL_FACTS)
_FALSE_CLAUSES = tuple(false for _true, false in _ALL_FACTS)


def _wrong_letter(answer: str, choices: tuple[str, ...]) -> str:
    """Any valid letter within range that is not the gold answer."""
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    return next(letter for letter in letters if letter != answer)


def test_build_is_deterministic():
    a = build(0)
    b = build(0)
    triples_a = [(q.question_id, q.prompt, q.answer, q.choices) for q in a.questions]
    triples_b = [(q.question_id, q.prompt, q.answer, q.choices) for q in b.questions]
    assert triples_a == triples_b


def test_pack_shape():
    pack = build(0)
    assert isinstance(pack, QuestionPack)
    assert pack.pack_id == PACK_ID
    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert pack.tier == "librarian"
    assert pack.system_prompt
    assert 10 <= len(pack.questions) <= 16

    ids = [q.question_id for q in pack.questions]
    assert len(ids) == len(set(ids))  # unique

    for i, q in enumerate(pack.questions):
        assert q.question_id == f"{PACK_ID}-s0-{i}"
        assert q.answer_source == "librarian:compress"
        assert "librarian" in q.tags
        assert "compress" in q.tags
        assert any(t.startswith("k_facts=") for t in q.tags)
        # Exactly four choices; gold is a single letter A-D within range.
        assert q.choices is not None
        assert len(q.choices) == 4
        assert q.answer in ("A", "B", "C", "D")
        assert ord(q.answer) - ord("A") < len(q.choices)


def test_scorer_round_trip_correct_and_wrong():
    for seed in (0, 1, 2, 7):
        pack = build(seed)
        for q in pack.questions:
            assert q.choices is not None
            good = f"Reasoning about the note.\nFinal Answer: {q.answer}"
            assert score_answer(good, q.answer, AnswerType.MULTIPLE_CHOICE, q.accept) is True
            bad = f"Final Answer: {_wrong_letter(q.answer, q.choices)}"
            assert score_answer(bad, q.answer, AnswerType.MULTIPLE_CHOICE, q.accept) is False


def test_gold_sanity():
    """The gold summary preserves every planted clause and adds no false token;
    every non-gold summary either drops a clause or injects a false token."""
    for seed in range(6):
        pack = build(seed)
        for q in pack.questions:
            assert q.choices is not None
            gold_idx = ord(q.answer) - ord("A")
            gold = q.choices[gold_idx]

            # Recover the planted clauses: the true clauses present in the gold
            # summary (the faithful one mentions all K).
            planted = [c for c in _TRUE_CLAUSES if c in gold]
            assert len(planted) >= 3  # K is 3 or 4

            # Gold contains every planted clause (substring check) ...
            for clause in planted:
                assert clause in gold
            # ... and introduces no false token.
            for false in _FALSE_CLAUSES:
                assert false not in gold

            # Each non-gold summary omits a planted clause OR injects a false token.
            for idx, summary in enumerate(q.choices):
                if idx == gold_idx:
                    continue
                omits_fact = any(clause not in summary for clause in planted)
                injects_false = any(false in summary for false in _FALSE_CLAUSES)
                assert omits_fact or injects_false


def test_k_facts_tag_matches_planted():
    pack = build(0)
    for q in pack.questions:
        assert q.choices is not None
        gold = q.choices[ord(q.answer) - ord("A")]
        planted = [c for c in _TRUE_CLAUSES if c in gold]
        k_tag = next(t for t in q.tags if t.startswith("k_facts="))
        assert k_tag == f"k_facts={len(planted)}"


def test_seeds_vary_selection():
    sets = {tuple((q.prompt, q.answer) for q in build(s).questions) for s in range(6)}
    assert len(sets) > 1
