import json

import pytest

from gguf_limit_bench.packs import BUILTIN_PACK_IDS, BenchmarkPack, load_benchmark_packs
from gguf_limit_bench.packs import (
    AnswerType,
    DEFAULT_PACKS,
    available_packs,
    load_pack,
)


def test_builtin_benchmark_packs_cover_autoresearch_plan():
    packs = load_benchmark_packs()

    assert BUILTIN_PACK_IDS <= set(packs)
    assert packs["hermes-pilot"].safety_policy == "local_deterministic"
    assert "tool" in packs["tool-calling"].scoring_categories


def test_user_pack_manifest_is_loaded_and_versioned(tmp_path):
    plugin_dir = tmp_path / "plugins" / "benchmarks"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "private-pack.json").write_text(
        json.dumps(
            {
                "id": "private-pack",
                "version": "2026.05.26",
                "description": "Private local prompt pack.",
                "tasks": ["local_fixture"],
                "settings_space": {"context": [16384]},
                "scoring_categories": ["quality"],
                "safety_policy": "local_fixture_only",
                "receipt_schema": "pack.v1",
            }
        ),
        encoding="utf-8",
    )

    packs = load_benchmark_packs(plugin_dir)

    assert isinstance(packs["private-pack"], BenchmarkPack)
    assert packs["private-pack"].version == "2026.05.26"


# ---------------------------------------------------------------------------
# QuestionPack tests (Task 1)
# ---------------------------------------------------------------------------


def test_load_simple_bench_question_pack():
    pack = load_pack("simple-bench")

    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert len(pack.questions) == 10
    assert all(q.prompt for q in pack.questions)
    assert all(q.answer for q in pack.questions)


def test_load_easy_gotcha_question_pack():
    pack = load_pack("easy-gotcha")

    assert pack.answer_type is AnswerType.EXACT
    assert len(pack.questions) >= 20
    assert all(q.answer_source == "curated_fact" for q in pack.questions)


def test_load_easy_mc_question_pack():
    pack = load_pack("easy-mc")

    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert len(pack.questions) >= 20
    assert all(q.answer_source.startswith("dataset_label:") for q in pack.questions)


def test_available_packs_covers_default_packs():
    packs = available_packs()

    assert set(DEFAULT_PACKS) <= set(packs)


def test_load_pack_raises_key_error_for_unknown_id():
    with pytest.raises(KeyError):
        load_pack("nope")


# ---------------------------------------------------------------------------
# Task 8: easy-gotcha pack hardening tests
# ---------------------------------------------------------------------------


def test_easy_gotcha_does_not_contain_staircase_steps():
    pack = load_pack("easy-gotcha")

    ids = {q.question_id for q in pack.questions}
    assert "staircase-steps" not in ids


def test_easy_gotcha_sister_riddle_has_accept_variants():
    pack = load_pack("easy-gotcha")

    sister = next(q for q in pack.questions if q.question_id == "sister-riddle")
    assert len(sister.accept) > 0


def test_easy_gotcha_doctor_riddle_has_accept_variants():
    pack = load_pack("easy-gotcha")

    doctor = next(q for q in pack.questions if q.question_id == "doctor-riddle")
    assert len(doctor.accept) > 0
