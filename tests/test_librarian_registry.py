from __future__ import annotations

import pytest

from gguf_limit_bench import packs
from gguf_limit_bench.librarian.registry import (
    LIBRARIAN_BUILDERS,
    LIBRARIAN_PACK_IDS,
    build_librarian_pack,
)

EXPECTED_IDS = {
    "librarian-write-entry",
    "librarian-triage",
    "librarian-dedupe",
    "librarian-gate",
    "librarian-query",
    "librarian-rerank",
    "librarian-compress",
    "librarian-contradiction",
}


def test_registry_lists_all_jobs() -> None:
    assert set(LIBRARIAN_PACK_IDS) == EXPECTED_IDS
    assert len(LIBRARIAN_PACK_IDS) == len(set(LIBRARIAN_PACK_IDS))
    assert set(LIBRARIAN_BUILDERS) == EXPECTED_IDS
    assert all(pid.startswith("librarian-") for pid in LIBRARIAN_PACK_IDS)


@pytest.mark.parametrize("pack_id", sorted(EXPECTED_IDS))
def test_build_returns_matching_pack(pack_id: str) -> None:
    pack = build_librarian_pack(pack_id, seed=0)
    assert pack.pack_id == pack_id
    assert pack.tier == "librarian"
    assert 10 <= len(pack.questions) <= 16
    assert len({q.question_id for q in pack.questions}) == len(pack.questions)


@pytest.mark.parametrize("pack_id", sorted(EXPECTED_IDS))
def test_build_is_deterministic(pack_id: str) -> None:
    def sig(p: object) -> list[tuple[str, str, str, tuple[str, ...] | None]]:
        return [(q.question_id, q.prompt, q.answer, q.choices) for q in p.questions]  # type: ignore[attr-defined]

    assert sig(build_librarian_pack(pack_id, seed=0)) == sig(build_librarian_pack(pack_id, seed=0))


def test_unknown_pack_raises() -> None:
    with pytest.raises(KeyError):
        build_librarian_pack("librarian-nope", seed=0)


def test_packs_discovery_includes_librarian() -> None:
    available = set(packs.available_packs())
    assert EXPECTED_IDS <= available


@pytest.mark.parametrize("pack_id", sorted(EXPECTED_IDS))
def test_load_pack_builds_librarian(pack_id: str) -> None:
    pack = packs.load_pack(pack_id)
    assert pack.pack_id == pack_id
    assert pack.questions
