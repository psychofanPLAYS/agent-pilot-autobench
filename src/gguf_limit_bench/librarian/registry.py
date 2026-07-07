"""Registry of librarian benchmark job packs.

Each job module exposes ``PACK_ID`` and ``build(seed) -> QuestionPack``. This
module collects them into a single mapping so the rest of the app can discover
and build librarian packs by id without importing each module directly.

Builders are referenced explicitly (not by iterating module objects) so the
static type checker can see each ``PACK_ID``/``build`` attribute.
"""

from __future__ import annotations

from typing import Callable

from gguf_limit_bench.librarian import (
    compress,
    contradiction,
    dedupe,
    gate,
    query,
    rerank,
    triage,
    write_entry,
)
from gguf_limit_bench.packs import QuestionPack

LibrarianBuilder = Callable[[int], QuestionPack]

LIBRARIAN_BUILDERS: dict[str, LibrarianBuilder] = {
    write_entry.PACK_ID: write_entry.build,
    triage.PACK_ID: triage.build,
    dedupe.PACK_ID: dedupe.build,
    gate.PACK_ID: gate.build,
    query.PACK_ID: query.build,
    rerank.PACK_ID: rerank.build,
    compress.PACK_ID: compress.build,
    contradiction.PACK_ID: contradiction.build,
}

LIBRARIAN_PACK_IDS: tuple[str, ...] = tuple(LIBRARIAN_BUILDERS)


def build_librarian_pack(pack_id: str, seed: int = 0) -> QuestionPack:
    """Build a librarian pack by id. Raises KeyError if the id is unknown."""
    try:
        builder = LIBRARIAN_BUILDERS[pack_id]
    except KeyError:
        raise KeyError(f"Unknown librarian pack: {pack_id!r}") from None
    return builder(seed)
