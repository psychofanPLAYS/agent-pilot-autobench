"""Librarian benchmark suite: deterministic generators that measure a local model
acting as the memory/RAG "librarian" serving Claude Code / Codex.

Each job module exposes ``PACK_ID: str`` and ``build(seed: int = 0) -> QuestionPack``
and is graded by the existing EXACT / MULTIPLE_CHOICE scorer. The registry below is
wired after the individual job modules land (see docs/feature-requests/
wiki-librarian-bench/). Keep this package import-light: job modules depend only on
``gguf_limit_bench.packs``, ``gguf_limit_bench.answer_scoring``, and
``gguf_limit_bench.librarian._common``.
"""

from __future__ import annotations

from gguf_limit_bench.librarian.registry import (
    LIBRARIAN_BUILDERS,
    LIBRARIAN_PACK_IDS,
    build_librarian_pack,
)

__all__ = [
    "LIBRARIAN_BUILDERS",
    "LIBRARIAN_PACK_IDS",
    "build_librarian_pack",
]
