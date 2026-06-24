"""Shared deterministic helpers for the librarian benchmark generators.

The librarian suite measures a local model acting as the memory/RAG "librarian"
that decides what a coding agent should remember and recall. Every job is a pure,
seed-deterministic generator that returns a :class:`gguf_limit_bench.packs.QuestionPack`
graded by the existing scorer in :mod:`gguf_limit_bench.answer_scoring`. No server
is needed and the generators are fully unit-testable.

Scorer contract (mirror of answer_scoring.py — read it before authoring a job):

* ``EXACT`` — the model must end with a line ``Final Answer: <value>``. Comparison
  goes through ``normalize_exact``: lowercased, surrounding punctuation stripped,
  number-words ``zero..twenty`` mapped to digits, internal whitespace collapsed,
  plus a phrase-containment fallback and an ``accept`` tuple of alternative forms.
  Keep gold answers short and unambiguous: a single word, a number, a short token.
  Avoid answers whose correctness depends on punctuation or casing.
* ``MULTIPLE_CHOICE`` — the model must emit a letter ``A``-``F`` (e.g.
  ``Final Answer: B``). Provide ``choices`` and a single uppercase-letter ``answer``.

Determinism rule: a generator called twice with the same seed MUST return byte-identical
questions. Use :func:`make_rng` for all randomness; never call the global ``random``.
"""

from __future__ import annotations

import random

from gguf_limit_bench.packs import AnswerType, PackQuestion, QuestionPack

__all__ = [
    "AnswerType",
    "PackQuestion",
    "QuestionPack",
    "MEMORY_TYPES",
    "LIBRARIAN_SYSTEM_PROMPT",
    "make_rng",
    "build_filler",
    "approx_token_count",
    "kebab_slug",
    "shuffle_choices",
]

# The memory-entry schema the librarian writes against (see the productivity
# memory system: one fact per file with this frontmatter `type`).
MEMORY_TYPES: tuple[str, ...] = ("user", "feedback", "project", "reference")

LIBRARIAN_SYSTEM_PROMPT: str = (
    "You are the librarian: the memory layer that decides what a coding agent "
    "should remember and recall. Follow the instructions exactly. Think only as "
    "much as you need, then end your reply with a single line of the form "
    "'Final Answer: X' where X is exactly the requested value."
)

# Neutral filler with no digits and no leakable tokens, so it never collides with
# a planted fact, code, or label. Borrowed in spirit from procedural_packs.py.
_FILLER_SENTENCES: tuple[str, ...] = (
    "The maintenance crew recorded routine inspections across the facility.",
    "Operators rotated through their shifts without any unusual incident.",
    "A gentle breeze moved through the corridor during the afternoon.",
    "The archive shelves were dusted and returned to their usual order.",
    "Visitors noted the quiet hum of the ventilation overhead.",
    "Reports from the eastern wing described calm and steady conditions.",
    "The logistics team confirmed that supplies arrived on schedule.",
    "Nothing of particular note disturbed the morning routine.",
    "The garden beyond the window held its familiar stillness.",
    "Staff exchanged brief greetings as they passed in the hallway.",
)


def make_rng(seed: int) -> random.Random:
    """Return a seeded RNG. All generator randomness must flow through this."""
    return random.Random(seed)


def build_filler(rng: random.Random, char_budget: int) -> list[str]:
    """Return neutral filler sentences totalling roughly ``char_budget`` chars."""
    sentences: list[str] = []
    length = 0
    while length < char_budget:
        sentence = rng.choice(_FILLER_SENTENCES)
        sentences.append(sentence)
        length += len(sentence) + 1
    return sentences


def approx_token_count(text: str) -> int:
    """Heuristic token count (~4 characters per token). Deterministic, no tokenizer."""
    return round(len(text) / 4)


def kebab_slug(text: str) -> str:
    """Lowercase kebab-case slug (used by the write-entry schema)."""
    out = []
    prev_dash = False
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def shuffle_choices(
    rng: random.Random, labels: tuple[str, ...], gold_index: int
) -> tuple[tuple[str, ...], str]:
    """Randomize the display order of MC ``labels`` so the answer letter does not
    encode the class.

    ``gold_index`` is the index of the correct label in ``labels``. Returns
    ``(display_choices, gold_letter)`` where ``gold_letter`` is the A/B/C... letter
    of the position the correct label was shuffled to. Multiple-choice packs should
    use this instead of a fixed label->letter mapping, so a model cannot exploit a
    constant "the answer is usually B" pattern. See docs 09-hardening-spec.
    """
    order = list(range(len(labels)))
    rng.shuffle(order)
    display = tuple(labels[i] for i in order)
    gold_letter = chr(ord("A") + order.index(gold_index))
    return display, gold_letter
