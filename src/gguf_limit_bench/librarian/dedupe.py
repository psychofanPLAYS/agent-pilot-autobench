"""Librarian job: memory de-duplication classification.

Given a NEW candidate memory and an EXISTING stored memory, the librarian must
classify the relationship between them so the agent's memory store stays clean:

* ``A`` — Duplicate: the same fact, just paraphrased. Drop or merge the new one.
* ``B`` — Related: same topic, but a genuinely different fact. Keep both.
* ``C`` — New: unrelated to the existing memory. Keep separately.

This is a pure, seed-deterministic generator returning a
:class:`gguf_limit_bench.packs.QuestionPack` graded by the existing
``MULTIPLE_CHOICE`` scorer (the model emits ``Final Answer: B``). No server is
needed and the module is fully unit-testable. ``build(seed)`` called twice with
the same seed returns byte-identical questions.
"""

from __future__ import annotations

from gguf_limit_bench.librarian._common import (
    LIBRARIAN_SYSTEM_PROMPT,
    AnswerType,
    PackQuestion,
    QuestionPack,
    make_rng,
    shuffle_choices,
)

PACK_ID = "librarian-dedupe"

# Fixed A/B/C choices. The wording never changes across questions; the gold label
# is *which class* the (existing, new) pair falls into.
_CHOICES: tuple[str, ...] = (
    "Duplicate — same fact; drop or merge",
    "Related — same topic, different fact; keep both",
    "New — unrelated; keep separately",
)

# Class index -> answer letter. Index 0 -> "A", 1 -> "B", 2 -> "C".
_DUPLICATE = 0
_RELATED = 1
_NEW = 2

# Curated item bank. Each item is (existing_memory, new_memory, gold_class).
# Authored so the gold is unambiguous; balanced across the three classes.
_ITEM_BANK: tuple[tuple[str, str, int], ...] = (
    # --- Duplicate (A): paraphrase of the same fact -------------------------
    (
        "The user prefers tabs over spaces.",
        "User likes tabs, not spaces.",
        _DUPLICATE,
    ),
    (
        "The user's name is Dave.",
        "The user goes by Dave.",
        _DUPLICATE,
    ),
    (
        "The project uses pytest for testing.",
        "Tests in this project are run with pytest.",
        _DUPLICATE,
    ),
    (
        "The default branch is named main.",
        "main is the repository's default branch.",
        _DUPLICATE,
    ),
    (
        "The user is based in the US Eastern timezone.",
        "User's timezone is US Eastern.",
        _DUPLICATE,
    ),
    # --- Related (B): same topic, a different fact --------------------------
    (
        "Deploys happen on Fridays.",
        "Deploys are blocked during a code freeze.",
        _RELATED,
    ),
    (
        "The user prefers tabs over spaces.",
        "The user prefers 4-wide indentation.",
        _RELATED,
    ),
    (
        "The API uses bearer-token authentication.",
        "The API rate-limits clients to 100 requests per minute.",
        _RELATED,
    ),
    (
        "The frontend is built with React.",
        "The frontend is bundled with Vite.",
        _RELATED,
    ),
    (
        "The staging database is in us-east-1.",
        "The staging database is restored from a nightly snapshot.",
        _RELATED,
    ),
    # --- New (C): unrelated ------------------------------------------------
    (
        "The user prefers tabs over spaces.",
        "The staging database is in us-east-1.",
        _NEW,
    ),
    (
        "Deploys happen on Fridays.",
        "The user's name is Dave.",
        _NEW,
    ),
    (
        "The project uses pytest for testing.",
        "The office Wi-Fi password rotates monthly.",
        _NEW,
    ),
    (
        "The API uses bearer-token authentication.",
        "The user prefers dark mode in their editor.",
        _NEW,
    ),
    (
        "The frontend is built with React.",
        "Standup is at 9:30 every weekday.",
        _NEW,
    ),
)

_PROMPT_TEMPLATE = (
    "You maintain the agent's long-term memory store. A new candidate memory has "
    "arrived. Decide how it relates to the existing stored memory so the store "
    "stays clean.\n\n"
    "EXISTING memory:\n"
    "{existing}\n\n"
    "NEW candidate memory:\n"
    "{new}\n\n"
    "Classify the relationship by choosing one option:\n"
    "{choice_lines}\n\n"
    "Answer with the single letter of the best option."
)


def _build_prompt(existing: str, new: str, choices: tuple[str, ...]) -> str:
    choice_lines = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    return _PROMPT_TEMPLATE.format(existing=existing, new=new, choice_lines=choice_lines)


def build(seed: int = 0) -> QuestionPack:
    """Build the de-duplication classification pack for ``seed``.

    Deterministic: the same ``seed`` always yields byte-identical questions.
    Produces 10..16 questions drawn (shuffled) from the curated item bank, with
    each of the three classes appearing at least twice.
    """
    rng = make_rng(seed)

    # Bucket the bank by gold class so we can guarantee balance.
    buckets: dict[int, list[tuple[str, str, int]]] = {_DUPLICATE: [], _RELATED: [], _NEW: []}
    for item in _ITEM_BANK:
        buckets[item[2]].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    # Reserve at least two of every class (gold-sanity guarantee), then fill up
    # to a deterministic count from the remaining pool. The reserved items are
    # never truncated away, so every class always appears at least twice.
    reserved: list[tuple[str, str, int]] = []
    remaining: list[tuple[str, str, int]] = []
    for bucket in buckets.values():
        reserved.extend(bucket[:2])
        remaining.extend(bucket[2:])
    rng.shuffle(remaining)

    # Pick a deterministic count in [10, 16], never exceeding the bank size.
    count = rng.randint(10, min(16, len(_ITEM_BANK)))
    fill = remaining[: max(0, count - len(reserved))]

    chosen = reserved + fill
    rng.shuffle(chosen)

    questions: list[PackQuestion] = []
    for i, (existing, new, gold_class) in enumerate(chosen):
        # Randomize which letter carries the correct class so the answer letter
        # does not encode the class (hardening: see docs 09-hardening-spec).
        choices, letter = shuffle_choices(rng, _CHOICES, gold_class)
        subtype = ("duplicate", "related", "new")[gold_class]
        questions.append(
            PackQuestion(
                question_id=f"{PACK_ID}-s{seed}-{i}",
                prompt=_build_prompt(existing, new, choices),
                answer=letter,
                answer_source="librarian:dedupe",
                choices=choices,
                tags=("librarian", "dedupe", subtype),
                accept=(),
            )
        )

    return QuestionPack(
        pack_id=PACK_ID,
        title="Librarian: memory de-duplication",
        tier="librarian",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        system_prompt=LIBRARIAN_SYSTEM_PROMPT,
        questions=tuple(questions),
    )
