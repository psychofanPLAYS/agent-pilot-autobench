"""Librarian *contradiction* job: staleness / conflict detection.

A memory layer must notice when a NEW statement invalidates something it already
remembers. Given an EXISTING memory and a NEW statement, the model classifies the
relationship: the new statement confirms the memory, contradicts it (the memory
is now stale or wrong), or is about an unrelated subject. Detecting contradictions
is the important capability here -- a stale memory silently injected into the
agent's context is worse than no memory at all.

Pure and seed-deterministic: ``build(seed)`` returns a
:class:`gguf_limit_bench.packs.QuestionPack` scored by the existing
MULTIPLE_CHOICE scorer in :mod:`gguf_limit_bench.answer_scoring`. No server, no
network, fully unit-testable.

The three choices are FIXED (gold is the letter, not the wording):

* ``A`` -- confirms (consistent with the existing memory)
* ``B`` -- contradicts (the existing memory is now stale or wrong)
* ``C`` -- unrelated (different subject)
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

PACK_ID = "librarian-contradiction"

# Fixed three-way choices. Order is load-bearing: gold answers reference index.
_CHOICES: tuple[str, ...] = (
    "Confirms — consistent with the existing memory",
    "Contradicts — the existing memory is now stale or wrong",
    "Unrelated — different subject",
)
_CONFIRMS = "A"
_CONTRADICTS = "B"
_UNRELATED = "C"

# Item bank: (existing, new, gold_letter, subtype). Each case has a KNOWN gold.
# Subtypes mirror the three classes so the gold-sanity tests can assert coverage.
#   - confirms:    the new statement is consistent with the existing memory -> A.
#   - contradicts: the new statement makes the existing memory stale/wrong -> B.
#   - unrelated:   the new statement is about a different subject -> C.
_ITEM_BANK: tuple[tuple[str, str, str, str], ...] = (
    # --- confirms -> A -------------------------------------------------------
    (
        "Deploys happen on Fridays.",
        "Friday's deploy went out on schedule.",
        _CONFIRMS,
        "confirms",
    ),
    (
        "The prod DB is in us-east-1.",
        "Latency from us-east-1 to the app servers stayed low again today.",
        _CONFIRMS,
        "confirms",
    ),
    (
        "The user prefers tabs for indentation.",
        "The user reiterated they want tabs, not spaces, in new files.",
        _CONFIRMS,
        "confirms",
    ),
    (
        "The default branch is named trunk.",
        "I merged the feature into trunk as usual.",
        _CONFIRMS,
        "confirms",
    ),
    # --- contradicts -> B ----------------------------------------------------
    (
        "The prod DB is in us-east-1.",
        "We migrated the prod DB to eu-west-2.",
        _CONTRADICTS,
        "contradicts",
    ),
    (
        "Deploys happen on Fridays.",
        "The team moved the weekly deploy window to Tuesdays.",
        _CONTRADICTS,
        "contradicts",
    ),
    (
        "The user prefers tabs for indentation.",
        "The user now wants spaces everywhere and added a formatter to enforce it.",
        _CONTRADICTS,
        "contradicts",
    ),
    (
        "The API returns responses in snake_case.",
        "The API was changed to return camelCase fields in v2.",
        _CONTRADICTS,
        "contradicts",
    ),
    (
        "The default branch is named master.",
        "We renamed the default branch from master to main.",
        _CONTRADICTS,
        "contradicts",
    ),
    (
        "The on-call rotation switches every Monday.",
        "On-call now rotates on Wednesdays instead of Mondays.",
        _CONTRADICTS,
        "contradicts",
    ),
    # --- unrelated -> C ------------------------------------------------------
    (
        "The user prefers tabs for indentation.",
        "The cafeteria changed its menu.",
        _UNRELATED,
        "unrelated",
    ),
    (
        "The prod DB is in us-east-1.",
        "The office plants were watered this morning.",
        _UNRELATED,
        "unrelated",
    ),
    (
        "Deploys happen on Fridays.",
        "A new espresso machine arrived in the break room.",
        _UNRELATED,
        "unrelated",
    ),
    (
        "The API returns responses in snake_case.",
        "The team booked a venue for the offsite next month.",
        _UNRELATED,
        "unrelated",
    ),
)

# Sanity self-check at import time: the bank must satisfy the gold constraints the
# tests enforce, so a malformed edit fails fast rather than silently.
assert len(_ITEM_BANK) >= 12
assert sum(1 for *_, sub in _ITEM_BANK if sub == "confirms") >= 1
assert sum(1 for *_, sub in _ITEM_BANK if sub == "contradicts") >= 4
assert sum(1 for *_, sub in _ITEM_BANK if sub == "unrelated") >= 1
assert all(gold in (_CONFIRMS, _CONTRADICTS, _UNRELATED) for *_, gold, _ in _ITEM_BANK)

_MIN_QUESTIONS = 10
_MAX_QUESTIONS = 16


def _format_prompt(existing: str, new: str, choices: tuple[str, ...]) -> str:
    choice_lines = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    return (
        "You maintain the coding agent's memory. A NEW statement has arrived. "
        "Decide how it relates to an EXISTING memory: does it confirm the memory, "
        "contradict it (making the memory stale or wrong), or is it about an "
        "unrelated subject?\n\n"
        f"Existing memory: {existing}\n"
        f"New statement: {new}\n\n"
        "Choices:\n"
        f"{choice_lines}\n\n"
        "Answer with the single letter of the best choice."
    )


def build(seed: int = 0) -> QuestionPack:
    """Build the staleness/conflict detection pack for ``seed`` (deterministic)."""
    rng = make_rng(seed)

    # Deterministic selection: shuffle a copy of the bank, then take a count in
    # [10, 16] driven by the same rng. The bank is large enough to honour the
    # required class floors regardless of count because those items are
    # guaranteed present in the full bank and we always keep >= 10.
    bank = list(_ITEM_BANK)
    rng.shuffle(bank)

    upper = min(_MAX_QUESTIONS, len(bank))
    count = rng.randint(_MIN_QUESTIONS, upper)

    # Guarantee the gold-sanity floors survive the slice: at least one confirms,
    # at least four contradicts, and at least one unrelated. Pull required items
    # to the front, then fill from the remainder.
    confirms = [it for it in bank if it[3] == "confirms"]
    contradicts = [it for it in bank if it[3] == "contradicts"]
    unrelated = [it for it in bank if it[3] == "unrelated"]
    required = confirms[:1] + contradicts[:4] + unrelated[:1]
    required_ids = {id(it) for it in required}
    remainder = [it for it in bank if id(it) not in required_ids]
    ordered = required + remainder
    selected = ordered[:count]
    # Re-shuffle the selected slice so required items are not always first.
    rng.shuffle(selected)

    questions: list[PackQuestion] = []
    for i, (existing, new, gold, subtype) in enumerate(selected):
        # Randomize which letter carries the correct choice so the answer letter
        # does not encode the class (hardening: see docs 09-hardening-spec).
        gold_index = ord(gold) - ord("A")
        choices, letter = shuffle_choices(rng, _CHOICES, gold_index)
        questions.append(
            PackQuestion(
                question_id=f"{PACK_ID}-s{seed}-{i}",
                prompt=_format_prompt(existing, new, choices),
                answer=letter,
                answer_source="librarian:contradiction",
                choices=choices,
                tags=("librarian", "contradiction", subtype),
                accept=(),
            )
        )

    return QuestionPack(
        pack_id=PACK_ID,
        title="Librarian: contradiction / staleness detection",
        tier="librarian",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        system_prompt=LIBRARIAN_SYSTEM_PROMPT,
        questions=tuple(questions),
    )
