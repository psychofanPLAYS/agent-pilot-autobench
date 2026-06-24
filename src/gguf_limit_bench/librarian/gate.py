"""Librarian job: the inject/skip gate (the highest-value librarian decision).

Given the coding agent's CURRENT TASK and ONE candidate memory entry, the model
must decide whether to inject that memory into the agent's context now or skip it.
Injecting an irrelevant memory pollutes the agent's working context, so the gate
is graded on precision as much as recall.

Pure and seed-deterministic: ``build(seed)`` returns a
:class:`gguf_limit_bench.packs.QuestionPack` scored by the existing
MULTIPLE_CHOICE scorer in :mod:`gguf_limit_bench.answer_scoring`. No server, no
network, fully unit-testable.

The two choices are FIXED (gold is the letter, not the wording):

* ``A`` -- inject (relevant and useful for this task now)
* ``B`` -- skip (not relevant to this task)
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

PACK_ID = "librarian-gate"

# Fixed two-way choices. Order is load-bearing: gold answers reference index.
_CHOICES: tuple[str, ...] = (
    "Inject — relevant and useful for this task now",
    "Skip — not relevant to this task",
)
_INJECT = "A"
_SKIP = "B"

# Item bank: (task, memory, gold_letter, subtype). Each case has a KNOWN gold.
# Subtypes: relevant -> A; irrelevant / distractor / stale -> B.
#   - relevant:   the memory directly bears on the task.
#   - irrelevant: unrelated memory, no shared keywords.
#   - distractor: shares surface keywords with the task but is not useful.
#   - stale:      deprecated / removed information that must not be injected.
_ITEM_BANK: tuple[tuple[str, str, str, str], ...] = (
    # --- relevant -> A -------------------------------------------------------
    (
        "Fixing the OAuth token refresh bug",
        "Auth tokens must be refreshed 60 seconds before expiry.",
        _INJECT,
        "relevant",
    ),
    (
        "Optimize the slow database query in reports.py",
        "reports.py joins orders and users without an index on user_id.",
        _INJECT,
        "relevant",
    ),
    (
        "Write the deploy script for the staging environment",
        "Staging deploys must run database migrations before restarting the app.",
        _INJECT,
        "relevant",
    ),
    (
        "Add retry logic to the payment webhook handler",
        "The payment provider returns HTTP 429 and expects exponential backoff.",
        _INJECT,
        "relevant",
    ),
    (
        "Set up logging for the new ingestion service",
        "Project standard: emit structured JSON logs at INFO level.",
        _INJECT,
        "relevant",
    ),
    # --- irrelevant -> B -----------------------------------------------------
    (
        "Writing the release notes",
        "The user prefers vim keybindings.",
        _SKIP,
        "irrelevant",
    ),
    (
        "Refactor the CSS for the landing page",
        "The on-call rotation switches every Monday at 9am.",
        _SKIP,
        "irrelevant",
    ),
    (
        "Add unit tests for the date parser",
        "The office coffee machine was replaced last quarter.",
        _SKIP,
        "irrelevant",
    ),
    # --- distractor (shares keywords but irrelevant) -> B --------------------
    (
        "Optimize the slow database query in reports.py",
        "The #reports Slack channel is muted.",
        _SKIP,
        "distractor",
    ),
    (
        "Fixing the OAuth token refresh bug",
        "The auth team sits on the third floor near the kitchen.",
        _SKIP,
        "distractor",
    ),
    (
        "Speed up the payment reconciliation job",
        "The payment for the team lunch is still pending reimbursement.",
        _SKIP,
        "distractor",
    ),
    # --- stale / deprecated -> B ---------------------------------------------
    (
        "Deploy to production",
        "DEPRECATED: the old deploy.sh script was removed.",
        _SKIP,
        "stale",
    ),
    (
        "Configure the API client timeout",
        "DEPRECATED: the legacy v1 API endpoints were shut down last year.",
        _SKIP,
        "stale",
    ),
)

# Sanity self-check at import time: the bank must satisfy the gold constraints the
# tests enforce, so a malformed edit fails fast rather than silently.
assert len(_ITEM_BANK) >= 12
assert sum(1 for *_, sub in _ITEM_BANK if sub == "distractor") >= 2
assert sum(1 for *_, sub in _ITEM_BANK if sub == "stale") >= 1
assert all(gold in (_INJECT, _SKIP) for *_, gold, _ in _ITEM_BANK)
assert all(gold == _SKIP for *_, gold, sub in _ITEM_BANK if sub != "relevant")

_MIN_QUESTIONS = 10
_MAX_QUESTIONS = 16


def _format_prompt(task: str, memory: str, choices: tuple[str, ...]) -> str:
    choice_lines = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    return (
        "You are deciding whether to inject one candidate memory into the coding "
        "agent's context for its current task. Injecting an irrelevant memory "
        "pollutes the agent's context, so inject only when the memory is relevant "
        "and useful for this task right now.\n\n"
        f"Current task: {task}\n"
        f"Candidate memory: {memory}\n\n"
        "Choices:\n"
        f"{choice_lines}\n\n"
        "Answer with the single letter of the better choice."
    )


def build(seed: int = 0) -> QuestionPack:
    """Build the inject/skip gate pack for ``seed`` (deterministic)."""
    rng = make_rng(seed)

    # Deterministic selection: shuffle a copy of the bank, then take a count in
    # [10, 16] driven by the same rng. The bank is large enough to honour the
    # required distractor/stale floors regardless of count because those items
    # are guaranteed present in the full bank and we always keep >= 10.
    bank = list(_ITEM_BANK)
    rng.shuffle(bank)

    upper = min(_MAX_QUESTIONS, len(bank))
    count = rng.randint(_MIN_QUESTIONS, upper)

    # Guarantee the gold-sanity floors (>=2 distractor, >=1 stale) survive the
    # slice: pull required items to the front, then fill from the remainder.
    distractors = [it for it in bank if it[3] == "distractor"]
    stales = [it for it in bank if it[3] == "stale"]
    required = distractors[:2] + stales[:1]
    required_ids = {id(it) for it in required}
    remainder = [it for it in bank if id(it) not in required_ids]
    ordered = required + remainder
    selected = ordered[:count]
    # Re-shuffle the selected slice so required items are not always first.
    rng.shuffle(selected)

    questions: list[PackQuestion] = []
    for i, (task, memory, gold, subtype) in enumerate(selected):
        # Randomize which letter carries the correct choice so the answer letter
        # does not encode the class (hardening: see docs 09-hardening-spec).
        gold_index = ord(gold) - ord("A")
        choices, letter = shuffle_choices(rng, _CHOICES, gold_index)
        questions.append(
            PackQuestion(
                question_id=f"{PACK_ID}-s{seed}-{i}",
                prompt=_format_prompt(task, memory, choices),
                answer=letter,
                answer_source="librarian:gate",
                choices=choices,
                tags=("librarian", "gate", subtype),
                accept=(),
            )
        )

    return QuestionPack(
        pack_id=PACK_ID,
        title="Librarian: inject/skip gate",
        tier="librarian",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        system_prompt=LIBRARIAN_SYSTEM_PROMPT,
        questions=tuple(questions),
    )
