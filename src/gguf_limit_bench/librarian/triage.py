"""Librarian *triage* job: salience — should this incoming snippet be remembered?

A coding agent streams a lot of conversation past the librarian. Most of it is
ephemeral chatter; a little of it is durable and worth committing to memory (a
stated preference, a decision, a constraint, a stable fact about the user or
project). This job measures whether the local model can tell the two apart.

Two EXACT-scored question kinds:

* ``keep_drop`` — a single snippet; the model replies ``keep`` (durable) or
  ``drop`` (ephemeral).
* ``count`` — a short text mixing a known number (0..5) of durable facts with
  neutral filler; the model replies with that count.

Pure and seed-deterministic: :func:`build` called twice with the same seed
returns byte-identical questions. All randomness flows through
:func:`gguf_limit_bench.librarian._common.make_rng`.
"""

from __future__ import annotations

import random

from gguf_limit_bench.librarian._common import (
    LIBRARIAN_SYSTEM_PROMPT,
    AnswerType,
    PackQuestion,
    QuestionPack,
    build_filler,
    make_rng,
)

PACK_ID = "librarian-triage"

_KEEP_DROP_INSTRUCTION = "Should the librarian remember this? Reply 'keep' or 'drop'."
_COUNT_INSTRUCTION = (
    "How many distinct durable facts worth remembering are in this text? Reply with the number."
)

# Durable snippets: a stated preference, a decision, a constraint, or a stable
# fact about the user or the project. These are the things a memory layer should
# commit. Each is a single self-contained statement with no digits, so it never
# collides with the EXACT number-word mapping in the count task.
_KEEP_SNIPPETS: tuple[str, ...] = (
    "The user prefers tabs over spaces for indentation.",
    "We decided to use PostgreSQL as the primary datastore for this project.",
    "Never deploy on Fridays is a hard team rule.",
    "The user's name is Dana and they go by they/them pronouns.",
    "All API responses in this codebase must be snake_case.",
    "The project targets Python and drops support for older runtimes.",
    "The user wants concise answers without preamble.",
    "We agreed that the staging environment mirrors production exactly.",
    "The repository's default branch is named trunk, not master.",
    "The user is colorblind and relies on high-contrast themes.",
)

# Ephemeral snippets: transient status, small talk, one-off acknowledgements,
# time-of-day chatter. A memory layer should discard these.
_DROP_SNIPPETS: tuple[str, ...] = (
    "ok thanks, that works.",
    "Running the tests now, give me a sec.",
    "Good morning! How's it going today?",
    "Hmm, let me think about that for a moment.",
    "Sounds good, sounds good.",
    "I'm grabbing a coffee, back in five.",
    "Cool, no worries.",
    "Just kicked off the build, watching the logs.",
    "Yeah that makes sense, got it.",
    "Brb, my dog needs to go out.",
)

_NUMBER_WORDS: tuple[str, ...] = ("zero", "one", "two", "three", "four", "five")


def _keep_drop_question(rng: random.Random, index: int, seed: int, gold: str) -> PackQuestion:
    snippet = rng.choice(_KEEP_SNIPPETS if gold == "keep" else _DROP_SNIPPETS)
    prompt = f"Snippet: {snippet}\n\n{_KEEP_DROP_INSTRUCTION}"
    return PackQuestion(
        question_id=f"{PACK_ID}-s{seed}-{index}",
        prompt=prompt,
        answer=gold,
        answer_source="librarian:triage",
        choices=None,
        tags=("librarian", "triage", "keep_drop", gold),
        accept=(),
    )


def _count_question(rng: random.Random, index: int, seed: int, count: int) -> PackQuestion:
    facts = rng.sample(_KEEP_SNIPPETS, count)
    filler = build_filler(rng, char_budget=rng.choice((140, 200, 260)))
    parts: list[str] = list(facts) + filler
    rng.shuffle(parts)
    text = " ".join(parts)
    prompt = f"Text:\n{text}\n\n{_COUNT_INSTRUCTION}"
    return PackQuestion(
        question_id=f"{PACK_ID}-s{seed}-{index}",
        prompt=prompt,
        answer=str(count),
        answer_source="librarian:triage",
        # accept the number-word form too; the EXACT scorer also maps it, but
        # being explicit documents intent for the 0..5 band.
        accept=(_NUMBER_WORDS[count],),
        choices=None,
        tags=("librarian", "triage", "count", f"facts={count}"),
    )


def build(seed: int = 0) -> QuestionPack:
    """Build the deterministic ``librarian-triage`` pack (10..16 EXACT questions)."""
    rng = make_rng(seed)
    questions: list[PackQuestion] = []
    index = 0

    # (a) KEEP/DROP — at least 6 keep and 6 drop, order shuffled deterministically.
    golds: list[str] = ["keep"] * 6 + ["drop"] * 6
    rng.shuffle(golds)
    for gold in golds:
        questions.append(_keep_drop_question(rng, index, seed, gold))
        index += 1

    # (b) COUNT extraction — a spread of known durable-fact counts in 0..5.
    counts = (0, 2, 3, 5)
    for count in counts:
        questions.append(_count_question(rng, index, seed, count))
        index += 1

    return QuestionPack(
        pack_id=PACK_ID,
        title="Librarian triage (salience)",
        tier="librarian",
        answer_type=AnswerType.EXACT,
        system_prompt=LIBRARIAN_SYSTEM_PROMPT,
        questions=tuple(questions),
    )
