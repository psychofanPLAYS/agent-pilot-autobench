"""Librarian job: ``write-entry`` — schema-conformant memory metadata.

This generator probes whether the local model, acting as the memory librarian,
can emit the structured metadata a memory entry needs before it is filed:

* **TYPE classification** — given a one-line memory, pick its ``type`` from the
  fixed schema vocabulary in :data:`gguf_limit_bench.librarian._common.MEMORY_TYPES`
  (``user`` / ``feedback`` / ``project`` / ``reference``).
* **SLUG formatting** — given a human title, produce the kebab-case slug used as
  the entry's filename, matching :func:`_common.kebab_slug` exactly.

Both kinds are EXACT-scored single tokens, so the gold is unambiguous and the
existing scorer round-trips cleanly. The generator is pure and
seed-deterministic: ``build(seed)`` called twice returns byte-identical questions.
"""

from __future__ import annotations

import random
from typing import cast

from gguf_limit_bench.librarian._common import (
    LIBRARIAN_SYSTEM_PROMPT,
    AnswerType,
    PackQuestion,
    QuestionPack,
    kebab_slug,
    make_rng,
)

PACK_ID = "librarian-write-entry"

_SYSTEM_HINT = (
    " For TYPE questions answer with one of: user, feedback, project, reference. "
    "For SLUG questions answer with the kebab-case slug only."
)

# --- TYPE item bank ---------------------------------------------------------
# Each entry is (one-line memory, gold type). Every gold is drawn from
# MEMORY_TYPES and every example is intended to be unambiguous: identity vs.
# how-to-work guidance vs. ongoing work vs. an external pointer.
_TYPE_BANK: tuple[tuple[str, str], ...] = (
    # user — who the user is: identity, role, durable personal preferences.
    ("The user is a security researcher.", "user"),
    ("The user is a senior backend engineer based in Berlin.", "user"),
    ("The user prefers tabs over spaces in all source files.", "user"),
    ("The user's name is Dana and they go by she/her.", "user"),
    ("The user is colorblind and prefers high-contrast output.", "user"),
    # feedback — guidance on how Claude should work.
    ("Always run the tests before claiming a task is done.", "feedback"),
    ("Never push directly to the main branch.", "feedback"),
    ("Prefer small, focused commits over one large commit.", "feedback"),
    ("Ask before installing any new dependency.", "feedback"),
    ("Keep responses concise and skip the preamble.", "feedback"),
    # project — ongoing work, goals, constraints.
    ("The Q3 data migration must finish before the audit on 2026-09-01.", "project"),
    ("The checkout rewrite is blocked on the payments API redesign.", "project"),
    ("The mobile app must ship to the App Store by the end of the quarter.", "project"),
    ("The billing service is being split out into its own repository.", "project"),
    # reference — pointer to an external resource, URL, ticket, or dashboard.
    ("The ops runbook lives at https://wiki.example/ops/runbook.", "reference"),
    ("The release checklist is tracked in ticket JIRA-4821.", "reference"),
    (
        "Latency metrics are on the Grafana dashboard at https://grafana.example/latency.",
        "reference",
    ),
    ("The API contract is documented at https://docs.example/api/v2.", "reference"),
)

# --- SLUG item bank ---------------------------------------------------------
# Letters-and-spaces titles only, so kebab_slug is a clean lowercase-join. Gold
# is computed from kebab_slug at build time (never hand-transcribed).
_SLUG_TITLES: tuple[str, ...] = (
    "Prefers Tabs Over Spaces",
    "Run Tests Before Done",
    "Quarterly Data Migration Plan",
    "Ops Runbook Location",
    "User Identity And Role",
    "Never Push To Main",
    "Payments API Redesign",
    "High Contrast Output Preference",
    "Mobile App Release Window",
    "Billing Service Extraction",
    "Concise Responses Always",
    "Latency Dashboard Pointer",
)


def _make_type_question(seed: int, index: int, memory: str, gold: str) -> PackQuestion:
    prompt = (
        f"Memory: {memory}\n\n"
        "Which memory `type` is this? Reply with exactly one of: "
        "user, feedback, project, reference."
    )
    return PackQuestion(
        question_id=f"{PACK_ID}-s{seed}-{index}",
        prompt=prompt,
        answer=gold,
        answer_source="librarian:write-entry",
        choices=None,
        tags=("librarian", "write-entry", "type"),
        accept=(),
    )


def _make_slug_question(seed: int, index: int, title: str) -> PackQuestion:
    gold = kebab_slug(title)
    prompt = f"Give the kebab-case slug for this title: '{title}'. Reply with the slug only."
    return PackQuestion(
        question_id=f"{PACK_ID}-s{seed}-{index}",
        prompt=prompt,
        answer=gold,
        answer_source="librarian:write-entry",
        choices=None,
        tags=("librarian", "write-entry", "slug"),
        accept=(),
    )


def build(seed: int = 0) -> QuestionPack:
    """Build the deterministic ``librarian-write-entry`` pack for ``seed``."""
    rng: random.Random = make_rng(seed)

    # Deterministically choose how many of each kind, totalling 10..16.
    total = rng.randint(10, 16)
    n_type = rng.randint(5, total - 5)  # leave >= 5 slugs; both kinds present
    n_slug = total - n_type

    type_bank = list(_TYPE_BANK)
    rng.shuffle(type_bank)
    chosen_types = type_bank[:n_type]

    slug_titles = list(_SLUG_TITLES)
    rng.shuffle(slug_titles)
    chosen_slugs = slug_titles[:n_slug]

    specs: list[tuple[str, object]] = [("type", item) for item in chosen_types]
    specs += [("slug", title) for title in chosen_slugs]
    rng.shuffle(specs)

    questions: list[PackQuestion] = []
    for index, (kind, payload) in enumerate(specs):
        if kind == "type":
            memory, gold = cast("tuple[str, str]", payload)
            questions.append(_make_type_question(seed, index, memory, gold))
        else:
            title = cast("str", payload)
            questions.append(_make_slug_question(seed, index, title))

    return QuestionPack(
        pack_id=PACK_ID,
        title="Librarian: write schema-conformant memory metadata",
        tier="librarian",
        answer_type=AnswerType.EXACT,
        system_prompt=LIBRARIAN_SYSTEM_PROMPT + _SYSTEM_HINT,
        questions=tuple(questions),
    )
