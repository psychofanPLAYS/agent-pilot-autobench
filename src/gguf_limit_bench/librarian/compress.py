"""Librarian job: faithful compression of a memory note.

A coding agent often asks the librarian to *summarize* a memory note so it fits a
tight context budget. A good summary preserves every load-bearing fact and adds
nothing false. This job measures exactly that judgement.

Each question presents a SOURCE note containing ``K`` (3 or 4) load-bearing facts
plus neutral filler, and four candidate summaries (choices A-D). Exactly one
summary is *faithful* — it mentions all ``K`` facts and introduces nothing false.
The other three are flawed:

* one **drops** a fact (omits a planted clause);
* one **adds** a hallucinated fact not present in the source;
* one **drops and adds** (both faults at once).

The model picks the faithful summary. Graded by the existing ``MULTIPLE_CHOICE``
scorer (the model emits ``Final Answer: D``). Pure and seed-deterministic:
:func:`build` called twice with the same seed returns byte-identical questions.
All randomness flows through
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

PACK_ID = "librarian-compress"

# A "fact" is an atomic clause paired with a *false* counterpart that contradicts
# it. The clause is written so a plain substring check verifies its presence in a
# summary, and the false variant shares no incidental substring with the true one
# (the contradicting token, e.g. ``eu-west-2`` vs ``us-east-1``, is the marker).
#
# Each scenario is a self-contained note theme with its own pool of facts. We
# plant K facts from one scenario per question so the summaries read coherently.
_Fact = tuple[str, str]  # (true_clause, false_clause)

# (scenario_name, facts) — every scenario carries >=4 facts so we can plant K=3 or 4.
_SCENARIOS: tuple[tuple[str, tuple[_Fact, ...]], ...] = (
    (
        "deploy",
        (
            ("deploys run on Fridays", "deploys run on Mondays"),
            ("the prod DB is in us-east-1", "the prod DB is in eu-west-2"),
            ("rollbacks need two approvals", "rollbacks need four approvals"),
            ("the on-call rotation is weekly", "the on-call rotation is daily"),
        ),
    ),
    (
        "project",
        (
            ("the default branch is named trunk", "the default branch is named master"),
            ("the backend is written in Go", "the backend is written in Rust"),
            ("CI runs on every pull request", "CI runs only nightly"),
            ("the API returns snake_case keys", "the API returns camelCase keys"),
        ),
    ),
    (
        "user",
        (
            ("the user prefers tabs over spaces", "the user prefers spaces over tabs"),
            ("the user goes by Dana", "the user goes by Morgan"),
            ("the user works in US Eastern time", "the user works in US Pacific time"),
            ("the user wants concise replies", "the user wants verbose replies"),
        ),
    ),
)

_K_VALUES: tuple[int, ...] = (3, 4)

_PROMPT_TEMPLATE = (
    "You are compressing a memory note for the agent. A faithful summary must "
    "preserve EVERY load-bearing fact in the source and introduce nothing that "
    "the source does not state.\n\n"
    "SOURCE note:\n"
    "{source}\n\n"
    "Candidate summaries:\n"
    "A. {choice_a}\n"
    "B. {choice_b}\n"
    "C. {choice_c}\n"
    "D. {choice_d}\n\n"
    "Choose the single summary that preserves all the facts and adds nothing "
    "false. Answer with its letter."
)


def _join_clauses(clauses: list[str]) -> str:
    """Render fact clauses as one summary sentence: 'a; b; c.'"""
    return "; ".join(clauses) + "."


def _build_source(rng: random.Random, facts: list[_Fact]) -> str:
    """Interleave the K true clauses with neutral filler into the source note."""
    true_clauses = [f"Note: {true}." for true, _false in facts]
    filler = build_filler(rng, char_budget=rng.choice((160, 220, 280)))
    parts: list[str] = true_clauses + filler
    rng.shuffle(parts)
    return " ".join(parts)


def _build_summaries(
    rng: random.Random, facts: list[_Fact], extra_false: str
) -> tuple[tuple[str, ...], str]:
    """Return (choices, gold_letter).

    Four summaries are built then placed at deterministic letter positions:

    * faithful  — all K true clauses, nothing false;
    * drop      — omits one true clause (a randomly chosen one);
    * add       — all K true clauses plus one hallucinated false clause;
    * drop_add  — omits one true clause and adds one false clause.
    """
    true_clauses = [true for true, _false in facts]

    drop_idx = rng.randrange(len(facts))
    dropped = [c for i, c in enumerate(true_clauses) if i != drop_idx]

    faithful = _join_clauses(list(true_clauses))
    drop = _join_clauses(dropped)
    add = _join_clauses(list(true_clauses) + [extra_false])
    drop_add = _join_clauses(dropped + [extra_false])

    # Place the faithful summary at a seed-chosen letter; fill the other slots
    # with the flawed variants in their fixed relative order (deterministic).
    gold_pos = rng.randrange(4)
    flawed = [drop, add, drop_add]
    slots: list[str | None] = [None, None, None, None]
    slots[gold_pos] = faithful
    flawed_iter = iter(flawed)
    for i in range(4):
        if slots[i] is None:
            slots[i] = next(flawed_iter)
    choices = tuple(s for s in slots if s is not None)
    gold_letter = chr(ord("A") + gold_pos)
    return choices, gold_letter


def _pick_extra_false(
    rng: random.Random, scenario_facts: tuple[_Fact, ...], used: list[_Fact]
) -> str:
    """A hallucinated false clause: the false variant of a scenario fact NOT planted."""
    used_set = set(used)
    candidates = [false for fact in scenario_facts if fact not in used_set for _t, false in (fact,)]
    if candidates:
        return rng.choice(candidates)
    # Fallback (should not happen given >=4 facts and K<=4 minus at least one held
    # out below): use the false variant of a planted fact — still source-absent.
    return rng.choice([false for _t, false in used])


def build(seed: int = 0) -> QuestionPack:
    """Build the ``librarian-compress`` pack for ``seed``.

    Deterministic: the same ``seed`` always yields byte-identical questions.
    Produces 10..16 MULTIPLE_CHOICE questions, each with exactly four summaries.
    """
    rng = make_rng(seed)

    count = rng.randint(10, 16)
    questions: list[PackQuestion] = []

    for i in range(count):
        scenario_name, scenario_facts = _SCENARIOS[rng.randrange(len(_SCENARIOS))]
        k = rng.choice(_K_VALUES)
        # Hold out at least one fact so the hallucinated clause is genuinely
        # source-absent: never plant all four when K could equal the pool size.
        max_plant = min(k, len(scenario_facts) - 1)
        planted = rng.sample(scenario_facts, max_plant)

        extra_false = _pick_extra_false(rng, scenario_facts, planted)
        source = _build_source(rng, planted)
        choices, gold_letter = _build_summaries(rng, planted, extra_false)

        prompt = _PROMPT_TEMPLATE.format(
            source=source,
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3],
        )
        questions.append(
            PackQuestion(
                question_id=f"{PACK_ID}-s{seed}-{i}",
                prompt=prompt,
                answer=gold_letter,
                answer_source="librarian:compress",
                choices=choices,
                tags=("librarian", "compress", f"k_facts={max_plant}", scenario_name),
                accept=(),
            )
        )

    return QuestionPack(
        pack_id=PACK_ID,
        title="Librarian: faithful compression",
        tier="librarian",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        system_prompt=LIBRARIAN_SYSTEM_PROMPT,
        questions=tuple(questions),
    )
