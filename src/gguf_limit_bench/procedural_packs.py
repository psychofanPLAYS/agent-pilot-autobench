"""RULER-style procedural long-context task generators (vendored, no deps).

Adapted from the task *design* of NVIDIA RULER ("What's the Real Context Size of
Your Long-Context Language Models?", Apache-2.0): synthetic needle-in-context
retrieval and multi-hop variable tracking. We re-implement the generators with
neutral local filler (no third-party corpus) so the tasks are:

- **uncontaminated** — generated per run from a seed, never a memorizable fixture;
- **length-controllable** — sized to a target context so we can measure quality
  falloff across 16k / 64k / 128k / 256k;
- **exactly gradeable** — answers are recovered with the existing EXACT scorer.

Pure functions: fully unit-testable without a server.
"""

from __future__ import annotations

import random

from gguf_limit_bench.packs import AnswerType, PackQuestion, QuestionPack

# Charset with no vowels and no 0/1 -> codes are alphanumeric, unambiguous, and
# can never accidentally spell a number-word the EXACT scorer would rewrite.
_CODE_ALPHABET = "BCDFGHJKLMNPQRSTVWXYZ23456789"
_CODE_LEN = 6

_TOPICS = (
    "harbor",
    "glacier",
    "meadow",
    "canyon",
    "lantern",
    "orchard",
    "comet",
    "reef",
    "summit",
    "delta",
)

# Neutral filler with no digits, no "VAR", no "code" — so it never collides with
# an embedded needle code or a variable-tracking assignment.
_FILLER_SENTENCES = (
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

_SYSTEM_PROMPT = (
    "You are reading a long document. Read the whole document carefully, then "
    "answer the question using only what the document states. Take as much "
    "reasoning as you need, and end with a line exactly of the form "
    "'Final Answer: X' where X is the requested value."
)


def approx_token_count(text: str) -> int:
    """Heuristic token count (~4 characters per token). Deterministic, no tokenizer."""
    return round(len(text) / 4)


def _make_code(rng: random.Random) -> str:
    return "".join(rng.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))


def _build_filler(rng: random.Random, char_budget: int) -> list[str]:
    sentences: list[str] = []
    length = 0
    while length < char_budget:
        sentence = rng.choice(_FILLER_SENTENCES)
        sentences.append(sentence)
        length += len(sentence) + 1
    return sentences


def generate_needle_single(
    *, target_tokens: int, depth_fraction: float = 0.5, seed: int = 0
) -> PackQuestion:
    """A single needle-in-context retrieval task sized to ``target_tokens``."""
    rng = random.Random(seed)
    code = _make_code(rng)
    topic = rng.choice(_TOPICS)
    needle = f"The secret access code for the {topic} vault is {code}."
    question_line = (
        f"Question: What is the secret access code for the {topic} vault? Reply with the code only."
    )
    char_budget = max(0, target_tokens * 4 - len(needle) - len(question_line) - 4)
    sentences = _build_filler(rng, char_budget)
    index = min(len(sentences), max(0, round(depth_fraction * len(sentences))))
    sentences.insert(index, needle)
    haystack = " ".join(sentences)
    prompt = f"{haystack}\n\n{question_line}"
    return PackQuestion(
        question_id=f"ruler-needle-s{seed}-t{target_tokens}-d{int(depth_fraction * 100)}",
        prompt=prompt,
        answer=code,
        answer_source="procedural:ruler-needle-single",
        choices=None,
        tags=(
            "ruler",
            "needle_single",
            f"depth={depth_fraction}",
            f"target_tokens={target_tokens}",
        ),
        accept=(),
    )


def generate_variable_tracking(*, target_tokens: int, hops: int = 4, seed: int = 0) -> PackQuestion:
    """A multi-hop variable-tracking task: resolve VAR_hops back to its value."""
    rng = random.Random(seed)
    value = rng.randint(10000, 99999)
    assignments = [f"VAR_1 = {value}."]
    assignments += [f"VAR_{i} = VAR_{i - 1}." for i in range(2, hops + 1)]
    question_line = (
        f"Question: What is the numeric value of VAR_{hops}? Reply with the number only."
    )
    assign_chars = sum(len(a) + 1 for a in assignments)
    char_budget = max(0, target_tokens * 4 - assign_chars - len(question_line) - 4)
    sentences = _build_filler(rng, char_budget)
    base_positions = [round((k + 0.5) / hops * len(sentences)) for k in range(hops)]
    for offset, (pos, assignment) in enumerate(zip(base_positions, assignments)):
        sentences.insert(pos + offset, assignment)
    haystack = " ".join(sentences)
    prompt = f"{haystack}\n\n{question_line}"
    return PackQuestion(
        question_id=f"ruler-vartrack-s{seed}-t{target_tokens}-h{hops}",
        prompt=prompt,
        answer=str(value),
        answer_source="procedural:ruler-variable-tracking",
        choices=None,
        tags=("ruler", "variable_tracking", f"hops={hops}", f"target_tokens={target_tokens}"),
        accept=(),
    )


def build_long_context_pack(
    *,
    target_tokens: int,
    count: int = 4,
    seed: int = 0,
    depths: tuple[float, ...] = (0.1, 0.35, 0.6, 0.85),
) -> QuestionPack:
    """Assemble an EXACT-scored long-context pack: needles at spread depths plus
    one variable-tracking task (when count > 1)."""
    questions: list[PackQuestion] = []
    for i in range(count):
        if count > 1 and i == count - 1:
            questions.append(
                generate_variable_tracking(target_tokens=target_tokens, hops=5, seed=seed + i)
            )
        else:
            depth = depths[i % len(depths)]
            questions.append(
                generate_needle_single(
                    target_tokens=target_tokens, depth_fraction=depth, seed=seed + i
                )
            )
    return QuestionPack(
        pack_id=f"ruler-longctx-{target_tokens}",
        title=f"RULER long-context @ ~{target_tokens} tokens",
        tier="long-context",
        answer_type=AnswerType.EXACT,
        system_prompt=_SYSTEM_PROMPT,
        questions=tuple(questions),
    )
