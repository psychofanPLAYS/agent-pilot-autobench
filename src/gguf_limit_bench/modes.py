from __future__ import annotations

from dataclasses import dataclass

from gguf_limit_bench.evaluation_mode import EvaluationMode


# Karpathy's autoresearch program uses a fixed time budget per round. We keep his
# 5-minute round as the unit of work; a mode's budget is just a number of rounds.
KARPATHY_ROUND_MINUTES = 5
KARPATHY_ROUND_SECONDS = KARPATHY_ROUND_MINUTES * 60


@dataclass(frozen=True)
class RunMode:
    """A goal-shaped run preset the novice picks in the cockpit.

    Each mode maps a friendly intent to concrete run parameters so the user never
    has to reason about llama.cpp flags or budgets directly.
    """

    id: str
    label: str
    description: str
    budget_minutes: int
    evaluation: EvaluationMode
    context_ladder: tuple[int, ...] = ()


RUN_MODES: tuple[RunMode, ...] = (
    RunMode(
        id="quick",
        label="Quick check",
        description="Does it load, and how fast? No questions asked.",
        budget_minutes=5,
        evaluation=EvaluationMode.SPEED_SCOUT,
    ),
    RunMode(
        id="best_settings",
        label="Find best settings",
        description="Walk the flag ladder, ask the questions, crown the best settings.",
        budget_minutes=30,
        evaluation=EvaluationMode.BENCHMARK,
    ),
    RunMode(
        id="librarian_bench",
        label="Librarian bot test",
        description="Compare Gemma and Qwen as local memory/RAG workers for coding agents.",
        budget_minutes=30,
        evaluation=EvaluationMode.BENCHMARK,
    ),
    RunMode(
        id="flag_effect",
        label="How flags affect speed",
        description="See how each llama.cpp flag changes tok/s and TTFT for this model.",
        budget_minutes=30,
        evaluation=EvaluationMode.BENCHMARK,
    ),
    RunMode(
        id="context_limits",
        label="Context limits",
        description="How much context fits, and how long context affects tok/s.",
        budget_minutes=25,
        evaluation=EvaluationMode.BENCHMARK,
        context_ladder=(4096, 8192, 16384, 32768),
    ),
    RunMode(
        id="deep",
        label="Deep / overnight",
        description="Everything: full ladder, big budget, context ladder.",
        budget_minutes=60,
        evaluation=EvaluationMode.BENCHMARK,
        context_ladder=(4096, 8192, 16384, 32768, 65536, 131072),
    ),
    RunMode(
        id="custom",
        label="Custom (set your time)",
        description="You choose the minutes; we do the most useful work that fits.",
        budget_minutes=15,
        evaluation=EvaluationMode.BENCHMARK,
    ),
)

DEFAULT_RUN_MODE: RunMode = RUN_MODES[1]  # Find best settings


def mode_by_id(mode_id: str) -> RunMode:
    for mode in RUN_MODES:
        if mode.id == mode_id:
            return mode
    raise KeyError(f"unknown run mode: {mode_id}")


def next_mode(current: RunMode) -> RunMode:
    index = RUN_MODES.index(current)
    return RUN_MODES[(index + 1) % len(RUN_MODES)]


def previous_mode(current: RunMode) -> RunMode:
    index = RUN_MODES.index(current)
    return RUN_MODES[(index - 1) % len(RUN_MODES)]
