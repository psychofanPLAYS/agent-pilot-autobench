from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


MIN_SERIOUS_CONTEXT_SIZE = 16_384
MIN_SPEED_CONTEXT_SIZE = MIN_SERIOUS_CONTEXT_SIZE
FIT_START_CONTEXT_SIZE = 32_768
FIT_ASCENT_STEP = 32_768
FIT_BACKOFF_STEP = 16_384
FIT_REFINE_STEP = 8_192
INTELLIGENCE_CONTEXT_SIZE = 65_536
STANDARD_KV_CACHE_TYPE = "q8_0"
LONG_CONTEXT_TIERS = (16_384, 65_536, 131_072, 262_144)


class ProgramId(StrEnum):
    FIT = "fit"
    SPEED = "speed"
    INTELLIGENCE = "intelligence"
    FLAG_ABLATION = "flag-ablation"
    LONG_CONTEXT_DROPOFF = "long-context-dropoff"
    DEEP = "deep"


@dataclass(frozen=True)
class ProgramSpec:
    id: ProgramId
    label: str
    purpose: str
    min_context_size: int
    default_context_size: int
    prompt_kind: str
    asks_questions: bool
    one_question_per_window: bool = False
    unlimited_thinking: bool = False


PROGRAMS: dict[ProgramId, ProgramSpec] = {
    ProgramId.FIT: ProgramSpec(
        id=ProgramId.FIT,
        label="Find fit",
        purpose="Climb context from 32k upward with q8_0 KV, then back off/refine after OOM.",
        min_context_size=MIN_SERIOUS_CONTEXT_SIZE,
        default_context_size=FIT_START_CONTEXT_SIZE,
        prompt_kind="graded-fit-probe",
        asks_questions=False,
    ),
    ProgramId.SPEED: ProgramSpec(
        id=ProgramId.SPEED,
        label="Speed probe",
        purpose="Generate the same long text at 16k+ and measure throughput, TTFT, and metrics.",
        min_context_size=MIN_SPEED_CONTEXT_SIZE,
        default_context_size=MIN_SPEED_CONTEXT_SIZE,
        prompt_kind="repeatable-generation",
        asks_questions=False,
    ),
    ProgramId.INTELLIGENCE: ProgramSpec(
        id=ProgramId.INTELLIGENCE,
        label="Intelligence",
        purpose="Ask benchmark questions one at a time in fresh 64k windows.",
        min_context_size=INTELLIGENCE_CONTEXT_SIZE,
        default_context_size=INTELLIGENCE_CONTEXT_SIZE,
        prompt_kind="question-pack",
        asks_questions=True,
        one_question_per_window=True,
        unlimited_thinking=True,
    ),
    ProgramId.FLAG_ABLATION: ProgramSpec(
        id=ProgramId.FLAG_ABLATION,
        label="Flag ablation",
        purpose="Keep standard flags on, then change one variable at a time.",
        min_context_size=MIN_SERIOUS_CONTEXT_SIZE,
        default_context_size=MIN_SERIOUS_CONTEXT_SIZE,
        prompt_kind="repeatable-generation",
        asks_questions=False,
    ),
    ProgramId.LONG_CONTEXT_DROPOFF: ProgramSpec(
        id=ProgramId.LONG_CONTEXT_DROPOFF,
        label="Long-context dropoff",
        purpose="Run matched checks across increasing contexts to measure quality and speed dropoff.",
        min_context_size=MIN_SERIOUS_CONTEXT_SIZE,
        default_context_size=INTELLIGENCE_CONTEXT_SIZE,
        prompt_kind="question-pack",
        asks_questions=True,
        one_question_per_window=True,
        unlimited_thinking=True,
    ),
    ProgramId.DEEP: ProgramSpec(
        id=ProgramId.DEEP,
        label="Deep campaign",
        purpose="Run fit, speed, ablation, intelligence, and long-context dropoff as one campaign.",
        min_context_size=MIN_SERIOUS_CONTEXT_SIZE,
        default_context_size=INTELLIGENCE_CONTEXT_SIZE,
        prompt_kind="campaign",
        asks_questions=True,
        one_question_per_window=True,
        unlimited_thinking=True,
    ),
}


def program_by_id(program_id: ProgramId | str) -> ProgramSpec:
    return PROGRAMS[ProgramId(program_id)]


def enforce_min_context(context_size: int, program_id: ProgramId | str) -> int:
    program = program_by_id(program_id)
    return max(context_size, program.min_context_size)


def speed_probe_prompt() -> str:
    return (
        "Generate the same text every run: write a 500 word poem about a local AI "
        "workbench tuning a model through measured experiments. Use plain language, "
        "complete sentences, and no markdown table. Do not stop early."
    )


def fit_probe_prompt() -> str:
    return (
        "Read this task carefully and answer in a structured way. Write a short field "
        "report about a benchmark tool finding the largest usable context window. "
        "Include exactly three numbered observations, exactly two risks, and end with "
        "the line 'FIT PROBE COMPLETE'."
    )
