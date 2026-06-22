from __future__ import annotations

from enum import StrEnum


class EvaluationMode(StrEnum):
    """How a run measures a model.

    BENCHMARK asks and scores real questions through llama-server.
    SPEED_SCOUT runs the fast synthetic llama-bench probe and asks nothing.
    """

    BENCHMARK = "benchmark"
    SPEED_SCOUT = "speed_scout"


def resolve_evaluation_mode(*, speed_scout: bool, flag_ladder: bool) -> EvaluationMode:
    """Benchmark (asks questions) is the default; ``--speed-scout`` opts out.

    An explicit ``--flag-ladder`` always means benchmark, even if ``speed_scout``
    is set, so the legacy flag keeps working.
    """
    if flag_ladder:
        return EvaluationMode.BENCHMARK
    return EvaluationMode.SPEED_SCOUT if speed_scout else EvaluationMode.BENCHMARK


def asks_questions(mode: EvaluationMode) -> bool:
    return mode is EvaluationMode.BENCHMARK
