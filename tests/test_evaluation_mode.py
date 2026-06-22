from gguf_limit_bench.evaluation_mode import (
    EvaluationMode,
    asks_questions,
    resolve_evaluation_mode,
)


def test_default_is_benchmark():
    assert resolve_evaluation_mode(speed_scout=False, flag_ladder=False) is EvaluationMode.BENCHMARK


def test_speed_scout_opt_out():
    assert resolve_evaluation_mode(speed_scout=True, flag_ladder=False) is EvaluationMode.SPEED_SCOUT


def test_flag_ladder_forces_benchmark():
    assert resolve_evaluation_mode(speed_scout=True, flag_ladder=True) is EvaluationMode.BENCHMARK


def test_benchmark_asks_questions():
    assert asks_questions(EvaluationMode.BENCHMARK) is True
    assert asks_questions(EvaluationMode.SPEED_SCOUT) is False
