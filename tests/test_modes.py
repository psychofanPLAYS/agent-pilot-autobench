import pytest

from gguf_limit_bench.evaluation_mode import EvaluationMode
from gguf_limit_bench.modes import (
    DEFAULT_RUN_MODE,
    RUN_MODES,
    mode_by_id,
    next_mode,
    previous_mode,
)


def test_default_mode_is_find_best_settings():
    assert DEFAULT_RUN_MODE.id == "best_settings"
    assert DEFAULT_RUN_MODE.evaluation is EvaluationMode.BENCHMARK


def test_quick_mode_is_a_speed_scout():
    quick = mode_by_id("quick")
    assert quick.evaluation is EvaluationMode.SPEED_SCOUT
    assert quick.context_ladder == ()


def test_context_modes_carry_a_ladder():
    assert mode_by_id("context_limits").context_ladder[0] == 4096
    assert mode_by_id("deep").context_ladder[-1] == 131072


def test_mode_cycling_wraps_around():
    first = RUN_MODES[0]
    last = RUN_MODES[-1]
    assert next_mode(last) is first
    assert previous_mode(first) is last
    assert previous_mode(next_mode(first)) is first


def test_mode_by_id_rejects_unknown():
    with pytest.raises(KeyError):
        mode_by_id("does-not-exist")
