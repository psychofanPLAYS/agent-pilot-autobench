from gguf_limit_bench.programs import (
    FIT_ASCENT_STEP,
    FIT_BACKOFF_STEP,
    FIT_REFINE_STEP,
    FIT_START_CONTEXT_SIZE,
    INTELLIGENCE_CONTEXT_SIZE,
    MIN_SPEED_CONTEXT_SIZE,
    ProgramId,
    enforce_min_context,
    fit_probe_prompt,
    program_by_id,
    speed_probe_prompt,
)


def test_speed_program_never_uses_less_than_16k():
    program = program_by_id(ProgramId.SPEED)

    assert program.min_context_size == MIN_SPEED_CONTEXT_SIZE
    assert program.min_context_size == 16_384
    assert program.default_context_size == 16_384
    assert program.asks_questions is False


def test_intelligence_program_uses_64k_fresh_question_windows():
    program = program_by_id(ProgramId.INTELLIGENCE)

    assert program.default_context_size == INTELLIGENCE_CONTEXT_SIZE
    assert program.default_context_size == 65_536
    assert program.one_question_per_window is True
    assert program.unlimited_thinking is True


def test_speed_prompt_is_repeatable_and_not_a_question_pack():
    program = program_by_id("speed")
    prompt = speed_probe_prompt()

    assert program.prompt_kind == "repeatable-generation"
    assert "500 word" in prompt.lower()
    assert "poem" in prompt.lower()
    assert "same text every run" in prompt.lower()


def test_program_context_floor_is_enforced():
    assert enforce_min_context(4096, ProgramId.SPEED) == 16_384
    assert enforce_min_context(16_384, ProgramId.SPEED) == 16_384
    assert enforce_min_context(4096, ProgramId.INTELLIGENCE) == 65_536


def test_fit_program_starts_at_32k_and_refines_after_oom():
    program = program_by_id(ProgramId.FIT)

    assert program.min_context_size == 16_384
    assert program.default_context_size == FIT_START_CONTEXT_SIZE == 32_768
    assert FIT_ASCENT_STEP == 32_768
    assert FIT_BACKOFF_STEP == 16_384
    assert FIT_REFINE_STEP == 8_192
    assert program.prompt_kind == "graded-fit-probe"


def test_fit_probe_prompt_is_gradable_and_not_a_tiny_readiness_ping():
    prompt = fit_probe_prompt()

    assert "FIT PROBE COMPLETE" in prompt
    assert "three numbered observations" in prompt
    assert "two risks" in prompt
