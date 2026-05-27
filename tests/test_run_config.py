from gguf_limit_bench.run_config import PRESETS, RunConfig, RunStatus


def test_beginner_presets_have_plain_language_and_safe_caps():
    assert PRESETS["quick"].budget_minutes <= 5
    assert PRESETS["quick"].max_attempts and PRESETS["quick"].max_attempts > 1
    assert PRESETS["quick"].context_ladder == ()
    assert PRESETS["normal"].context_ladder == (4096, 8192, 16384, 32768)
    assert PRESETS["deep"].budget_minutes == 20
    assert PRESETS["deep"].context_ladder[-1] == 131072
    assert PRESETS["deep"].adaptive is True
    assert "Serious agent pilot test" in PRESETS["deep"].description


def test_run_config_can_be_built_from_preset_and_mark_extensions():
    config = RunConfig.from_preset("normal")

    assert config.status == RunStatus.PENDING
    assert config.context_ladder == (4096, 8192, 16384, 32768)
    assert config.min_generation_tps == 20.0
    assert config.min_ttft_target_ms == 10_000
    assert config.should_allow_extension(unfinished_required_pack=True, healthy=True) is True
    assert config.should_allow_extension(unfinished_required_pack=False, healthy=True) is False
    assert config.should_allow_extension(unfinished_required_pack=True, healthy=False) is False
