from pathlib import Path

from gguf_limit_bench.hf_recommended_settings import (
    recommended_sampler_flags,
    recommended_sampler_presets,
    sampling_payload_from_server_args,
)


def test_qwen_researched_settings_emit_sampler_flags():
    model = Path("G:/AI/models/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf")

    flags = recommended_sampler_flags(model)

    assert flags[:2] == ("--temp", "1")
    assert "--top-p" in flags
    assert "--top-k" in flags
    assert "--presence-penalty" in flags


def test_named_hf_presets_are_available_for_preflight_selection():
    model = Path("G:/AI/models/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf")

    names = [preset["name"] for preset in recommended_sampler_presets(model)]

    assert "thinking_general" in names
    assert "thinking_precise_coding" in names
    assert "non_thinking_instruct" in names


def test_sampler_payload_reads_server_args_for_question_engine():
    payload = sampling_payload_from_server_args(
        ("--temp", "0.6", "--top-p=0.95", "--top-k", "20", "--repeat-penalty", "1.05")
    )

    assert payload == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.05,
    }
