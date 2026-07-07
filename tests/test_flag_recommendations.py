import json
from pathlib import Path

from gguf_limit_bench.flag_recommendations import write_flag_recommendations


def test_chat_model_recommendations_start_at_128k_and_include_200k_mode(tmp_path):
    model = tmp_path / "gemma-4-26B-A4B-it-uncensored-Q4_K_M.gguf"
    model.touch()

    result = write_flag_recommendations(
        model=model,
        llama_server=Path("llama-server.exe"),
        output_dir=tmp_path / "out",
        gpu_name="NVIDIA GeForce RTX 4090",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    profiles = {profile["id"]: profile for profile in payload["profiles"]}
    assert profiles["bare_minimum"]["context_size"] == 131_072
    assert profiles["standard"]["context_size"] == 131_072
    assert profiles["long_agent"]["context_size"] == 200_000
    assert profiles["over_the_top"]["context_size"] == 262_144
    assert "--cache-type-k" not in profiles["bare_minimum"]["command"]
    assert "--cache-type-k" in profiles["standard"]["command"]
    assert "q8_0" in profiles["standard"]["command"]
    assert profiles["standard"]["kv_cache"]["v"] == "q8_0"
    assert profiles["long_agent"]["kv_cache"]["k"] == "q8_0"
    assert profiles["long_agent"]["kv_cache"]["v"] == "q8_0"
    v_cache_index = profiles["long_agent"]["command"].index("--cache-type-v")
    assert profiles["long_agent"]["command"][v_cache_index + 1] == "q8_0"
    _assert_all_profiles_pair_kv_cache(profiles.values())
    assert "--jinja" in profiles["standard"]["command"]
    assert "--cache-ram" in profiles["long_agent"]["command"]
    cache_ram_index = profiles["long_agent"]["command"].index("--cache-ram")
    assert profiles["long_agent"]["command"][cache_ram_index + 1] == "0"
    assert "--ctx-checkpoints" in profiles["long_agent"]["command"]
    checkpoint_index = profiles["long_agent"]["command"].index("--ctx-checkpoints")
    assert profiles["long_agent"]["command"][checkpoint_index + 1] == "0"
    assert "over-the-top" in result.markdown_path.read_text(encoding="utf-8").lower()


def test_qe_model_recommendations_use_helper_context_and_q4_kv(tmp_path):
    model = tmp_path / "qmd-query-expansion-qwen3.5-2B.Q8_0.gguf"
    model.touch()

    result = write_flag_recommendations(
        model=model,
        llama_server=Path("llama-server.exe"),
        output_dir=tmp_path / "out",
        gpu_name="NVIDIA GeForce RTX 4090",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    profile = payload["profiles"][0]
    assert payload["lane_type"] == "query_expansion"
    assert profile["context_size"] == 20_480
    assert "--cache-type-k" in profile["command"]
    assert "q4_0" in profile["command"]
    _assert_all_profiles_pair_kv_cache(payload["profiles"])
    assert "--parallel" in profile["command"]
    assert profile["command"][profile["command"].index("--parallel") + 1] == "1"


def test_small_chat_variants_cap_over_the_top_at_128k(tmp_path):
    model = tmp_path / "gemma-4-4B-it-Q8_0.gguf"
    model.touch()

    result = write_flag_recommendations(
        model=model,
        llama_server=Path("llama-server.exe"),
        output_dir=tmp_path / "out",
        gpu_name="NVIDIA GeForce RTX 4090",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    profiles = {profile["id"]: profile for profile in payload["profiles"]}
    assert profiles["over_the_top"]["context_size"] == 131_072
    _assert_all_profiles_pair_kv_cache(profiles.values())


def test_mtp_profiles_keep_standard_parallel_at_one_and_quote_json(tmp_path):
    model = tmp_path / "Qwen3.6-27B-Native-MTP-Q5_K_M.gguf"
    model.touch()

    result = write_flag_recommendations(
        model=model,
        llama_server=Path("llama-server.exe"),
        output_dir=tmp_path / "out",
        gpu_name="NVIDIA GeForce RTX 4090",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    profiles = {profile["id"]: profile for profile in payload["profiles"]}
    assert profiles["standard"]["parallel"] == 1
    _assert_all_profiles_pair_kv_cache(profiles.values())
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert '--chat-template-kwargs \'{"enable_thinking":true' in markdown


def _assert_all_profiles_pair_kv_cache(profiles):
    for profile in profiles:
        kv_cache = profile["kv_cache"]
        assert kv_cache["k"] == kv_cache["v"]
        command = profile["command"]
        if "--cache-type-k" in command or "--cache-type-v" in command:
            assert "--cache-type-k" in command
            assert "--cache-type-v" in command
            assert (
                command[command.index("--cache-type-k") + 1]
                == command[command.index("--cache-type-v") + 1]
            )
