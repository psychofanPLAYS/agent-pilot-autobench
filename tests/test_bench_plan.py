from pathlib import Path

from gguf_limit_bench.bench_plan import BenchProfile, build_llama_bench_command


def test_build_quick_llama_bench_command_uses_jsonl_and_safe_repetitions():
    command = build_llama_bench_command(
        llama_bench=Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-bench.exe"),
        model=Path("G:/AI/models/example.Q4_K_M.gguf"),
        profile=BenchProfile.quick(),
    )

    assert command[:2] == [
        "G:\\AI\\llamaCPP-server\\_internal\\runtime\\llama.cpp\\llama-bench.exe",
        "--model",
    ]
    assert "-o" in command
    assert "jsonl" in command
    assert "-r" in command
    assert "1" in command
    assert "-pg" in command
    assert "512,128" in command
    assert "-fa" in command
    assert "1" in command


def test_limit_profile_expands_context_depths_without_unbounded_values():
    profile = BenchProfile.limit(max_depth=32768)

    assert profile.depths == [0, 4096, 8192, 16384, 32768]
    assert max(profile.depths) == 32768

