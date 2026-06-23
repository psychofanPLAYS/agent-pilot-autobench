from __future__ import annotations

from gguf_limit_bench.gpu_profiles import describe, recommended_always_on, recommended_parallel


class TestRecommendedAlwaysOn:
    def test_4090_flags_include_flash_attn(self):
        flags = recommended_always_on("RTX 4090")
        assert "--flash-attn" in flags
        assert "on" in flags

    def test_4090_flags_include_cache_type_k(self):
        flags = recommended_always_on("NVIDIA GeForce RTX 4090")
        assert "--cache-type-k" in flags

    def test_4090_flags_include_q8_0_cache(self):
        flags = recommended_always_on("RTX 4090 24GB")
        assert "q8_0" in flags

    def test_4090_flags_include_gpu_layers_99(self):
        flags = recommended_always_on("RTX 4090")
        assert "--gpu-layers" in flags
        idx = list(flags).index("--gpu-layers")
        assert flags[idx + 1] == "99"

    def test_4090_case_insensitive(self):
        flags_lower = recommended_always_on("rtx 4090")
        flags_upper = recommended_always_on("RTX 4090")
        assert "--flash-attn" in flags_lower
        assert "--flash-attn" in flags_upper

    def test_unknown_gpu_includes_flash_attn(self):
        flags = recommended_always_on("Unknown GPU XYZ")
        assert "--flash-attn" in flags
        assert "on" in flags

    def test_unknown_gpu_includes_gpu_layers(self):
        flags = recommended_always_on("SomeBrand 3070")
        assert "--gpu-layers" in flags

    def test_unknown_gpu_no_cache_type_k(self):
        flags = recommended_always_on("Unknown GPU")
        # Conservative profile should NOT include cache-type-k (no q8_0 cache)
        assert "--cache-type-k" not in flags

    def test_returns_tuple(self):
        flags = recommended_always_on("RTX 4090")
        assert isinstance(flags, tuple)


class TestRecommendedParallel:
    def test_4090_returns_4(self):
        assert recommended_parallel("RTX 4090") == 4

    def test_4090_case_insensitive(self):
        assert recommended_parallel("rtx 4090") == 4
        assert recommended_parallel("RTX 4090 24GB") == 4

    def test_unknown_returns_1(self):
        assert recommended_parallel("Unknown GPU") == 1

    def test_other_gpu_returns_1(self):
        assert recommended_parallel("RTX 3090") == 1


class TestDescribe:
    def test_4090_describe_mentions_flash_attn(self):
        desc = describe("RTX 4090")
        assert "flash-attn" in desc.lower() or "flash_attn" in desc.lower()

    def test_4090_describe_mentions_q8_0(self):
        desc = describe("RTX 4090")
        assert "q8_0" in desc

    def test_4090_describe_mentions_parallel_slots(self):
        desc = describe("RTX 4090")
        assert "4" in desc

    def test_unknown_describe_says_not_available(self):
        desc = describe("Unknown GPU")
        assert len(desc) > 0  # Not empty
        # Should mention something about tuned recommendations
        assert any(
            word in desc.lower()
            for word in ["unknown", "generic", "tuned", "available", "conservative"]
        )

    def test_returns_string(self):
        assert isinstance(describe("RTX 4090"), str)
        assert isinstance(describe("Unknown GPU XYZ"), str)


def test_detect_gpu_name_parses_nvidia_smi(monkeypatch):

    from gguf_limit_bench import gpu_profiles

    class _R:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 4090\n"

    monkeypatch.setattr(gpu_profiles.subprocess, "run", lambda *a, **k: _R())
    assert gpu_profiles.detect_gpu_name() == "NVIDIA GeForce RTX 4090"


def test_detect_gpu_name_returns_empty_when_missing(monkeypatch):
    from gguf_limit_bench import gpu_profiles

    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu_profiles.subprocess, "run", _boom)
    assert gpu_profiles.detect_gpu_name() == ""
