from gguf_limit_bench.gguf_metadata import ModelArch
from gguf_limit_bench.vram import (
    detect_vram_mb,
    kv_cache_bytes,
    max_fitting_context,
    plan_context_fit,
)


def _arch(**overrides) -> ModelArch:
    base = dict(
        architecture="test",
        n_layers=32,
        n_heads=32,
        n_heads_kv=8,
        embedding_length=4096,
        key_length=128,
        value_length=128,
        train_context_length=8192,
    )
    base.update(overrides)
    return ModelArch(**base)


def test_kv_cache_bytes_uses_kv_heads_and_lengths():
    arch = _arch(n_layers=2, n_heads_kv=1, key_length=512, value_length=512)
    # per token = 1 * 512 * 2 bytes (k) + 1 * 512 * 2 (v) = 2048 B; * 4 ctx * 2 layers
    assert kv_cache_bytes(arch, 4, k_bits=16, v_bits=16) == 2048 * 4 * 2


def test_kv_cache_bytes_does_not_scale_with_parallel_slots():
    # KV is sized by total ctx-size; there is no parallel multiplier in the API.
    arch = _arch()
    assert kv_cache_bytes(arch, 8192) > 0


def test_q8_kv_is_half_of_f16():
    arch = _arch()
    f16 = kv_cache_bytes(arch, 16384, k_bits=16, v_bits=16)
    q8 = kv_cache_bytes(arch, 16384, k_bits=8, v_bits=8)
    assert q8 * 2 == f16


def test_plan_context_fit_skips_tiers_over_budget():
    arch = _arch(n_layers=40, n_heads_kv=8, key_length=128, value_length=128)
    size_bytes = 5 * 1024**3  # 5 GB weights
    contexts = [16384, 65536, 262144]

    plan = plan_context_fit(arch, size_bytes, contexts, budget_mb=12000, k_bits=16, v_bits=16)

    by_ctx = {fit.context_size: fit for fit in plan}
    assert by_ctx[16384].fits is True
    assert by_ctx[262144].fits is False
    assert "budget" in by_ctx[262144].reason


def test_max_fitting_context_returns_largest_or_none():
    arch = _arch()
    size_bytes = 4 * 1024**3
    plan = plan_context_fit(arch, size_bytes, [16384, 32768], budget_mb=100000)
    assert max_fitting_context(plan) == 32768

    impossible = plan_context_fit(arch, size_bytes, [262144], budget_mb=10)
    assert max_fitting_context(impossible) is None


def test_detect_vram_parses_nvidia_smi():
    def fake_runner(*args, **kwargs):
        class _Done:
            returncode = 0
            stdout = "24564, 22695\n"
            stderr = ""

        return _Done()

    info = detect_vram_mb(runner=fake_runner)
    assert info is not None
    assert info.total_mb == 24564
    assert info.free_mb == 22695


def test_detect_vram_returns_none_when_nvidia_smi_missing():
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    assert detect_vram_mb(runner=boom) is None
