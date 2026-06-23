"""VRAM-headroom guard for the context ladder.

Pushing context toward 256k on a 24 GB card can OOM-crash llama-server, which
wastes a benchmark round. This module estimates how much VRAM a model + KV
cache needs at each context tier and plans which tiers are worth attempting.

The KV estimate is a **dense upper bound**: it assumes every layer keeps the
full context (real K/V dims and GQA head count are used, but sliding-window
attention and shared-KV layers are NOT modelled yet, so SWA models will use
*less* than predicted). Erring high keeps the guard safe — it will skip a tier
before it OOMs, never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
import subprocess

from gguf_limit_bench.gguf_metadata import ModelArch

_MB = 1024 * 1024


@dataclass(frozen=True)
class VramInfo:
    total_mb: int
    free_mb: int


def detect_vram_mb(runner=subprocess.run) -> VramInfo | None:
    """Best-effort total/free VRAM of the primary GPU via nvidia-smi (MB).

    Returns None when nvidia-smi is unavailable or fails. Never raises.
    """
    try:
        completed = runner(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    try:
        total_str, free_str = (part.strip() for part in line[0].split(","))
        return VramInfo(total_mb=int(float(total_str)), free_mb=int(float(free_str)))
    except (ValueError, IndexError):
        return None


def _bytes_per_element(kv_bits: int) -> float:
    # q8_0 is ~1.06 B/elem incl. scales; treat as 1 for an estimate. f16 = 2.
    return kv_bits / 8.0


def kv_cache_bytes(
    arch: ModelArch,
    context_size: int,
    *,
    k_bits: int = 16,
    v_bits: int = 16,
) -> int:
    """KV-cache size in bytes for a given context and arch.

    ``context_size`` is the total llama-server ``--ctx-size``; with
    ``--parallel N`` that total is shared across slots, so the KV cache is sized
    by the total context, not multiplied by the slot count.

    For sliding-window models (Gemma 3/4 etc.) only the global layers store the
    full context — recent llama.cpp's interleaved-SWA cache caps the windowed
    layers at the sliding window with their smaller K/V dims, which is a large
    memory saving at long context. Dense models size every layer at full context.
    """
    k_bytes = _bytes_per_element(k_bits)
    v_bytes = _bytes_per_element(v_bits)
    heads = arch.n_heads_kv

    def layer_bytes(tokens: int, key_len: int, value_len: int) -> float:
        return (heads * key_len * k_bytes + heads * value_len * v_bytes) * tokens

    if arch.is_sliding_window:
        window_tokens = min(context_size, arch.sliding_window)
        global_bytes = arch.n_global_layers * layer_bytes(
            context_size, arch.key_length, arch.value_length
        )
        swa_bytes = arch.n_swa_layers * layer_bytes(
            window_tokens, arch.key_length_swa, arch.value_length_swa
        )
        return int(global_bytes + swa_bytes)

    return int(arch.n_layers * layer_bytes(context_size, arch.key_length, arch.value_length))


def model_weights_mb(size_bytes: int) -> int:
    """Approximate weight VRAM as the on-disk size (full GPU offload)."""
    return int(size_bytes / _MB)


@dataclass(frozen=True)
class ContextFit:
    context_size: int
    needed_mb: int
    fits: bool
    reason: str


def plan_context_fit(
    arch: ModelArch,
    size_bytes: int,
    contexts: list[int],
    budget_mb: int,
    *,
    k_bits: int = 16,
    v_bits: int = 16,
    overhead_mb: int = 1024,
    headroom_mb: int = 1024,
) -> list[ContextFit]:
    """Decide which context tiers fit within ``budget_mb`` (e.g. total VRAM).

    ``needed = weights + kv(context) + overhead``; a tier *fits* when
    ``needed + headroom <= budget``. The headroom leaves room for fragmentation
    and other processes so the guard stays conservative.
    """
    weights = model_weights_mb(size_bytes)
    plan: list[ContextFit] = []
    for context in sorted(contexts):
        kv_mb = int(kv_cache_bytes(arch, context, k_bits=k_bits, v_bits=v_bits) / _MB)
        needed = weights + kv_mb + overhead_mb
        fits = needed + headroom_mb <= budget_mb
        if fits:
            reason = f"~{needed} MB needed (weights {weights} + kv {kv_mb} + {overhead_mb})"
        else:
            reason = (
                f"~{needed} MB needed > {budget_mb} MB budget "
                f"(kv {kv_mb} MB at {context // 1024}k)"
            )
        plan.append(ContextFit(context_size=context, needed_mb=needed, fits=fits, reason=reason))
    return plan


def max_fitting_context(plan: list[ContextFit]) -> int | None:
    """Largest context tier predicted to fit, or None if none fit."""
    fitting = [fit.context_size for fit in plan if fit.fits]
    return max(fitting) if fitting else None
