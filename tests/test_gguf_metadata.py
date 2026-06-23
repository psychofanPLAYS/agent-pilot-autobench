import struct

import pytest

from gguf_limit_bench.gguf_metadata import (
    GGUFFormatError,
    model_arch_from_metadata,
    read_gguf_metadata,
    read_model_arch,
)

_GGUF_MAGIC = 0x46554747


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_u32_array(key: str, values: list[int]) -> bytes:
    # type 9 = array; elem type 4 = uint32
    body = struct.pack("<I", 4) + struct.pack("<Q", len(values))
    body += b"".join(struct.pack("<I", v) for v in values)
    return _gguf_string(key) + struct.pack("<I", 9) + body


def _build_gguf(pairs: list[bytes]) -> bytes:
    header = struct.pack("<I", _GGUF_MAGIC) + struct.pack("<I", 3)
    header += struct.pack("<Q", 0)  # tensor count
    header += struct.pack("<Q", len(pairs))  # kv count
    return header + b"".join(pairs)


def _write(tmp_path, pairs):
    path = tmp_path / "model.gguf"
    path.write_bytes(_build_gguf(pairs))
    return path


def test_read_gguf_metadata_parses_scalars_and_small_arrays(tmp_path):
    path = _write(
        tmp_path,
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 32),
            # Small numeric arrays (e.g. a sliding-window pattern) are captured.
            _kv_u32_array("llama.small.array", [1, 0, 1, 1]),
            # A huge array (e.g. tokenizer vocab) is streamed past, not stored.
            _kv_u32_array("llama.big.array", list(range(5000))),
            _kv_u32("llama.embedding_length", 4096),
        ],
    )

    meta = read_gguf_metadata(path)

    assert meta["general.architecture"] == "llama"
    assert meta["llama.block_count"] == 32
    assert meta["llama.embedding_length"] == 4096
    assert meta["llama.small.array"] == [1, 0, 1, 1]
    assert meta["llama.big.array"] is None  # too large -> skipped


def test_model_arch_uses_explicit_key_value_lengths(tmp_path):
    path = _write(
        tmp_path,
        [
            _kv_string("general.architecture", "gemma4"),
            _kv_u32("gemma4.block_count", 35),
            _kv_u32("gemma4.attention.head_count", 8),
            _kv_u32("gemma4.attention.head_count_kv", 1),
            _kv_u32("gemma4.embedding_length", 1536),
            _kv_u32("gemma4.attention.key_length", 512),
            _kv_u32("gemma4.attention.value_length", 512),
            _kv_u32("gemma4.context_length", 131072),
        ],
    )

    arch = read_model_arch(path)

    assert arch is not None
    assert arch.n_layers == 35
    assert arch.n_heads_kv == 1
    # Explicit 512, NOT embedding_length / head_count (= 192).
    assert arch.key_length == 512
    assert arch.value_length == 512
    assert arch.train_context_length == 131072


def test_model_arch_falls_back_to_head_dim_when_lengths_absent():
    meta = {
        "general.architecture": "llama",
        "llama.block_count": 32,
        "llama.attention.head_count": 32,
        "llama.embedding_length": 4096,
    }

    arch = model_arch_from_metadata(meta)

    assert arch is not None
    assert arch.n_heads_kv == 32  # no GQA metadata -> equals head_count
    assert arch.key_length == 128  # 4096 / 32
    assert arch.value_length == 128


def test_model_arch_resolves_sliding_window_split_from_pattern():
    # Gemma-style pattern: [T,T,T,T,F] repeated -> 4 SWA : 1 global.
    pattern = [True, True, True, True, False] * 7  # 35 layers
    meta = {
        "general.architecture": "gemma4",
        "gemma4.block_count": 35,
        "gemma4.attention.head_count": 8,
        "gemma4.attention.head_count_kv": 1,
        "gemma4.embedding_length": 1536,
        "gemma4.attention.key_length": 512,
        "gemma4.attention.value_length": 512,
        "gemma4.attention.key_length_swa": 256,
        "gemma4.attention.value_length_swa": 256,
        "gemma4.attention.sliding_window": 512,
        "gemma4.attention.sliding_window_pattern": pattern,
    }

    arch = model_arch_from_metadata(meta)

    assert arch is not None
    assert arch.is_sliding_window is True
    assert arch.n_global_layers == 7
    assert arch.n_swa_layers == 28
    assert arch.sliding_window == 512
    assert arch.key_length_swa == 256


def test_model_arch_without_sliding_window_is_dense():
    meta = {
        "general.architecture": "llama",
        "llama.block_count": 32,
        "llama.attention.head_count": 32,
        "llama.embedding_length": 4096,
    }
    arch = model_arch_from_metadata(meta)
    assert arch is not None
    assert arch.is_sliding_window is False
    assert arch.n_swa_layers == 0


def test_model_arch_returns_none_when_required_fields_missing():
    assert model_arch_from_metadata({"general.architecture": "llama"}) is None


def test_read_gguf_metadata_rejects_non_gguf(tmp_path):
    bad = tmp_path / "not.gguf"
    bad.write_bytes(b"NOTGGUF!")
    with pytest.raises(GGUFFormatError):
        read_gguf_metadata(bad)


def test_read_model_arch_returns_none_on_unreadable_file(tmp_path):
    assert read_model_arch(tmp_path / "missing.gguf") is None
