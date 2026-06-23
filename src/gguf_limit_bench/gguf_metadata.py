"""Minimal, dependency-free reader for GGUF header metadata.

We only need a handful of architecture fields (layer/head counts, embedding
length) to estimate KV-cache VRAM, so this parses just the key/value metadata
block at the front of a ``.gguf`` file and stops before the tensor table. Large
array values (e.g. tokenizer vocab) are streamed past, not loaded.

Reference: GGUF v2/v3 layout — magic, version, tensor_count, kv_count, then
``kv_count`` typed key/value pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO

_GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF metadata value type enum.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL = range(8)
_STRING = 8
_ARRAY = 9
_UINT64, _INT64, _FLOAT64 = 10, 11, 12

_FIXED_FORMATS = {
    _UINT8: "<B",
    _INT8: "<b",
    _UINT16: "<H",
    _INT16: "<h",
    _UINT32: "<I",
    _INT32: "<i",
    _FLOAT32: "<f",
    _BOOL: "<?",
    _UINT64: "<Q",
    _INT64: "<q",
    _FLOAT64: "<d",
}
_FIXED_SIZES = {value_type: struct.calcsize(fmt) for value_type, fmt in _FIXED_FORMATS.items()}


class GGUFFormatError(ValueError):
    """Raised when a file does not look like a parseable GGUF."""


def _read(stream: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    data = stream.read(size)
    if len(data) < size:
        raise GGUFFormatError("unexpected end of file while reading header")
    return struct.unpack(fmt, data)[0]


def _read_string(stream: BinaryIO) -> str:
    length = _read(stream, "<Q")
    raw = stream.read(length)
    if len(raw) < length:
        raise GGUFFormatError("unexpected end of file while reading string")
    return raw.decode("utf-8", errors="replace")


def _skip_value(stream: BinaryIO, value_type: int) -> None:
    """Consume a value of the given type without storing it (used for arrays)."""
    if value_type in _FIXED_SIZES:
        stream.seek(_FIXED_SIZES[value_type], 1)
    elif value_type == _STRING:
        length = _read(stream, "<Q")
        stream.seek(length, 1)
    elif value_type == _ARRAY:
        elem_type = _read(stream, "<I")
        count = _read(stream, "<Q")
        if elem_type in _FIXED_SIZES:
            stream.seek(_FIXED_SIZES[elem_type] * count, 1)
        else:
            for _ in range(count):
                _skip_value(stream, elem_type)
    else:
        raise GGUFFormatError(f"unknown gguf value type {value_type}")


def _read_value(stream: BinaryIO, value_type: int):
    if value_type in _FIXED_FORMATS:
        return _read(stream, _FIXED_FORMATS[value_type])
    if value_type == _STRING:
        return _read_string(stream)
    if value_type == _ARRAY:
        return _read_array(stream)
    raise GGUFFormatError(f"unknown gguf value type {value_type}")


# Small numeric/bool arrays (e.g. a 35-layer sliding-window pattern) are useful
# for VRAM math; huge arrays (tokenizer vocab) are skipped to stay fast/cheap.
_SMALL_ARRAY_LIMIT = 4096


def _read_array(stream: BinaryIO):
    elem_type = _read(stream, "<I")
    count = _read(stream, "<Q")
    numeric = elem_type in _FIXED_FORMATS and elem_type != _STRING
    if numeric and count <= _SMALL_ARRAY_LIMIT:
        fmt = _FIXED_FORMATS[elem_type]
        return [_read(stream, fmt) for _ in range(count)]
    # Too large or string elements: skip without storing.
    if elem_type in _FIXED_SIZES:
        stream.seek(_FIXED_SIZES[elem_type] * count, 1)
    else:
        for _ in range(count):
            _skip_value(stream, elem_type)
    return None


def _skip_value_array_body(stream: BinaryIO) -> None:
    elem_type = _read(stream, "<I")
    count = _read(stream, "<Q")
    if elem_type in _FIXED_SIZES:
        stream.seek(_FIXED_SIZES[elem_type] * count, 1)
    else:
        for _ in range(count):
            _skip_value(stream, elem_type)


def read_gguf_metadata(path: Path | str) -> dict[str, object]:
    """Return the scalar/string GGUF metadata key/value pairs (arrays -> None)."""
    path = Path(path)
    with path.open("rb") as stream:
        magic = _read(stream, "<I")
        if magic != _GGUF_MAGIC:
            raise GGUFFormatError(f"not a GGUF file: {path}")
        _version = _read(stream, "<I")
        _tensor_count = _read(stream, "<Q")
        kv_count = _read(stream, "<Q")
        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = _read_string(stream)
            value_type = _read(stream, "<I")
            metadata[key] = _read_value(stream, value_type)
    return metadata


@dataclass(frozen=True)
class ModelArch:
    """The architecture fields needed to size a KV cache.

    ``key_length`` / ``value_length`` are the per-head K/V dimensions actually
    stored in the cache. Modern models set these explicitly (Gemma uses 512,
    not embedding_length / head_count), so we keep them rather than deriving a
    single ``head_dim``.
    """

    architecture: str
    n_layers: int
    n_heads: int
    n_heads_kv: int
    embedding_length: int
    key_length: int
    value_length: int
    train_context_length: int
    # Sliding-window attention (Gemma 3/4 etc.). When ``sliding_window`` and
    # ``n_swa_layers`` are set, only ``n_global_layers`` keep full-context KV;
    # the rest cap at the window with the (usually smaller) *_swa dims. Defaults
    # of 0 mean "no SWA" → every layer is treated as full-context (dense).
    sliding_window: int = 0
    key_length_swa: int = 0
    value_length_swa: int = 0
    n_global_layers: int = 0
    n_swa_layers: int = 0

    @property
    def head_dim(self) -> int:
        if self.n_heads <= 0:
            return 0
        return self.embedding_length // self.n_heads

    @property
    def is_sliding_window(self) -> bool:
        return self.sliding_window > 0 and self.n_swa_layers > 0


def model_arch_from_metadata(metadata: dict[str, object]) -> ModelArch | None:
    """Resolve the arch-prefixed KV fields (e.g. ``llama.block_count``).

    Returns None when the required fields are absent so callers can fall back to
    a coarse estimate instead of guessing wrong.
    """
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None

    def _int(suffix: str) -> int | None:
        value = metadata.get(f"{arch}.{suffix}")
        return int(value) if isinstance(value, int) else None

    n_layers = _int("block_count")
    n_heads = _int("attention.head_count")
    embedding_length = _int("embedding_length")
    if n_layers is None or n_heads is None or embedding_length is None:
        return None
    n_heads_kv = _int("attention.head_count_kv")
    if n_heads_kv is None:
        n_heads_kv = n_heads  # no GQA metadata -> assume multi-head attention
    # Per-head K/V dims default to embedding_length / head_count when not set.
    fallback_head_dim = embedding_length // n_heads if n_heads else 0
    key_length = _int("attention.key_length") or fallback_head_dim
    value_length = _int("attention.value_length") or fallback_head_dim
    train_context = _int("context_length") or 0

    # Sliding-window attention: resolve how many layers keep full-context KV.
    sliding_window = _int("attention.sliding_window") or 0
    key_length_swa = _int("attention.key_length_swa") or key_length
    value_length_swa = _int("attention.value_length_swa") or value_length
    n_global, n_swa = _resolve_swa_layer_split(
        metadata.get(f"{arch}.attention.sliding_window_pattern"),
        n_layers=n_layers,
        sliding_window=sliding_window,
    )

    return ModelArch(
        architecture=arch,
        n_layers=n_layers,
        n_heads=n_heads,
        n_heads_kv=n_heads_kv,
        embedding_length=embedding_length,
        key_length=key_length,
        value_length=value_length,
        train_context_length=train_context,
        sliding_window=sliding_window,
        key_length_swa=key_length_swa,
        value_length_swa=value_length_swa,
        n_global_layers=n_global,
        n_swa_layers=n_swa,
    )


def _resolve_swa_layer_split(
    pattern: object, *, n_layers: int, sliding_window: int
) -> tuple[int, int]:
    """Return (n_global_layers, n_swa_layers) from the sliding-window pattern.

    The pattern is either a per-layer bool array (True = sliding window) or an
    integer interval (1 global every N layers, the Gemma convention). With no
    pattern or no window, every layer is global (dense).
    """
    if sliding_window <= 0:
        return n_layers, 0
    if isinstance(pattern, list) and pattern:
        n_swa = sum(1 for flag in pattern if flag)
        n_global = len(pattern) - n_swa
        return n_global, n_swa
    if isinstance(pattern, int) and pattern > 1:
        # 1 global layer every `pattern` layers (e.g. Gemma 3 uses 6).
        n_global = max(1, n_layers // pattern)
        return n_global, n_layers - n_global
    # Window set but no usable pattern: be safe and treat all layers as global.
    return n_layers, 0


def read_model_arch(path: Path | str) -> ModelArch | None:
    """Convenience: read a GGUF and resolve its architecture, or None on failure."""
    try:
        metadata = read_gguf_metadata(path)
    except (OSError, GGUFFormatError):
        return None
    return model_arch_from_metadata(metadata)
