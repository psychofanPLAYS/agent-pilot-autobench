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
        # We never need array contents for VRAM math; skip them and record None.
        _skip_value_array_body(stream)
        return None
    raise GGUFFormatError(f"unknown gguf value type {value_type}")


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

    @property
    def head_dim(self) -> int:
        if self.n_heads <= 0:
            return 0
        return self.embedding_length // self.n_heads


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
    return ModelArch(
        architecture=arch,
        n_layers=n_layers,
        n_heads=n_heads,
        n_heads_kv=n_heads_kv,
        embedding_length=embedding_length,
        key_length=key_length,
        value_length=value_length,
        train_context_length=train_context,
    )


def read_model_arch(path: Path | str) -> ModelArch | None:
    """Convenience: read a GGUF and resolve its architecture, or None on failure."""
    try:
        metadata = read_gguf_metadata(path)
    except (OSError, GGUFFormatError):
        return None
    return model_arch_from_metadata(metadata)
