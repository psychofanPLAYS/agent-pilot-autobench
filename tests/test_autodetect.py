from pathlib import Path
import os

from gguf_limit_bench.autodetect import (
    find_llama_binaries,
    find_model_roots,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _llama_bin_name(stem: str) -> str:
    return f"{stem}.exe" if os.name == "nt" else stem


def test_find_model_roots_returns_folders_that_contain_gguf(tmp_path):
    has_models = tmp_path / "AI" / "models"
    _touch(has_models / "Gemma-4-E2B-Q8.gguf")
    empty = tmp_path / "Documents"
    empty.mkdir(parents=True)

    found = find_model_roots([has_models, empty])

    assert has_models in found
    assert empty not in found


def test_find_model_roots_finds_gguf_in_nested_subfolders(tmp_path):
    root = tmp_path / "AI"
    _touch(root / "LM_Studio-gguf" / "publisher" / "model" / "weights.gguf")

    found = find_model_roots([root], max_depth=4)

    assert found == [root]


def test_find_model_roots_prefers_specific_folder_over_parent(tmp_path):
    ai = tmp_path / "AI"
    models = ai / "models"
    _touch(models / "m.gguf")

    # Both `AI` (recursively) and `AI/models` contain a gguf; keep only models.
    found = find_model_roots([models, ai])

    assert found == [models]


def test_find_model_roots_respects_limit(tmp_path):
    roots = []
    for i in range(4):
        r = tmp_path / f"root{i}"
        _touch(r / "m.gguf")
        roots.append(r)

    found = find_model_roots(roots, limit=2)

    assert len(found) == 2


def test_find_llama_binaries_scans_roots_when_not_on_path(tmp_path):
    build = tmp_path / "AI" / "llama.cpp" / "cuda12"
    server = _touch(build / _llama_bin_name("llama-server"))
    _touch(build / _llama_bin_name("llama-bench"))

    found = find_llama_binaries([tmp_path / "AI"], which=lambda _name: None)

    assert found["llama-server"] == server
    assert "llama-bench" in found


def test_find_llama_binaries_prefers_path_over_scan(tmp_path):
    on_path = tmp_path / "bin" / "llama-server"
    _touch(on_path)
    scan_root = tmp_path / "elsewhere"
    _touch(scan_root / "llama-server.exe")

    found = find_llama_binaries(
        [scan_root],
        which=lambda name: str(on_path) if name == "llama-server" else None,
    )

    assert found["llama-server"] == on_path


def test_find_llama_binaries_missing_binary_is_simply_absent(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()

    found = find_llama_binaries([empty], which=lambda _name: None)

    assert found == {}
