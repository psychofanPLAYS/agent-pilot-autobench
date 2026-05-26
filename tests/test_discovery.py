from pathlib import Path

from gguf_limit_bench.discovery import discover_models, parse_model_name


def test_parse_model_name_detects_qwen_family_quant_and_moe():
    info = parse_model_name(
        Path("Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf")
    )

    assert info.family == "qwen"
    assert info.parameters == "35B-A3B"
    assert info.quant == "Q4_K_M"
    assert info.is_moe is True
    assert info.has_mtp is True


def test_parse_model_name_does_not_mark_plain_qwen35_moe_as_mtp():
    info = parse_model_name(Path("Qwen3.6-35B-A3B-Q4_K_M.gguf"))

    assert info.family == "qwen"
    assert info.parameters == "35B-A3B"
    assert info.is_moe is True
    assert info.has_mtp is False


def test_discover_models_pairs_mmproj_with_weight(tmp_path):
    model_dir = tmp_path / "lmstudio-community" / "Qwen3.6-35B-A3B-GGUF"
    model_dir.mkdir(parents=True)
    weight = model_dir / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    mmproj = model_dir / "mmproj-Qwen3.6-35B-A3B-BF16.gguf"
    weight.write_bytes(b"fake")
    mmproj.write_bytes(b"fake")

    models = discover_models([tmp_path])

    assert len(models) == 1
    assert models[0].path == weight
    assert models[0].vision_mmproj == mmproj
    assert models[0].has_vision is True


def test_discover_models_records_size_and_sorts_heaviest_first(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    small = model_dir / "Qwen3.6-7B-Q4_K_M.gguf"
    large = model_dir / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    small.write_bytes(b"1" * 10)
    large.write_bytes(b"1" * 30)

    models = discover_models([model_dir])

    assert [model.path for model in models] == [large, small]
    assert [model.size_bytes for model in models] == [30, 10]
    assert models[0].size_gb > models[1].size_gb


def test_discover_models_skips_mmproj_anywhere_in_name_and_deduplicates_nested_roots(tmp_path):
    root = tmp_path / "models"
    lmstudio = root / "LM_Studio-gguf" / "repo"
    lmstudio.mkdir(parents=True)
    weight = lmstudio / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    mmproj = lmstudio / "Qwen3.6-35B-A3B-mmproj-BF16.gguf"
    weight.write_bytes(b"fake")
    mmproj.write_bytes(b"fake")

    models = discover_models([root, root / "LM_Studio-gguf"])

    assert [model.path for model in models] == [weight]
    assert models[0].vision_mmproj == mmproj
