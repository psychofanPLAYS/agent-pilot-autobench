from pathlib import Path

from gguf_limit_bench.discovery import discover_models, parse_model_name
from gguf_limit_bench.model_identity import IdentityConfidence


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


def test_parse_model_name_detects_gemma_family():
    info = parse_model_name(Path("google-gemma-4-26B-A4B-it-Q4_K_M.gguf"))

    assert info.family == "gemma"
    assert info.parameters == "26B-A4B"
    assert info.quant == "Q4_K_M"


def test_parse_model_name_classifies_qwopus_as_qwen_derivative():
    info = parse_model_name(Path("Qwopus3.5-4B-v3-heretic.Q8_0.gguf"))

    assert info.family == "qwen"
    assert info.parameters == "4B"
    assert info.quant == "Q8_0"


def test_parse_model_name_detects_qat_q4_zero_quant():
    info = parse_model_name(Path("gemma-4-31B-it-QAT-Q4_0.gguf"))

    assert info.family == "gemma"
    assert info.parameters == "31B"
    assert info.quant == "Q4_0"


def test_parse_model_name_uses_apex_release_tier_when_quant_suffix_is_absent():
    info = parse_model_name(Path("Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf"))

    assert info.family == "qwen"
    assert info.parameters == "35B-A3B"
    assert info.quant == "APEX-I-QUALITY"
    assert info.has_mtp is True


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


def test_discover_models_skips_non_generative_ggufs(tmp_path):
    root = tmp_path / "models"
    chat = root / "LM_Studio-gguf" / "Qwen3.6-7B-Q4_K_M.gguf"
    embedding = root / "_Embedding-server" / "Qwen3-Embedding-0.6B-Q8_0.gguf"
    reranker = root / "_Reranking-server" / "rerankers" / "zerank-2.Q4_K_M.gguf"
    imatrix = root / "LM_Studio-gguf" / "Qwen3-0.6B" / "imatrix.gguf"
    for path in (chat, embedding, reranker, imatrix):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")

    models = discover_models([root])

    assert [model.path for model in models] == [chat]


def test_discover_models_attaches_path_identity(tmp_path):
    model_dir = tmp_path / "LM_Studio-gguf" / "Publisher" / "Repo"
    model_dir.mkdir(parents=True)
    weight = model_dir / "Qwen3.6-7B-Q4_K_M.gguf"
    weight.write_bytes(b"fake")

    models = discover_models([tmp_path])

    assert models[0].identity is not None
    assert models[0].identity.repo_id == "Publisher/Repo"
    assert models[0].identity.confidence is IdentityConfidence.CANDIDATE
