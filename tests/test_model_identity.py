from pathlib import Path

from gguf_limit_bench.model_identity import IdentityConfidence, resolve_path_identity


def test_resolve_lm_studio_layout_to_exact_repo_and_file():
    path = Path(
        r"G:\AI\models\LM_Studio-gguf\bytkim\Qwen3.6-27B-MTP-pi-reasoning-GGUF"
        r"\Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf"
    )

    identity = resolve_path_identity(path)

    assert identity.repo_id == "bytkim/Qwen3.6-27B-MTP-pi-reasoning-GGUF"
    assert identity.filename == path.name
    assert identity.confidence is IdentityConfidence.CANDIDATE
    assert identity.source == "lm_studio_path"


def test_plain_filename_stays_unresolved():
    identity = resolve_path_identity(Path("Qwen3.6-27B-Q5_K_M.gguf"))

    assert identity.repo_id is None
    assert identity.confidence is IdentityConfidence.UNRESOLVED
