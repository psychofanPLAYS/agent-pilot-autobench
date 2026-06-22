from pathlib import Path, PurePosixPath, PureWindowsPath
import os

import pytest

from gguf_limit_bench.model_identity import IdentityConfidence, resolve_path_identity


def test_resolve_lm_studio_layout_to_exact_repo_and_file():
    path = Path(
        r"G:\AI\models\LM_Studio-gguf\bytkim\Qwen3.6-27B-MTP-pi-reasoning-GGUF"
        r"\Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf"
    )

    identity = resolve_path_identity(path)

    assert identity.repo_id == "bytkim/Qwen3.6-27B-MTP-pi-reasoning-GGUF"
    assert identity.filename == "Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf"
    assert identity.confidence is IdentityConfidence.CANDIDATE
    assert identity.source == "lm_studio_path"
    assert identity.evidence == (str(path),)


def test_resolve_lm_studio_marker_case_insensitively():
    identity = resolve_path_identity(Path("/models/lm_STUDIO-GGUF/Publisher/Repo/weights.gguf"))

    assert identity.repo_id == "Publisher/Repo"
    assert identity.filename == "weights.gguf"
    assert identity.confidence is IdentityConfidence.CANDIDATE


def test_plain_filename_stays_unresolved():
    identity = resolve_path_identity(Path("Qwen3.6-27B-Q5_K_M.gguf"))

    assert identity.repo_id is None
    assert identity.filename == "Qwen3.6-27B-Q5_K_M.gguf"
    assert identity.confidence is IdentityConfidence.UNRESOLVED
    assert identity.source == "filename"


def test_lm_studio_path_with_insufficient_segments_stays_unresolved():
    identity = resolve_path_identity(Path("/models/LM_Studio-gguf/Publisher/weights.gguf"))

    assert identity.repo_id is None
    assert identity.confidence is IdentityConfidence.UNRESOLVED


@pytest.mark.parametrize(
    "path",
    [
        PurePosixPath("/models/LM_Studio-gguf/../Repo/file.gguf"),
        PurePosixPath("/models/LM_Studio-gguf/.Owner/Repo/file.gguf"),
        PurePosixPath("/models/LM_Studio-gguf/Owner/Repo-/file.gguf"),
        PurePosixPath("/models/LM_Studio-gguf/Owner/bad..repo/file.gguf"),
        PurePosixPath("/models/LM_Studio-gguf/Owner/bad repo/file.gguf"),
    ],
)
def test_invalid_or_traversal_repo_segments_stay_unresolved(path):
    identity = resolve_path_identity(path)

    assert identity.repo_id is None
    assert identity.confidence is IdentityConfidence.UNRESOLVED


def test_overlong_combined_repo_id_stays_unresolved():
    owner = "a" * 50
    repo = "b" * 50
    identity = resolve_path_identity(
        PurePosixPath(f"/models/LM_Studio-gguf/{owner}/{repo}/file.gguf")
    )

    assert identity.repo_id is None


def test_repeated_marker_uses_deepest_structurally_valid_layout():
    path = PurePosixPath("/models/LM_Studio-gguf/cache/LM_Studio-gguf/Owner/Repo/file.gguf")

    identity = resolve_path_identity(path)

    assert identity.repo_id == "Owner/Repo"
    assert identity.filename == "file.gguf"


def test_unc_lm_studio_layout_resolves_identity():
    path = PureWindowsPath(r"\\server\models\LM_Studio-gguf\Owner\Repo\Weights.Q5_K_M.gguf")

    identity = resolve_path_identity(path)

    assert identity.repo_id == "Owner/Repo"
    assert identity.filename == "Weights.Q5_K_M.gguf"
    assert identity.evidence == (str(path),)


def test_posix_filename_with_literal_backslashes_is_not_reinterpreted_as_windows_path():
    path = PurePosixPath(r"/models/LM_Studio-gguf\Owner\Repo\file.gguf")

    identity = resolve_path_identity(path)

    assert identity.repo_id is None
    assert identity.filename == r"LM_Studio-gguf\Owner\Repo\file.gguf"


@pytest.mark.skipif(os.name == "nt", reason="backslash is a native separator on Windows")
def test_native_posix_path_with_literal_backslashes_stays_unresolved():
    path = Path(r"LM_Studio-gguf\Owner\Repo\file.gguf")

    identity = resolve_path_identity(path)

    assert identity.repo_id is None
    assert identity.filename == r"LM_Studio-gguf\Owner\Repo\file.gguf"
