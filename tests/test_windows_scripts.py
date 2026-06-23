from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_first_run_delegates_to_one_command_installer():
    script = (ROOT / "FIRST_RUN.bat").read_text(encoding="utf-8")

    # First run is now just the one-command installer.
    assert "install.ps1" in script
    assert "apb" in script


def test_start_launches_apb_without_setup():
    script = (ROOT / "START.bat").read_text(encoding="utf-8")

    # START just opens the app via bare `apb`; it never re-runs setup.
    assert ".venv\\Scripts\\apb.exe" in script
    assert "uv run --extra dev --extra bench apb" in script
    assert " setup" not in script
    assert "--first-run" not in script


def test_one_command_installer_exists_and_bootstraps_everything():
    bat = (ROOT / "INSTALL.bat").read_text(encoding="utf-8")
    ps1 = (ROOT / "install.ps1").read_text(encoding="utf-8")

    assert "install.ps1" in bat
    # The installer auto-provisions uv, builds the env, and installs the command.
    assert "astral.sh/uv/install.ps1" in ps1
    assert "uv sync" in ps1
    assert "apb setup" in ps1


def test_redundant_beginner_launchers_are_not_kept_in_repo_root():
    assert not (ROOT / "START-HERE.bat").exists()
    assert not (ROOT / "INSTALL-COMMAND.bat").exists()
