from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_first_run_prefers_local_venv_before_uv():
    script = (ROOT / "FIRST_RUN.bat").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\agent-autobench.exe" in script
    assert 'set "AGENT_CMD=' in script
    assert "uv run --extra dev --extra bench agent-autobench" in script
    assert " --first-run" in script
    assert "apb --start" in script


def test_start_opens_existing_install_without_setup():
    script = (ROOT / "START.bat").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\agent-autobench.exe" in script
    assert 'set "AGENT_CMD=' in script
    assert "uv run --extra dev --extra bench agent-autobench" in script
    assert " --start" in script
    assert " setup" not in script


def test_redundant_beginner_launchers_are_not_kept_in_repo_root():
    assert not (ROOT / "START-HERE.bat").exists()
    assert not (ROOT / "INSTALL-COMMAND.bat").exists()
