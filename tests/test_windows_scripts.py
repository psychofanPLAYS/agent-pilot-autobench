from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_start_here_prefers_local_venv_before_uv():
    script = (ROOT / "START-HERE.bat").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\agent-autobench.exe" in script
    assert 'set "AGENT_CMD=' in script
    assert "uv run --extra dev --extra bench agent-autobench" in script


def test_install_command_shims_prefer_local_venv_before_uv():
    script = (ROOT / "INSTALL-COMMAND.bat").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\agent-autobench.exe" in script
    assert ".venv\\Scripts\\apb.exe" in script
    assert "uv run --extra dev --extra bench agent-autobench" in script
    assert "uv run --extra dev --extra bench apb" in script
