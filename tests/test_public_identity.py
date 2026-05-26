from pathlib import Path
import tomllib


def test_public_package_name_and_command_are_agent_pilot_autobench():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "agent-pilot-autobench"
    assert "pilotbench" in pyproject["project"]["scripts"]
    assert pyproject["project"]["scripts"]["gguf-limit-bench"] == pyproject["project"]["scripts"]["pilotbench"]


def test_readme_uses_public_command_name():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "# Agent Pilot Autobench" in readme
    assert "pilotbench --start" in readme
    assert "Legacy command" in readme
    assert "`gguf-limit-bench`" in readme
    assert "LlamaLab" not in readme
