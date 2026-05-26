from pathlib import Path
import tomllib


def test_public_package_name_and_command_are_agent_pilot_autobench():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "agent-pilot-autobench"
    assert "pilotbench" in pyproject["project"]["scripts"]
    assert "agent-autobench" in pyproject["project"]["scripts"]
    assert "apb" in pyproject["project"]["scripts"]
    assert pyproject["project"]["scripts"]["agent-autobench"] == "gguf_limit_bench.cli:app"
    assert pyproject["project"]["scripts"]["apb"] == "gguf_limit_bench.cli:app"
    assert "gguf-limit-bench" not in pyproject["project"]["scripts"]


def test_readme_uses_public_command_name():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "# Agent Pilot Autobench" in readme
    assert "agent-autobench first-run" in readme
    assert "PilotBENCHY" in readme
    assert "apb first-run" in readme
    assert "pilotbench --start" in readme
    assert "Legacy command" not in readme
    assert "`gguf-limit-bench`" not in readme
    assert "LlamaLab" not in readme


def test_docs_use_only_new_public_command_name():
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("docs/COMMAND-BOARD.md"),
            Path("docs/START-FOR-NORMAL-PEOPLE.md"),
        ]
    )

    assert "agent-autobench" in docs
    assert "pilotbench" in docs
    assert "gguf-limit-bench" not in docs
    assert "Legacy command" not in docs


def test_root_gitignore_hides_accidental_npm_lockfile():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "/package-lock.json" in gitignore


def test_public_docs_do_not_use_personal_name():
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("README.md"),
            Path("docs/COMMAND-BOARD.md"),
            Path("docs/START-FOR-NORMAL-PEOPLE.md"),
            Path("docs/IMPLEMENTATION-PLAN.md"),
        ]
    )

    assert "David" not in public_text
    assert "David's" not in public_text


def test_public_package_docstring_does_not_leak_private_workstation_name():
    package_init = Path("src/gguf_limit_bench/__init__.py").read_text(encoding="utf-8")

    assert "David" not in package_init
    assert "XTREME" not in package_init
