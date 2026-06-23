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
    # New onboarding: one-command install, then bare `apb` opens the app.
    assert "INSTALL.bat" in readme
    assert "install.ps1" in readme
    assert "PilotBENCHY" not in readme
    assert "Hero command" not in readme
    assert "apb" in readme
    assert "pilotbench" in readme
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
    assert "PilotBENCHY" not in docs
    assert "Hero command" not in docs
    assert "gguf-limit-bench" not in docs
    assert "Legacy command" not in docs
    assert "first-run" in docs.lower()


def test_product_design_doc_is_linked_from_readme():
    readme = Path("README.md").read_text(encoding="utf-8")
    product_design = Path("docs/PRODUCT-DESIGN.md").read_text(encoding="utf-8")
    product_design_lower = product_design.lower()

    assert "docs/PRODUCT-DESIGN.md" in readme
    assert "# Agent Pilot Autobench Product Design" in product_design
    assert "readiness wizard" in product_design_lower
    assert "campaign builder" in product_design_lower
    assert "champion promotion" in product_design_lower


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


def test_readme_links_release_trust_and_architecture_docs():
    readme = Path("README.md").read_text(encoding="utf-8")

    for required in (
        "docs/ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "## Verification",
        "## Project Status",
        "## Limitations",
    ):
        assert required in readme


def test_readme_links_a_sanitized_generated_example():
    readme = Path("README.md").read_text(encoding="utf-8")
    example = Path("docs/examples/flag-ladder-dry-run.md")

    assert example.is_file()
    assert "docs/examples/flag-ladder-dry-run.md" in readme
    assert "dry run" in example.read_text(encoding="utf-8").lower()


def test_readme_does_not_overclaim_release_readiness():
    readme = Path("README.md").read_text(encoding="utf-8")
    status = readme.split("## Project Status", 1)[1].split("## Limitations", 1)[0]

    assert "release candidate" in status.lower()
    assert "live" in status.lower()
    assert "production-ready" not in readme.lower()


def test_public_release_docs_exist():
    for path in (
        "docs/ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
    ):
        assert Path(path).is_file(), path


def test_github_community_health_files_exist():
    for path in (
        "CODE_OF_CONDUCT.md",
        "SUPPORT.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/pull_request_template.md",
    ):
        assert Path(path).is_file(), path


def test_ci_smoke_tests_the_supported_windows_path():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "windows-latest" in workflow
    assert "tests/test_windows_scripts.py" in workflow
    assert "agent-autobench --help" in workflow


def test_public_operating_docs_do_not_require_private_machine_context():
    public_paths = [
        Path("AGENTS.md"),
        Path("README.md"),
        Path("docs/AUTORESEARCH-PROGRAM.md"),
        Path("docs/IMPLEMENTATION-PLAN.md"),
        Path("docs/COMMAND-BOARD.md"),
        Path("docs/START-FOR-NORMAL-PEOPLE.md"),
    ]
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in public_paths)

    for private_term in (
        "David",
        "G:\\_codex_global",
        "G:\\AI\\_codex_projects",
        "CLAMSHELL",
        "XTREME",
        "SemLoc",
    ):
        assert private_term not in public_text
