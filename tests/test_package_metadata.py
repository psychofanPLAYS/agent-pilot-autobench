from pathlib import Path
import tomllib

from gguf_limit_bench.cli import app
from gguf_limit_bench.simple_bench import (
    DEFAULT_SIMPLE_BENCH_PATH,
    DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT,
    load_simple_bench_questions,
)


def test_public_package_metadata_is_release_ready():
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]

    assert project["description"] == (
        "Local-first GGUF and llama.cpp benchmarking for agent workloads."
    )
    assert project["license"] == "MIT"
    assert project["urls"]["Repository"] == (
        "https://github.com/psychofanPLAYS/agent-pilot-autobench"
    )
    assert project["urls"]["Issues"].endswith("/agent-pilot-autobench/issues")
    assert {"gguf", "llama.cpp", "benchmarking", "local-ai"} <= set(project["keywords"])
    assert "Programming Language :: Python :: 3.13" in project["classifiers"]
    assert "Operating System :: Microsoft :: Windows" in project["classifiers"]


def test_built_in_simple_bench_assets_are_present_and_attributed():
    assert DEFAULT_SIMPLE_BENCH_PATH.is_file()
    assert DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT.is_file()
    assert len(load_simple_bench_questions()) == 10

    notice = DEFAULT_SIMPLE_BENCH_PATH.with_name("SIMPLEBENCH_NOTICE.md")
    notice_text = notice.read_text(encoding="utf-8")
    assert "License: MIT" in notice_text
    assert "Impulse2000/simple_bench_public-20-12-2024" in notice_text


def test_cli_help_uses_the_public_product_description():
    assert app.info.help == "Local-first GGUF and llama.cpp benchmarking for agent workloads."
