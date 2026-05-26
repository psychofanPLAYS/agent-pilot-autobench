from pathlib import Path

from gguf_limit_bench.config import load_config, with_cli_overrides


def test_load_config_reads_pilotbench_toml(tmp_path, monkeypatch):
    config_path = tmp_path / "pilotbench.toml"
    config_path.write_text(
        """
[paths]
model_roots = ["D:/models", "E:/more-models"]
llama_bench = "D:/llama/llama-bench.exe"
llama_cli = "D:/llama/llama-cli.exe"
llama_server = "D:/llama/llama-server.exe"
runs_root = "D:/runs"

[benchmark]
default_preset = "normal"
parallel_max = 2
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.paths.model_roots == (Path("D:/models"), Path("E:/more-models"))
    assert config.paths.llama_bench == Path("D:/llama/llama-bench.exe")
    assert config.paths.llama_cli == Path("D:/llama/llama-cli.exe")
    assert config.paths.llama_server == Path("D:/llama/llama-server.exe")
    assert config.paths.runs_root == Path("D:/runs")
    assert config.benchmark.default_preset == "normal"
    assert config.benchmark.parallel_max == 2


def test_cli_overrides_config_and_env_overrides_cli(tmp_path, monkeypatch):
    config_path = tmp_path / "pilotbench.toml"
    config_path.write_text(
        """
[paths]
model_roots = ["D:/models"]
llama_bench = "D:/llama/llama-bench.exe"
llama_cli = "D:/llama/llama-cli.exe"
llama_server = "D:/llama/llama-server.exe"
runs_root = "D:/runs"

[benchmark]
default_preset = "normal"
parallel_max = 2
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PILOTBENCH_RUNS_ROOT", "F:/env-runs")
    monkeypatch.setenv("PILOTBENCH_PARALLEL_MAX", "9")

    config = with_cli_overrides(
        load_config(),
        model_roots=[Path("E:/cli-models")],
        runs_root=Path("E:/cli-runs"),
        parallel_max=5,
    )

    assert config.paths.model_roots == (Path("E:/cli-models"),)
    assert config.paths.runs_root == Path("F:/env-runs")
    assert config.benchmark.parallel_max == 9
