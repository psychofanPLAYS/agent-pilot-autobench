from pathlib import Path

from gguf_limit_bench.config import load_config, with_cli_overrides


def test_default_config_uses_deep_preset_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.benchmark.default_preset == "deep"


def test_load_config_reads_config_toml(tmp_path, monkeypatch):
    config_path = tmp_path / "_CONFIG.toml"
    config_path.write_text(
        """
[paths]
model_roots = ["D:/models", "E:/more-models"]
llama_bench = "D:/llama/llama-bench.exe"
llama_cli = "D:/llama/llama-cli.exe"
llama_server = "D:/llama/llama-server.exe"
llama_perplexity = "D:/llama/llama-perplexity.exe"
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
    assert config.paths.llama_perplexity == Path("D:/llama/llama-perplexity.exe")
    assert config.paths.runs_root == Path("D:/runs")
    assert config.benchmark.default_preset == "normal"
    assert config.benchmark.parallel_max == 2


def test_cli_overrides_config_and_env_overrides_cli(tmp_path, monkeypatch):
    config_path = tmp_path / "_CONFIG.toml"
    config_path.write_text(
        """
[paths]
model_roots = ["D:/models"]
llama_bench = "D:/llama/llama-bench.exe"
llama_cli = "D:/llama/llama-cli.exe"
llama_server = "D:/llama/llama-server.exe"
llama_perplexity = "D:/llama/llama-perplexity.exe"
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


def test_config_parses_forced_server_args(tmp_path):
    from gguf_limit_bench.config import load_config

    cfg = tmp_path / "_CONFIG.toml"
    cfg.write_text(
        '[benchmark]\nforced_server_args = ["--no-mmap", "--mlock"]\n',
        encoding="utf-8",
    )
    config = load_config(cfg)
    assert config.benchmark.forced_server_args == ("--no-mmap", "--mlock")


def test_config_forced_server_args_default_empty(tmp_path):
    from gguf_limit_bench.config import load_config

    cfg = tmp_path / "_CONFIG.toml"
    cfg.write_text("[benchmark]\nparallel_max = 4\n", encoding="utf-8")
    assert load_config(cfg).benchmark.forced_server_args == ()


def test_config_question_sample_size_default(tmp_path):
    from gguf_limit_bench.config import DEFAULT_QUESTION_SAMPLE_SIZE, load_config

    cfg = tmp_path / "_CONFIG.toml"
    cfg.write_text("[benchmark]\nparallel_max = 4\n", encoding="utf-8")
    config = load_config(cfg)
    assert config.benchmark.question_sample_size == DEFAULT_QUESTION_SAMPLE_SIZE


def test_config_question_sample_size_from_toml(tmp_path):
    from gguf_limit_bench.config import load_config

    cfg = tmp_path / "_CONFIG.toml"
    cfg.write_text("[benchmark]\nquestion_sample_size = 10\n", encoding="utf-8")
    config = load_config(cfg)
    assert config.benchmark.question_sample_size == 10


def test_config_question_selection_default(tmp_path):
    from gguf_limit_bench.config import DEFAULT_QUESTION_SELECTION, load_config

    cfg = tmp_path / "_CONFIG.toml"
    cfg.write_text("[benchmark]\nparallel_max = 4\n", encoding="utf-8")
    config = load_config(cfg)
    assert config.benchmark.question_selection == DEFAULT_QUESTION_SELECTION


def test_config_question_selection_from_toml(tmp_path):
    from gguf_limit_bench.config import load_config

    cfg = tmp_path / "_CONFIG.toml"
    cfg.write_text('[benchmark]\nquestion_selection = "random"\n', encoding="utf-8")
    config = load_config(cfg)
    assert config.benchmark.question_selection == "random"


def test_config_question_sample_size_env_override(tmp_path, monkeypatch):
    from gguf_limit_bench.config import load_config

    monkeypatch.setenv("PILOTBENCH_QUESTION_SAMPLE_SIZE", "7")
    cfg = tmp_path / "_CONFIG.toml"
    cfg.write_text("[benchmark]\nquestion_sample_size = 3\n", encoding="utf-8")
    config = load_config(cfg)
    assert config.benchmark.question_sample_size == 7


def test_config_question_selection_env_override(tmp_path, monkeypatch):
    from gguf_limit_bench.config import load_config

    monkeypatch.setenv("PILOTBENCH_QUESTION_SELECTION", "random")
    cfg = tmp_path / "_CONFIG.toml"
    cfg.write_text('[benchmark]\nquestion_selection = "sequential"\n', encoding="utf-8")
    config = load_config(cfg)
    assert config.benchmark.question_selection == "random"
