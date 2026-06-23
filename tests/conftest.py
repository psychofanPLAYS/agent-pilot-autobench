import pytest

# Environment variables that override resolved config. Tests must not be
# influenced by a developer's machine setup (e.g. real model/llama paths a user
# exported so `apb` works from anywhere), so we clear them for every test.
_PILOTBENCH_ENV = (
    "PILOTBENCH_MODEL_ROOTS",
    "PILOTBENCH_LLAMA_BENCH",
    "PILOTBENCH_LLAMA_CLI",
    "PILOTBENCH_LLAMA_SERVER",
    "PILOTBENCH_LLAMA_PERPLEXITY",
    "PILOTBENCH_RUNS_ROOT",
    "PILOTBENCH_DEFAULT_PRESET",
    "PILOTBENCH_PARALLEL_MAX",
    "PILOTBENCH_LEARNING",
    "PILOTBENCH_WORKFLOW_EVAL",
    "PILOTBENCH_TTFT_PROBE",
    "PILOTBENCH_PERPLEXITY_CORPUS",
    "PILOTBENCH_PERPLEXITY_CONTEXTS",
    "PILOTBENCH_FORCED_SERVER_ARGS",
    "PILOTBENCH_QUESTION_SAMPLE_SIZE",
    "PILOTBENCH_QUESTION_SELECTION",
)


@pytest.fixture(autouse=True)
def _isolate_pilotbench_env(monkeypatch):
    for name in _PILOTBENCH_ENV:
        monkeypatch.delenv(name, raising=False)
