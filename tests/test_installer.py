import os

from gguf_limit_bench.installer import (
    is_setup_complete,
    local_script,
    mark_setup_complete,
    persist_user_env,
    setup_marker_path,
    sync_project_environment,
)


def test_persist_user_env_sets_values_in_current_process(monkeypatch):
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))

        class _Done:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Done()

    # Uses a PILOTBENCH_ name so the autouse conftest fixture cleans it up.
    step = persist_user_env(
        {"PILOTBENCH_LLAMA_SERVER": "X:/llama/llama-server.exe"}, runner=fake_runner
    )

    assert os.environ["PILOTBENCH_LLAMA_SERVER"] == "X:/llama/llama-server.exe"
    assert step.ok
    assert step.required is False


def test_persist_user_env_no_values_is_a_skipped_noop():
    step = persist_user_env({})
    assert step.status == "skipped"
    assert step.required is False


def _make_local_command(repo_root):
    local_command = local_script(repo_root, "apb")
    local_command.parent.mkdir(parents=True, exist_ok=True)
    local_command.write_text("", encoding="utf-8")
    return local_command


def test_setup_is_incomplete_before_marker_is_written(tmp_path):
    _make_local_command(tmp_path)
    assert is_setup_complete(tmp_path) is False


def test_mark_setup_complete_makes_setup_detectable(tmp_path):
    _make_local_command(tmp_path)
    mark_setup_complete(tmp_path)

    assert setup_marker_path(tmp_path).exists()
    assert is_setup_complete(tmp_path) is True


def test_setup_is_incomplete_when_venv_command_is_missing(tmp_path):
    # Marker present but the local environment was wiped: re-run setup.
    mark_setup_complete(tmp_path)
    assert is_setup_complete(tmp_path) is False


def test_sync_project_environment_reuses_existing_local_command(tmp_path, monkeypatch):
    local_command = local_script(tmp_path, "agent-autobench")
    local_command.parent.mkdir(parents=True)
    local_command.write_text("", encoding="utf-8")
    monkeypatch.setattr("gguf_limit_bench.installer.shutil.which", lambda name: None)

    step = sync_project_environment(tmp_path)

    assert step.status == "ok"
    assert "local command already exists" in step.detail
