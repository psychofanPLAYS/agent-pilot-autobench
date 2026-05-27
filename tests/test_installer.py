from gguf_limit_bench.installer import sync_project_environment


def test_sync_project_environment_reuses_existing_local_command(tmp_path, monkeypatch):
    local_command = tmp_path / ".venv" / "Scripts" / "agent-autobench.exe"
    local_command.parent.mkdir(parents=True)
    local_command.write_text("", encoding="utf-8")
    monkeypatch.setattr("gguf_limit_bench.installer.shutil.which", lambda name: None)

    step = sync_project_environment(tmp_path)

    assert step.status == "ok"
    assert "local command already exists" in step.detail
