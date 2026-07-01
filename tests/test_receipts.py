import json

from gguf_limit_bench.receipts import RunReceipt


def test_run_receipt_writes_jsonl_events_and_recovery_file(tmp_path):
    receipt = RunReceipt.create(tmp_path, slug="qwen-test")

    receipt.event("command_started", {"model": "qwen", "args": ["llama-bench"]})
    receipt.mark_recovery(step="benchmark", status="running")

    events_path = receipt.path / "events.jsonl"
    recovery_path = receipt.path / "recovery.json"

    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))

    assert event["type"] == "command_started"
    assert event["data"]["model"] == "qwen"
    assert recovery["step"] == "benchmark"
    assert recovery["status"] == "running"


def test_run_receipt_create_avoids_same_second_folder_collisions(tmp_path, monkeypatch):
    class FakeDateTime:
        @classmethod
        def now(cls):
            return cls()

        def strftime(self, _format: str) -> str:
            return "20260526-120000"

        def isoformat(self, timespec: str = "seconds") -> str:
            return "2026-05-26T12:00:00"

    monkeypatch.setattr("gguf_limit_bench.receipts.datetime", FakeDateTime)

    first = RunReceipt.create(tmp_path, slug="qwen-test")
    second = RunReceipt.create(tmp_path, slug="qwen-test")

    assert first.path.name == "20260526-120000-qwen-test"
    assert second.path.name == "20260526-120000-qwen-test-2"


def test_run_receipt_writes_reproducible_plan_command_and_status(tmp_path):
    receipt = RunReceipt.create(tmp_path, slug="qwen-test")

    receipt.write_resolved_plan(
        {"schema_version": 1, "program": "autoresearch", "model": "model.gguf"},
        [
            {
                "argv": ["agent-autobench", "autoresearch", "--model", "model.gguf"],
                "display_command": "agent-autobench autoresearch --model model.gguf",
            }
        ],
    )
    receipt.write_status("running", step="autoresearch", detail="attempt:1")

    resolved = json.loads((receipt.path / "resolved-plan.json").read_text(encoding="utf-8"))
    command = (receipt.path / "command.txt").read_text(encoding="utf-8")
    status = json.loads((receipt.path / "status.json").read_text(encoding="utf-8"))

    assert resolved["schema_version"] == 1
    assert resolved["program"] == "autoresearch"
    assert resolved["commands"][0]["argv"] == [
        "agent-autobench",
        "autoresearch",
        "--model",
        "model.gguf",
    ]
    assert command == "agent-autobench autoresearch --model model.gguf\n"
    assert status["status"] == "running"
    assert status["step"] == "autoresearch"
    assert status["detail"] == "attempt:1"


def test_run_receipt_command_fallback_quotes_argv_with_spaces(tmp_path):
    receipt = RunReceipt.create(tmp_path, slug="qwen-test")

    receipt.write_resolved_plan(
        {"schema_version": 1, "program": "autoresearch"},
        [
            {
                "argv": [
                    "agent-autobench",
                    "autoresearch",
                    "--model",
                    "G:\\models\\model with spaces.gguf",
                ],
            }
        ],
    )

    command = (receipt.path / "command.txt").read_text(encoding="utf-8")

    assert '"G:\\models\\model with spaces.gguf"' in command


def test_run_receipt_rejects_unsafe_json_filenames(tmp_path):
    receipt = RunReceipt.create(tmp_path, slug="qwen-test")

    for filename in ("../outside.json", "nested/file.json", "CON.json"):
        try:
            receipt.write_json(filename, {})
        except ValueError:
            pass
        else:  # pragma: no cover - assertion clarity
            raise AssertionError(f"expected unsafe filename to be rejected: {filename}")
