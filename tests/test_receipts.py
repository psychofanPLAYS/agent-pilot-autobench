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

