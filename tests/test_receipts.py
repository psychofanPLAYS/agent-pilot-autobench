import json

from gguf_limit_bench import events
from gguf_limit_bench.receipts import RunReceipt


def test_run_receipt_event_forwards_to_live_sink(tmp_path):
    receipt = RunReceipt.create(tmp_path, slug="qwen-test")
    seen: list = []
    with events.set_event_sink(lambda t, d: seen.append((t, d))):
        receipt.event("autoresearch_started", {"model": "m"})
    # forwarded to the live sink (so the cockpit sees the full behind-the-scenes)
    assert ("autoresearch_started", {"model": "m"}) in seen
    # and still persisted to the receipt's own events.jsonl
    assert (receipt.path / "events.jsonl").read_text(encoding="utf-8").strip()


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
