from __future__ import annotations

from io import BytesIO
import json

from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.librarian.preflight import run_librarian_preflight


class FakeResponse(BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _identity_model(tmp_path, name: str):
    model = tmp_path / "LM_Studio-gguf" / "Owner" / "Repo" / name
    model.parent.mkdir(parents=True)
    model.touch()
    return model


def test_librarian_preflight_passes_required_common_gates_for_qwen(tmp_path, monkeypatch):
    model = _identity_model(tmp_path, "Qwen3.5-9B-Q4_K_M.gguf")

    monkeypatch.setattr(
        "gguf_limit_bench.librarian.preflight._chat",
        lambda **_kwargs: ("Final Answer: A", 1.0, 1.0, 1.0, 1),
    )

    receipt = run_librarian_preflight(
        model=model,
        settings=AutoresearchSettings(extra_server_args=("--jinja",)),
        base_url="http://127.0.0.1:8080",
        timeout_seconds=5,
    )

    gates = {gate.name: gate for gate in receipt.gates}
    assert receipt.ok is True
    assert gates["identity"].status == "pass"
    assert gates["template_load"].status == "pass"
    assert gates["thinking_sanity"].status == "skip"
    assert gates["answer_channel"].status == "pass"


def test_librarian_preflight_fails_unidentified_model_before_scoring(tmp_path, monkeypatch):
    model = tmp_path / "Qwen3.5-9B-Q4_K_M.gguf"
    model.touch()
    monkeypatch.setattr(
        "gguf_limit_bench.librarian.preflight._chat",
        lambda **_kwargs: ("Final Answer: A", 1.0, 1.0, 1.0, 1),
    )

    receipt = run_librarian_preflight(
        model=model,
        settings=AutoresearchSettings(extra_server_args=("--jinja",)),
        base_url="http://127.0.0.1:8080",
        timeout_seconds=5,
    )

    assert receipt.ok is False
    assert receipt.failure_class == "preflight_fail"
    assert receipt.failure == "identity"


def test_librarian_preflight_gemma_single_bos_uses_tokenize_endpoint(tmp_path, monkeypatch):
    model = _identity_model(tmp_path, "Gemma-4-26B-A4B-Q4_K_M.gguf")

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        tokens = [1, 10, 11, 12] if payload["add_special"] else [10, 11, 12]
        return FakeResponse(json.dumps({"tokens": tokens}).encode("utf-8"))

    monkeypatch.setattr("gguf_limit_bench.librarian.preflight.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "gguf_limit_bench.librarian.preflight._chat",
        lambda **_kwargs: ("Final Answer: A", 1.0, 1.0, 1.0, 1),
    )

    receipt = run_librarian_preflight(
        model=model,
        settings=AutoresearchSettings(extra_server_args=("--jinja",)),
        base_url="http://127.0.0.1:8080",
        timeout_seconds=5,
    )

    gates = {gate.name: gate for gate in receipt.gates}
    assert gates["single_bos"].status == "pass"


def test_librarian_preflight_qwen_thinking_gate_rejects_missing_think_block(tmp_path, monkeypatch):
    model = _identity_model(tmp_path, "Qwen3.5-9B-Q4_K_M.gguf")
    monkeypatch.setattr(
        "gguf_limit_bench.librarian.preflight._chat",
        lambda **_kwargs: ("Final Answer: A", 1.0, 1.0, 1.0, 1),
    )

    receipt = run_librarian_preflight(
        model=model,
        settings=AutoresearchSettings(
            extra_server_args=("--jinja", "--chat-template-kwargs", '{"enable_thinking":true}')
        ),
        base_url="http://127.0.0.1:8080",
        timeout_seconds=5,
    )

    assert receipt.ok is False
    assert receipt.failure == "thinking_sanity"


def test_librarian_preflight_qwen_thinking_accepts_reasoning_content(tmp_path, monkeypatch):
    model = _identity_model(tmp_path, "Qwen3.6-35B-A3B-Q4_K_M.gguf")

    def fake_chat_completion_message(**_kwargs):
        return "Final Answer: A", {
            "content": "Final Answer: A",
            "reasoning_content": "I should inspect the evidence.",
        }

    monkeypatch.setattr(
        "gguf_limit_bench.librarian.preflight._chat_completion_message",
        fake_chat_completion_message,
    )
    monkeypatch.setattr(
        "gguf_limit_bench.librarian.preflight._chat",
        lambda **_kwargs: ("Final Answer: A", 1.0, 1.0, 1.0, 1),
    )

    receipt = run_librarian_preflight(
        model=model,
        settings=AutoresearchSettings(
            extra_server_args=(
                "--jinja",
                "--chat-template-kwargs",
                '{"enable_thinking":true,"preserve_thinking":true}',
                "--reasoning",
                "on",
                "--reasoning-format",
                "deepseek",
            )
        ),
        base_url="http://127.0.0.1:8080",
        timeout_seconds=5,
    )

    gates = {gate.name: gate for gate in receipt.gates}
    assert gates["thinking_sanity"].status == "pass"
    assert gates["thinking_sanity"].evidence["reasoning"]["source"] == "reasoning_content"


def test_librarian_preflight_answer_channel_rejects_unparseable_output(tmp_path, monkeypatch):
    model = _identity_model(tmp_path, "Qwen3.5-9B-Q4_K_M.gguf")
    monkeypatch.setattr(
        "gguf_limit_bench.librarian.preflight._chat",
        lambda **_kwargs: ("I cannot answer this.", 1.0, 1.0, 1.0, 1),
    )

    receipt = run_librarian_preflight(
        model=model,
        settings=AutoresearchSettings(extra_server_args=("--jinja",)),
        base_url="http://127.0.0.1:8080",
        timeout_seconds=5,
    )

    assert receipt.ok is False
    assert receipt.failure == "answer_channel"
