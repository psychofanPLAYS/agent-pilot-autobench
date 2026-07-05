from __future__ import annotations

from gguf_limit_bench.runtime_doctor import (
    RuntimeDoctorReceipt,
    detect_template_version,
    flag_supported,
    live_template_status,
    reasoning_status_from_message,
)


HELP = """
--chat-template-kwargs STRING
--reasoning [on|off|auto]
--reasoning-format FORMAT
--chat-template-file JINJA_TEMPLATE_FILE
"""


def test_flag_supported_finds_long_options():
    assert flag_supported(HELP, "--reasoning") is True
    assert flag_supported(HELP, "--reasoning-format") is True
    assert flag_supported(HELP, "--missing") is False


def test_detect_template_version_reads_froggeric_constant(tmp_path):
    template = tmp_path / "chat_template.jinja"
    template.write_text(
        '{%- set template_version = "qwen3.6-froggeric-v21.3" %}\n',
        encoding="utf-8",
    )

    assert detect_template_version(template) == "qwen3.6-froggeric-v21.3"


def test_live_template_status_detects_stale_loaded_template(tmp_path):
    template = tmp_path / "chat_template.jinja"
    template.write_text(
        '{%- set template_version = "qwen3.6-froggeric-v21.3" %}',
        encoding="utf-8",
    )
    props = {"chat_template": '{%- set template_version = "qwen3.6-froggeric-v19" %}'}

    status = live_template_status(template, props)

    assert status["disk_version"] == "qwen3.6-froggeric-v21.3"
    assert status["live_version"] == "qwen3.6-froggeric-v19"
    assert status["matches_disk"] is False


def test_reasoning_status_accepts_reasoning_content():
    message = {
        "content": "Final Answer: A",
        "reasoning_content": "I should inspect the evidence.",
    }

    assert reasoning_status_from_message(message) == {
        "has_reasoning": True,
        "source": "reasoning_content",
        "content_has_think_tags": False,
    }


def test_runtime_doctor_receipt_serializes_checks():
    receipt = RuntimeDoctorReceipt(ok=True, checks=({"name": "help", "status": "pass"},))

    assert receipt.to_dict() == {
        "ok": True,
        "checks": ({"name": "help", "status": "pass"},),
    }
