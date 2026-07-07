from __future__ import annotations

from pathlib import Path

from gguf_limit_bench.template_recommend import (
    discover_chat_template,
    merge_flags,
    recommended_model_flags,
)


def _make_template(root: Path) -> Path:
    template = root / "Qwen-Fixed-Chat-Templates" / "chat_template.jinja"
    template.parent.mkdir(parents=True)
    template.write_text("{{ messages }}", encoding="utf-8")
    return template


def test_qwen_recommends_jinja_and_custom_template_when_present(tmp_path):
    template = _make_template(tmp_path)
    model = tmp_path / "models" / "Qwen3.5-9B-Q8_0.gguf"
    model.parent.mkdir(parents=True)
    model.touch()

    flags = recommended_model_flags(model, search_roots=(tmp_path,))

    assert flags == (
        "--jinja",
        "--chat-template-file",
        str(template),
        "--chat-template-kwargs",
        '{"enable_thinking":true,"preserve_thinking":true}',
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
    )


def test_qwen_recommends_froggeric_v21_reasoning_defaults_when_template_present(tmp_path):
    template = _make_template(tmp_path)
    model = tmp_path / "models" / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model.parent.mkdir(parents=True)
    model.touch()

    flags = recommended_model_flags(model, search_roots=(tmp_path,))

    assert flags == (
        "--jinja",
        "--chat-template-file",
        str(template),
        "--chat-template-kwargs",
        '{"enable_thinking":true,"preserve_thinking":true}',
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
    )


def test_qwen_falls_back_to_jinja_only_without_template(tmp_path):
    model = tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model.touch()

    assert recommended_model_flags(model, search_roots=(tmp_path,)) == (
        "--jinja",
        "--chat-template-kwargs",
        '{"enable_thinking":true,"preserve_thinking":true}',
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
    )


def test_gemma_recommends_jinja(tmp_path):
    model = tmp_path / "gemma-4-26B-A4B-it-Q4_K_M.gguf"
    model.touch()

    assert recommended_model_flags(model) == ("--jinja",)


def test_gemma_recommends_official_template_when_present(tmp_path):
    template = tmp_path / "Gemma-4-Templates" / "chat_template.jinja"
    template.parent.mkdir(parents=True)
    template.write_text("{{ messages }}", encoding="utf-8")
    model = tmp_path / "gemma-4-26B-A4B-it-Q4_K_M.gguf"
    model.touch()

    assert recommended_model_flags(model, search_roots=(tmp_path,)) == (
        "--jinja",
        "--chat-template-file",
        str(template),
    )


def test_unknown_family_gets_no_template_flags(tmp_path):
    model = tmp_path / "some-random-model-Q4_K_M.gguf"
    model.touch()

    assert recommended_model_flags(model) == ()


def test_qwen_template_override_keeps_reasoning_flags(tmp_path):
    override = tmp_path / "custom.jinja"
    override.write_text("{{ messages }}", encoding="utf-8")
    model = tmp_path / "Qwen3.5-9B-Q8_0.gguf"
    model.touch()

    flags = recommended_model_flags(model, search_roots=(tmp_path,), template_override=override)

    assert flags == (
        "--jinja",
        "--chat-template-file",
        str(override),
        "--chat-template-kwargs",
        '{"enable_thinking":true,"preserve_thinking":true}',
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
    )


def test_gemma_ignores_qwen_template_folder(tmp_path):
    _make_template(tmp_path)
    assert discover_chat_template("gemma", (tmp_path,)) is None


def test_gemma_template_override_wins(tmp_path):
    override = tmp_path / "custom-gemma.jinja"
    override.write_text("{{ messages }}", encoding="utf-8")
    model = tmp_path / "gemma-4-26B-A4B-it-Q4_K_M.gguf"
    model.touch()

    assert recommended_model_flags(model, template_override=override) == (
        "--jinja",
        "--chat-template-file",
        str(override),
    )


def test_merge_flags_skips_already_present_flags():
    base = ("--flash-attn", "on", "--jinja")
    extra = ("--jinja", "--chat-template-file", "/path/t.jinja")

    assert merge_flags(base, extra) == (
        "--flash-attn",
        "on",
        "--jinja",
        "--chat-template-file",
        "/path/t.jinja",
    )
