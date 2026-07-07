"""Small llama.cpp runtime checks used before expensive benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from pathlib import Path
from typing import Any

_VERSION_RE = re.compile(r'template_version\s*=\s*"([^"]+)"')


@dataclass(frozen=True)
class RuntimeDoctorReceipt:
    ok: bool
    checks: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def flag_supported(help_text: str, option: str) -> bool:
    """Return whether a llama.cpp help string advertises *option*."""
    return any(
        line.lstrip().startswith(option) or f", {option}" in line for line in help_text.splitlines()
    )


def detect_template_version(template_path: Path) -> str | None:
    """Read a froggeric-style ``template_version`` constant from disk."""
    try:
        text = template_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return _template_version_from_text(text)


def live_template_status(template_path: Path, props: dict[str, Any]) -> dict[str, Any]:
    """Compare the template on disk with the template exposed by llama.cpp ``/props``."""
    disk_version = detect_template_version(template_path)
    live_version = _template_version_from_text(str(props.get("chat_template") or ""))
    return {
        "template_file": str(template_path),
        "disk_version": disk_version,
        "live_version": live_version,
        "live_template_set": bool(props.get("chat_template")),
        "matches_disk": bool(disk_version and live_version and disk_version == live_version),
    }


def reasoning_status_from_message(message: dict[str, Any]) -> dict[str, Any]:
    """Classify whether an OpenAI-compatible chat message contains Qwen reasoning."""
    reasoning = message.get("reasoning_content")
    content = str(message.get("content") or "")
    has_tags = "<think>" in content.lower()
    if isinstance(reasoning, str) and reasoning:
        return {
            "has_reasoning": True,
            "source": "reasoning_content",
            "content_has_think_tags": has_tags,
        }
    return {
        "has_reasoning": has_tags,
        "source": "content" if has_tags else "none",
        "content_has_think_tags": has_tags,
    }


def _template_version_from_text(text: str) -> str | None:
    match = _VERSION_RE.search(text or "")
    return match.group(1) if match else None
