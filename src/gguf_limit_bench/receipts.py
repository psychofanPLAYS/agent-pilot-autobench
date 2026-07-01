from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any


@dataclass(frozen=True)
class RunReceipt:
    path: Path

    @classmethod
    def create(cls, root: Path, slug: str) -> "RunReceipt":
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = root / f"{timestamp}-{slug}"
        suffix = 2
        while True:
            try:
                path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                path = root / f"{timestamp}-{slug}-{suffix}"
                suffix += 1
        return cls(path=path)

    def event(self, event_type: str, data: dict) -> None:
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            "data": data,
        }
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def mark_recovery(self, step: str, status: str, detail: str | None = None) -> None:
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "step": step,
            "status": status,
            "detail": detail,
        }
        (self.path / "recovery.json").write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def write_summary(self, lines: list[str]) -> None:
        (self.path / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_json(self, filename: str, data: dict) -> None:
        target = self.path / _safe_receipt_filename(filename)
        target.write_text(
            json.dumps(data, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def write_resolved_plan(self, data: dict[str, Any], commands: list[dict[str, Any]]) -> None:
        payload = {**data, "commands": commands}
        self.write_json("resolved-plan.json", payload)
        (self.path / "command.txt").write_text(
            "\n".join(_display_command(command) for command in commands).rstrip() + "\n",
            encoding="utf-8",
        )

    def write_status(self, status: str, *, step: str, detail: str | None = None) -> None:
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "step": step,
            "detail": detail,
        }
        self.write_json("status.json", payload)


def _safe_receipt_filename(filename: str) -> str:
    path = Path(filename)
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
    if path.name != filename or filename in {"", ".", ".."} or path.stem.upper() in reserved:
        raise ValueError(f"Receipt filename must be a safe basename: {filename}")
    return filename


def _display_command(command: dict[str, Any]) -> str:
    display = command.get("display_command")
    if isinstance(display, str) and display:
        return display
    argv = command.get("argv")
    if isinstance(argv, list):
        return subprocess.list2cmdline([str(part) for part in argv])
    return ""
