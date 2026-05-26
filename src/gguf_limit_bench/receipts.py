from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


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
        (self.path / filename).write_text(
            json.dumps(data, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
