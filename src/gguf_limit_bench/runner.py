from __future__ import annotations

import subprocess
from pathlib import Path

from gguf_limit_bench.bench_plan import BenchProfile, build_llama_bench_command
from gguf_limit_bench.receipts import RunReceipt
from gguf_limit_bench.telemetry import classify_failure, sample_telemetry


class BenchmarkRunner:
    def __init__(self, llama_bench: Path, runs_root: Path) -> None:
        self.llama_bench = llama_bench
        self.runs_root = runs_root

    def run_model(self, model: Path, profile: BenchProfile) -> RunReceipt:
        receipt = RunReceipt.create(self.runs_root, slug=_safe_slug(model.stem))
        receipt.event("model_started", {"model": str(model), "profile": profile.name})
        for depth in profile.depths:
            command = build_llama_bench_command(self.llama_bench, model, profile, depth=depth)
            receipt.mark_recovery(step=f"depth:{depth}", status="running")
            receipt.event(
                "command_started", {"args": command, "telemetry": sample_telemetry().to_dict()}
            )
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=profile.timeout_seconds,
            )
            receipt.event(
                "command_finished",
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-8000:],
                    "stderr": completed.stderr[-8000:],
                    "telemetry": sample_telemetry().to_dict(),
                    "failure": classify_failure(completed.stderr + "\n" + completed.stdout),
                },
            )
            if completed.returncode != 0:
                receipt.mark_recovery(step=f"depth:{depth}", status="failed")
                break
        receipt.mark_recovery(step="benchmark", status="finished")
        receipt.write_summary(
            [f"# {model.name}", "", f"Profile: `{profile.name}`", "", "See `events.jsonl`."]
        )
        return receipt


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)[:80]
