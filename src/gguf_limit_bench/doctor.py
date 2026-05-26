from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    path: str
    detail: str
    required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DoctorReport:
    checks: list[DoctorCheck]

    @property
    def ready(self) -> bool:
        return all(check.status == "ok" for check in self.checks if check.required)

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "checks": [check.to_dict() for check in self.checks],
        }


def build_doctor_report(
    model_roots: list[Path],
    llama_bench: Path,
    llama_cli: Path,
    runs_root: Path,
    llama_server: Path | None = None,
) -> DoctorReport:
    checks: list[DoctorCheck] = []
    for root in model_roots:
        checks.append(_path_check(name=f"model root: {root}", path=root, expected="directory"))
    checks.append(_path_check(name="llama-bench", path=llama_bench, expected="file"))
    checks.append(_path_check(name="llama-cli", path=llama_cli, expected="file"))
    if llama_server is not None:
        checks.append(_path_check(name="llama-server", path=llama_server, expected="file"))
    checks.append(_runs_root_check(runs_root))
    return DoctorReport(checks=checks)


def _path_check(name: str, path: Path, expected: str) -> DoctorCheck:
    exists = path.exists()
    if expected == "directory":
        ok = exists and path.is_dir()
        detail = "directory exists" if ok else "directory was not found"
    else:
        ok = exists and path.is_file()
        detail = "file exists" if ok else "file was not found"
    return DoctorCheck(
        name=name,
        status="ok" if ok else "missing",
        path=str(path),
        detail=detail,
    )


def _runs_root_check(path: Path) -> DoctorCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return DoctorCheck(
            name="runs root",
            status="error",
            path=str(path),
            detail=str(exc),
        )
    return DoctorCheck(
        name="runs root",
        status="ok",
        path=str(path),
        detail="directory is ready for receipts",
    )
