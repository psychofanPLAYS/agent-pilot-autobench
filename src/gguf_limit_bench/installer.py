from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import shutil
import subprocess


DEFAULT_SHIM_DIR = Path("G:/_codex_global/bin")


@dataclass(frozen=True)
class InstallStep:
    name: str
    status: str
    path: str
    detail: str
    required: bool = True

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}

    def to_dict(self) -> dict:
        return asdict(self)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_script(repo_root: Path, command: str) -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return repo_root / ".venv" / scripts_dir / f"{command}{suffix}"


def sync_project_environment(repo_root: Path, skip: bool = False) -> InstallStep:
    local_agent = local_script(repo_root, "agent-autobench")
    if skip:
        return InstallStep(
            name="python environment",
            status="skipped",
            path=str(repo_root / ".venv"),
            detail="environment sync was skipped by option",
        )

    uv = shutil.which("uv")
    if uv is None:
        if local_agent.exists():
            return InstallStep(
                name="python environment",
                status="ok",
                path=str(local_agent),
                detail="local command already exists; uv was not needed",
            )
        return InstallStep(
            name="python environment",
            status="missing",
            path=str(repo_root / ".venv"),
            detail="uv is missing and the local .venv command is not installed yet",
        )

    try:
        completed = subprocess.run(
            [uv, "sync", "--extra", "dev", "--extra", "bench"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return InstallStep(
            name="python environment",
            status="error",
            path=str(repo_root / ".venv"),
            detail=f"uv sync could not run: {exc}",
        )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return InstallStep(
            name="python environment",
            status="error",
            path=str(repo_root / ".venv"),
            detail=f"uv sync failed: {stderr[:300]}",
        )

    if not local_agent.exists():
        return InstallStep(
            name="python environment",
            status="error",
            path=str(local_agent),
            detail="uv sync finished, but the local agent-autobench command was not created",
        )

    return InstallStep(
        name="python environment",
        status="ok",
        path=str(repo_root / ".venv"),
        detail="dev and benchmark-suite dependencies are synced with uv",
    )


def install_command_shims(repo_root: Path, shim_dir: Path = DEFAULT_SHIM_DIR) -> list[InstallStep]:
    steps: list[InstallStep] = []
    try:
        shim_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return [
            InstallStep(
                name="command shim folder",
                status="error",
                path=str(shim_dir),
                detail=str(exc),
            )
        ]

    steps.append(
        InstallStep(
            name="command shim folder",
            status="ok",
            path=str(shim_dir),
            detail="folder is ready",
        )
    )
    steps.append(_write_windows_shim(repo_root, shim_dir, "agent-autobench"))
    steps.append(_write_windows_shim(repo_root, shim_dir, "apb"))
    return steps


def _write_windows_shim(repo_root: Path, shim_dir: Path, command: str) -> InstallStep:
    shim_file = shim_dir / f"{command}.bat"
    local_command = local_script(repo_root, command)
    lines = [
        "@echo off",
        f'cd /d "{repo_root}"',
        f'if exist "{local_command}" (',
        f'  "{local_command}" %*',
        ") else (",
        f"  uv run --extra dev --extra bench {command} %*",
        ")",
        "",
    ]
    try:
        shim_file.write_text("\r\n".join(lines), encoding="utf-8")
    except OSError as exc:
        return InstallStep(
            name=f"{command} command",
            status="error",
            path=str(shim_file),
            detail=str(exc),
        )
    return InstallStep(
        name=f"{command} command",
        status="ok",
        path=str(shim_file),
        detail="command shim points to this repo",
    )


def check_user_path(shim_dir: Path) -> InstallStep:
    current_path = os.environ.get("PATH", "")
    parts = [Path(part).resolve() for part in current_path.split(os.pathsep) if part.strip()]
    try:
        target = shim_dir.resolve()
    except OSError:
        target = shim_dir
    if target in parts:
        return InstallStep(
            name="user PATH",
            status="ok",
            path=str(shim_dir),
            detail="this terminal can already find the command shim folder",
            required=False,
        )
    return InstallStep(
        name="user PATH",
        status="missing",
        path=str(shim_dir),
        detail="new terminals may not find agent-autobench until this folder is added to PATH",
        required=False,
    )


def add_shim_dir_to_user_path(shim_dir: Path) -> InstallStep:
    if os.name != "nt":
        return InstallStep(
            name="user PATH",
            status="skipped",
            path=str(shim_dir),
            detail="automatic user PATH editing is only supported on Windows",
            required=False,
        )

    command = (
        "$dir = $env:SHIM_DIR; "
        "$old = [Environment]::GetEnvironmentVariable('Path', 'User'); "
        "if ([string]::IsNullOrWhiteSpace($old)) { $new = $dir } "
        "elseif (($old -split ';') -contains $dir) { $new = $old } "
        "else { $new = $old.TrimEnd(';') + ';' + $dir }; "
        "[Environment]::SetEnvironmentVariable('Path', $new, 'User')"
    )
    env = {**os.environ, "SHIM_DIR": str(shim_dir)}
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return InstallStep(
            name="user PATH",
            status="error",
            path=str(shim_dir),
            detail=f"could not update user PATH: {exc}",
            required=False,
        )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return InstallStep(
            name="user PATH",
            status="error",
            path=str(shim_dir),
            detail=f"could not update user PATH: {stderr[:300]}",
            required=False,
        )
    return InstallStep(
        name="user PATH",
        status="ok",
        path=str(shim_dir),
        detail="folder was added to the Windows user PATH; open a new terminal",
        required=False,
    )
