"""Best-effort discovery of where llama.cpp binaries and GGUF models live.

A fresh "home llama.cpp enthusiast" should not have to hand-set environment
variables before `apb` works. On first run we scan a small set of common
locations so the app can open against a real model folder and a real
llama-server instead of dead-ending on "Something is missing".

Everything here is pure and injectable: callers pass the search roots (real
defaults come from :func:`default_llama_search_roots` /
:func:`default_model_search_roots`) so tests run against a tmp tree and never
touch the developer's machine.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import os
import shutil

# Binary stems we care about, in llama.cpp's naming. The order is the order we
# report them; llama-server is the one that actually matters for benchmarking.
LLAMA_BINARIES = ("llama-server", "llama-bench", "llama-cli", "llama-perplexity")

# Map each binary stem to the PILOTBENCH_* env var that overrides its path.
LLAMA_ENV_VARS = {
    "llama-server": "PILOTBENCH_LLAMA_SERVER",
    "llama-bench": "PILOTBENCH_LLAMA_BENCH",
    "llama-cli": "PILOTBENCH_LLAMA_CLI",
    "llama-perplexity": "PILOTBENCH_LLAMA_PERPLEXITY",
}


def _exe_name(stem: str) -> str:
    return f"{stem}.exe" if os.name == "nt" else stem


def _windows_drives() -> list[Path]:
    if os.name != "nt":
        return []
    drives: list[Path] = []
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(f"{letter}:/")
        if root.exists():
            drives.append(root)
    return drives


def _dedupe_existing(candidates: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            key = str(candidate.resolve()).lower()
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def default_llama_search_roots() -> list[Path]:
    """Likely parent folders of a llama.cpp build, kept tight so scans are fast.

    Only llama.cpp-named and AI folders are scanned (never a whole home or
    Program Files tree), so discovery stays quick on real machines.
    """
    home = Path.home()
    candidates: list[Path] = [
        home / "llama.cpp",
        home / "llama-cpp",
        Path("C:/llama.cpp"),
        Path("C:/tools/llama.cpp"),
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "llama.cpp")
    for drive in _windows_drives():
        candidates += [drive / "AI" / "llama.cpp", drive / "AI", drive / "llama.cpp"]
    return _dedupe_existing(candidates)


def default_model_search_roots() -> list[Path]:
    """Likely folders that hold GGUF models (LM Studio, common AI/model dirs)."""
    home = Path.home()
    candidates: list[Path] = [
        home / ".cache" / "lm-studio" / "models",
        home / ".lmstudio" / "models",
        home / "models",
        home / "gguf",
    ]
    for drive in _windows_drives():
        candidates += [
            drive / "AI" / "models",
            drive / "AI" / "LM_Studio-gguf",
            drive / "models",
            drive / "AI",
        ]
    return _dedupe_existing(candidates)


def _find_file(root: Path, filename: str, max_depth: int) -> Path | None:
    """Return the first file named ``filename`` within ``max_depth`` of root."""
    target = filename.lower()
    for path in _iter_bounded(root, max_depth):
        if path.name.lower() == target:
            return path
    return None


def _contains_gguf(root: Path, max_depth: int) -> bool:
    try:
        for path in _iter_bounded(root, max_depth):
            if path.suffix.lower() == ".gguf" and path.is_file():
                return True
    except OSError:
        return False
    return False


def _iter_bounded(root: Path, max_depth: int) -> Iterable[Path]:
    """Depth-bounded walk that yields files, skipping unreadable subtrees."""
    base_depth = len(root.parts)
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if len(entry.parts) - base_depth < max_depth:
                        stack.append(entry)
                else:
                    yield entry
            except OSError:
                continue


def find_llama_binaries(
    search_roots: Iterable[Path],
    *,
    which=shutil.which,
    max_depth: int = 5,
) -> dict[str, Path]:
    """Locate llama.cpp executables by PATH first, then by scanning roots."""
    roots = list(search_roots)
    found: dict[str, Path] = {}
    for stem in LLAMA_BINARIES:
        on_path = which(stem)
        if on_path:
            found[stem] = Path(on_path)
            continue
        exe = _exe_name(stem)
        for root in roots:
            match = _find_file(root, exe, max_depth)
            if match is not None:
                found[stem] = match
                break
    return found


def _drop_ancestors(paths: list[Path]) -> list[Path]:
    """Drop any path that is an ancestor of another kept path.

    A more specific model folder (``G:\\AI\\models``) is preferred over its
    parent (``G:\\AI``) so model discovery does not scan the same tree twice.
    """
    kept: list[Path] = []
    for path in paths:
        if any(other != path and _is_ancestor(path, other) for other in paths):
            continue
        kept.append(path)
    return kept


def _is_ancestor(maybe_parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(maybe_parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def find_model_roots(
    search_roots: Iterable[Path],
    *,
    max_depth: int = 4,
    limit: int = 5,
) -> list[Path]:
    """Return search roots that contain at least one ``*.gguf`` file.

    When both a folder and its parent match, only the more specific folder is
    kept so callers do not scan overlapping trees.
    """
    matches: list[Path] = []
    for root in search_roots:
        if _contains_gguf(root, max_depth):
            matches.append(root)
    return _drop_ancestors(matches)[:limit]
