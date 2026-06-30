"""The package must be runnable as `python -m gguf_limit_bench` so the web server
can spawn a detached engine via [sys.executable, '-m', 'gguf_limit_bench', ...].
"""

from __future__ import annotations

import subprocess
import sys


def test_python_m_package_shows_help():
    result = subprocess.run(
        [sys.executable, "-m", "gguf_limit_bench", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "engine" in result.stdout
