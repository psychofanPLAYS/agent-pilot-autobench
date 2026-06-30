"""Run the CLI as ``python -m gguf_limit_bench``.

The web server uses this form to spawn a detached engine process via
``[sys.executable, "-m", "gguf_limit_bench", "engine", "--run-dir", <dir>]``,
which avoids depending on a console script being on PATH.
"""

from __future__ import annotations

from gguf_limit_bench.cli import app

if __name__ == "__main__":
    app()
