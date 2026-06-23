"""Recognise out-of-memory (OOM) failures from llama-server output.

When a context size is too large for the GPU, llama-server does not return a
tidy error code — it dies while allocating the KV cache or compute buffers with
a CUDA out-of-memory message on stderr. We detect that here so the context
ladder can *note it and back off to a smaller context* instead of treating it as
a mysterious crash that aborts the whole run.
"""

from __future__ import annotations

def is_oom_failure(stderr: str, returncode: int | None = None) -> bool:
    """True when *stderr* shows a memory-allocation failure (not a plain timeout).

    ``returncode`` is accepted for callers that have it but is advisory only —
    the stderr text is the reliable signal across platforms.
    """
    if not stderr:
        return False
    text = stderr.lower()

    # The pinned-memory fallback warning alone is not an OOM.
    only_pinned = "host_malloc" in text and "failed to allocate" in text
    hard_markers = (
        "out of memory",
        "cudamalloc failed",
        "cuda error: out of memory",
        "unable to allocate",
        "insufficient memory",
        "alloc_buffer: failed",
        "failed to allocate compute",
        "failed to allocate buffer for kv",
        "failed to allocate kv",
    )
    if any(marker in text for marker in hard_markers):
        return True

    # A bare "failed to allocate ... MiB on device" (the classic KV/buffer OOM)
    # counts, unless the only such line is the non-fatal pinned-host warning.
    if "failed to allocate" in text and "on device" in text and not only_pinned:
        return True
    return False


def oom_failure_label(context_size: int) -> str:
    """A short, durable failure label noting the OOM and the context that caused it."""
    return f"oom_context_too_large: {context_size // 1024}k context did not fit in VRAM"
