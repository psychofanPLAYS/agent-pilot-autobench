"""Model-aware llama.cpp flag recommendations.

GPU-level flags come from :mod:`gpu_profiles`; this module adds the *per-model*
flag combinations a given model family needs to be served correctly — most
importantly the chat template. Qwen3.5/3.6 reasoning models need ``--jinja`` plus
a custom froggeric-style ``--chat-template-file``; serving them with the builtin
chatml template silently degrades thinking-on scores.
"""

from __future__ import annotations

from pathlib import Path

from gguf_limit_bench.discovery import parse_model_name

# Directory names and filenames that hold a custom Qwen chat template the user
# keeps alongside their models (e.g. froggeric-v19).
_QWEN_TEMPLATE_DIR_HINTS = ("Qwen-Fixed-Chat-Templates",)
_TEMPLATE_FILENAMES = ("chat_template.jinja",)


def discover_chat_template(family: str, search_roots: tuple[Path, ...]) -> Path | None:
    """Return a custom chat-template ``.jinja`` for *family*, if one is on disk.

    Only Qwen has a known custom template convention today. Looks for a
    ``Qwen-Fixed-Chat-Templates/chat_template.jinja`` under any search root, then
    falls back to any ``*/chat_template.jinja`` whose path mentions ``qwen``.
    """
    if family != "qwen":
        return None
    for raw_root in search_roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        for hint in _QWEN_TEMPLATE_DIR_HINTS:
            for filename in _TEMPLATE_FILENAMES:
                candidate = root / hint / filename
                if candidate.is_file():
                    return candidate
        for filename in _TEMPLATE_FILENAMES:
            for candidate in sorted(root.glob(f"*/{filename}")):
                if "qwen" in str(candidate).lower() and candidate.is_file():
                    return candidate
    return None


def recommended_model_flags(
    model: Path,
    *,
    search_roots: tuple[Path, ...] = (),
    template_override: Path | None = None,
) -> tuple[str, ...]:
    """Recommend llama.cpp flag *combinations* specific to *model*'s family.

    - Qwen3.5/3.6: ``--jinja`` plus the froggeric custom ``--chat-template-file``
      when one can be found (or *template_override* when given).
    - Gemma: ``--jinja`` (Jinja template handling; single-BOS correctness).
    - Everything else: no extra flags (GPU/always-on flags are recommended
      separately by :func:`gpu_profiles.recommended_always_on`).
    """
    family = parse_model_name(model).family
    if family == "qwen":
        template = template_override or discover_chat_template(family, search_roots)
        if template is not None:
            return ("--jinja", "--chat-template-file", str(template))
        return ("--jinja",)
    if family == "gemma":
        return ("--jinja",)
    return ()


def merge_flags(base: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
    """Append *extra* flag tokens to *base*, skipping any flag already present.

    A flag and its following value are treated as a unit: if a ``--flag`` is
    already in *base*, neither it nor its value is appended again.
    """
    present = {token for token in base if token.startswith("-")}
    merged = list(base)
    index = 0
    extras = list(extra)
    while index < len(extras):
        token = extras[index]
        if token.startswith("-"):
            takes_value = index + 1 < len(extras) and not extras[index + 1].startswith("-")
            if token in present:
                index += 2 if takes_value else 1
                continue
            merged.append(token)
            present.add(token)
            if takes_value:
                merged.append(extras[index + 1])
                index += 2
            else:
                index += 1
        else:
            merged.append(token)
            index += 1
    return tuple(merged)
