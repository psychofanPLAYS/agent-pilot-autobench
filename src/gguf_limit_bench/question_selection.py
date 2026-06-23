from __future__ import annotations

import random as _random


def select_questions(
    questions: list,
    size: int,
    mode: str,
    *,
    seed: int | None = None,
    cursor: int = 0,
) -> tuple[list, int]:
    """Select questions from a list using either sequential or random sampling.

    Parameters
    ----------
    questions:
        Source list of questions to sample from.
    size:
        Number of items to return.
    mode:
        ``"sequential"`` advances a cursor through the list with wrap-around;
        ``"random"`` picks ``size`` distinct items using a seeded RNG.
    seed:
        RNG seed used only for ``mode="random"``; ignored for ``"sequential"``.
    cursor:
        Starting index used only for ``mode="sequential"``; ignored for
        ``"random"``.

    Returns
    -------
    tuple[list, int]
        ``(selected_items, new_cursor)``.  For ``"random"`` mode the returned
        cursor is always ``0``.
    """
    if not questions:
        return [], 0

    n = len(questions)

    if mode == "sequential":
        # Clamp size so we never return duplicates.
        effective_size = min(size, n)
        selected = [questions[(cursor + i) % n] for i in range(effective_size)]
        new_cursor = (cursor + effective_size) % n
        return selected, new_cursor

    if mode == "random":
        effective_size = min(size, n)
        rng = _random.Random(seed)
        selected = rng.sample(questions, effective_size)
        return selected, 0

    raise ValueError(f"Unknown selection mode: {mode!r}")
