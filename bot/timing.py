"""Human-like thinking time, based only on the move itself (not on the clock).

Very fast: book moves, forced moves (only one legal or one obvious move), recaptures and
moving the same piece again right after moving it. Normal moves take a few seconds.
Hard moves are deliberately inconsistent: sometimes a quick decision, often a few seconds,
sometimes a long think. The only clock rule is a safety net so the bot never loses on time.
"""
import random

import chess

DEFAULTS = {
    "book": [0.2, 0.8], "forced": [0.1, 0.5], "obvious": [0.2, 1.0], "follow_up": [0.3, 1.2],
    "normal": [1.0, 4.0], "hard_quick": [0.8, 2.5], "hard": [3.0, 9.0], "long_think": [10.0, 25.0],
    "hard_quick_chance": 0.35, "long_think_chance": 0.2,
}


def think_seconds(clock_ms, inc_ms, ply, top_prob, legal_moves, from_book, cfg, rng=random,
                  board=None, move=None, base_ms=None, is_top=None):
    if not cfg.get("enabled", True):
        return 0.0
    c = {**DEFAULTS, **cfg}
    recapture = follow_up = False
    if board is not None and move is not None and board.move_stack:
        last_opp = board.move_stack[-1]
        recapture = board.is_capture(move) and move.to_square == last_opp.to_square
        if len(board.move_stack) >= 2:
            follow_up = move.from_square == board.move_stack[-2].to_square   # same piece again

    if from_book:
        t = rng.uniform(*c["book"])
    elif legal_moves == 1 or top_prob >= 0.95:
        t = rng.uniform(*c["forced"])
    elif recapture or top_prob >= 0.8:
        t = rng.uniform(*c["obvious"])
    elif top_prob >= 0.45:
        t = rng.uniform(*c["normal"])
    else:                                                   # hard: inconsistent on purpose
        r = rng.random()
        if r < c["hard_quick_chance"]:
            t = rng.uniform(*c["hard_quick"])
        elif r < c["hard_quick_chance"] + c["long_think_chance"] and ply > 12:
            t = rng.uniform(*c["long_think"])
        else:
            t = rng.uniform(*c["hard"])
    if follow_up and not from_book:
        t = min(t * 0.5, rng.uniform(*c["follow_up"]) + 0.5)   # continuing with the same piece
    if is_top is not None and not from_book:   # humans play their first instinct fast, surprising moves after longer thought
        t *= float(c.get("top_factor", 1.0)) if is_top else float(c.get("surprise_factor", 1.0))
    t *= float(c.get("speed", 1.0)) * rng.lognormvariate(0, 0.25)
    ref = float(c.get("reference_game_seconds", 0) or 0)   # times above were measured at this game length
    if ref and base_ms is not None:                         # slower in longer games, like a human
        game = base_ms / 1000 + 40 * (inc_ms or 0) / 1000
        t *= min(4.0, max(0.5, game / ref)) ** float(c.get("clock_scale_power", 0.8))

    if clock_ms is not None:                       # safety net only: never lose on time
        clock, inc = clock_ms / 1000, (inc_ms or 0) / 1000
        if clock < 30:
            t = min(t, rng.uniform(0.3, 1.2) + inc * 0.5)
        if clock < 10:
            t = min(t, rng.uniform(0.15, 0.5))
        t = min(t, clock * 0.12 + inc * 0.5)
    return max(0.1, t)
