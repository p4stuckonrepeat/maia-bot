"""Human-like thinking time, based only on how hard the move is (not on the clock).

Book moves and obvious moves are quick, ordinary moves take a few seconds, hard positions
(Maia sees several plausible moves) take longer, and now and then the bot has a long think.
The only clock rule is a safety net so the bot never loses on time: under 30 s it plays fast,
and it never spends more than 12% of its remaining time on one move.
"""
import random

DEFAULTS = {"book": [0.4, 1.8], "obvious": [0.5, 1.5], "normal": [1.5, 5.0], "hard": [4.0, 10.0],
            "long_think": [12.0, 25.0], "long_think_chance": 0.05}


def think_seconds(clock_ms, inc_ms, ply, top_prob, legal_moves, from_book, cfg, rng=random):
    if not cfg.get("enabled", True):
        return 0.0
    c = {**DEFAULTS, **cfg}
    if legal_moves == 1:
        t = rng.uniform(0.3, 0.9)
    elif from_book:
        t = rng.uniform(*c["book"])
    elif top_prob >= 0.85:
        t = rng.uniform(*c["obvious"])
    elif top_prob >= 0.5:
        t = rng.uniform(*c["normal"])
    else:
        t = rng.uniform(*c["hard"])
    if not from_book and top_prob < 0.85 and ply > 20 and rng.random() < c["long_think_chance"]:
        t = rng.uniform(*c["long_think"])
    t *= float(c.get("speed", 1.0)) * rng.lognormvariate(0, 0.2)

    if clock_ms is not None:                       # safety net only: never lose on time
        clock, inc = clock_ms / 1000, (inc_ms or 0) / 1000
        if clock < 30:
            t = min(t, rng.uniform(0.3, 1.2) + inc * 0.5)
        if clock < 10:
            t = min(t, rng.uniform(0.15, 0.5))
        t = min(t, clock * 0.12 + inc * 0.5)
    return max(0.15, t)
