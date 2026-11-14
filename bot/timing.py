"""Human-like thinking time.

Fast for book moves and obvious moves (one move Maia is very sure about, e.g. recaptures),
longer when Maia sees several plausible moves, occasional deep thinks in the middlegame,
and quick, hectic moves when the clock runs low. Returns seconds to spend on a move.
"""
import random


def think_seconds(clock_ms, inc_ms, ply, top_prob, legal_moves, from_book, cfg, rng=random):
    if not cfg.get("enabled", True) or clock_ms is None:
        return 0.0
    clock, inc = clock_ms / 1000, (inc_ms or 0) / 1000
    speed = float(cfg.get("speed", 1.0))       # >1 = slower, more deliberate; <1 = faster

    if legal_moves == 1:
        t = rng.uniform(0.3, 0.9)
    elif from_book:
        t = rng.uniform(0.3, 0.9) if ply < 6 else rng.uniform(0.6, 2.2)   # known theory
    else:
        budget = clock / 40 + inc * 0.8                                     # typical human pace
        if top_prob >= 0.85:
            factor = rng.uniform(0.15, 0.35)                                # obvious move
        else:
            factor = 0.6 + 1.6 * (1 - top_prob)                             # harder = longer
            if ply > 20 and rng.random() < 0.06:
                factor *= rng.uniform(2.0, 3.0)                             # a long think
        t = budget * factor * rng.lognormvariate(0, 0.35)

    t *= speed
    if clock < 30:                                                          # time trouble
        t = min(t, rng.uniform(0.3, 1.2) + inc * 0.5)
    if clock < 10:
        t = min(t, rng.uniform(0.15, 0.5))
    return max(0.15, min(t, clock * 0.12))
