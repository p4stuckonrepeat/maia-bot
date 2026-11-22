"""Tal v8 measurements on held-out Tal games (runs on a CPU, no GPU needed).

1. Temperature of the Tal rules (tal_rules2): which sampling temperature makes the bot's move
   probabilities match Tal's real choices best (lowest log-loss).
2. Feel-based search (bot.search, Maia only) as a filter before the Tal rules: does dropping moves that
   Maia's judgement says lose quickly make the bot more or less Tal-like, and how many sacrifices survive?
   Variants: no filter / filter / filter that always keeps sacrifices.

  PLAYER=tal TAL_CKPT=tal-79m.pt python -m tal.tune_tal  -> tal-tune.json (also printed)
"""
import json
import math
import os
import random
import threading
import types

import chess
import yaml

from bot.engine import Guard
from bot.search import Search
from tal.imitate import FEATURES, features
from tal.train import download_games, log, samples, tal_games

N = int(os.environ.get("TUNE_POSITIONS", "500"))
CKPT = os.environ.get("TAL_CKPT", "tal-79m.pt")
MARGIN = float(os.environ.get("TUNE_MARGIN", "0.10"))
TS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.4]


def cands_of(eng, board, k):
    eng.self_elo = eng.oppo_elo = 2600
    eng.multipv = k
    moves = " ".join(m.uci() for m in board.move_stack)
    eng.cmd_position("position startpos" + (f" moves {moves}" if moves else ""))
    return eng.score_moves()[1]


def softmax(ls, t, keep=None):
    keep = keep if keep is not None else [True] * len(ls)
    mx = max(l for l, k in zip(ls, keep) if k)
    ps = [math.exp((l - mx) / t) if k else 0.0 for l, k in zip(ls, keep)]
    s = sum(ps)
    return [p / s for p in ps]


def main():
    from maia3.uci import Maia3UCIEngine, parse_args
    cfg = yaml.safe_load(open("config-tal.yml"))
    rcfg = {**cfg["tal_rules2"], "human_curve": {"enabled": False}, "lapse_chance": 0}
    k = int(rcfg.get("candidates", 10))
    t_now = float(rcfg.get("temperature", 0.5))

    games = tal_games(download_games())
    rng = random.Random(1)
    rng.shuffle(games)
    hold = games[:max(50, len(games) // 10)]
    items = [(gi, p) for gi, p in samples(hold) if p >= 16]
    rng.shuffle(items)
    items = items[:N]

    eng = Maia3UCIEngine(parse_args(["--model", "maia3-79m", "--device", "cpu", "--checkpoint", CKPT]))
    eng.ensure_model_loaded()
    guard = Guard(threads=4)
    srch = Search(types.SimpleNamespace(eng=eng, lock=threading.Lock()))

    nll = {t: 0.0 for t in TS}
    var = {"no filter": [0.0, 0, 0.0], "search filter": [0.0, 0, 0.0], "search filter, keep sacrifices": [0.0, 0, 0.0]}
    n = covered = 0
    for i, (gi, ply) in enumerate(items):
        g = hold[gi]
        b = chess.Board()
        for m in g["moves"][:ply]:
            b.push_uci(m)
        real = g["moves"][ply]
        cands = cands_of(eng, b, k)
        if len(cands) < 2:
            continue
        n += 1
        idx = next((j for j, c in enumerate(cands) if c["move"].uci() == real), -1)
        _, scored = guard.tal_rules(b, cands, rcfg)
        logits = [l for _, l, _ in scored]
        sac = []
        for c in cands:
            wdl = c.get("wdl", (500, 0, 500))
            f = dict(zip(FEATURES, features(b, c["move"], c["policy"], (wdl[0] + 0.5 * wdl[1]) / 1000)))
            sac.append(f.get("sacrifice", 0) > 0)
        vals = []
        for c in cands:
            b.push(c["move"])
            vals.append(1.0 if b.is_checkmate() else 1.0 - srch.value(b, 1, (2600, 2600), [3, 2]))
            b.pop()
        best = max(vals)
        ok = [v >= best - MARGIN for v in vals]
        keeps = {"no filter": None, "search filter": ok, "search filter, keep sacrifices": [o or s for o, s in zip(ok, sac)]}
        if idx >= 0:
            covered += 1
            for t in TS:
                nll[t] += -math.log(max(1e-9, softmax(logits, t)[idx]))
        for name, keep in keeps.items():
            ps = softmax(logits, t_now, keep)
            if idx >= 0:
                var[name][0] += ps[idx]                            # chance the bot plays Tal's real move
                var[name][1] += int(max(range(len(ps)), key=lambda j: ps[j]) == idx)
            var[name][2] += sum(p for p, s in zip(ps, sac) if s)    # chance the bot sacrifices
        if i % 50 == 0:
            log(f"  {i}/{len(items)}")
    out = {"positions": n, "tal_move_in_candidates": round(covered / n, 3), "current_temperature": t_now,
           "temperature_logloss": {t: round(v / covered, 4) for t, v in nll.items()},
           "best_temperature": min(nll, key=nll.get),
           "search": {name: {"chance_tal_move": round(v[0] / n, 4), "top_pick_is_tal_move": round(v[1] / n, 4),
                             "sacrifice_chance": round(v[2] / n, 4)} for name, v in var.items()},
           "tal_real_sacrifice_share": None}
    real_sac = 0
    for gi, ply in items:
        g = hold[gi]
        b = chess.Board()
        for m in g["moves"][:ply]:
            b.push_uci(m)
        mv = chess.Move.from_uci(g["moves"][ply])
        f = dict(zip(FEATURES, features(b, mv, 0.1, 0.5)))
        real_sac += f.get("sacrifice", 0) > 0
    out["tal_real_sacrifice_share"] = round(real_sac / max(1, len(items)), 4)
    guard.sf.quit()
    json.dump(out, open("tal-tune.json", "w"), indent=1)
    print("TUNE_RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
