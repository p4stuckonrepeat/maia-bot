"""Measure how the bot should be tuned to play more like a player (e.g. Carlsen).

1. Temperature: how random the move choice should be so the bot's move probabilities match the
   player's real choices best (lowest log-loss on held-out games).
2. Timing: the player's real thinking time per move (from %clk in online games), split into the
   same categories the bot's timing uses (book, forced, obvious, normal, hard).

  PLAYER=... IMIT_NAME=carlsen TAL_CKPT=prev/carlsen-79m.pt python -m tal.tune   -> <name>-tune.json
"""
import io
import json
import math
import os
import random
import re

import chess
import chess.pgn
import torch

from tal.imitate import NAME, CKPT, build_book, candidates, dataset, key
from tal.train import download_games, is_player, log, samples, tal_games, PLAYER

N_POS = int(os.environ.get("TUNE_POSITIONS", "3000"))
N_TIME = int(os.environ.get("TUNE_TIME_MOVES", "4000"))
CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 2) if xs else None


def timed_games(text):
    """Online games with clock comments: moves, player colour, seconds spent per player move, time control."""
    out, pgn = [], io.StringIO(text)
    user = PLAYER.split(":")[1].lower() if ":" in PLAYER and not PLAYER.startswith("multi:") else None
    while True:
        g = chess.pgn.read_game(pgn)
        if g is None:
            break
        tc = g.headers.get("TimeControl", "")
        m = re.fullmatch(r"(\d+)\+(\d+)", tc) or re.fullmatch(r"(\d+)", tc)
        if not m:
            continue
        base, inc = int(m.group(1)), int(m.group(2)) if m.lastindex == 2 else 0
        w, b = g.headers.get("White", ""), g.headers.get("Black", "")
        isw, isb = ((w.lower() == user, b.lower() == user) if user else (is_player(w), is_player(b)))
        if isw == isb:
            continue
        color = chess.WHITE if isw else chess.BLACK
        moves, clocks, node = [], [], g
        for node in g.mainline():
            c = CLK.search(node.comment or "")
            moves.append(node.move.uci())
            clocks.append(int(c.group(1)) * 3600 + int(c.group(2)) * 60 + float(c.group(3)) if c else None)
        if len(moves) < 20 or sum(c is not None for c in clocks) < len(moves) * 0.9:
            continue
        spent = {}
        start = 0 if color == chess.WHITE else 1
        prev = base
        for ply in range(start, len(moves), 2):
            if clocks[ply] is None:
                prev = None
                continue
            if prev is not None:
                spent[ply] = max(0.0, prev - clocks[ply] + inc)
            prev = clocks[ply]
        out.append({"moves": moves, "color": color, "spent": spent, "base": base, "inc": inc})
    return out


def category(board, move, top_prob, from_book):
    recapture = follow_up = False
    if board.move_stack:
        last = board.move_stack[-1]
        recapture = board.is_capture(move) and move.to_square == last.to_square
        if len(board.move_stack) >= 2:
            follow_up = move.from_square == board.move_stack[-2].to_square
    if from_book:
        return "book"
    if board.legal_moves.count() == 1 or top_prob >= 0.95:
        return "forced"
    if recapture or top_prob >= 0.8:
        return "obvious"
    if follow_up:
        return "follow_up"
    if top_prob >= 0.45:
        return "normal"
    return "hard"


def main():
    from maia3.uci import Maia3UCIEngine, parse_args
    text = download_games()
    games = tal_games(text)
    rng = random.Random(1)                       # same split as tal.imitate
    rng.shuffle(games)
    n_hold = max(50, len(games) // 10)
    hold, train = games[:n_hold], games[n_hold:]
    s_hold = samples(hold)
    rng.shuffle(s_hold)
    s_hold = s_hold[:N_POS]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eng = Maia3UCIEngine(parse_args(["--model", "maia3-79m", "--device", dev, "--checkpoint", CKPT]))
    eng.ensure_model_loaded()
    rules = json.load(open(f"tal/{NAME}-rules.json"))
    w = torch.tensor([rules["weights"][f] for f in rules["features"]])
    out = {"player": PLAYER}

    # 1. temperature
    X, Y, _ = dataset(eng, hold, s_hold)
    logits = [torch.tensor(x) @ w for x in X]
    covered = [(l, y) for l, y in zip(logits, Y) if y >= 0]
    grid = {}
    for T in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
        nll = sum(-torch.log_softmax(l / T, 0)[y].item() for l, y in covered) / len(covered)
        hit = sum(torch.softmax(l / T, 0)[y].item() for l, y in covered) / len(Y)
        grid[T] = {"logloss": round(nll, 4), "chance_same_move": round(hit, 4)}
    best = min(grid, key=lambda t: grid[t]["logloss"])
    out["temperature"] = {"grid": grid, "best": best, "player_move_in_top": round(len(covered) / len(Y), 4)}
    log("temperature", json.dumps(out["temperature"]))

    # 1b. human-like calculation (bot.search): does checking the opponent's replies with Maia's own
    #     judgement make the bot pick Carlsen's move more often than the rules alone?
    import threading
    import types
    from bot.search import Search
    from tal.imitate import features
    srch = Search(types.SimpleNamespace(eng=eng, lock=threading.Lock()))
    n_s = int(os.environ.get("TUNE_SEARCH_POSITIONS", "800"))
    res = {"rules only": 0}
    configs = [(d, m, c) for d in (1, 2) for m in (0.05, 0.10, 0.20) for c in (4, 6)]
    for cfg in configs:
        res[f"search depth {cfg[0]} margin {cfg[1]} candidates {cfg[2]}"] = 0
    counted = 0
    for n, (gi, ply) in enumerate(s_hold[:n_s]):
        g = hold[gi]
        b = chess.Board()
        for m in g["moves"][:ply]:
            b.push_uci(m)
        real = g["moves"][ply]
        top = candidates(eng, b)
        if not top:
            continue
        counted += 1
        lg = lambda c: float(torch.tensor(features(b, c["move"], c["policy"], (c["wdl"][0] + 0.5 * c["wdl"][1]) / 1000)) @ w)
        plain = max(top, key=lg)["move"].uci()
        res["rules only"] += plain == real
        for d, m, c in configs:
            key_ = f"search depth {d} margin {m} candidates {c}"
            if top[0]["policy"] >= 0.9:
                res[key_] += plain == real
                continue
            root = top[:c]
            for cd in root:
                b.push(cd["move"])
                cd["value"] = 1.0 if b.is_checkmate() else 1.0 - srch.value(b, d, (2600, 2600), [3, 2])
                b.pop()
            best = max(cd["value"] for cd in root)
            ok = [cd for cd in root if cd["value"] >= best - m]
            res[key_] += max(ok, key=lg)["move"].uci() == real
        if n % 200 == 0:
            log(f"  search {n}/{n_s}")
    out["search"] = {k: round(v / max(1, counted), 4) for k, v in res.items()}
    out["search_positions"] = counted
    log("search", json.dumps(out["search"]))

    # 2. timing (blitz games with clocks)
    book = build_book(train)
    tg = [g for g in timed_games(text) if 150 <= g["base"] + 40 * g["inc"] <= 300]   # ~3+0 .. 3+2 / 5+0
    tcs = {}
    for g in tg:
        tcs[f'{g["base"]}+{g["inc"]}'] = tcs.get(f'{g["base"]}+{g["inc"]}', 0) + 1
    out["timing_games"] = tcs
    items = [(gi, ply) for gi, g in enumerate(tg) for ply in g["spent"]]
    rng.shuffle(items)
    items = items[:N_TIME]
    cats, ply_times, game_secs = {}, {}, []
    for g in tg:
        game_secs.append(g["base"] + 40 * g["inc"])
    for n, (gi, ply) in enumerate(items):
        g = tg[gi]
        b = chess.Board()
        for m in g["moves"][:ply]:
            b.push_uci(m)
        mv = chess.Move.from_uci(g["moves"][ply])
        side = "white" if g["color"] == chess.WHITE else "black"
        entry = book[side].get(key(b), {})
        from_book = ply < 24 and entry.get(g["moves"][ply], 0) >= 3
        top = candidates(eng, b)
        cat = category(b, mv, top[0]["policy"] if top else 1.0, from_book)
        cats.setdefault(cat, []).append(g["spent"][ply])
        ply_times.setdefault(min(ply // 10, 8), []).append(g["spent"][ply])
        if n % 1000 == 0:
            log(f"  timing {n}/{len(items)}")
    out["timing"] = {c: {"n": len(v), **{f"p{p}": pct(v, p) for p in (5, 10, 25, 50, 75, 90, 95, 98)},
                         "share_over_10s": round(sum(t > 10 for t in v) / len(v), 3)}
                     for c, v in cats.items()}
    out["timing_by_ply"] = {f"ply {k * 10}-{k * 10 + 9}": {"n": len(v), "p50": pct(v, 50), "p90": pct(v, 90)}
                            for k, v in sorted(ply_times.items())}
    out["reference_game_seconds"] = pct(game_secs, 50)
    log("timing", json.dumps(out["timing"]))
    json.dump(out, open(f"{NAME}-tune.json", "w"), indent=1)


if __name__ == "__main__":
    main()
