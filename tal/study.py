"""Overnight study of Mikhail Tal's games: what makes his moves different, how sound his
sacrifices are, how accurate he is in which situations, and which rules bring the bot closer.

STUDY_MODE=collect  (STUDY_SHARD=i, STUDY_SHARDS=n): for a slice of positions from Tal's games,
    the Tal model's top candidates + Stockfish's verdict on each + Tal-style features
    -> study-shard-<i>.jsonl
STUDY_MODE=report: merges the shards, writes tal/study-report.md and tal/tal-rules2.json
    (feature weights fitted on training games, tested on games the bot never saw).
"""
import json
import glob
import math
import os
import random
import statistics
import time

import chess
import chess.engine

from bot.style import VALUE, phase as game_phase
from tal.imitate import FEATURES as BASE, features as base_features, key
from tal.train import download_games, log, samples, tal_games

K = int(os.environ.get("STUDY_K", "10"))
N_TRAIN = int(os.environ.get("STUDY_TRAIN", "20000"))
N_HOLD = int(os.environ.get("STUDY_HOLD", "4000"))
MIN_PLY = 16                        # skip the opening (the book handles it)
DEPTH = int(os.environ.get("STUDY_DEPTH", "11"))
CKPT = os.environ.get("TAL_CKPT", "prev/tal-79m.pt")

EXTRA = ["plan_worst", "plan_break", "plan_king", "threat", "forcing", "even_trade", "opens_king_file",
         "complexity", "sf_loss", "sound_sac", "spec_sac", "unsound_sac"]
FEATURES = BASE + EXTRA
GROUPS = {
    "Tal model only": ["logp"],
    "+ basic Tal traits": BASE,
    "+ plans (worst piece, breaks, king)": BASE + EXTRA[:3],
    "+ tactics (threats, forcing, trades, open files, complexity)": BASE + EXTRA[:8],
    "+ Stockfish (move quality, sound/speculative sacrifices)": FEATURES,
}


# ---------------------------------------------------------------- features
def king_files_open(board, color):
    """Files next to the enemy king without our own pawns (open lines for our rooks)."""
    k = board.king(not color)
    if k is None:
        return 0
    f = chess.square_file(k)
    own = board.pieces(chess.PAWN, color)
    return sum(1 for x in range(max(0, f - 1), min(7, f + 1) + 1)
               if not any(chess.square_file(s) == x for s in own))


def worst_piece(board, color):
    worst, wa = None, 99
    for sq in board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color) | board.pieces(chess.ROOK, color):
        a = len(board.attacks(sq))
        if a < wa:
            worst, wa = sq, a
    return worst, wa


def kpress(board, color):
    k = board.king(not color)
    if k is None:
        return 0
    return sum(len(board.attackers(color, s)) for s in chess.SquareSet(chess.BB_KING_ATTACKS[k] | chess.BB_SQUARES[k]))


def material(board, color):
    return sum(VALUE[p.piece_type] * (1 if p.color == color else -1) for p in board.piece_map().values()
               if p.piece_type != chess.KING)


def is_sacrifice(board, move, pv):
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type in (chess.KING, chess.PAWN) or not pv or pv[0].to_square != move.to_square:
        return False
    color, start = board.turn, material(board, board.turn)
    b, deltas = board.copy(stack=False), []
    for m in [move] + pv[:2]:
        if m not in b.legal_moves:
            break
        b.push(m)
        deltas.append(material(b, color) - start)
    if len(deltas) < 2:
        return False
    settled = deltas[2] if len(deltas) >= 3 else deltas[1]
    return min(deltas[1], settled) <= -2 and settled <= -2


def extra_features(board, move, cp, best_cp, pv):
    me = board.turn
    f = dict.fromkeys(EXTRA, 0.0)
    piece = board.piece_at(move.from_square)
    captured = board.piece_type_at(move.to_square)
    worst, wa = worst_piece(board, me)
    kp0, files0 = kpress(board, me), king_files_open(board, me)
    if captured and piece and VALUE[captured] == VALUE[piece.piece_type] and board.attackers(not me, move.to_square):
        f["even_trade"] = 1.0
    b = board.copy(stack=False)
    b.push(move)
    if worst is not None and wa <= 4:
        sq = move.to_square if move.from_square == worst else worst
        p = b.piece_at(sq)
        if p and p.color == me:
            f["plan_worst"] = max(0, min(len(b.attacks(sq)) - wa, 6)) / 6.0
    if piece and piece.piece_type == chess.PAWN and not captured:
        if any(b.piece_at(s) and b.piece_at(s).piece_type == chess.PAWN and b.piece_at(s).color != me
               for s in b.attacks(move.to_square)):
            f["plan_break"] = 1.0
    f["plan_king"] = max(0, min(kpress(b, me) - kp0, 3)) / 3.0
    if piece and piece.piece_type != chess.KING:
        for s in b.attacks(move.to_square):
            t = b.piece_at(s)
            if t and t.color != me and t.piece_type != chess.KING and \
                    (VALUE[t.piece_type] > VALUE[piece.piece_type] or not b.attackers(not me, s)):
                f["threat"] = 1.0
                break
    f["forcing"] = 1.0 if (b.is_check() or captured or f["threat"]) else 0.0
    f["opens_king_file"] = 1.0 if king_files_open(b, me) > files0 else 0.0
    f["complexity"] = sum(1 for m in b.legal_moves if b.is_capture(m)) / 5.0
    loss = max(0, best_cp - cp)
    f["sf_loss"] = min(loss, 500) / 100.0
    sac = is_sacrifice(board, move, pv)
    if sac and loss <= 30:
        f["sound_sac"] = 1.0
    elif sac and loss <= 200:
        f["spec_sac"] = 1.0
    elif sac:
        f["unsound_sac"] = 1.0
    return [f[n] for n in EXTRA]


# ---------------------------------------------------------------- collect
def group_positions(games, per_player):
    """Similar-players study: the same number of positions from every player (all used for statistics)."""
    rng = random.Random(1)
    by = {}
    for i, g in enumerate(games):
        by.setdefault(g.get("player", "?"), []).append(i)
    out = []
    for who in sorted(by):
        s = [(gi, ply) for gi, ply in samples([games[i] for i in by[who]]) if ply >= MIN_PLY]
        rng.shuffle(s)
        sub = [games[i] for i in by[who]]
        out += [("train", sub, gi, ply) for gi, ply in s[:per_player]]
        log(f"{who}: {len(by[who])} games, {min(per_player, len(s))} positions")
    return out


def split_positions(games):
    if os.environ.get("STUDY_PER_PLAYER"):
        return group_positions(games, int(os.environ["STUDY_PER_PLAYER"]))
    rng = random.Random(1)                      # same split as tal/imitate.py and training
    rng.shuffle(games)
    n_hold = max(50, len(games) // 10)
    hold, train = games[:n_hold], games[n_hold:]
    out = []
    for name, gs, n in (("train", train, N_TRAIN), ("hold", hold, N_HOLD)):
        s = [(gi, ply) for gi, ply in samples(gs) if ply >= MIN_PLY]
        rng.shuffle(s)
        out += [(name, gs, gi, ply) for gi, ply in s[:n]]
    return out


def collect():
    import faulthandler
    import sys
    if os.environ.get("STUDY_DEBUG"):
        faulthandler.dump_traceback_later(int(os.environ["STUDY_DEBUG"]), exit=True, file=sys.stdout)
    from maia3.uci import Maia3UCIEngine, parse_args
    shard, shards = int(os.environ["STUDY_SHARD"]), int(os.environ["STUDY_SHARDS"])
    t0 = time.time()
    games = tal_games(download_games())
    log(f"downloaded {len(games)} games ({time.time() - t0:.0f}s)")
    items = split_positions(games)[shard::shards]
    log(f"shard {shard}/{shards}: {len(items)} positions")
    eng = Maia3UCIEngine(parse_args(["--model", "maia3-79m", "--checkpoint", CKPT]))
    eng.ensure_model_loaded()
    sf = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH", "/usr/games/stockfish"))
    sf.configure({"Threads": 4, "Hash": 256})
    out = open(f"study-shard-{shard}.jsonl", "w")
    t0 = time.time()
    try:
        _collect_loop(items, eng, sf, out, t0)
    finally:
        sf.quit()        # otherwise a crash leaves the process hanging
        out.close()


def _collect_loop(items, eng, sf, out, t0):
    for n, (split, gs, gi, ply) in enumerate(items):
        g = gs[gi]
        b = chess.Board()
        for m in g["moves"][:ply]:
            b.push_uci(m)
        tal = chess.Move.from_uci(g["moves"][ply])
        eng.self_elo = eng.oppo_elo = 2600
        eng.multipv = K
        moves = " ".join(m.uci() for m in b.move_stack)
        eng.cmd_position("position startpos" + (f" moves {moves}" if moves else ""))
        _, top = eng.score_moves()
        cands = [(c["move"], c["policy"], (c["wdl"][0] + 0.5 * c["wdl"][1]) / 1000) for c in top]
        if tal not in [c[0] for c in cands]:
            cands.append((tal, min(c[1] for c in cands) / 2 if cands else 0.001, 0.5))
        sfres = []
        for m, _, _ in cands:
            b.push(m)
            if b.is_checkmate():
                sfres.append((10000, []))
            elif b.is_game_over():
                sfres.append((0, []))
            else:
                info = sf.analyse(b, chess.engine.Limit(depth=DEPTH))
                sfres.append((info["score"].pov(not b.turn).score(mate_score=10000), list(info.get("pv", []))[:2]))
            b.pop()
        best = max(s for s, _ in sfres)
        rows = []
        for (m, p, e), (cp, pv) in zip(cands, sfres):
            f = base_features(b, m, p, e) + extra_features(b, m, cp, best, pv)
            rows.append({"m": m.uci(), "p": round(p, 5), "cp": cp, "f": [round(x, 4) for x in f]})
        tal_row = next(x for x in rows if x["m"] == tal.uci())
        sac_kind = next((k for k in ("sound_sac", "spec_sac", "unsound_sac") if tal_row["f"][FEATURES.index(k)] > 0), None)
        motif = None
        if sac_kind:
            from tal.motifs import motif_key
            motif = motif_key(b, tal)
        out.write(json.dumps({"split": split, "ply": ply, "phase": game_phase(b), "best": best,
                              "player": g.get("player", "Tal"), "sac": sac_kind, "motif": motif,
                              "tal": tal.uci(), "in_top": tal in [c[0] for c in cands[:len(top)]], "n_top": len(top),
                              "cands": rows}) + "\n")
        if n % 200 == 0 or n < 3:
            log(f"  {n}/{len(items)} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- report
def fit_eval(train, hold, names, ctx=False, epochs=500):
    import torch
    idx = [FEATURES.index(n) for n in names]

    def tensor(rows):
        mx = max(r["n_top"] for r in rows)
        X = torch.zeros(len(rows), mx, len(idx) * (3 if ctx else 1))
        M = torch.full((len(rows), mx), -1e9)
        Y = torch.zeros(len(rows), dtype=torch.long)
        for i, r in enumerate(rows):
            state = 0 if r["best"] < -80 else (2 if r["best"] > 80 else 1)
            for j, c in enumerate(r["cands"][:r["n_top"]]):   # never the appended Tal move (would leak the answer)
                v = torch.tensor([c["f"][k] for k in idx])
                if ctx:
                    X[i, j, state * len(idx):(state + 1) * len(idx)] = v
                else:
                    X[i, j] = v
                M[i, j] = 0
                if c["m"] == r["tal"]:
                    Y[i] = j
        return X, M, Y

    tr = [r for r in train if r["in_top"]]
    Xt, Mt, Yt = tensor(tr)
    Xh, Mh, Yh = tensor(hold)
    w = torch.zeros(Xt.shape[2], requires_grad=True)
    with torch.no_grad():
        for s in range(3 if ctx else 1):
            w[s * len(idx)] = 1.0                 # start from the Tal model (logp)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ w + Mt, Yt) + 0.003 * (w ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (Xh @ w + Mh).argmax(1)
        tal_ok = torch.tensor([r["in_top"] for r in hold])
        acc = float(((pred == Yh) & tal_ok).float().mean())
    return acc, {n: round(float(w[i]), 3) for i, n in enumerate(names)} if not ctx else \
        {s: {n: round(float(w[k * len(idx) + i]), 3) for i, n in enumerate(names)}
         for k, s in enumerate(["behind", "equal", "ahead"])}


def group_report(rows):
    import collections
    F = {n: i for i, n in enumerate(FEATURES)}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["player"]].append(r)
    L = [f"# Attacking players compared ({len(rows)} positions, Stockfish-checked)", "",
         "Brilliant-type = a piece sacrifice within 0.3 pawns of the best natural move.", "",
         "| Player | Positions | Avg loss (cp) | Blunders | Sacrifices / 100 moves | Brilliant-type / 100 | Speculative / 100 | Unsound / 100 | Took an available sound sac |",
         "|---|---|---|---|---|---|---|---|---|"]
    stats = []
    for who, rs in by.items():
        n = len(rs)
        loss = [max(0, r["best"] - next(c for c in r["cands"] if c["m"] == r["tal"])["cp"]) for r in rs]
        k = collections.Counter(r["sac"] for r in rs if r["sac"])
        av = [r for r in rs if any(c["f"][F["sound_sac"]] for c in r["cands"][:r["n_top"]])]
        took = sum(1 for r in av if r["sac"] == "sound_sac") / max(1, len(av))
        stats.append((k["sound_sac"] / n, who, n, statistics.mean(min(x, 500) for x in loss),
                      sum(x > 200 for x in loss) / n, sum(k.values()) / n, k["spec_sac"] / n, k["unsound_sac"] / n, took))
    for br, who, n, al, bl, sac, spec, uns, took in sorted(stats, reverse=True):
        L.append(f"| {who} | {n} | {al:.0f} | {bl:.1%} | {sac * 100:.2f} | {br * 100:.2f} | {spec * 100:.2f} | {uns * 100:.2f} | {took:.0%} |")
    motifs = collections.Counter(r["motif"] for r in rows if r["sac"] == "sound_sac" and r["motif"])
    spec_m = collections.Counter(r["motif"] for r in rows if r["sac"] == "spec_sac" and r["motif"])
    L += ["", "## Most common brilliant-type sacrifices (all players)", ""] + \
         [f"- {m}: {c}" for m, c in motifs.most_common(30)]
    L += ["", "## Most common speculative sacrifices", ""] + [f"- {m}: {c}" for m, c in spec_m.most_common(15)]
    json.dump({"favourite_sacrifices": {m: c for m, c in (motifs + spec_m).items() if c >= 3},
               "players": {s[1]: {"brilliant_per_100": round(s[0] * 100, 2), "avg_loss": round(s[3], 1)} for s in stats}},
              open("tal/attack-motifs.json", "w"), indent=1)
    text = "\n".join(L)
    open("tal/attack-report.md", "w").write(text + "\n")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


def report():
    rows = [json.loads(l) for f in sorted(glob.glob("study-shard-*.jsonl")) for l in open(f)]
    if any(r.get("player", "Tal") != "Tal" for r in rows):
        return group_report(rows)
    train = [r for r in rows if r["split"] == "train"]
    hold = [r for r in rows if r["split"] == "hold"]
    F = {n: i for i, n in enumerate(FEATURES)}
    L = [f"# Mikhail Tal study ({len(rows)} positions from his games, move 9 onwards)", ""]

    def tal_c(r):
        return next(c for c in r["cands"] if c["m"] == r["tal"])

    def top_c(r):
        return max(r["cands"][:r["n_top"]], key=lambda c: c["p"])

    # 1. accuracy
    loss = [max(0, r["best"] - tal_c(r)["cp"]) for r in rows]
    mloss = [max(0, r["best"] - top_c(r)["cp"]) for r in rows]
    L += ["## 1. How accurate Tal was (Stockfish, among the natural candidate moves)", "",
          "| | Tal | Tal model's first choice |", "|---|---|---|",
          f"| average loss (centipawns) | {statistics.mean(min(x, 500) for x in loss):.0f} | {statistics.mean(min(x, 500) for x in mloss):.0f} |",
          f"| best or within 0.2 pawns | {sum(x <= 20 for x in loss) / len(loss):.0%} | {sum(x <= 20 for x in mloss) / len(mloss):.0%} |",
          f"| mistakes (1-2 pawns) | {sum(100 < x <= 200 for x in loss) / len(loss):.1%} | {sum(100 < x <= 200 for x in mloss) / len(mloss):.1%} |",
          f"| blunders (> 2 pawns) | {sum(x > 200 for x in loss) / len(loss):.1%} | {sum(x > 200 for x in mloss) / len(mloss):.1%} |", ""]
    for label, cond in (("behind (< -0.8)", lambda r: r["best"] < -80), ("equal", lambda r: -80 <= r["best"] <= 80),
                        ("ahead (> +0.8)", lambda r: r["best"] > 80),
                        ("sharp (3+ captures available)", lambda r: tal_c(r)["f"][F["complexity"]] >= 0.6)):
        sub = [max(0, r["best"] - tal_c(r)["cp"]) for r in rows if cond(r)]
        if sub:
            L.append(f"- When {label}: average loss {statistics.mean(min(x, 500) for x in sub):.0f} cp, "
                     f"blunders {sum(x > 200 for x in sub) / len(sub):.1%} ({len(sub)} positions)")
    # 2. what Tal does more / less than the model
    L += ["", "## 2. What Tal does more or less often than the Tal model's first choice", "",
          "| Trait | Tal | Model | Ratio |", "|---|---|---|---|"]
    for n in FEATURES[2:]:
        a = statistics.mean(1.0 if tal_c(r)["f"][F[n]] > 0 else 0.0 for r in rows)
        b = statistics.mean(1.0 if top_c(r)["f"][F[n]] > 0 else 0.0 for r in rows)
        if a + b > 0.002:
            L.append(f"| {n} | {a:.1%} | {b:.1%} | {a / b if b else float('inf'):.2f}x |")
    # 3. sacrifices
    sacs = [r for r in rows if any(tal_c(r)["f"][F[n]] for n in ("sound_sac", "spec_sac", "unsound_sac"))]
    sound_av = [r for r in rows if any(c["f"][F["sound_sac"]] for c in r["cands"][:r["n_top"]])]
    spec_av = [r for r in rows if any(c["f"][F["spec_sac"]] for c in r["cands"][:r["n_top"]])]
    kinds = {n: sum(tal_c(r)["f"][F[n]] > 0 for r in sacs) for n in ("sound_sac", "spec_sac", "unsound_sac")}
    L += ["", "## 3. Sacrifices", "",
          f"- Tal sacrificed a piece in {len(sacs)} of {len(rows)} positions ({len(sacs) / len(rows):.1%}, about once every {len(rows) / max(1, len(sacs)):.0f} moves).",
          f"- Of those: sound (within 0.3 pawns of best) {kinds['sound_sac']}, speculative (0.3-2 pawns worse) {kinds['spec_sac']}, unsound (2+ pawns worse) {kinds['unsound_sac']}.",
          f"- Sound sacrifice available among natural moves: {len(sound_av)} positions; Tal played one in "
          f"{sum(tal_c(r)['f'][F['sound_sac']] > 0 for r in sound_av) / max(1, len(sound_av)):.0%} of them.",
          f"- Speculative sacrifice (0.3-2 pawns worse) available: {len(spec_av)} positions; Tal played one in "
          f"{sum(tal_c(r)['f'][F['spec_sac']] > 0 for r in spec_av) / max(1, len(spec_av)):.0%} of them."]
    after = sorted(tal_c(r)["cp"] for r in sacs)
    before = sorted(r["best"] for r in sacs)
    if after:
        q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
        L += [f"- Position after Tal's sacrifices (Stockfish): 10% {q(after, .1):+}, median {q(after, .5):+}, 90% {q(after, .9):+} cp.",
              f"- Position before them: 10% {q(before, .1):+}, median {q(before, .5):+}, 90% {q(before, .9):+} cp."]
    # 4. fitted rules
    L += ["", "## 4. Rules fitted on Tal's games, tested on games the bot never saw", "",
          "| Rules | Exact Tal move |", "|---|---|"]
    results = {}
    for label, names in GROUPS.items():
        acc, w = fit_eval(train, hold, names)
        results[label] = (acc, w)
        L.append(f"| {label} | {acc:.1%} |")
        log(f"{label}: {acc:.1%}")
    acc_ctx, w_ctx = fit_eval(train, hold, FEATURES, ctx=True)
    L.append(f"| all rules, separate weights when behind / equal / ahead | {acc_ctx:.1%} |")
    best_label = max(results, key=lambda k: results[k][0])
    best = {"features": GROUPS[best_label], "weights": results[best_label][1], "accuracy": results[best_label][0],
            "baseline": results["Tal model only"][0], "context": {"accuracy": acc_ctx, "weights": w_ctx}}
    json.dump(best, open("tal/tal-rules2.json", "w"), indent=1)
    L += ["", f"Best: **{best_label}**. Weights (positive = Tal likes it more than the model does):", "```",
          json.dumps(results[best_label][1], indent=1), "```"]
    text = "\n".join(L)
    open("tal/study-report.md", "w").write(text + "\n")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


if __name__ == "__main__":
    collect() if os.environ.get("STUDY_MODE", "collect") == "collect" else report()
