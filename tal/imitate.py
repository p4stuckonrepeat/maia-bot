"""Get the bot as close to Tal as possible, and measure how close.

1. Book: every position Tal reached in the first 30 half-moves, with the moves he played there.
2. Rules: Tal-style features (attack the king, sacrifices, checks, open lines, activity, ...)
   whose weights are FITTED on Tal's games so that his actual move ranks first as often as possible.
3. Evaluation on held-out Tal games: how often the bot plays exactly Tal's move with
   normal Maia / the Tal model / + rules / + book.
Outputs: tal/tal-rules.json (feature weights), book-tal.json (built from all games).
"""
import json
import math
import os
import random
import time

import chess
import torch

from bot.style import VALUE, positional_score
from tal.train import download_games, log, samples, tal_games

N_TRAIN = int(os.environ.get("IMIT_TRAIN_POSITIONS", "6000"))
N_EVAL = int(os.environ.get("IMIT_EVAL_POSITIONS", "2500"))
K = 12
BOOK_PLIES = 30
CKPT = os.environ.get("TAL_CKPT", "prev/tal-79m.pt")
NAME = os.environ.get("IMIT_NAME", "tal")


def key(board):
    return " ".join(board.fen().split()[:4])


def king_zone(board, color):
    k = board.king(color)
    return chess.SquareSet(chess.BB_KING_ATTACKS[k] | chess.BB_SQUARES[k]) if k is not None else chess.SquareSet()


def zone_pressure(board, color, zone):
    return sum(len(board.attackers(color, sq)) for sq in zone)


FEATURES = ["logp", "exp", "capture", "check", "sacrifice", "king_pressure", "near_king", "pawn_storm",
            "mobility", "retreat", "queen_trade", "castle", "develop", "center_pawn", "pawn_break", "safety"]


def features(board, move, policy, exp):
    color = board.turn
    piece = board.piece_type_at(move.from_square)
    captured = board.piece_type_at(move.to_square) or (chess.PAWN if board.is_en_passant(move) else None)
    gain = VALUE[captured] if captured else 0
    ek = board.king(not color)
    zone = king_zone(board, not color)
    pressure_before = zone_pressure(board, color, zone)
    mob_before = board.legal_moves.count()
    rel = (lambda sq: chess.square_rank(sq)) if color == chess.WHITE else (lambda sq: 7 - chess.square_rank(sq))
    f = dict.fromkeys(FEATURES, 0.0)
    f["logp"] = math.log(policy + 1e-6)
    f["exp"] = exp - 0.5
    f["capture"] = 1.0 if captured else 0.0
    f["check"] = 1.0 if board.gives_check(move) else 0.0
    f["castle"] = 1.0 if board.is_castling(move) else 0.0
    f["develop"] = 1.0 if piece in (chess.KNIGHT, chess.BISHOP) and rel(move.from_square) == 0 else 0.0
    f["center_pawn"] = 1.0 if piece == chess.PAWN and move.to_square in (chess.D4, chess.E4, chess.D5, chess.E5) else 0.0
    f["retreat"] = 1.0 if piece not in (chess.PAWN, chess.KING) and rel(move.to_square) < rel(move.from_square) else 0.0
    f["queen_trade"] = 1.0 if piece == chess.QUEEN and captured == chess.QUEEN else 0.0
    if ek is not None:
        f["near_king"] = 1.0 if chess.square_distance(move.to_square, ek) <= 2 else 0.0
        if piece == chess.PAWN and abs(chess.square_file(move.to_square) - chess.square_file(ek)) <= 1 and not captured:
            f["pawn_storm"] = 1.0
    f["safety"] = positional_score(board, move)
    board.push(move)
    try:
        f["king_pressure"] = (zone_pressure(board, color, zone) - pressure_before) / 3.0
        if piece != chess.KING:
            attackers = board.attackers(not color, move.to_square)
            if attackers:
                least = min(VALUE[board.piece_type_at(a)] for a in attackers)
                defended = bool(board.attackers(color, move.to_square))
                loss = VALUE[piece] - gain if (least < VALUE[piece] or not defended) else 0
                f["sacrifice"] = max(0.0, loss) / 3.0
        if piece == chess.PAWN:
            f["pawn_break"] = 1.0 if any(board.piece_type_at(s) == chess.PAWN and board.color_at(s) != color
                                         for s in board.attacks(move.to_square)) else 0.0
        board.push(chess.Move.null()) if not board.is_check() else None
        if board.turn == color:
            f["mobility"] = (board.legal_moves.count() - mob_before) / 10.0
            board.pop()
    finally:
        board.pop()
    return [f[n] for n in FEATURES]


def candidates(eng, board):
    e = eng
    e.self_elo = e.oppo_elo = 2600
    e.multipv = K
    moves = " ".join(m.uci() for m in board.move_stack)
    root = board.root().fen()
    base = "position startpos" if root == chess.STARTING_FEN else f"position fen {root}"
    e.cmd_position(base + (f" moves {moves}" if moves else ""))
    _, top = e.score_moves()
    return top


def build_book(games):
    book = {"white": {}, "black": {}}
    for g in games:
        b = chess.Board()
        side = "white" if g["tal"] == chess.WHITE else "black"
        for ply, m in enumerate(g["moves"][:BOOK_PLIES]):
            if b.turn == g["tal"]:
                d = book[side].setdefault(key(b), {})
                d[m] = d.get(m, 0) + 1
            b.push_uci(m)
    return book


def dataset(eng, games, items):
    X, Y, P = [], [], []   # features per candidate, index of Tal's move (or -1), candidate moves
    for n, (gi, ply) in enumerate(items):
        g = games[gi]
        b = chess.Board()
        for m in g["moves"][:ply]:
            b.push_uci(m)
        tal_move = chess.Move.from_uci(g["moves"][ply])
        top = candidates(eng, b)
        feats, idx = [], -1
        for i, c in enumerate(top):
            w, d, _ = c["wdl"]
            feats.append(features(b, c["move"], c["policy"], (w + 0.5 * d) / 1000))
            if c["move"] == tal_move:
                idx = i
        X.append(feats)
        Y.append(idx)
        P.append((key(b), g["tal"], g["moves"][ply]))
        if n % 1000 == 0:
            log(f"  features {n}/{len(items)}")
    return X, Y, P


def fit(X, Y, epochs=400):
    data = [(torch.tensor(x), y) for x, y in zip(X, Y) if y >= 0]
    w = torch.zeros(len(FEATURES), requires_grad=True)
    with torch.no_grad():
        w[0] = 1.0                                   # start from pure Maia
    opt = torch.optim.Adam([w], lr=0.05)
    for ep in range(epochs):
        opt.zero_grad()
        loss = sum(-torch.log_softmax(x @ w, 0)[y] for x, y in data) / len(data) + 0.01 * (w[1:] ** 2).sum()
        loss.backward()
        opt.step()
        if ep % 100 == 0:
            log(f"  fit epoch {ep}: loss {loss.item():.4f}")
    return w.detach()


def match(X, Y, w):
    hit = 0
    for x, y in zip(X, Y):
        s = torch.tensor(x) @ w
        hit += int(y >= 0 and int(s.argmax()) == y)
    return hit / len(X)


def main():
    from maia3.uci import Maia3UCIEngine, parse_args
    games = tal_games(download_games())
    rng = random.Random(1)
    rng.shuffle(games)
    n_hold = max(50, len(games) // 10)
    hold, train = games[:n_hold], games[n_hold:]
    s_train, s_hold = samples(train), samples(hold)
    rng.shuffle(s_train)
    rng.shuffle(s_hold)
    s_train, s_hold = s_train[:N_TRAIN], s_hold[:N_EVAL]

    results = {}
    for label, ckpt in [("normal Maia-3", None), (f"{NAME} model", CKPT)]:
        args = ["--model", "maia3-79m", "--device", "cuda" if torch.cuda.is_available() else "cpu"] + (["--checkpoint", ckpt] if ckpt else [])
        eng = Maia3UCIEngine(parse_args(args))
        eng.ensure_model_loaded()
        t = time.time()
        Xh, Yh, Ph = dataset(eng, hold, s_hold)
        base_w = torch.zeros(len(FEATURES)); base_w[0] = 1.0
        results[label] = match(Xh, Yh, base_w)
        log(f"{label}: {results[label]:.1%} exact player moves on held-out games ({time.time() - t:.0f}s); "
            f"Tal's move among top {K}: {sum(y >= 0 for y in Yh) / len(Yh):.1%}")
    # eng is now the Tal model: learn the rule weights on training games
    Xt, Yt, _ = dataset(eng, train, s_train)
    w = fit(Xt, Yt)
    results[f"{NAME} model + learned rules"] = match(Xh, Yh, w)
    # book from the training games only (fair test), then check held-out positions it covers
    book = build_book(train)
    hit = 0
    for (k, color, tal_uci), x, y in zip(Ph, Xh, Yh):
        side = "white" if color == chess.WHITE else "black"
        if k in book[side]:
            hit += int(max(book[side][k], key=book[side][k].get) == tal_uci)
        else:
            hit += int(y >= 0 and int((torch.tensor(x) @ w).argmax()) == y)
    results[f"+ {NAME}'s own opening book"] = hit / len(Ph)

    weights = {n: round(float(v), 4) for n, v in zip(FEATURES, w)}
    json.dump({"features": FEATURES, "weights": weights, "results": results}, open(f"tal/{NAME}-rules.json", "w"), indent=1)
    json.dump(build_book(games), open(f"book-{NAME}.json", "w"), separators=(",", ":"))
    lines = [f"## {NAME} imitation", f"{len(s_hold)} positions from {len(hold)} {NAME} games the bot never saw", "",
             "| Version | Plays exactly the player's move |", "|---|---|"] + \
            [f"| {k} | {v:.1%} |" for k, v in results.items()] + ["", "Learned rule weights:", "```",
             json.dumps(weights, indent=1), "```"]
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


if __name__ == "__main__":
    main()
