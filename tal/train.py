"""Fine-tune Maia-3 79M on Mikhail Tal's moves.

Steps: download Tal's games -> keep positions where Tal is to move -> 90/10 split by game ->
measure how often the model predicts Tal's actual move on the held-out games (before) ->
train (policy = Tal's move, value = game result) -> measure again (after) -> save checkpoint.

Env: TAL_MINUTES (training time budget), TAL_EPOCHS, TAL_LR, TAL_RESUME (path to checkpoint).
"""
import io
import json
import os
import random
import re
import sys
import time
import zipfile
from collections import deque

import chess
import chess.pgn
import requests
import torch
import torch.nn.functional as F
from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.uci import Maia3UCIEngine, parse_args
from maia3.utils import mirror_move

SOURCES = ["https://www.pgnmentor.com/players/Tal.zip"]
ELO = 2600                     # the bot plays with elo 2600, so Tal's style is attached to that setting
OUT = os.environ.get("TAL_OUT", "tal-79m.pt")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download_games():
    for url in SOURCES:
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                text = "\n".join(z.read(n).decode("latin-1") for n in z.namelist() if n.lower().endswith(".pgn"))
            log(f"downloaded {url}: {len(text) // 1024} KB")
            return text
        except Exception as e:
            log(f"source failed {url}: {e}")
    sys.exit("No source for Tal's games worked.")


def tal_games(text):
    pgn, games = io.StringIO(text), []
    while True:
        g = chess.pgn.read_game(pgn)
        if g is None:
            break
        w, b = g.headers.get("White", ""), g.headers.get("Black", "")
        is_w, is_b = bool(re.match(r"^Tal\b", w)), bool(re.match(r"^Tal\b", b))
        if is_w == is_b or g.headers.get("Variant") or g.headers.get("SetUp") == "1":
            continue
        moves = list(g.mainline_moves())
        if len(moves) < 10:
            continue
        res = g.headers.get("Result", "*")
        games.append({"moves": [m.uci() for m in moves], "tal": chess.WHITE if is_w else chess.BLACK,
                      "result": res})
    return games


def samples(games):
    """(game index, ply) for every position where Tal is to move."""
    out = []
    for gi, g in enumerate(games):
        for ply in range(len(g["moves"])):
            if (ply % 2 == 0) == (g["tal"] == chess.WHITE):
                out.append((gi, ply))
    return out


class Encoder:
    def __init__(self, eng):
        self.eng, self.cfg = eng, eng.cfg

    def encode(self, g, ply):
        board, hist = chess.Board(), deque(maxlen=self.cfg.history)
        hist.append(tokenize_board(board))
        for m in g["moves"][:ply]:
            board.push_uci(m)
            hist.append(tokenize_board(board))
        tokens = get_historical_tokens(hist, self.cfg, 0.0, 0.0, 0.0, 0.0)
        mask = get_legal_moves_mask(board, self.eng.all_moves_dict)
        uci = g["moves"][ply]
        if board.turn == chess.BLACK:
            uci = mirror_move(uci)
        target = self.eng.all_moves_dict.get(uci)
        res = g["result"]
        tal_won = (res == "1-0" and g["tal"] == chess.WHITE) or (res == "0-1" and g["tal"] == chess.BLACK)
        value = 1 if res == "1/2-1/2" else (2 if tal_won else 0) if res in ("1-0", "0-1") else -100
        return tokens, mask, target, value

    def batch(self, games, items):
        rows = [self.encode(games[gi], ply) for gi, ply in items]
        rows = [r for r in rows if r[2] is not None]
        t = torch.stack([r[0] for r in rows])
        m = torch.stack([r[1] for r in rows])
        y = torch.tensor([r[2] for r in rows])
        v = torch.tensor([r[3] for r in rows])
        return t, m, y, v


def forward(model, t, m):
    elos = torch.full((t.size(0),), ELO, dtype=torch.float)
    logits_move, logits_value, _ = model(t, elos, elos)
    return logits_move.masked_fill(~m, float("-inf")), logits_value


@torch.no_grad()
def accuracy(model, enc, games, items, bs=64):
    model.eval()
    hit = n = 0
    for i in range(0, len(items), bs):
        t, m, y, _ = enc.batch(games, items[i:i + bs])
        lm, _ = forward(model, t, m)
        hit += (lm.argmax(-1) == y).sum().item()
        n += len(y)
    return hit / max(1, n)


def main():
    torch.set_num_threads(os.cpu_count())
    minutes = float(os.environ.get("TAL_MINUTES", "30"))
    epochs = float(os.environ.get("TAL_EPOCHS", "2"))
    lr = float(os.environ.get("TAL_LR", "2e-5"))
    bs = int(os.environ.get("TAL_BATCH", "32"))
    eval_n = int(os.environ.get("TAL_EVAL", "3000"))

    games = tal_games(download_games())
    rng = random.Random(1)
    rng.shuffle(games)
    n_hold = max(50, len(games) // 10)
    hold, train = games[:n_hold], games[n_hold:]
    s_train, s_hold = samples(train), samples(hold)
    rng.shuffle(s_hold)
    s_hold = s_hold[:eval_n]
    log(f"Tal games: {len(games)} (train {len(train)}, held out {len(hold)}); "
        f"Tal moves: train {len(s_train)}, eval {len(s_hold)}")

    eng = Maia3UCIEngine(parse_args(["--model", "maia3-79m"]))
    eng.ensure_model_loaded()
    model = eng.model
    if os.environ.get("TAL_RESUME") and os.path.exists(os.environ["TAL_RESUME"]):
        ck = torch.load(os.environ["TAL_RESUME"], map_location="cpu")
        model.load_state_dict(ck["model_state_dict"])
        log("resumed from", os.environ["TAL_RESUME"], ck.get("info"))
    enc = Encoder(eng)

    t0 = time.time()
    base = accuracy(model, enc, hold, s_hold)
    log(f"held-out Tal move match BEFORE training: {base:.1%} ({time.time() - t0:.0f}s)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = int(len(s_train) * epochs / bs)
    deadline, step, seen, t_start = time.time() + minutes * 60, 0, 0, time.time()
    model.train()
    while step < total_steps and time.time() < deadline:
        rng.shuffle(s_train)
        for i in range(0, len(s_train), bs):
            if step >= total_steps or time.time() >= deadline:
                break
            t, m, y, v = enc.batch(train, s_train[i:i + bs])
            lm, lv = forward(model, t, m)
            loss = F.cross_entropy(lm, y)
            keep = v >= 0
            if keep.any():
                loss = loss + 0.25 * F.cross_entropy(lv[keep], v[keep])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            seen += len(y)
            if step % 50 == 0:
                rate = seen / (time.time() - t_start)
                log(f"step {step}/{total_steps} loss {loss.item():.3f} | {rate:.1f} positions/s | "
                    f"epoch {seen / len(s_train):.2f}")
    rate = seen / max(1, time.time() - t_start)
    after = accuracy(model, enc, hold, s_hold)
    info = {"before": base, "after": after, "positions_trained": seen, "epochs": seen / len(s_train),
            "positions_per_s": rate, "train_moves": len(s_train), "games": len(games)}
    log(f"held-out Tal move match AFTER training: {after:.1%} (before {base:.1%})")
    torch.save({"model_state_dict": model.state_dict(), "info": info}, OUT)
    json.dump(info, open("tal-info.json", "w"), indent=1)
    hours_epoch = len(s_train) / rate / 3600
    text = (f"## Tal training\n- games: {len(games)}, Tal moves for training: {len(s_train)}\n"
            f"- trained on {seen} positions ({seen / len(s_train):.2f} epochs) at {rate:.1f} positions/s "
            f"(one epoch ≈ {hours_epoch:.1f} h on this machine)\n"
            f"- predicts Tal's move on held-out games: **{base:.1%} → {after:.1%}**\n")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text)


if __name__ == "__main__":
    main()
