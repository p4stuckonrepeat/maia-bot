"""Anti-blunder fine-tuning of the Tal model, using the overnight study (tal/study.py).

For every studied position we know Tal's move and Stockfish's verdict on the Tal model's natural
candidates. Training keeps the style (cross-entropy on Tal's move) and pushes down the probability
of natural-looking moves that Stockfish says lose 2+ pawns (unlikelihood loss).
Tested on positions from games the model never saw: exact Tal move %, and how often the model's
first choice is a blunder.

Env: STUDY_FILE (merged study jsonl, gz ok), TAL_RESUME (current tal model), AB_EPOCHS, AB_LR,
AB_BLOCKS, AB_LAMBDA, AB_BLUNDER_CP, AB_OUT.
"""
import gzip
import json
import os
import random
import re
import time

import chess
import torch
import torch.nn.functional as F
from maia3.uci import Maia3UCIEngine, parse_args
from maia3.utils import mirror_move

import tal.study as study
from tal.train import DEVICE, Encoder, download_games, forward, log, tal_games

BLUNDER = int(os.environ.get("AB_BLUNDER_CP", "200"))


def load_rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f]


def rebuild(rows_by_shard, games, shards=8):
    """Positions in the same order tal/study.py used: shard s, row j = items[s + shards*j]."""
    study.N_TRAIN, study.N_HOLD = 40000, 6000
    items = study.split_positions(games)
    out, bad = [], 0
    for s in range(shards):
        mine = items[s::shards]
        for r, (split, gs, gi, ply) in zip(rows_by_shard.get(s, []), mine):
            g = gs[gi]
            if g["moves"][ply] != r["tal"] or ply != r["ply"]:
                bad += 1
                continue
            out.append((r, g, ply))
    log(f"rebuilt {len(out)} positions ({bad} mismatches)")
    return out


def targets(enc, r, g, ply):
    """(tokens, legal mask, Tal move index, blunder move indices)"""
    t, m, y, _ = enc.encode(g, ply)
    board = chess.Board()
    for mv in g["moves"][:ply]:
        board.push_uci(mv)
    best = r["best"]
    bl = []
    for c in r["cands"][:r["n_top"]]:
        if best - c["cp"] >= BLUNDER and c["m"] != r["tal"] and c["p"] >= 0.01:
            u = c["m"] if board.turn == chess.WHITE else mirror_move(c["m"])
            j = enc.all_moves_dict.get(u)
            if j is not None:
                bl.append(j)
    return t, m, y, bl


def batches(data, bs, shuffle):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [data[k] for k in idx[i:i + bs]]
        t = torch.stack([c[0] for c in chunk]).to(DEVICE)
        m = torch.stack([c[1] for c in chunk]).to(DEVICE)
        y = torch.tensor([c[2] if c[2] is not None else -1 for c in chunk]).to(DEVICE)
        B = torch.zeros(len(chunk), m.shape[1], dtype=torch.bool)
        for n, c in enumerate(chunk):
            for j in c[3]:
                B[n, j] = True
        yield t, m, y, B.to(DEVICE), [c[4] for c in chunk]


@torch.no_grad()
def evaluate(model, data):
    """exact Tal move %, blunder rate of the model's first choice (where Stockfish judged it)."""
    model.eval()
    hit = n = bl = judged = 0
    for t, m, y, B, rows in batches(data, 256, False):
        lm, _ = forward(model, t, m)
        top = lm.argmax(-1)
        for k, r in enumerate(rows):
            if y[k] >= 0:
                n += 1
                hit += int(top[k] == y[k])
            j = int(top[k])
            if B[k, j]:
                bl += 1
            judged += 1
    return hit / max(1, n), bl / max(1, judged)


def main():
    path = os.environ.get("STUDY_FILE", "study.jsonl.gz")
    rows = load_rows(path)
    by_shard = {}
    for r in rows:
        by_shard.setdefault(r["shard"], []).append(r)
    games = tal_games(download_games())
    data = rebuild(by_shard, games)
    eng = Maia3UCIEngine(parse_args(["--model", "maia3-79m", "--device", DEVICE]))
    eng.ensure_model_loaded()
    model = eng.model
    ck = torch.load(os.environ.get("TAL_RESUME", "tal-79m.pt"), map_location=DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    enc = Encoder(eng)
    t0 = time.time()
    enc_data = {"train": [], "hold": []}
    for r, g, ply in data:
        t, m, y, bl = targets(enc, r, g, ply)
        enc_data[r["split"]].append((t, m, y, bl, r))
    log(f"encoded train {len(enc_data['train'])}, held out {len(enc_data['hold'])} "
        f"({time.time() - t0:.0f}s); positions with a natural blunder: "
        f"{sum(1 for x in enc_data['train'] if x[3])}")
    before = evaluate(model, enc_data["hold"])
    log(f"BEFORE: exact Tal move {before[0]:.1%}, first choice is a blunder {before[1]:.2%}")

    blocks, lam = int(os.environ.get("AB_BLOCKS", "4")), float(os.environ.get("AB_LAMBDA", "1.0"))
    n_blocks = len(model.transformer.layers)
    for name, p in model.named_parameters():
        mt = re.match(r"transformer\.layers\.(\d+)\.", name)
        p.requires_grad = not (name.startswith(("token_projection", "elo_embedding", "gab_shared", "abs_pe")) or
                               (mt is not None and int(mt.group(1)) < n_blocks - blocks))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(os.environ.get("AB_LR", "3e-6")), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    best, best_state = before, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [("before",) + before]
    for ep in range(int(os.environ.get("AB_EPOCHS", "4"))):
        model.train()
        for t, m, y, B, _ in batches(enc_data["train"], int(os.environ.get("AB_BATCH", "256")), True):
            lm, _ = forward(model, t, m)
            ok = y >= 0
            loss = F.cross_entropy(lm[ok], y[ok]) if ok.any() else lm.sum() * 0
            if B.any():
                p = torch.softmax(lm, -1).clamp(max=1 - 1e-4)
                loss = loss + lam * (-(torch.log1p(-p[B]))).sum() / t.size(0)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        res = evaluate(model, enc_data["hold"])
        history.append((f"epoch {ep + 1}",) + res)
        log(f"epoch {ep + 1}: exact Tal move {res[0]:.1%}, first choice is a blunder {res[1]:.2%}")
        # better = fewer blunders without losing more than 0.3 points of Tal-likeness
        if res[0] >= before[0] - 0.003 and res[1] < best[1]:
            best, best_state = res, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = os.environ.get("AB_OUT", "antiblunder.pt")
    info = {"before": before, "after": best, "history": history, "improved": best != before}
    torch.save({"model_state_dict": best_state, "info": info}, out)
    json.dump(info, open("antiblunder-info.json", "w"), indent=1)
    log("done", info)


if __name__ == "__main__":
    main()
