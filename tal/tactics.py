"""Tactics training for the Tal model (pod). Stockfish only provides training labels.

Data: self-play positions (tools/selfplay_positions.py): the Tal model's 8 most natural moves with
Stockfish's verdict. Target = the model's own move probabilities with the blunders (2+ pawns worse than
the best natural move) removed -> its instinct keeps its style but stops liking natural-looking blunders.
Retention: Tal's real moves from the study positions keep the Tal style.
Saved only if blunders of the model's first choice drop clearly and Tal-likeness stays.

Env: TACTICS_FILE, STUDY_FILE, TAL_RESUME, TT_EPOCHS, TT_LR, TT_BLOCKS, TT_BLUNDER_CP, TT_OUT
"""
import gzip
import json
import os
import random
import re

import chess
import torch
import torch.nn.functional as F
from maia3.uci import Maia3UCIEngine, parse_args
from maia3.utils import mirror_move

from tal.antiblunder import batches as tal_batches, evaluate as tal_evaluate, load_rows, rebuild, targets
from tal.train import DEVICE, Encoder, download_games, forward, log, tal_games

BL = int(os.environ.get("TT_BLUNDER_CP", "150"))


def encode_sp(enc, r):
    moves = r["moves"]
    g = {"moves": moves + [r["cands"][0]["m"]], "tal": chess.WHITE, "result": "*"}
    t, m, _, _ = enc.encode(g, len(moves))
    black = len(moves) % 2 == 1
    idx, q, blunder = [], [], []
    for c in r["cands"]:
        u = mirror_move(c["m"]) if black else c["m"]
        j = enc.all_moves_dict.get(u)
        if j is None:
            continue
        bad = r["best"] - c["cp"] >= BL
        idx.append(j)
        q.append(0.0 if bad else c["p"])
        blunder.append(bad)
    if not idx or sum(q) <= 0 or not any(blunder):
        return None
    tgt = torch.zeros(m.shape[0])
    s = sum(q)
    for j, w in zip(idx, q):
        tgt[j] = w / s
    return t, m, tgt, idx, blunder


@torch.no_grad()
def blunder_rate(model, data):
    model.eval()
    bad = judged = 0
    for i in range(0, len(data), 256):
        chunk = data[i:i + 256]
        t = torch.stack([c[0] for c in chunk]).to(DEVICE)
        m = torch.stack([c[1] for c in chunk]).to(DEVICE)
        lm, _ = forward(model, t, m)
        top = lm.argmax(-1).tolist()
        for k, c in enumerate(chunk):
            if top[k] in c[3]:
                judged += 1
                bad += c[4][c[3].index(top[k])]
    return bad / max(1, judged)


def main():
    op = gzip.open if os.environ.get("TACTICS_FILE", "tactics.jsonl.gz").endswith(".gz") else open
    with op(os.environ.get("TACTICS_FILE", "tactics.jsonl.gz"), "rt") as f:
        sp = [json.loads(l) for l in f]
    random.Random(7).shuffle(sp)
    eng = Maia3UCIEngine(parse_args(["--model", "maia3-79m", "--device", DEVICE]))
    eng.ensure_model_loaded()
    model = eng.model
    model.load_state_dict(torch.load(os.environ.get("TAL_RESUME", "tal-79m.pt"), map_location=DEVICE)["model_state_dict"])
    enc = Encoder(eng)
    data = [x for x in (encode_sp(enc, r) for r in sp) if x is not None]
    n_hold = max(200, len(data) // 10)
    hold, train = data[:n_hold], data[n_hold:]
    # Tal-likeness retention + measurement (study positions)
    rows = load_rows(os.environ.get("STUDY_FILE", "study.jsonl.gz"))
    by = {}
    for r in rows:
        by.setdefault(r["shard"], []).append(r)
    tal = {"train": [], "hold": []}
    for r, g, ply in rebuild(by, tal_games(download_games())):
        t, m, y, bl = targets(enc, r, g, ply)
        tal[r["split"]].append((t, m, y, bl, r))
    log(f"tactics positions with a natural blunder: train {len(train)}, held out {len(hold)}; "
        f"Tal positions train {len(tal['train'])}, held out {len(tal['hold'])}")
    b0, t0 = blunder_rate(model, hold), tal_evaluate(model, tal["hold"])[0]
    log(f"BEFORE: first choice is a blunder in {b0:.1%} of tactics positions; exact Tal move {t0:.1%}")
    blocks = int(os.environ.get("TT_BLOCKS", "4"))
    nb = len(model.transformer.layers)
    for name, p in model.named_parameters():
        mt = re.match(r"transformer\.layers\.(\d+)\.", name)
        p.requires_grad = not (name.startswith(("token_projection", "elo_embedding", "gab_shared", "abs_pe")) or
                               (mt is not None and int(mt.group(1)) < nb - blocks))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(os.environ.get("TT_LR", "5e-6")), weight_decay=0.01)
    best = (b0, t0, None)
    history = [("before", b0, t0)]
    tal_iter = None
    for ep in range(int(os.environ.get("TT_EPOCHS", "4"))):
        model.train()
        random.shuffle(train)
        tal_iter = iter(tal_batches(tal["train"], 128, True))
        for i in range(0, len(train), 128):
            chunk = train[i:i + 128]
            t = torch.stack([c[0] for c in chunk]).to(DEVICE)
            m = torch.stack([c[1] for c in chunk]).to(DEVICE)
            tg = torch.stack([c[2] for c in chunk]).to(DEVICE)
            lm, _ = forward(model, t, m)
            loss = -(tg * F.log_softmax(lm, -1).clamp(min=-1e4)).sum(-1).mean()
            try:
                tt, tm, ty, _, _ = next(tal_iter)
            except StopIteration:
                tal_iter = iter(tal_batches(tal["train"], 128, True))
                tt, tm, ty, _, _ = next(tal_iter)
            lt, _ = forward(model, tt, tm)
            ok = ty >= 0
            if ok.any():
                loss = loss + float(os.environ.get("TT_RETAIN", "1.0")) * F.cross_entropy(lt[ok], ty[ok])
            opt.zero_grad()
            loss.backward()
            opt.step()
        b, tl = blunder_rate(model, hold), tal_evaluate(model, tal["hold"])[0]
        history.append((f"epoch {ep + 1}", b, tl))
        log(f"epoch {ep + 1}: blunder {b:.1%}, exact Tal move {tl:.1%}")
        if b < best[0] and tl >= t0 - 0.005:
            best = (b, tl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    improved = best[2] is not None and best[0] <= b0 * 0.85
    info = {"before": [b0, t0], "after": [best[0], best[1]], "history": history, "improved": improved}
    if best[2] is not None:
        torch.save({"model_state_dict": best[2], "info": info}, os.environ.get("TT_OUT", "tactics.pt"))
    json.dump(info, open("tactics-info.json", "w"), indent=1)
    log("done", info)


if __name__ == "__main__":
    main()
