"""Tal's move patterns, learned from the study data (tal/study.py) on top of the fitted rules.

Pattern = (piece, target square seen from the mover's side, game phase), plus capture/check.
For each studied position the current rules (tal-rules2.json) score the natural candidates;
pattern weights are fitted on the training games so the combined score picks Tal's move more often,
then tested on games the model never saw. Writes tal/tal-patterns.json and tal/patterns-report.md.
"""
import gzip
import json
import os

import chess
import torch

import tal.study as study
from tal.train import download_games, log, tal_games

NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
         chess.QUEEN: "queen", chess.KING: "king"}
PHASES = ["opening", "middlegame", "endgame"]


def pattern_keys(board, move, phase):
    p = board.piece_at(move.from_square)
    if p is None:
        return []
    to = move.to_square if board.turn == chess.WHITE else chess.square_mirror(move.to_square)
    name = f"{NAMES[p.piece_type]} to {chess.square_name(to)}"
    keys = [name, f"{name} ({phase})"]
    if board.is_capture(move):
        keys.append(f"{NAMES[p.piece_type]} captures ({phase})")
    if board.gives_check(move):
        keys.append(f"{NAMES[p.piece_type]} check ({phase})")
    return keys


def main():
    rows = []
    with gzip.open(os.environ.get("STUDY_FILE", "study.jsonl.gz"), "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    by_shard = {}
    for r in rows:
        by_shard.setdefault(r["shard"], []).append(r)
    from tal.antiblunder import rebuild
    data = rebuild(by_shard, tal_games(download_games()))
    rules = json.load(open("tal/tal-rules2.json"))
    names, w = rules["features"], rules["weights"]
    fidx = {n: i for i, n in enumerate(study.FEATURES)}

    vocab, counts = {}, {}
    samples = []          # (split, [(base_logit, [key ids]) per candidate], tal index or -1)
    for r, g, ply in data:
        b = chess.Board()
        for m in g["moves"][:ply]:
            b.push_uci(m)
        phase = r["phase"] if r["phase"] in PHASES else "middlegame"
        cands, y = [], -1
        for j, c in enumerate(r["cands"][:r["n_top"]]):
            mv = chess.Move.from_uci(c["m"])
            base = sum(w[n] * c["f"][fidx[n]] for n in names)
            ids = []
            for k in pattern_keys(b, mv, phase):
                if k not in vocab:
                    vocab[k] = len(vocab)
                ids.append(vocab[k])
                if r["split"] == "train":
                    counts[k] = counts.get(k, 0) + 1
            cands.append((base, ids))
            if c["m"] == r["tal"]:
                y = j
        samples.append((r["split"], cands, y))
    keep = {k for k, n in counts.items() if n >= int(os.environ.get("PAT_MIN_COUNT", "40"))}
    kid = {vocab[k]: i for i, k in enumerate(sorted(keep))}
    inv = sorted(keep)
    log(f"{len(samples)} positions, {len(inv)} patterns kept")

    def tensors(split):        # sparse: (position*cands, pattern) pairs instead of a huge dense tensor
        S = [s for s in samples if s[0] == split]
        mx = max(len(s[1]) for s in S)
        base = torch.full((len(S), mx), -1e9)
        Y = torch.full((len(S),), -1, dtype=torch.long)
        pos, pat = [], []
        for i, (_, cands, y) in enumerate(S):
            for j, (bl, ids) in enumerate(cands):
                base[i, j] = bl
                for t in ids:
                    if t in kid:
                        pos.append(i * mx + j)
                        pat.append(kid[t])
            Y[i] = y
        return base, torch.tensor(pos, dtype=torch.long), torch.tensor(pat, dtype=torch.long), Y

    def score(base, pos, pat, wp):
        add = torch.zeros(base.numel()).index_add(0, pos, wp[pat]).view_as(base)
        return base + add

    bt, post, patt, Yt = tensors("train")
    bh, posh, path_, Yh = tensors("hold")

    def acc(wp):
        pred = score(bh, posh, path_, wp).argmax(1)
        return float(((pred == Yh) & (Yh >= 0)).float().mean())

    base_acc = acc(torch.zeros(len(inv)))
    best = (base_acc, torch.zeros(len(inv)), 0.0)
    ok = Yt >= 0
    for reg in (0.3, 0.1, 0.03, 0.01):
        wp = torch.zeros(len(inv), requires_grad=True)
        opt = torch.optim.Adam([wp], lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(score(bt, post, patt, wp)[ok], Yt[ok]) + reg * (wp ** 2).mean()
            loss.backward()
            opt.step()
        a = acc(wp.detach())
        log(f"regularisation {reg}: held-out exact Tal move {a:.1%} (rules only {base_acc:.1%})")
        if a > best[0]:
            best = (a, wp.detach().clone(), reg)
    a, wp, reg = best
    weights = {inv[i]: round(float(wp[i]), 3) for i in range(len(inv)) if abs(float(wp[i])) >= 0.02}
    json.dump({"accuracy": a, "baseline": base_acc, "regularisation": reg, "min_count": int(os.environ.get("PAT_MIN_COUNT", "40")),
               "weights": weights}, open("tal/tal-patterns.json", "w"), indent=1)
    top = sorted(weights.items(), key=lambda kv: -kv[1])
    L = ["# Tal's move patterns (learned from 46,000 positions of his games)", "",
         f"Exact Tal move on games the model never saw: **{base_acc:.1%} -> {a:.1%}** with the patterns.", "",
         "## Tal plays these more often than the model", ""] + \
        [f"- {k}  (+{v:.2f})" for k, v in top[:25] if v > 0] + \
        ["", "## ...and these less often", ""] + [f"- {k}  ({v:.2f})" for k, v in top[::-1][:25] if v < 0]
    text = "\n".join(L)
    open("tal/patterns-report.md", "w").write(text + "\n")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


if __name__ == "__main__":
    main()
