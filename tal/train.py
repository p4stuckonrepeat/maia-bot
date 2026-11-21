"""Fine-tune Maia-3 79M on Mikhail Tal's moves.

Steps: download Tal's games -> keep positions where Tal is to move -> 90/10 split by game ->
measure how often the model predicts Tal's actual move on the held-out games (before) ->
train (policy = Tal's move, value = game result) -> measure again (after) -> save checkpoint.

Env: TAL_MINUTES (training time budget), TAL_EPOCHS, TAL_LR, TAL_RESUME (path to checkpoint).
"""
import gzip
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

# PLAYER = "tal" (pgnmentor over-the-board games) or "chesscom:<user>:<time classes>",
# e.g. "chesscom:hikaru:blitz,rapid", or "lichess:<user>:<speeds>", or several sources of ONE player
# joined with "+" after "multi:", e.g.
# "multi:group:Carlsen+chesscom:magnuscarlsen:blitz,rapid+lichess:DrNykterstein:blitz,rapid".
PLAYER = os.environ.get("PLAYER", "tal")
MAX_GAMES = int(os.environ.get("PLAYER_MAX_GAMES", "0"))     # 0 = all
SOURCES = ["https://www.pgnmentor.com/players/Tal.zip"]
ELO = 2600                     # the bot plays with elo 2600, so the style is attached to that setting
OUT = os.environ.get("TAL_OUT", "tal-79m.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UA = {"User-Agent": "maia-bot style training (github.com/p4stuckonrepeat/maia-bot)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download_chesscom(user, classes):
    for attempt in range(5):
        r = requests.get(f"https://api.chess.com/pub/player/{user}/games/archives", headers=UA, timeout=60)
        if r.status_code == 200 and "archives" in r.text:
            break
        log(f"chess.com {user}: archives HTTP {r.status_code}, retry {attempt + 1}")
        time.sleep(10 * (attempt + 1))
    archives = r.json()["archives"]
    parts = []
    for url in reversed(archives):          # newest first
        for _ in range(3):
            try:
                data = requests.get(url, headers=UA, timeout=60).json()
                break
            except Exception:
                time.sleep(5)
        else:
            continue
        for g in data.get("games", []):
            if g.get("rules") == "chess" and g.get("time_class") in classes and g.get("pgn"):
                parts.append(g["pgn"])
        if MAX_GAMES and len(parts) >= MAX_GAMES * 1.05:
            break
        time.sleep(0.3)
    log(f"chess.com {user}: {len(parts)} games ({','.join(classes)}) from {len(archives)} months available")
    return "\n\n".join(parts)


def download_lichess(user, speeds):
    r = requests.get(f"https://lichess.org/api/games/user/{user}", headers={**UA, "Accept": "application/x-chess-pgn"},
                     params={"perfType": ",".join(speeds), "max": MAX_GAMES or None}, timeout=600)
    if r.status_code != 200:
        log(f"lichess {user}: HTTP {r.status_code}, skipped")
        return ""
    log(f"lichess {user}: {r.text.count('[Event ')} games ({','.join(speeds)})")
    return r.text


def download_group(names):
    parts = []
    for n in names:
        url = f"https://www.pgnmentor.com/players/{n}.zip"
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                parts += [z.read(x).decode("latin-1") for x in z.namelist() if x.lower().endswith(".pgn")]
            log(f"downloaded {n}")
        except Exception as e:
            log(f"skipped {n}: {e}")
    return "\n".join(parts)


def multi_sources():
    return PLAYER.split(":", 1)[1].split("+") if PLAYER.startswith("multi:") else []


def download_games(spec=None):
    gf = os.environ.get("GAMES_FILE")      # games downloaded beforehand (e.g. on GitHub, where chess.com works)
    if spec is None and gf and os.path.exists(gf):
        text = gzip.open(gf, "rt").read()
        log(f"games file {gf}: {text.count('[Event ')} games")
        return text
    spec = spec or PLAYER
    if spec.startswith("multi:"):
        parts = []
        for src in multi_sources():
            try:
                parts.append(download_games(src))
            except Exception as e:
                log(f"source failed {src}: {e}")
        return "\n\n".join(p for p in parts if p)
    if spec.startswith("group:"):
        return download_group(spec.split(":", 1)[1].split(","))
    if spec.startswith("chesscom:"):
        _, user, classes = spec.split(":")
        return download_chesscom(user, classes.split(","))
    if spec.startswith("lichess:"):
        _, user, speeds = spec.split(":")
        return download_lichess(user, speeds.split(","))
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


def is_player(name):
    """multi: the name belongs to the player in any of the sources (OTB surname or online account)."""
    for src in multi_sources():
        kind, rest = src.split(":", 1)
        if kind == "group" and any(re.match(rf"^{x}\b", name) for x in rest.split(",")):
            return True
        if kind in ("chesscom", "lichess") and name.lower() == rest.split(":")[0].lower():
            return True
    return False


def tal_games(text):
    pgn, games = io.StringIO(text), []
    while True:
        g = chess.pgn.read_game(pgn)
        if g is None:
            break
        w, b = g.headers.get("White", ""), g.headers.get("Black", "")
        if PLAYER.startswith("group:"):
            names = PLAYER.split(":", 1)[1].split(",")
            sides = [(c, next(x for x in names if re.match(rf"^{x}\b", nm))) for c, nm in ((chess.WHITE, w), (chess.BLACK, b))
                     if any(re.match(rf"^{x}\b", nm) for x in names)]
            if not sides or g.headers.get("Variant") or g.headers.get("SetUp") == "1":
                continue
            moves = [m.uci() for m in g.mainline_moves()]
            if len(moves) < 10:
                continue
            for c, who in sides:
                games.append({"moves": moves, "tal": c, "result": g.headers.get("Result", "*"), "player": who})
            if MAX_GAMES and len(games) >= MAX_GAMES:
                break
            continue
        if PLAYER.startswith("multi:"):
            is_w, is_b = is_player(w), is_player(b)
        elif ":" in PLAYER:
            user = PLAYER.split(":")[1].lower()
            is_w, is_b = w.lower() == user, b.lower() == user
        else:
            is_w, is_b = bool(re.match(r"^Tal\b", w)), bool(re.match(r"^Tal\b", b))
        if is_w == is_b or g.headers.get("Variant") or g.headers.get("SetUp") == "1":
            continue
        years = os.environ.get("TAL_YEARS")          # e.g. 1949-1962 = young, wild Tal only
        if years:
            lo, hi = (int(x) for x in years.split("-"))
            try:
                y = int(g.headers.get("Date", "?")[:4])
            except ValueError:
                continue
            if not lo <= y <= hi:
                continue
        moves = list(g.mainline_moves())
        if len(moves) < 10:
            continue
        res = g.headers.get("Result", "*")
        games.append({"moves": [m.uci() for m in moves], "tal": chess.WHITE if is_w else chess.BLACK,
                      "result": res})
        if MAX_GAMES and len(games) >= MAX_GAMES:
            break
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
        self.cfg = eng.cfg
        self.all_moves_dict = eng.all_moves_dict      # no model reference: safe for worker processes

    def encode(self, g, ply):
        board, hist = chess.Board(), deque(maxlen=self.cfg.history)
        hist.append(tokenize_board(board))
        for m in g["moves"][:ply]:
            board.push_uci(m)
            hist.append(tokenize_board(board))
        tokens = get_historical_tokens(hist, self.cfg, 0.0, 0.0, 0.0, 0.0)
        mask = get_legal_moves_mask(board, self.all_moves_dict)
        uci = g["moves"][ply]
        if board.turn == chess.BLACK:
            uci = mirror_move(uci)
        target = self.all_moves_dict.get(uci)
        res = g["result"]
        tal_won = (res == "1-0" and g["tal"] == chess.WHITE) or (res == "0-1" and g["tal"] == chess.BLACK)
        value = 1 if res == "1/2-1/2" else (2 if tal_won else 0) if res in ("1-0", "0-1") else -100
        return tokens, mask, target, value

    def batch(self, games, items):
        rows = [self.encode(games[gi], ply) for gi, ply in items]
        rows = [r for r in rows if r[2] is not None]
        t = torch.stack([r[0] for r in rows]).to(DEVICE)
        m = torch.stack([r[1] for r in rows]).to(DEVICE)
        y = torch.tensor([r[2] for r in rows]).to(DEVICE)
        v = torch.tensor([r[3] for r in rows]).to(DEVICE)
        return t, m, y, v


class PosDataset(torch.utils.data.Dataset):
    def __init__(self, enc, games, items):
        self.enc, self.games, self.items = enc, games, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        gi, ply = self.items[i]
        t, m, y, v = self.enc.encode(self.games[gi], ply)
        return t, m, -1 if y is None else y, v


try:
    _CPUS = len(os.sched_getaffinity(0))
except Exception:
    _CPUS = os.cpu_count() or 2
WORKERS = max(1, min(int(os.environ.get("TAL_WORKERS", "8")), _CPUS))   # cloud machines report hundreds of host CPUs
torch.multiprocessing.set_sharing_strategy("file_system")


def loader(enc, games, items, bs, shuffle):
    return torch.utils.data.DataLoader(PosDataset(enc, games, items), batch_size=bs, shuffle=shuffle,
                                       num_workers=WORKERS, pin_memory=DEVICE == "cuda",
                                       persistent_workers=False)


def forward(model, t, m):
    elos = torch.full((t.size(0),), ELO, dtype=torch.float, device=t.device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        logits_move, logits_value, _ = model(t, elos, elos)
    return logits_move.float().masked_fill(~m, float("-inf")), logits_value.float()


@torch.no_grad()
def accuracy(model, enc, games, items, bs=128):
    model.eval()
    hit = n = 0
    for t, m, y, _ in loader(enc, games, items, bs, False):
        t, m, y = t.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True), y.to(DEVICE)
        ok = y >= 0
        lm, _ = forward(model, t, m)
        hit += (lm.argmax(-1)[ok] == y[ok]).sum().item()
        n += ok.sum().item()
    return hit / max(1, n)


def main():
    torch.set_num_threads(os.cpu_count())
    minutes = float(os.environ.get("TAL_MINUTES", "30"))
    epochs = float(os.environ.get("TAL_EPOCHS", "2"))
    lr = float(os.environ.get("TAL_LR", "5e-6"))
    train_blocks = int(os.environ.get("TAL_TRAIN_BLOCKS", "2"))   # only the last N transformer blocks + heads
    value_w = float(os.environ.get("TAL_VALUE_WEIGHT", "0.1"))
    eval_every = int(os.environ.get("TAL_EVAL_EVERY", "400"))
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
    log(f"{PLAYER} games: {len(games)} (train {len(train)}, held out {len(hold)}); "
        f"player moves: train {len(s_train)}, eval {len(s_hold)}; device {DEVICE}")

    eng = Maia3UCIEngine(parse_args(["--model", "maia3-79m", "--device", DEVICE]))
    eng.ensure_model_loaded()
    model = eng.model
    enc = Encoder(eng)
    base_normal = None
    if os.environ.get("TAL_RESUME") and os.path.exists(os.environ["TAL_RESUME"]):
        base_normal = accuracy(model, enc, hold, s_hold)
        log(f"normal Maia on this player's held-out moves: {base_normal:.1%}")
        ck = torch.load(os.environ["TAL_RESUME"], map_location=DEVICE)
        model.load_state_dict(ck["model_state_dict"])
        log("resumed from", os.environ["TAL_RESUME"], ck.get("info"))

    t0 = time.time()
    base = accuracy(model, enc, hold, s_hold)
    log(f"held-out player move match BEFORE training: {base:.1%} ({time.time() - t0:.0f}s)")

    # Train only the top of the network: Tal's style on top of Maia's general chess knowledge.
    n_blocks = len(model.transformer.layers)
    for name, p in model.named_parameters():
        mt = re.match(r"transformer\.layers\.(\d+)\.", name)
        frozen = name.startswith(("token_projection", "elo_embedding", "gab_shared", "abs_pe")) or \
            (mt is not None and int(mt.group(1)) < n_blocks - train_blocks)
        p.requires_grad = not frozen
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"training the last {train_blocks} of {n_blocks} blocks + heads: {trainable / 1e6:.1f}M parameters")
    quick = s_hold[:800]
    best_acc = accuracy(model, enc, hold, quick)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    log(f"quick eval (800 positions) at start: {best_acc:.1%}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    total_steps = int(len(s_train) * epochs / bs)
    deadline, step, seen, t_start = time.time() + minutes * 60, 0, 0, time.time()
    model.train()
    while step < total_steps and time.time() < deadline:
        for t, m, y, v in loader(enc, train, s_train, bs, True):
            if step >= total_steps or time.time() >= deadline:
                break
            t, m = t.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True)
            y, v = y.to(DEVICE), v.to(DEVICE)
            if not (y >= 0).any():
                continue
            lm, lv = forward(model, t, m)
            loss = F.cross_entropy(lm, y, ignore_index=-1)
            keep = v >= 0
            if keep.any():
                loss = loss + value_w * F.cross_entropy(lv[keep], v[keep])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            seen += len(y)
            if step % 50 == 0:
                rate = seen / (time.time() - t_start)
                log(f"step {step}/{total_steps} loss {loss.item():.3f} | {rate:.1f} positions/s | "
                    f"epoch {seen / len(s_train):.2f}")
            if step % eval_every == 0:
                acc = accuracy(model, enc, hold, quick)
                model.train()
                if acc > best_acc:
                    best_acc, best_state = acc, {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}
                log(f"quick eval at step {step}: {acc:.1%} (best {best_acc:.1%})")
    rate = seen / max(1, time.time() - t_start)
    acc = accuracy(model, enc, hold, quick)
    if acc > best_acc:
        best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)   # keep the best version seen during training
    after = accuracy(model, enc, hold, s_hold)
    info = {"before": base, "after": after, "positions_trained": seen, "epochs": seen / len(s_train),
            "positions_per_s": rate, "train_moves": len(s_train), "games": len(games)}
    log(f"held-out player move match AFTER training: {after:.1%} (before {base:.1%})")
    if base_normal is not None:
        info["before_resume"], info["before"] = base, base_normal   # compare against normal Maia
    info["improved"] = after > info["before"]
    info["player"] = PLAYER
    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "info": info}, OUT)
    json.dump(info, open("tal-info.json", "w"), indent=1)
    hours_epoch = len(s_train) / rate / 3600
    text = (f"## Style training: {PLAYER}\n- games: {len(games)}, player moves for training: {len(s_train)}\n"
            f"- trained on {seen} positions ({seen / len(s_train):.2f} epochs) at {rate:.1f} positions/s "
            f"(one epoch ≈ {hours_epoch:.1f} h on this machine)\n"
            f"- predicts the player's move on held-out games: **{base:.1%} → {after:.1%}**\n")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text)


if __name__ == "__main__":
    main()
