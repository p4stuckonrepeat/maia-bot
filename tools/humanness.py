"""How human vs how accurate: compare bot versions on real strong-human positions.

Positions come from the Lichess Elite Database (2400+ vs 2200+). For each version we measure
  match: how often it plays the move the human actually played (higher = more human)
  loss:  average centipawn loss per move by Stockfish (lower = more accurate)
"""
import copy
import os
import random
import statistics as st

import chess
import chess.engine
import requests
import io
import zipfile

from bot.book import Book
from bot.engine import Guard, Maia
from bot.main import Bot
from bot.settings import Settings
from tools.elite_index import games_from_pgn, month_urls

N = int(os.environ.get("HUMAN_POSITIONS", "500"))


def sample_positions(n, rng):
    url = month_urls(1)[0]
    data = requests.get(url, timeout=600).content
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8", "ignore")
    games = [g for g, _ in games_from_pgn(text) if len(g) >= 30]
    rng.shuffle(games)
    out = []
    for sans in games:
        b = chess.Board()
        ply = rng.randint(16, len(sans) - 1)  # middlegame and later, past most opening theory
        try:
            for s in sans[:ply]:
                b.push_san(s)
            human = b.parse_san(sans[ply])
        except ValueError:
            continue
        out.append((b, human))
        if len(out) >= n:
            break
    return out


def main():
    rng = random.Random(7)
    positions = sample_positions(N, rng)
    settings = Settings()
    base = settings.cfg
    maia, guard = Maia(base["maia"]["model"]), Guard()
    sf = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    sf.configure({"Threads": os.cpu_count(), "Hash": 256})

    def bot_with(**changes):
        cfg = copy.deepcopy(base)
        cfg["openings"]["book"]["enabled"] = False
        for path, val in changes.items():
            sec, key = path.split(".")
            cfg[sec][key] = val
        s = copy.copy(settings)
        s.cfg = cfg
        b = Bot(s, None, maia, guard, "me", Book(path="/nonexistent"))
        return lambda board: b.pick_move("x", board, {}, board.turn, 2600)[0]

    versions = {
        "Bot now (filter only on non-top moves)": bot_with(),
        "Bot before (filter on all moves, 1 pawn)": bot_with(**{"assist.only_when_not_top": False,
                                                                "assist.max_loss_cp": 100}),
        "Bot without filter": bot_with(**{"assist.enabled": False}),
        "Pure Maia-3 (top choice)": lambda board: maia.choose(board, {**base["maia"], "temperature": 0.0,
                                                                    "top_p": 1.0}, 2600)[0],
        "Stockfish (depth 12)": lambda board: sf.play(board, chess.engine.Limit(depth=12)).move,
    }
    evals = {}

    def loss(board, move):
        key = board.fen()
        if key not in evals:
            evals[key] = sf.analyse(board, chess.engine.Limit(depth=14))["score"].pov(board.turn).score(mate_score=2000)
        board.push(move)
        after = -sf.analyse(board, chess.engine.Limit(depth=14))["score"].pov(board.turn).score(mate_score=2000)
        board.pop()
        return max(0, min(1000, evals[key] - after))

    human_loss = st.mean(loss(b, m) for b, m in positions)
    rows = [("Strong humans (the actual moves)", 100.0, human_loss)]
    for name, fn in versions.items():
        match, losses = 0, []
        for b, human in positions:
            mv = fn(b.copy())
            match += mv == human
            losses.append(loss(b, mv))
        rows.append((name, 100 * match / len(positions), st.mean(losses)))
        print(name, "done", flush=True)
    sf.quit()
    guard.close()
    lines = ["## Human-likeness test", f"{len(positions)} positions from 2400+ vs 2200+ Lichess games", "",
             "| Version | Same move as human | Avg loss (cp) |", "|---|---|---|"]
    lines += [f"| {n} | {m:.1f}% | {l:.0f} |" for n, m, l in rows]
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


if __name__ == "__main__":
    main()
