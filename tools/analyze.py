"""Analyse the bot's recent Lichess games: where do its bad moves come from?

For every bot move (after the book): Stockfish centipawn loss, and Maia's own ranking of
the move (rank 1 = Maia's top choice). If bad moves are mostly Maia rank 2+, the extra
randomness/style is to blame; if they are rank 1, it is Maia itself.
"""
import io
import json
import os
import statistics as st

import chess
import chess.engine
import chess.pgn
import requests
import yaml

from bot.engine import Maia
from bot.style import positional_score

N = int(os.environ.get("ANALYZE_GAMES", "10"))
THRESH = int(os.environ.get("ANALYZE_THRESHOLD", "150"))
DEPTH = 16


def cp(info, color):
    return info["score"].pov(color).score(mate_score=3000)


def main():
    cfg = yaml.safe_load(open("config.yml"))
    tok = os.environ["LICHESS_TOKEN"]
    h = {"Authorization": f"Bearer {tok}"}
    me = requests.get("https://lichess.org/api/account", headers=h, timeout=20).json()["id"]
    r = requests.get(f"https://lichess.org/api/games/user/{me}", headers={**h, "Accept": "application/x-chess-pgn"},
                     params={"max": N, "moves": "true", "opening": "true", "clocks": "true"}, timeout=60)
    pgn = io.StringIO(r.text)
    maia = Maia(cfg["maia"]["model"])
    sf = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    sf.configure({"Threads": os.cpu_count(), "Hash": 512})
    book = json.load(open("book.json"))["openings"] if os.path.exists("book.json") else {}
    book_keys = {k for o in book.values() for k in o["positions"]}

    rows, out = [], []
    while True:
        g = chess.pgn.read_game(pgn)
        if g is None:
            break
        color = chess.WHITE if g.headers["White"].lower() == me else chess.BLACK
        opp = g.headers["Black" if color == chess.WHITE else "White"]
        opp_elo = g.headers.get("BlackElo" if color == chess.WHITE else "WhiteElo", "?")
        board = g.board()
        losses, notes = [], []
        for mv in g.mainline_moves():
            if board.turn == color:
                k = " ".join(board.fen().split()[:4])
                if k not in book_keys:
                    best = sf.analyse(board, chess.engine.Limit(depth=DEPTH))
                    before = cp(best, color)
                    _, top, _ = maia.choose(board, {**cfg["maia"], "temperature": 1.0, "top_p": 1.0},
                                            int(opp_elo) if opp_elo.isdigit() else None, candidates=20)
                    ranks = [c["move"] for c in top]
                    rank = ranks.index(mv) + 1 if mv in ranks else 99
                    prob = next((c["policy"] for c in top if c["move"] == mv), 0.0)
                    style = positional_score(board, mv)
                    san = board.san(mv)
                    board.push(mv)
                    after = cp(sf.analyse(board, chess.engine.Limit(depth=DEPTH)), color)
                    loss = max(0, before - after)
                    losses.append(loss)
                    rows.append((loss, rank, prob))
                    if loss >= THRESH:
                        top1 = board.copy(); top1.pop()
                        notes.append(f"  move {top1.fullmove_number}: {san} eval {before/100:+.1f} -> {after/100:+.1f}, lost {loss/100:.1f} "
                                     f"(Maia rank {rank}, prob {prob:.0%}, style {style:+.2f}; "
                                     f"Maia's top choice was {top1.san(top[0]['move'])} "
                                     f"{top[0]['policy']:.0%})")
                    continue
            board.push(mv)
        print(str(g), flush=True)
        res = g.headers["Result"]
        out.append(f"{'White' if color else 'Black'} vs {opp} ({opp_elo}) {res} "
                   f"{g.headers.get('Opening', '')} | avg loss {st.mean(losses) if losses else 0:.0f} cp | "
                   f"{g.headers.get('Site', '')}")
        out += notes
    sf.quit()

    def avg(sel):
        v = [r[0] for r in rows if sel(r)]
        return f"{st.mean(v):.0f} cp over {len(v)} moves" if v else "-"

    summary = ["## Game analysis", "```"] + out + ["", "Average loss per bot move (after book):",
               f"  all moves:            {avg(lambda r: True)}",
               f"  Maia's top choice:    {avg(lambda r: r[1] == 1)}",
               f"  Maia's 2nd-3rd:       {avg(lambda r: 2 <= r[1] <= 3)}",
               f"  Maia's rank 4+:       {avg(lambda r: r[1] >= 4)}",
               f"  share of moves not Maia's top choice: "
               f"{sum(r[1] > 1 for r in rows) / max(1, len(rows)):.0%}", "```"]
    text = "\n".join(summary)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


if __name__ == "__main__":
    main()
