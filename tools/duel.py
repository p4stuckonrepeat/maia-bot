"""Head-to-head: two bot versions play each other (colours alternate).

DUEL_A / DUEL_B: config files (default config-tal.yml = v5, config-tal-v43.yml = v4.3),
DUEL_NAMES: "v5,v4.3", DUEL_GAMES per machine, DUEL_SHARD.
Prints the PGNs (between =====PGN START===== / =====PGN END=====) and the score.
"""
import os
import time

import chess
import chess.pgn

from bot.book import Book
from bot.engine import Guard, Maia
from bot.main import Bot
from bot.settings import Settings


def make(path, maia, guard, name):
    s = Settings(path)
    return Bot(s, None, maia, guard, name, Book(s.cfg.get("openings", {}).get("book", {}).get("path", "book.json")))


def main():
    a_cfg, b_cfg = os.environ.get("DUEL_A", "config-tal.yml"), os.environ.get("DUEL_B", "config-tal-v43.yml")
    na, nb = os.environ.get("DUEL_NAMES", "v5,v4.3").split(",")
    shard, games = os.environ.get("DUEL_SHARD", "0"), int(os.environ.get("DUEL_GAMES", "4"))
    first = Settings(a_cfg).cfg
    maia = Maia(first["maia"]["model"], first["maia"].get("checkpoint"))
    guard = Guard()
    bots = {na: make(a_cfg, maia, guard, na), nb: make(b_cfg, maia, guard, nb)}
    score, pgns = {na: 0.0, nb: 0.0}, []
    try:
        for i in range(games):
            white, black = (na, nb) if (i + int(shard)) % 2 == 0 else (nb, na)
            gid = f"d{shard}-{i}"
            board, game = chess.Board(), chess.pgn.Game()
            node = game
            while not board.is_game_over(claim_draw=True) and board.ply() < 240:
                who = white if board.turn == chess.WHITE else black
                move, note, opening, _ = bots[who].pick_move(gid, board, {}, board.turn, 2500)
                comment = f"book: {opening}" if opening else (note or "")
                node = node.add_variation(move, comment=comment)
                board.push(move)
            res = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
            for bot in bots.values():
                bot.book.end(gid)
            game.headers.update({"Event": f"Duel {na} vs {nb}", "Site": "GitHub Actions",
                                 "Date": time.strftime("%Y.%m.%d"), "Round": gid,
                                 "White": white, "Black": black, "Result": res})
            pts = {"1-0": (1, 0), "0-1": (0, 1)}.get(res, (0.5, 0.5))
            score[white] += pts[0]
            score[black] += pts[1]
            pgns.append(str(game))
            print(f"game {i + 1}: {white} - {black} {res}", flush=True)
    finally:
        guard.close()
    print("=====PGN START=====\n" + "\n\n".join(pgns) + "\n=====PGN END=====", flush=True)
    print(f"DUEL SCORE {na} {score[na]} - {score[nb]} {nb}", flush=True)


if __name__ == "__main__":
    main()
