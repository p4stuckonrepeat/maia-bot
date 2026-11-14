"""Offline test matches: our bot (book + style + guard) vs plain Maia-3 and vs Stockfish."""
import os
import sys
import time

import chess
import chess.engine
import chess.pgn

from bot.book import Book
from bot.engine import Guard, Maia
from bot.main import Bot
from bot.settings import Settings

GAMES = int(os.environ.get("MATCH_GAMES", "3"))
SF_ELO = int(os.environ.get("MATCH_SF_ELO", "2500"))


def play(bot, opponent, opp_name, bot_white, gid):
    board, game = chess.Board(), chess.pgn.Game()
    node = game
    bot_color = chess.WHITE if bot_white else chess.BLACK
    while not board.is_game_over(claim_draw=True) and board.ply() < 240:
        if board.turn == bot_color:
            move, note, opening, _ = bot.pick_move(gid, board, {}, bot_color, 2600)
            comment = f"book: {opening}" if opening else (f"GUARD {note}" if note else "")
        else:
            move, comment = opponent(board), ""
        node = node.add_variation(move, comment=comment)
        board.push(move)
    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    game.headers.update({"Event": "maia-bot test match", "Site": "GitHub Actions",
                         "Date": time.strftime("%Y.%m.%d"), "Round": gid,
                         "White": "maia-bot" if bot_white else opp_name,
                         "Black": opp_name if bot_white else "maia-bot", "Result": result})
    if board.ply() >= 240 and not board.is_game_over(claim_draw=True):
        game.headers["Termination"] = "adjudicated draw after 120 moves"
    bot.book.end(gid)
    return game, result


def main():
    settings = Settings()
    cfg = settings.cfg
    maia, guard = Maia(cfg["maia"]["model"]), Guard()
    bot = Bot(settings, None, maia, guard, "maia-bot", Book())
    plain_cfg = {"elo": 2600, "temperature": 1.0, "top_p": 1.0}
    sf = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": SF_ELO, "Threads": 1})
    opponents = [
        ("Maia-3 2600 (plain)", lambda b: maia.choose(b, plain_cfg, 2600)[0]),
        (f"Stockfish (UCI_Elo {SF_ELO})", lambda b: sf.play(b, chess.engine.Limit(time=0.1)).move),
    ]
    pgns, score = [], {}
    for name, opp in opponents:
        for i in range(GAMES):
            game, result = play(bot, opp, name, bot_white=(i % 2 == 0), gid=f"{name[:5]}-{i + 1}")
            bot_white = i % 2 == 0
            pts = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
            pts = pts if bot_white else 1 - pts
            score.setdefault(name, []).append(pts)
            print(f"{game.headers['White']} - {game.headers['Black']}: {result} "
                  f"({game.end().board().fullmove_number} moves)", flush=True)
            pgns.append(str(game))
    sf.quit()
    guard.close()
    text = "\n\n".join(pgns) + "\n"
    open("test-matches.pgn", "w").write(text)
    lines = [f"{n}: bot scored {sum(p)}/{len(p)}" for n, p in score.items()]
    print("\n".join(lines))
    print("=====PGN START=====\n" + text + "=====PGN END=====", flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("## Test matches\n" + "\n".join(f"- {l}" for l in lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
