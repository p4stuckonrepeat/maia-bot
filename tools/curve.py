"""Accuracy curve: does the bot's accuracy over a game look like a strong human's?

CURVE_MODE=bot    : bot plays CURVE_GAMES games vs plain Maia 2600 -> curve-bot.pgn, then analysis
CURVE_MODE=human  : CURVE_GAMES games of 2300-2600 players from the Lichess Elite Database -> analysis
CURVE_MODE=compare: merges curve-*.json into a comparison table (curve-report.md)

Analysis (Stockfish, depth CURVE_DEPTH): for every move of the tracked player, the loss in centipawns,
whether it was Stockfish's top move, and the situation (move number, equal/ahead/behind, sharp/quiet).
"""
import glob
import io
import json
import os
import statistics
import zipfile

import chess
import chess.engine
import chess.pgn
import requests

DEPTH = int(os.environ.get("CURVE_DEPTH", "11"))
GAMES = int(os.environ.get("CURVE_GAMES", "8"))
BUCKETS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 999)]


def analyse(games, out_name):
    """games: list of (chess.pgn.Game, set of colors to track)."""
    sf = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH", "/usr/games/stockfish"))
    sf.configure({"Threads": 4, "Hash": 256})
    rows = []
    try:
        for gi, (g, colors) in enumerate(games):
            b = g.board()
            moves = list(g.mainline_moves())
            evals = []
            for m in moves + [None]:
                if b.is_game_over():
                    evals.append((-10000 if b.is_checkmate() else 0, None))
                else:
                    info = sf.analyse(b, chess.engine.Limit(depth=DEPTH))
                    pv = info.get("pv", [])
                    evals.append((info["score"].pov(b.turn).score(mate_score=10000), pv[0] if pv else None))
                if m is None:
                    break
                b.push(m)
            b = g.board()
            for i, m in enumerate(moves):
                if b.turn in colors:
                    before, best = evals[i]
                    after = -evals[i + 1][0]
                    if abs(before) <= 1000:
                        rows.append({"move": b.fullmove_number, "loss": max(0, min(500, before - after)),
                                     "top": m == best, "eval": before,
                                     "sharp": sum(1 for x in b.legal_moves if b.is_capture(x)) >= 3})
                b.push(m)
            if gi % 20 == 0:
                print(f"analysed {gi + 1}/{len(games)} games", flush=True)
    finally:
        sf.quit()
    json.dump(rows, open(out_name, "w"))
    print(f"{len(rows)} moves analysed -> {out_name}")


def bot_games():
    from tools.match import play
    from bot.book import Book
    from bot.engine import Guard, Maia
    from bot.main import Bot
    from bot.settings import Settings
    settings = Settings()
    cfg = settings.cfg
    maia, guard = Maia(cfg["maia"]["model"], cfg["maia"].get("checkpoint")), Guard()
    bot = Bot(settings, None, maia, guard, "maia-bot",
              Book(cfg.get("openings", {}).get("book", {}).get("path", "book.json")))
    plain = Maia(cfg["maia"]["model"]) if cfg["maia"].get("checkpoint") else maia
    pcfg = {"elo": 2600, "temperature": 1.0, "top_p": 1.0}
    shard = os.environ.get("CURVE_SHARD", "0")
    out = []
    for i in range(GAMES):
        white = i % 2 == 0
        g, res = play(bot, lambda b: plain.choose(b, pcfg, 2600)[0], "Maia-3 2600", white, f"c{shard}-{i}")
        print(f"game {i + 1}: {res}", flush=True)
        out.append((g, {chess.WHITE if white else chess.BLACK}))
    guard.close()
    open(f"curve-bot-{shard}.pgn", "w").write("\n\n".join(str(g) for g, _ in out) + "\n")
    analyse(out, f"curve-bot-{shard}.json")


def human_games():
    from tools.elite_index import month_urls
    url = month_urls(1)[0]
    print("downloading", url, flush=True)
    data = requests.get(url, timeout=900).content
    lo, hi = int(os.environ.get("CURVE_MIN_ELO", "2300")), int(os.environ.get("CURVE_MAX_ELO", "2600"))
    games = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith(".pgn"))
        with z.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            while len(games) < GAMES:
                g = chess.pgn.read_game(text)
                if g is None:
                    break
                h = g.headers
                try:
                    we, be = int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))
                    base = int(h.get("TimeControl", "0+0").split("+")[0])
                except ValueError:
                    continue
                if base < 180 or len(list(g.mainline_moves())) < 50:
                    continue
                colors = {c for c, e in ((chess.WHITE, we), (chess.BLACK, be)) if lo <= e <= hi}
                if colors:
                    games.append((g, colors))
    print(f"{len(games)} human games ({lo}-{hi}, 3+ minutes)", flush=True)
    analyse(games, "curve-human.json")


def summary(rows):
    if not rows:
        return None
    return (statistics.mean(r["loss"] for r in rows), sum(r["top"] for r in rows) / len(rows),
            sum(r["loss"] > 100 for r in rows) / len(rows), sum(r["loss"] > 200 for r in rows) / len(rows), len(rows))


def compare():
    sets = {}
    for f in glob.glob("curve-*.json"):
        who = "bot" if "bot" in f else "human"
        sets.setdefault(who, []).extend(json.load(open(f)))
    L = ["# Accuracy curve: bot vs strong humans (2300-2600, Lichess Elite Database)", "",
         "Loss = centipawns lost per move (Stockfish). Top = played Stockfish's first choice.", ""]
    cuts = [(f"moves {a}-{b if b < 999 else '+'}", lambda r, a=a, b=b: a <= r["move"] <= b) for a, b in BUCKETS]
    cuts += [("equal (-1 to +1)", lambda r: abs(r["eval"]) <= 100),
             ("ahead (> +3)", lambda r: r["eval"] > 300), ("behind (< -3)", lambda r: r["eval"] < -300),
             ("sharp (3+ captures possible)", lambda r: r["sharp"]), ("quiet", lambda r: not r["sharp"]),
             ("ALL MOVES", lambda r: True)]
    L += ["| Situation | Human loss | Bot loss | Human top move | Bot top move | Human mistakes >1 | Bot mistakes >1 | Human blunders >2 | Bot blunders >2 | moves (h/b) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for label, f in cuts:
        h = summary([r for r in sets.get("human", []) if f(r)])
        b = summary([r for r in sets.get("bot", []) if f(r)])
        if h and b:
            out[label] = {"human": h, "bot": b}
            L.append(f"| {label} | {h[0]:.0f} | {b[0]:.0f} | {h[1]:.0%} | {b[1]:.0%} | {h[2]:.1%} | {b[2]:.1%} | "
                     f"{h[3]:.1%} | {b[3]:.1%} | {h[4]}/{b[4]} |")
    json.dump(out, open("curve-summary.json", "w"), indent=1)
    text = "\n".join(L)
    open("curve-report.md", "w").write(text + "\n")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


if __name__ == "__main__":
    {"bot": bot_games, "human": human_games, "compare": compare}[os.environ.get("CURVE_MODE", "compare")]()
