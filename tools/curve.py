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


VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def is_sac(board, move, reply):
    """Piece sacrifice: a knight or more is left where the opponent's best reply takes it, losing 2+ points."""
    p = board.piece_at(move.from_square)
    if p is None or p.piece_type in (chess.PAWN, chess.KING) or reply is None or reply.to_square != move.to_square:
        return False
    b = board.copy(stack=False)
    cap = VAL[b.piece_type_at(move.to_square)] if b.piece_type_at(move.to_square) else 0
    b.push(move)
    att = b.piece_at(reply.from_square)
    if att is None:
        return False
    b.push(reply)
    back = VAL[att.piece_type] if b.attackers(p.color, move.to_square) else 0
    return VAL[p.piece_type] - cap - back >= 2


def win_pct(cp):
    """Lichess' public win-percentage formula."""
    import math
    cp = max(-1000, min(1000, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(wb, wa):
    """Lichess' public per-move accuracy formula (0-100)."""
    import math
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * max(0.0, wb - wa)) - 3.1669))


def game_accuracies(rows):
    by = {}
    for r in rows:
        if "game" in r:
            by.setdefault(r["game"], []).append(move_accuracy(r["wb"], r["wa"]))
    return [statistics.mean(v) for v in by.values() if len(v) >= 10]


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
            node = g
            for i, m in enumerate(moves):
                node = node.variations[0] if node.variations else node
                if b.turn in colors:
                    sac = is_sac(b, m, evals[i + 1][1])
                    note = node.comment or ""
                    before, best = evals[i]
                    after = -evals[i + 1][0]
                    if abs(before) <= 1000:
                        rows.append({"game": f"{out_name}:{gi}", "wb": win_pct(before), "wa": win_pct(after),
                                     "move": b.fullmove_number, "loss": max(0, min(500, before - after)),
                                     "top": m == best, "eval": before, "sac": sac,
                                     "brill": sac and before - after <= 20 and after >= -100,
                                     "hunter": "sacrifice:" in note,
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
    text = "\n\n".join(str(g) for g, _ in out) + "\n"
    open(f"curve-bot-{shard}.pgn", "w").write(text)
    print("=====PGN START=====\n" + text + "=====PGN END=====", flush=True)
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
    for who in ("human", "bot"):
        rs = sets.get(who, [])
        games = len({r.get("game") for r in rs})
        n_sac = sum(1 for r in rs if r.get("sac"))
        n_br = sum(1 for r in rs if r.get("brill"))
        out[f"sacrifices_{who}"] = {"games": games, "sacrifices": n_sac, "brilliant_like": n_br}
    hs, bs = out["sacrifices_human"], out["sacrifices_bot"]
    L += ["", "## Sacrifices (piece left to be taken, 2+ points down) and brilliant-like ones",
          "Brilliant-like = a sacrifice that is the best move or within 0.2 pawns, and the position stays OK (chess.com's idea of !!).", "",
          "| | Humans 2300-2600 | Bot |", "|---|---|---|",
          f"| sacrifices per game | {hs['sacrifices'] / max(1, hs['games']):.2f} | {bs['sacrifices'] / max(1, bs['games']):.2f} |",
          f"| brilliant-like per game | {hs['brilliant_like'] / max(1, hs['games']):.2f} | {bs['brilliant_like'] / max(1, bs['games']):.2f} |",
          f"| total (sacrifices / brilliant-like / games) | {hs['sacrifices']} / {hs['brilliant_like']} / {hs['games']} | {bs['sacrifices']} / {bs['brilliant_like']} / {bs['games']} |",
          f"| bot moves chosen by the sacrifice hunter | - | {sum(1 for r in sets.get('bot', []) if r.get('hunter'))} |"]
    ga_h, ga_b = game_accuracies(sets.get("human", [])), game_accuracies(sets.get("bot", []))
    if ga_h and ga_b:
        q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
        L += ["", "## Game accuracy (Lichess formula, per player per game)", "",
              "| | Humans 2300-2600 | Bot |", "|---|---|---|",
              f"| average | {statistics.mean(ga_h):.1f}% | {statistics.mean(ga_b):.1f}% |",
              f"| median | {q(ga_h, .5):.1f}% | {q(ga_b, .5):.1f}% |",
              f"| lowest 10% | {q(ga_h, .1):.1f}% | {q(ga_b, .1):.1f}% |",
              f"| top 10% | {q(ga_h, .9):.1f}% | {q(ga_b, .9):.1f}% |",
              f"| games | {len(ga_h)} | {len(ga_b)} |"]
        out["game_accuracy"] = {"human": statistics.mean(ga_h), "bot": statistics.mean(ga_b)}
    json.dump(out, open("curve-summary.json", "w"), indent=1)
    text = "\n".join(L)
    open("curve-report.md", "w").write(text + "\n")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


if __name__ == "__main__":
    {"bot": bot_games, "human": human_games, "compare": compare}[os.environ.get("CURVE_MODE", "compare")]()
