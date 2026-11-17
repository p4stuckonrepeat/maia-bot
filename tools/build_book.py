"""Build book.json: deep repertoire lines from strong Lichess games, checked by Stockfish.

For each opening in config.yml the tree starts from its line. At the opponent's turn every
common reply is followed (so sidelines are covered); at our turn we keep the moves strong
players actually choose, weighted by popularity, but drop any that Stockfish rates more
than MAX_LOSS_CP worse than the best candidate. Most-played positions are expanded first.
"""
import heapq
import json
import os
import sys
import time

import chess
import chess.engine
import requests
import yaml

EXPLORER = "https://explorer.lichess.ovh/lichess"
PARAMS = {"variant": "standard", "speeds": "blitz,rapid,classical", "ratings": "2200,2500",
          "moves": 15, "topGames": 0, "recentGames": 0}
SOURCE = os.environ.get("BOOK_SOURCE", "explorer")  # "elite" = downloaded Elite Database
MIN_NODE_GAMES = int(os.environ.get("BOOK_MIN_GAMES", "20" if SOURCE == "elite" else "60"))
MIN_MOVE_SHARE = 0.04    # opponent replies below this share of games are left to Maia
OUR_MIN_SHARE = 0.10     # our candidates must be >= 10% as popular as the top choice
MAX_LOSS_CP = 40         # ...and within 0.4 pawns of the best candidate (Stockfish)
MAX_PLY = 30
DEPTH = 14
TIME_LIMIT = float(os.environ.get("BOOK_TIME_LIMIT_MIN", "320")) * 60

session = requests.Session()
if os.environ.get("LICHESS_TOKEN") and os.environ.get("BOOK_AUTH", "1") == "1":
    session.headers["Authorization"] = f"Bearer {os.environ['LICHESS_TOKEN']}"
cache, start, requests_made = {}, time.time(), 0
delay = [1.0]  # seconds between explorer requests; grows after rate limits


def key(board):
    return " ".join(board.fen().split()[:4])


TABLE = None  # filled from the Elite Database when SOURCE == "elite"
DECISIVE = {}  # position -> {move: decisive games}
SHARP = float(os.environ.get("BOOK_SHARP", "0"))  # >0: prefer our moves that lead to decisive games (sharp lines)


def explore(board):
    global requests_made
    k = key(board)
    if k in cache:
        return cache[k]
    if TABLE is not None:
        c = TABLE.get(k)
        moves = sorted([(u, n) for u, n in c.items() if "|" not in u], key=lambda x: -x[1]) if c else []
        if c:
            DECISIVE[k] = {u: c.get(u + "|d", 0) for u, _ in moves}
        cache[k] = (sum(n for _, n in moves), moves)
        return cache[k]
    for attempt in range(8):
        try:
            r = session.get(EXPLORER, params={**PARAMS, "fen": board.fen()}, timeout=30)
        except requests.RequestException as e:
            print("network error", e, flush=True)
            time.sleep(5)
            continue
        requests_made += 1
        if r.status_code == 429:
            delay[0] = min(5.0, delay[0] + 0.5)
            print(f"rate limited, waiting 60s (delay now {delay[0]}s)", flush=True)
            time.sleep(60)
            continue
        if r.status_code != 200:
            print("explorer error", r.status_code, r.text[:200], flush=True)
            if r.status_code in (401, 403):
                sys.exit("Explorer refused the request (authentication).")
            time.sleep(5)
            continue
        d = r.json()
        moves = []
        for m in d.get("moves", []):
            try:
                mv = board.parse_uci(m["uci"])
            except ValueError:
                continue
            moves.append((mv.uci(), m["white"] + m["draws"] + m["black"]))
        cache[k] = (d["white"] + d["draws"] + d["black"], moves)
        time.sleep(delay[0])
        return cache[k]
    return 0, []


def evaluate(sf, board, move, color):
    board.push(chess.Move.from_uci(move))
    try:
        info = sf.analyse(board, chess.engine.Limit(depth=DEPTH))
        return info["score"].pov(color).score(mate_score=10000)
    finally:
        board.pop()


def build(name, spec, color, sf):
    line = spec["line"].split()
    budget = max(3, int(spec["budget"] * float(os.environ.get("BOOK_BUDGET_SCALE", "1"))))
    book, seen, heap, n, counter = {}, set(), [], 0, 0
    heapq.heappush(heap, (0, counter, [], 0))
    deepest = 0
    while heap and n < budget and time.time() - start < TIME_LIMIT:
        _, _, moves, idx = heapq.heappop(heap)
        board = chess.Board()
        for m in moves:
            board.push_uci(m)
        state = (key(board), min(idx, len(line)))
        if state in seen or len(moves) >= MAX_PLY:
            continue
        seen.add(state)
        total, cands = explore(board)
        n += 1
        if total < MIN_NODE_GAMES and moves:
            continue
        ours = board.turn == color
        deepest = max(deepest, len(moves))
        children = []
        if idx < len(line) and line[idx] != "*":
            mv = line[idx]
            if chess.Move.from_uci(mv) not in board.legal_moves:
                continue
            if ours:
                games = dict(cands).get(mv, 0)
                if moves and games < MIN_NODE_GAMES / 4:
                    continue  # setup move almost never played here: leave it to Maia
                book[key(board)] = [[mv, 1]]
            children = [(mv, total)]
        elif ours:
            if not cands:
                continue
            top = cands[0][1]
            pool = [(m, g) for m, g in cands[:4] if g >= OUR_MIN_SHARE * top]
            scores = {m: evaluate(sf, board, m, color) for m, _ in pool}
            best = max(scores.values())
            keep = [(m, g) for m, g in pool if best - scores[m] <= MAX_LOSS_CP]
            if SHARP > 0 and DECISIVE.get(key(board)):
                dec = DECISIVE[key(board)]
                tot = sum(g for _, g in keep) or 1
                avg = (sum(dec.get(m, 0) for m, _ in keep) / tot) or 0.5
                keep = [(m, max(1, int(g * ((dec.get(m, 0) / max(g, 1)) / avg) ** SHARP))) for m, g in keep]
            book[key(board)] = [[m, g] for m, g in keep]
            children = keep
        else:
            children = [(m, g) for m, g in cands if g >= MIN_MOVE_SHARE * total]
        nidx = idx + 1 if idx < len(line) else idx
        for m, g in children:
            counter += 1
            heapq.heappush(heap, (-g, counter, moves + [m], nidx))
    print(f"{name}: {len(book)} book positions, {n} explored, deepest ply {deepest}", flush=True)
    return book, deepest


def main_line(book, color):
    board, sans = chess.Board(), []
    for _ in range(MAX_PLY):
        k = key(board)
        if board.turn == color:
            if k not in book:
                break
            mv = max(book[k], key=lambda x: x[1])[0]
        else:
            # follow the most common opponent reply that keeps us in book
            nxt = None
            for m, _ in explore(board)[1]:
                b2 = board.copy()
                b2.push_uci(m)
                if key(b2) in book:
                    nxt = m
                    break
            if nxt is None:
                break
            mv = nxt
        sans.append(board.san(chess.Move.from_uci(mv)))
        board.push_uci(mv)
    return " ".join(sans)


def merge(paths):
    """Combine per-opening parts into book.json and write the summary table."""
    out = {"source": "lichess.org games, 2200+ rated, blitz/rapid/classical",
           "built": time.strftime("%Y-%m-%d"), "openings": {}}
    rows = []
    for p in sorted(paths):
        part = json.load(open(p))
        out["openings"].update(part["openings"])
        rows += part.get("summary", [])
    json.dump(out, open("book.json", "w"), separators=(",", ":"))
    text = "\n".join(["## Opening book", "", "| Opening | Positions | Deepest ply | Main line |",
                      "|---|---|---|---|"] + rows)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--merge":
        return merge(sys.argv[2:])
    only = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--only" else None
    cfg = yaml.safe_load(open(os.environ.get("BOOK_CONFIG", "config.yml")))["openings"]
    global TABLE
    if SOURCE == "elite":
        from tools.elite_index import build_table
        TABLE, n_games = build_table(int(os.environ.get("ELITE_MONTHS", "4")))
    sf = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    sf.configure({"Threads": os.cpu_count(), "Hash": 512})
    out = {"source": ("Lichess Elite Database (2400+ vs 2200+, no bullet)" if SOURCE == "elite"
                      else "lichess.org games, 2200+ rated, blitz/rapid/classical"),
           "built": time.strftime("%Y-%m-%d"), "openings": {}}
    summary = ["## Opening book", "", "| Opening | Positions | Deepest ply | Main line |", "|---|---|---|---|"]
    # main openings first so they always get built even if time runs short
    jobs = [(n, s, c) for c in ("white", "black") for n, s in cfg[c].items() if only in (None, n)]
    jobs.sort(key=lambda j: -j[1]["weight"])
    for name, spec, color in jobs:
        col = chess.WHITE if color == "white" else chess.BLACK
        book, deepest = build(name, spec, col, sf)
        out["openings"][name] = {"color": color, "positions": book}
        summary.append(f"| {name} | {len(book)} | {deepest} | {main_line(book, col)} |")
    sf.quit()
    if only:
        out["summary"] = summary[4:]
        json.dump(out, open(f"part-{only}.json", "w"), separators=(",", ":"))
        print(summary[-1], f"| requests {requests_made}, {(time.time() - start) / 60:.0f} min")
        return
    json.dump(out, open(os.environ.get("BOOK_OUT", "book.json"), "w"), separators=(",", ":"))
    summary.append(f"\nExplorer requests: {requests_made}, time {(time.time() - start) / 60:.0f} min")
    text = "\n".join(summary)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


if __name__ == "__main__":
    main()
