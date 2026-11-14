"""Count opening moves from the Lichess Elite Database (2400+ vs 2200+, no bullet).

Downloads the newest monthly files from database.nikonoel.fr and returns
{position key: {uci move: games}} for the first MAX_PLY half-moves of every game.
"""
import io
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool

import chess
import requests

SITE = "https://database.nikonoel.fr/"
MAX_PLY = 30
TOKEN_RE = re.compile(r"\{[^}]*\}|\([^)]*\)|\d+\.(\.\.)?|\$\d+|1-0|0-1|1/2-1/2|\*")


def key(board):
    return " ".join(board.fen().split()[:4])


def month_urls(n):
    html = requests.get(SITE, timeout=60).text
    files = sorted(set(re.findall(r'href="([^"]*lichess_elite_\d{4}-\d{2}\.zip)"', html)))
    if not files:
        sys.exit("Could not find any monthly files on " + SITE)
    return [f if f.startswith("http") else SITE + f.lstrip("./") for f in files[-n:]]


def games_from_pgn(text):
    """Yield the first MAX_PLY SAN moves of each game."""
    moves = []
    for line in text.splitlines():
        if line.startswith("["):
            if moves:
                yield moves
                moves = []
            continue
        if line.strip() and len(moves) < MAX_PLY:
            moves.extend(t.rstrip("!?") for t in TOKEN_RE.sub(" ", line).split()[: MAX_PLY - len(moves)])
    if moves:
        yield moves


def count_chunk(games):
    counts = defaultdict(Counter)
    for sans in games:
        board = chess.Board()
        try:
            for san in sans:
                k = key(board)
                mv = board.push_san(san)
                counts[k][mv.uci()] += 1
        except ValueError:
            continue
    return counts


def build_table(months=4):
    table = defaultdict(Counter)
    total_games = 0
    for url in month_urls(months):
        t = time.time()
        data = requests.get(url, timeout=600).content
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            text = z.read(z.namelist()[0]).decode("utf-8", "ignore")
        del data
        games = list(games_from_pgn(text))
        del text
        total_games += len(games)
        chunks = [games[i::os.cpu_count()] for i in range(os.cpu_count())]
        with Pool(os.cpu_count()) as pool:
            for part in pool.imap_unordered(count_chunk, chunks):
                for k, c in part.items():
                    table[k].update(c)
        # drop positions seen only once so far to keep memory in check
        for k in [k for k, c in table.items() if sum(c.values()) < 2]:
            del table[k]
        print(f"{url.rsplit('/', 1)[-1]}: {len(games)} games, {len(table)} positions kept, "
              f"{time.time() - t:.0f}s", flush=True)
    print(f"Elite database: {total_games} games from the newest {months} months", flush=True)
    return table, total_games
