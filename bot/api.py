"""HTTP API: Magnus v1 (pure Carlsen) analysis for a Safari userscript.

Uses the bot's own move selection (Bot.pick_move): Carlsen opening book -> light Maia search
-> Carlsen style rules -> Maia sampling. No Stockfish: the Guard is never created.

Start (from the repo root):  uvicorn bot.api:app --host 0.0.0.0 --port 8000
Call:  GET /analyze?fen=<url-encoded FEN>      optional: &n=5 (moves listed), &key=... (if API_KEY set)

Evals are Maia's own win/draw/loss feeling converted to pawns (White's view), not engine evals.
"""
import math
import os
import threading
import uuid

import chess
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from bot.book import Book
from bot.engine import Maia
from bot.main import Bot
from bot.search import expected
from bot.settings import Settings

CONFIG = "config-carlsen.yml"
STATUS = "Magnus v1"

settings = Settings(CONFIG)
cfg = settings.cfg
# Refuse to start with anything other than the pure Carlsen profile.
assert cfg["maia"]["checkpoint"] == "carlsen-79m.pt", "wrong model"
assert cfg["style"]["rules_file"] == "tal/carlsen-rules.json", "wrong style rules"
assert cfg["openings"]["book"]["path"] == "book-carlsen.json", "wrong opening book"
assert not cfg.get("assist", {}).get("enabled") and not cfg.get("guard", {}).get("enabled"), "Stockfish must be off"
assert not cfg["style"].get("plans", {}).get("enabled"), "plans = Magnus v2, not v1"

maia = Maia(cfg["maia"]["model"], cfg["maia"]["checkpoint"])
book = Book(cfg["openings"]["book"]["path"])
bot = Bot(settings, api=None, maia=maia, guard=None, me="api", book=book)   # guard=None: no Stockfish
lock = threading.Lock()

app = FastAPI(title="Maia bot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def fmt(score, white_to_move):
    """Expected score (0..1, side to move) -> '+0.4' in pawns from White's view (Lichess win% curve)."""
    s = min(0.999, max(0.001, score))
    pawns = math.log(s / (1 - s)) / 0.00368208 / 100
    return f"{pawns if white_to_move else -pawns:+.1f}"


def move_eval(board, c):
    """Eval after candidate c: mate if it mates, else the search value (if calculated) or Maia's wdl."""
    board.push(c["move"])
    mate = board.is_checkmate()
    board.pop()
    if mate:
        return "#1" if board.turn == chess.WHITE else "#-1"
    return fmt(c.get("value", expected(c["wdl"])), board.turn == chess.WHITE)


@app.get("/analyze")
def analyze(fen: str = Query(...), n: int = Query(5, ge=1, le=12), key: str = ""):
    if os.environ.get("API_KEY") and key != os.environ["API_KEY"]:
        raise HTTPException(401, "bad key")
    try:
        board = chess.Board(fen)
    except ValueError:
        raise HTTPException(400, "invalid FEN")
    if board.is_game_over():
        return {"eval": None, "moves": [], "best": None, "book": None, "status": STATUS}
    gid = uuid.uuid4().hex
    with lock:
        try:
            move, _, opening, _ = bot.pick_move(gid, board, {}, board.turn, None)
        finally:
            book.end(gid)
        top = bot.last_top
        if len(top) < 2:   # book move or one obvious move: still list Maia's candidates
            top = maia.choose(board, cfg["maia"], candidates=n)[1]
    top = sorted(top, key=lambda c: c["policy"], reverse=True)[:n]
    moves = [{"move": board.san(c["move"]), "eval": move_eval(board, c), "policy": round(c["policy"], 3)}
             for c in top]
    white = board.turn == chess.WHITE
    return {
        "eval": max((m["eval"] for m in moves), key=lambda e: _sortable(e, white)) if moves else None,
        "moves": moves,
        "best": board.san(move),   # the move Magnus v1 would actually play here
        "book": opening,           # opening name if it came from Carlsen's book, else null
        "status": STATUS,
    }


def _sortable(e, white_to_move):
    """Rank eval strings from the side to move's view (best line = position eval)."""
    if e.startswith("#"):
        v = 1000 if not e.startswith("#-") else -1000
    else:
        v = float(e)
    return v if white_to_move else -v


@app.get("/")
def health():
    return {"status": STATUS, "model": cfg["maia"]["checkpoint"]}
