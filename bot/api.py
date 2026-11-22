"""HTTP API: Tal v8 analysis for a Safari/userscript.

Uses the bot's own move selection (Bot.pick_move) with Stockfish-dependent features disabled.
Falls back to pure Maia if pick_move raises.

Start (from the repo root):  uvicorn bot.api:app --host 0.0.0.0 --port 8000
Call:  GET /analyze?fen=<url-encoded FEN>      optional: &n=5
"""
import math
import os
import threading
import uuid
import traceback

import chess
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from bot.book import Book
from bot.engine import Maia
from bot.main import Bot
from bot.search import expected
from bot.settings import Settings

CONFIG = "config-tal.yml"
STATUS = "Tal v8"

settings = Settings(CONFIG)
cfg = settings.cfg

# Force pure Tal path for the API (no Stockfish)
cfg.setdefault("guard", {})["enabled"] = False
cfg.setdefault("assist", {})["enabled"] = False
cfg.setdefault("tal_rules2", {})["enabled"] = False
cfg.setdefault("brilliant", {})["enabled"] = False
cfg.setdefault("search", {})["enabled"] = False
cfg.setdefault("mcts", {})["enabled"] = False

assert cfg["maia"]["checkpoint"] == "tal-79m.pt", "wrong model"
assert cfg["style"]["rules_file"] == "tal/tal-rules.json", "wrong style rules"
assert cfg["openings"]["book"]["path"] == "book-tal-sharp.json", "wrong opening book"

maia = Maia(cfg["maia"]["model"], cfg["maia"]["checkpoint"])
book = Book(cfg["openings"]["book"]["path"])
bot = Bot(settings, api=None, maia=maia, guard=None, me="api", book=book)
lock = threading.Lock()

app = FastAPI(title="Maia bot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def fmt(score, white_to_move):
    s = min(0.999, max(0.001, score))
    pawns = math.log(s / (1 - s)) / 0.00368208 / 100
    return f"{pawns if white_to_move else -pawns:+.1f}"


def move_eval(board, c):
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
    move = None
    opening = None
    with lock:
        try:
            move, _, opening, _ = bot.pick_move(gid, board, {}, board.turn, None)
        except Exception:
            traceback.print_exc()
            # fallback: pure Maia
            try:
                move, top = maia.choose(board, cfg["maia"], candidates=n)
                bot.last_top = top
            except Exception:
                traceback.print_exc()
                raise HTTPException(500, "analysis failed")
        finally:
            book.end(gid)
        top = getattr(bot, "last_top", []) or []
        if len(top) < 2:
            try:
                _, top = maia.choose(board, cfg["maia"], candidates=n)
            except Exception:
                top = []
    top = sorted(top, key=lambda c: c.get("policy", 0), reverse=True)[:n]
    moves = [{"move": board.san(c["move"]), "eval": move_eval(board, c), "policy": round(c.get("policy", 0), 3)}
             for c in top if "move" in c]
    white = board.turn == chess.WHITE
    return {
        "eval": max((m["eval"] for m in moves), key=lambda e: _sortable(e, white)) if moves else None,
        "moves": moves,
        "best": board.san(move) if move else (moves[0]["move"] if moves else None),
        "book": opening,
        "status": STATUS,
    }


def _sortable(e, white_to_move):
    if e.startswith("#"):
        v = 1000 if not e.startswith("#-") else -1000
    else:
        v = float(e)
    return v if white_to_move else -v


@app.get("/")
def health():
    return {"status": STATUS, "model": cfg["maia"]["checkpoint"]}
