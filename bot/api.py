"""HTTP API: Tal v8 analysis for a Safari/userscript.

No Stockfish on this runner. Uses the opening book, then Maia styled by the
Tal rules when that works, otherwise plain Maia. Never 500s.
"""
import math
import os
import threading
import traceback
import uuid

import chess
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bot.book import Book
from bot.engine import Maia
from bot.main import Bot
from bot.search import expected
from bot.settings import Settings

CONFIG = "config-tal.yml"
STATUS = "Tal v8"

settings = Settings(CONFIG)
cfg = settings.cfg

cfg.setdefault("guard", {})["enabled"] = False
cfg.setdefault("assist", {})["enabled"] = False
cfg.setdefault("brilliant", {})["enabled"] = False
cfg.setdefault("mcts", {})["enabled"] = False
# search stays as the profile sets it: one-ply check of the opponent's reply, no Stockfish

assert cfg["maia"]["checkpoint"] == "tal-79m.pt", "wrong model"

maia = Maia(cfg["maia"]["model"], cfg["maia"]["checkpoint"])
book = Book(cfg["openings"]["book"]["path"])
bot = Bot(settings, api=None, maia=maia, guard=None, me="api", book=book)
lock = threading.Lock()

app = FastAPI(title="Maia bot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.exception_handler(Exception)
async def any_error(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse({"eval": None, "moves": [], "best": None, "book": None, "status": "error"}, status_code=200)


def fmt(score, white_to_move):
    s = min(0.999, max(0.001, float(score)))
    pawns = math.log(s / (1 - s)) / 0.00368208 / 100
    return f"{pawns if white_to_move else -pawns:+.1f}"


def move_eval(board, c):
    try:
        board.push(c["move"])
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return "#1" if board.turn == chess.WHITE else "#-1"
        wdl = c.get("wdl")
        if not wdl:
            return "—"
        return fmt(c.get("value", expected(wdl)), board.turn == chess.WHITE)
    except Exception:
        return "—"


def safe_choose(board, n):
    try:
        move, top, _ = maia.choose(board, cfg["maia"], candidates=n)
        return move, top or []
    except Exception:
        traceback.print_exc()
        return None, []


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
    move, opening, top = None, None, []
    with lock:
        try:
            move, _, opening, _ = bot.pick_move(gid, board, {}, board.turn, None)
            top = getattr(bot, "last_top", []) or []
        except Exception:
            traceback.print_exc()
            move, opening = None, None
        finally:
            try:
                book.end(gid)
            except Exception:
                pass
        if move is None or len(top) < 2:
            m2, top2 = safe_choose(board, n)
            if move is None:
                move = m2
            if len(top2) >= len(top):
                top = top2

    try:
        top = sorted(top, key=lambda c: c.get("policy", 0), reverse=True)[:n]
        moves = []
        for c in top:
            if "move" not in c:
                continue
            try:
                moves.append({"move": board.san(c["move"]), "eval": move_eval(board, c),
                              "policy": round(float(c.get("policy", 0)), 3)})
            except Exception:
                continue
        best = None
        if move is not None:
            try:
                best = board.san(move)
            except Exception:
                best = None
        if not best and moves:
            best = moves[0]["move"]
        white = board.turn == chess.WHITE
        ev = None
        if moves:
            try:
                ev = max((m["eval"] for m in moves if m["eval"] not in (None, "—")),
                         key=lambda e: _sortable(e, white))
            except Exception:
                ev = moves[0]["eval"]
        return {"eval": ev, "moves": moves, "best": best, "book": opening, "status": STATUS}
    except Exception:
        traceback.print_exc()
        return {"eval": None, "moves": [], "best": None, "book": opening, "status": STATUS}


def _sortable(e, white_to_move):
    if str(e).startswith("#"):
        v = 1000 if not str(e).startswith("#-") else -1000
    else:
        v = float(e)
    return v if white_to_move else -v


@app.get("/")
def health():
    return {"status": STATUS, "model": cfg["maia"]["checkpoint"]}
