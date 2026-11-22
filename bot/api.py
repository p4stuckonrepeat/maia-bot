"""HTTP API: Tal v8 analysis for a Safari/userscript.

Uses the bot's own move selection, then the Tal rules with Stockfish off
(Maia's own feel only): attack, complications, sacrifices, still sampled
like a person. No engine moves.

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
from bot.engine import Guard, Maia
from bot.main import Bot
from bot.search import expected
from bot.settings import Settings

CONFIG = "config-tal.yml"
STATUS = "Tal v8"

settings = Settings(CONFIG)
cfg = settings.cfg

# This runner has no Stockfish. Tal rules still run, on Maia's own judgement.
cfg.setdefault("guard", {})["enabled"] = False
cfg.setdefault("assist", {})["enabled"] = False
cfg.setdefault("brilliant", {})["enabled"] = False
cfg.setdefault("search", {})["enabled"] = False
cfg.setdefault("mcts", {})["enabled"] = False
cfg.setdefault("tal_rules2", {})["enabled"] = True
cfg["tal_rules2"]["no_stockfish"] = True

assert cfg["maia"]["checkpoint"] == "tal-79m.pt", "wrong model"
assert cfg["style"]["rules_file"] == "tal/tal-rules.json", "wrong style rules"
assert cfg["openings"]["book"]["path"] == "book-tal-sharp.json", "wrong opening book"

maia = Maia(cfg["maia"]["model"], cfg["maia"]["checkpoint"])
book = Book(cfg["openings"]["book"]["path"])
bot = Bot(settings, api=None, maia=maia, guard=None, me="api", book=book)
_rules = object.__new__(Guard)  # tal_rules cache only; does not start Stockfish
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


def tal_feel(board, move):
    """Re-pick among Maia's natural moves with the Tal rules. No Stockfish."""
    rcfg = dict(cfg.get("tal_rules2", {}))
    rcfg["enabled"] = True
    rcfg["no_stockfish"] = True
    cands = maia.choose(board, cfg["maia"], candidates=int(rcfg.get("candidates", 10)))[1]
    sf = cfg.get("search_filter", {})
    if sf.get("enabled") and len(cands) >= 3:
        cands, _ = bot.feel_filter(board, cands, sf)
    bot.last_top = cands
    if len(cands) < 2:
        return move
    move2, _ = _rules.tal_rules(board, cands, rcfg)
    return move2 or move


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
            if not opening and move is not None:
                move = tal_feel(board, move)
        except Exception:
            traceback.print_exc()
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
