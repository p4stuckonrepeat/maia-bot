"""Small HTTP API: Maia's likely moves for a position, each checked by Stockfish.

    GET  /analyze?fen=<FEN>[&depth=12&n=5]      header X-API-Key: <key>  (or &key=<key>)
    POST /analyze  {"fen": "...", "depth": 12, "n": 5}
    GET  /health   (no key, for checking the server is up)

Start:  ANALYZE_KEY=<secret> python -m bot.api        (port 8000, or PORT=...)
Uses the same config file as the bot (BOT_CONFIG, default config.yml) for model / Elo.
Evaluations are from White's point of view, like a normal eval bar.
"""
import hmac
import os
import threading
from collections import OrderedDict

import chess
import chess.engine
import yaml
from flask import Flask, jsonify, request

KEY = os.environ.get("ANALYZE_KEY", "")
ORIGINS = {o.strip() for o in os.environ.get(
    "ANALYZE_ORIGINS", "https://www.chess.com,https://chess.com").split(",") if o.strip()}
DEFAULT_DEPTH = int(os.environ.get("ANALYZE_DEPTH", 12))
MAX_DEPTH, MAX_MOVES, CACHE_SIZE = 18, 8, 128

app = Flask(__name__)
_maia = _guard = _cfg = None
_cache, _cache_lock = OrderedDict(), threading.Lock()


def load():
    global _maia, _guard, _cfg
    from bot.engine import Guard, Maia
    _cfg = yaml.safe_load(open(os.environ.get("BOT_CONFIG", "config.yml")))
    _maia = Maia(_cfg["maia"]["model"], _cfg["maia"].get("checkpoint"))
    _guard = Guard(threads=int(os.environ.get("SF_THREADS", os.cpu_count() or 2)))


def fmt(score):
    """PovScore (White's view) -> '+0.80', '-1.25', '#3', '#-2'."""
    m = score.mate()
    if m is not None:
        return f"#{m}"
    return f"{score.score() / 100:+.2f}"


def sf_eval(board, depth):
    """Stockfish eval of `board` from White's view, as text."""
    if board.is_checkmate():
        return "#0" if board.turn == chess.BLACK else "#-0"
    if board.is_game_over():
        return "0.00"
    with _guard.lock:
        info = _guard.sf.analyse(board, chess.engine.Limit(depth=depth))
    return fmt(info["score"].white())


def analyze(fen, depth, n):
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("illegal position")
    if board.is_game_over():
        return {"eval": sf_eval(board, depth), "moves": [], "status": f"game over ({board.result()})"}
    _, top, _ = _maia.choose(board, _cfg["maia"], candidates=n)
    moves = []
    for c in top[:n]:
        mv = c["move"]
        san = board.san(mv)
        board.push(mv)
        try:
            ev = sf_eval(board, depth)
        finally:
            board.pop()
        moves.append({"move": san, "uci": mv.uci(), "eval": ev, "prob": round(float(c["policy"]), 3)})
    return {"eval": sf_eval(board, depth), "moves": moves, "status": f"Maia · depth {depth}"}


def _int(v, default, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


@app.after_request
def cors(resp):
    origin = request.headers.get("Origin")
    if origin in ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = "X-API-Key, Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.get("/health")
def health():
    return jsonify(ok=_maia is not None)


@app.route("/analyze", methods=["GET", "POST", "OPTIONS"])
def analyze_route():
    if request.method == "OPTIONS":                 # CORS preflight
        return "", 204
    given = request.headers.get("X-API-Key") or request.args.get("key") or ""
    if not hmac.compare_digest(given.encode(), KEY.encode()):
        return jsonify(error="bad or missing key"), 401
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
    else:
        data = request.args
    fen = (data.get("fen") or "").strip()
    if not fen or len(fen) > 100:
        return jsonify(error="send a FEN as ?fen=... or {\"fen\": ...}"), 400
    depth = _int(data.get("depth"), DEFAULT_DEPTH, 1, MAX_DEPTH)
    n = _int(data.get("n"), 5, 1, MAX_MOVES)
    ck = (fen, depth, n)
    with _cache_lock:
        if ck in _cache:
            _cache.move_to_end(ck)
            return jsonify(_cache[ck])
    try:
        out = analyze(fen, depth, n)
    except ValueError as e:
        return jsonify(error=f"bad FEN: {e}"), 400
    with _cache_lock:
        _cache[ck] = out
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return jsonify(out)


def main():
    if len(KEY) < 16:
        raise SystemExit("Set ANALYZE_KEY to a secret of at least 16 characters.")
    print("Loading Maia + Stockfish...", flush=True)
    load()
    port = int(os.environ.get("PORT", 8000))
    print(f"Ready on http://0.0.0.0:{port}/analyze", flush=True)
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=port, threaded=True)


if __name__ == "__main__":
    main()
