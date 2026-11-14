"""Maia-3 move selection (in-process) plus the optional Stockfish blunder guard."""
import threading
import time

import chess
import chess.engine
from maia3.uci import Maia3UCIEngine, parse_args


class Maia:
    """Runs the official Maia-3 inference code inside this process."""

    def __init__(self, model):
        self.lock = threading.Lock()
        self.eng = Maia3UCIEngine(parse_args(["--model", model, "--use-uci-history", "--multipv", "1"]))
        self.eng.ensure_model_loaded()

    def choose(self, board, maia_cfg, opp_rating=None, candidates=1):
        """Return (sampled move, candidates ranked by Maia probability, seconds).
        Each candidate: {"move", "policy", "wdl"}; wdl is Maia's own win/draw/loss estimate."""
        opp = int(min(2600, max(600, opp_rating or maia_cfg["elo"])))
        with self.lock:
            e = self.eng
            e.self_elo, e.oppo_elo = int(maia_cfg["elo"]), opp
            e.temperature, e.top_p = float(maia_cfg["temperature"]), float(maia_cfg["top_p"])
            e.multipv = max(1, min(20, candidates))
            moves = " ".join(m.uci() for m in board.move_stack)
            e.cmd_position("position startpos" + (f" moves {moves}" if moves else ""))
            t = time.time()
            move, top = e.score_moves()
            return move, top, time.time() - t


class Guard:
    """Stockfish safety check. Only vetoes moves that turn a playable position into a clearly
    lost one. Inaccuracies, positional mistakes and winning->equal are allowed."""

    def __init__(self, path="/usr/games/stockfish", threads=4):
        self.lock = threading.Lock()
        self.sf = chess.engine.SimpleEngine.popen_uci(path)
        self.sf.configure({"Threads": threads, "Hash": 256})

    def score(self, board, color, depth):
        """Evaluation in centipawns from `color`'s point of view (mate = +-10000)."""
        info = self.sf.analyse(board, chess.engine.Limit(depth=depth))
        return info["score"].pov(color).score(mate_score=10000)

    def after(self, board, move, color, depth):
        board.push(move)
        try:
            if board.is_checkmate():
                return 10000
            if board.is_game_over():
                return 0
            return self.score(board, color, depth)
        finally:
            board.pop()

    def check(self, board, move, gcfg, candidates_fn):
        """Return (final move, note or None)."""
        color, depth = board.turn, int(gcfg["depth"])
        with self.lock:
            before = self.score(board, color, depth)
            if before < gcfg["playable_cp"]:
                return move, None  # already bad: let Maia play on naturally
            after = self.after(board, move, color, depth)
            if after > gcfg["lost_cp"] or before - after < gcfg["min_drop_cp"]:
                return move, None
            # Blunder: try Maia's own alternatives in order of how likely Maia thinks they are.
            for cand in candidates_fn():
                m = cand["move"]
                if m == move:
                    continue
                a = self.after(board, m, color, depth)
                if a > gcfg["lost_cp"]:
                    return m, (f"vetoed {board.san(move)} ({before:+}->{after:+}), "
                               f"played Maia's {board.san(m)} ({a:+})")
            best = self.sf.play(board, chess.engine.Limit(depth=depth)).move
            return best, (f"vetoed {board.san(move)} ({before:+}->{after:+}), "
                          f"no safe Maia move, played Stockfish's {board.san(best)}")

    def close(self):
        try:
            self.sf.quit()
        except Exception:
            pass
