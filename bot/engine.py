"""Maia-3 move selection through the official maia3-uci engine."""
import threading
import time

import chess
import chess.engine


class Maia:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.eng = chess.engine.SimpleEngine.popen_uci(
            ["maia3-uci", "--model", cfg["model"], "--use-uci-history", "--multipv", "1"])

    def choose(self, board, opp_rating=None):
        """Return (move, seconds). Maia sees the full move history of the game."""
        opp = int(min(2600, max(600, opp_rating or self.cfg["elo"])))
        with self.lock:
            self.eng.configure({
                "SelfElo": int(self.cfg["elo"]),
                "OppoElo": opp,
                "Temperature": str(self.cfg["temperature"]),
                "TopP": str(self.cfg["top_p"]),
            })
            t = time.time()
            move = self.eng.play(board, chess.engine.Limit(nodes=1)).move
            return move, time.time() - t

    def close(self):
        try:
            self.eng.quit()
        except Exception:
            pass
