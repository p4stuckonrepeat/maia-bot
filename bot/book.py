"""Opening repertoire lookup. Main openings (weight > 0) are preferred; weight-0 openings are
only used when no main opening covers the position. Out of book -> Maia takes over."""
import json
import os
import random

import chess


def key(board):
    return " ".join(board.fen().split()[:4])


class Book:
    def __init__(self, path="book.json"):
        self.openings = {}
        if os.path.exists(path):
            data = json.load(open(path))
            if "openings" in data:
                self.openings = data["openings"]
            elif "white" in data and "black" in data:   # a player's own book: {side: {fen: {uci: count}}}
                for side in ("white", "black"):
                    self.openings[f"player_{side}"] = {
                        "color": side,
                        "positions": {k: [[m, c] for m, c in v.items()] for k, v in data[side].items()}}
        self.games = {}  # game id -> {"opening": name or None, "out": bool}

    def loaded(self):
        return sum(len(o["positions"]) for o in self.openings.values())

    def pick(self, gid, board, color, cfg):
        """Return (move, opening name) or (None, None) when out of book."""
        ocfg = cfg.get("openings", {})
        if not ocfg.get("book", {}).get("enabled", True) or not self.openings:
            return None, None
        st = self.games.setdefault(gid, {"opening": None, "out": False})
        if st["out"]:
            return None, None
        side = "white" if color == chess.WHITE else "black"
        weights = {n: s.get("weight", 0) for n, s in ocfg.get(side, {}).items()}
        k = key(board)
        for n in self.openings:
            weights.setdefault(n, 100)   # e.g. a player's own book, not listed in config
        fits = [n for n, o in self.openings.items()
                if o["color"] == side and k in o["positions"]]
        if not fits:
            st["out"] = True
            return None, None
        name = st["opening"]
        if name not in fits:
            main = [n for n in fits if weights[n] > 0]
            if main:
                name = random.choices(main, weights=[weights[n] for n in main])[0]
            else:
                name = random.choice(fits)  # fallback opening: only because no main one fits
            st["opening"] = name
        entries = [(m, w) for m, w in self.openings[name]["positions"][k]
                   if chess.Move.from_uci(m) in board.legal_moves]
        if not entries:
            st["out"] = True
            return None, None
        m = random.choices([e[0] for e in entries], weights=[e[1] for e in entries])[0]
        return chess.Move.from_uci(m), name

    def end(self, gid):
        return self.games.pop(gid, None)
