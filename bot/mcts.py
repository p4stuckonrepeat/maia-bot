"""Leela-style calculation with Maia only (no Stockfish), kept human.

How it works (like Leela / AlphaZero, but with a human brain):
- Maia's move probabilities decide WHICH moves get looked at (only natural, human candidates).
- Maia's own win/draw/loss judgement decides how good a position feels.
- The opponent is assumed to answer like a strong human (Maia at the opponent's rating).
- The search tree grows where it matters (PUCT), limited in time and depth like a human
  calculating a few moves ahead.

Human limits, so it never plays something a ~2500 player could not find:
- Only moves Maia gives at least `min_prior` (or its top 3) can be played. Engine-only moves are never chosen.
- Calculation depth is capped (`max_depth` plies), width too (`width` candidates per position).
- The natural move (Maia / Tal style) is kept unless the calculation shows it is clearly worse
  (more than `margin` expected score). Then a different natural candidate is chosen, weighted by Maia,
  not simply the best one.
"""
import math
import random
import time

import chess


def expected(wdl):
    w, d, _ = wdl
    return (w + 0.5 * d) / 1000.0


class Child:
    __slots__ = ("move", "prior", "q0", "n", "w", "node")

    def __init__(self, move, prior, q0):
        self.move, self.prior, self.q0, self.n, self.w, self.node = move, prior, q0, 0, 0.0, None

    def q(self):
        return self.w / self.n if self.n else self.q0


class Node:
    __slots__ = ("children", "n", "leaf")

    def __init__(self):
        self.children, self.n, self.leaf = [], 0, 0.5


class MCTS:
    def __init__(self, maia):
        self.maia = maia
        self.calls = 0

    def _top(self, board, self_elo, oppo_elo, width):
        e = self.maia.eng
        with self.maia.lock:
            e.self_elo, e.oppo_elo = int(self_elo), int(oppo_elo)
            e.multipv = max(1, min(20, width))
            moves = " ".join(m.uci() for m in board.move_stack)
            root = board.root().fen()
            base = "position startpos" if root == chess.STARTING_FEN else f"position fen {root}"
            e.cmd_position(base + (f" moves {moves}" if moves else ""))
            _, top = e.score_moves()
        self.calls += 1
        return top

    def _expand(self, board, node, elos, width, sharp):
        """Fill node's children (priors + first impressions). Returns the position's value for the side to move."""
        top = self._top(board, elos[0], elos[1], width)
        tot = sum(c["policy"] for c in top) or 1.0
        for c in top:
            board.push(c["move"])
            q0 = 1.0 if board.is_checkmate() else (0.5 if board.is_game_over(claim_draw=True) else expected(c["wdl"]))
            board.pop()
            node.children.append(Child(c["move"], c["policy"] / tot, q0))
        if not node.children:
            return 0.5
        # what a strong human would get from here: Maia's choices, a bit sharper than its raw probabilities
        ws = [ch.prior ** (1.0 / sharp) for ch in node.children]
        s = sum(ws)
        node.leaf = sum(w * ch.q0 for w, ch in zip(ws, node.children)) / s
        return node.leaf

    def think(self, board, natural, top_root, cfg, bot_elo, opp_elo, budget_s):
        """Return (move, note or None). `natural` = the move the bot wanted to play anyway."""
        width, max_depth = int(cfg.get("width", 6)), int(cfg.get("max_depth", 6))
        cpuct, sharp = float(cfg.get("cpuct", 1.5)), float(cfg.get("sharpness", 0.5))
        max_nodes = int(cfg.get("max_nodes", 400))
        me = board.turn
        t0, self.calls = time.time(), 0
        root = Node()
        self._expand(board, root, (bot_elo, opp_elo), max(width, int(cfg.get("root_width", 8))), sharp)
        if not root.children:
            return natural, None
        raw = {c["move"]: c["policy"] for c in top_root} if top_root else {}
        if natural not in [c.move for c in root.children]:
            board.push(natural)
            q0 = 1.0 if board.is_checkmate() else 0.5
            if not board.is_game_over(claim_draw=True):
                n = Node()
                q0 = 1.0 - self._expand(board, n, (opp_elo, bot_elo), width, sharp)
            board.pop()
            root.children.append(Child(natural, 0.02, q0))
        while self.calls < max_nodes and time.time() - t0 < budget_s:
            b = board.copy(stack=True)
            node, path = root, []
            while True:
                sq = math.sqrt(node.n + 1)
                c = max(node.children, key=lambda ch: ch.q() + cpuct * ch.prior * sq / (1 + ch.n))
                b.push(c.move)
                path.append((node, c))
                if b.is_checkmate():
                    v = 1.0
                    break
                if b.is_game_over(claim_draw=True):
                    v = 0.5
                    break
                if c.node is None:
                    if len(path) >= max_depth:      # calculation horizon: trust the first impression
                        v = c.q0
                        break
                    c.node = Node()
                    elos = (bot_elo, opp_elo) if b.turn == me else (opp_elo, bot_elo)
                    v = 1.0 - self._expand(b, c.node, elos, width, sharp)
                    if not c.node.children:
                        c.node = None
                    break
                node = c.node
            for nd, ch in reversed(path):           # v = value for the side that played ch.move
                ch.n += 1
                ch.w += v
                nd.n += 1
                v = 1.0 - v
        # human limits: only natural candidates can be played
        kids = sorted(root.children, key=lambda ch: -raw.get(ch.move, ch.prior))
        min_prior = float(cfg.get("min_prior", 0.04))
        allowed = [ch for i, ch in enumerate(kids) if i < 3 or raw.get(ch.move, ch.prior) >= min_prior or ch.move == natural]
        seen = [ch for ch in allowed if ch.n >= int(cfg.get("min_visits", 3))] or allowed
        best = max(seen, key=lambda ch: ch.q())
        nat = next((ch for ch in allowed if ch.move == natural), None)
        margin = float(cfg.get("margin", 0.05))
        spent = time.time() - t0
        if nat is not None and nat.q() >= best.q() - margin:
            return natural, None
        ok = [ch for ch in seen if ch.q() >= best.q() - margin]
        pick = random.choices(ok, weights=[raw.get(ch.move, ch.prior) + 1e-3 for ch in ok])[0]
        san = board.san
        return pick.move, (f"calc ({self.calls} positions, {spent:.1f}s): {san(natural)} "
                           f"{(nat.q() if nat else 0):.0%} -> {san(pick.move)} {pick.q():.0%}")
