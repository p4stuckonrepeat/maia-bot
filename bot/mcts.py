"""Leela-style calculation with Maia only (no Stockfish), kept human.

Like Leela / AlphaZero, but with a human brain:
- Maia's move probabilities decide WHICH moves get looked at (only natural, human candidates).
- Maia's win/draw/loss head judges positions; the opponent's moves are judged with the
  opponent's rating, so it expects human replies, not engine replies.
- PUCT tree search; leaves are evaluated in batches (several positions per network call),
  which is several times faster on a CPU.

Human limits, so it never plays something a ~2500 player could not find:
- Only moves Maia gives at least `min_prior` (or its top 3) can be played: no engine-only moves.
- Width (`width` candidates per position) and depth (`max_depth` plies) are capped.
- The natural move (Maia / Tal style) is kept unless the calculation shows it is clearly worse
  (more than `margin` expected score). Then another natural candidate is chosen, weighted by Maia,
  not simply the best one.
"""
import math
import random
import time
from collections import deque

import chess
import torch

try:
    from maia3.dataset import tokenize_board
    from maia3.utils import mirror_move
except Exception:          # tests without maia3 installed
    tokenize_board = mirror_move = None


class Child:
    __slots__ = ("move", "prior", "n", "w", "vl", "node")

    def __init__(self, move, prior):
        self.move, self.prior, self.n, self.w, self.vl, self.node = move, prior, 0, 0.0, 0, None


class Node:
    __slots__ = ("children", "n", "v", "hist", "terminal", "horizon", "ready")

    def __init__(self, hist):
        self.children, self.n, self.v, self.hist = [], 0, 0.5, hist
        self.terminal = self.horizon = self.ready = False


class MCTS:
    def __init__(self, maia):
        self.maia = maia
        self.calls = 0
        self.evals = 0

    # -- batched network evaluation ----------------------------------------
    def _evaluate(self, items, width):
        """items: list of (board, node, self_elo, oppo_elo). Fills node.v (side to move) and children priors."""
        e = self.maia.eng
        dev = e.cfg.device
        with self.maia.lock, torch.no_grad():
            toks = torch.stack([e._tokens_from_history(n.hist) for _, n, _, _ in items]).to(dev)
            se = torch.tensor([s for _, _, s, _ in items], dtype=torch.long, device=dev)
            oe = torch.tensor([o for _, _, _, o in items], dtype=torch.long, device=dev)
            logits_move, logits_value, _ = e.model(toks, se, oe)
            logits_move, logits_value = logits_move.float().cpu(), logits_value.float().cpu()
        self.calls += 1
        self.evals += len(items)
        for i, (b, node, _, _) in enumerate(items):
            loss, draw, win = torch.softmax(logits_value[i], dim=-1).tolist()
            node.v = win + 0.5 * draw
            node.ready = True
            moves, idx = [], []
            for m in b.legal_moves:
                u = m.uci() if b.turn == chess.WHITE else mirror_move(m.uci())
                j = e.all_moves_dict.get(u)
                if j is not None:
                    moves.append(m)
                    idx.append(j)
            if not moves:
                continue
            p = torch.softmax(logits_move[i][idx], dim=-1)
            order = torch.argsort(p, descending=True)[:width].tolist()
            node.children = [Child(moves[k], float(p[k])) for k in order]

    @staticmethod
    def _q(ch, fpu):
        n = ch.n + ch.vl
        return (ch.w / n) if n else fpu

    def think(self, board, natural, top_root, cfg, bot_elo, opp_elo, budget_s):
        """Return (move, note or None). `natural` = the move the bot wanted to play anyway."""
        e = self.maia.eng
        width, max_depth = int(cfg.get("width", 6)), int(cfg.get("max_depth", 6))
        cpuct, fpu_red = float(cfg.get("cpuct", 1.5)), float(cfg.get("fpu_reduction", 0.1))
        batch, max_nodes = int(cfg.get("batch", 8)), int(cfg.get("max_nodes", 400))
        me = board.turn
        t0, self.calls, self.evals = time.time(), 0, 0
        with self.maia.lock:   # Maia's own history of the real game (set by the last choose() call)
            moves = " ".join(m.uci() for m in board.move_stack)
            r = board.root().fen()
            base = "position startpos" if r == chess.STARTING_FEN else f"position fen {r}"
            e.cmd_position(base + (f" moves {moves}" if moves else ""))
            root_hist = deque(e.history, maxlen=e.cfg.history)
        root = Node(root_hist)
        self._evaluate([(board, root, bot_elo, opp_elo)], max(width, int(cfg.get("root_width", 10))))
        if not root.children:
            return natural, None
        raw = {ch.move: ch.prior for ch in root.children}
        if natural not in raw:
            root.children.append(Child(natural, 0.01))
        while self.evals < max_nodes and time.time() - t0 < budget_s:
            pending, paths = [], []
            for _ in range(batch):
                b = board.copy(stack=True)
                node, path = root, []
                leaf = None
                while True:
                    sq = math.sqrt(node.n + 1)
                    fpu = node.v - fpu_red
                    c = max(node.children, key=lambda ch: self._q(ch, fpu) + cpuct * ch.prior * sq / (1 + ch.n + ch.vl))
                    b.push(c.move)
                    path.append((node, c))
                    c.vl += 1
                    if c.node is None:
                        hist = deque(node.hist, maxlen=e.cfg.history)
                        hist.append(tokenize_board(b))
                        c.node = Node(hist)
                        if b.is_checkmate():
                            c.node.terminal, c.node.v, c.node.ready = True, 0.0, True
                        elif b.is_game_over(claim_draw=True):
                            c.node.terminal, c.node.v, c.node.ready = True, 0.5, True
                        else:
                            leaf = c.node
                            c.node.horizon = len(path) >= max_depth
                        break
                    if not c.node.ready:                      # already waiting in this batch
                        leaf = "collision"
                        break
                    if c.node.terminal or c.node.horizon or not c.node.children:
                        break
                    node = c.node
                if leaf == "collision":
                    for nd, ch in path:
                        ch.vl -= 1
                    continue
                if leaf is not None:
                    elos = (bot_elo, opp_elo) if b.turn == me else (opp_elo, bot_elo)
                    pending.append((b, leaf, elos[0], elos[1]))
                paths.append(path)
            if pending:
                self._evaluate(pending, width)
                for _, nd, _, _ in pending:
                    if nd.horizon:
                        nd.children = []          # calculation horizon: value only, no deeper look
            for path in paths:
                v = 1.0 - path[-1][1].node.v      # value for the side that played the last move
                for nd, ch in reversed(path):
                    ch.vl -= 1
                    ch.n += 1
                    ch.w += v
                    nd.n += 1
                    v = 1.0 - v
            if not pending and root.n > 20 * max_nodes:
                break      # the whole reachable tree is explored
        # human limits: only natural candidates can be played
        kids = sorted(root.children, key=lambda ch: -raw.get(ch.move, 0))
        min_prior = float(cfg.get("min_prior", 0.04))
        allowed = [ch for i, ch in enumerate(kids) if i < 3 or raw.get(ch.move, 0) >= min_prior or ch.move == natural]
        q = lambda ch: ch.w / ch.n if ch.n else 0.0
        seen = [ch for ch in allowed if ch.n >= int(cfg.get("min_visits", 3))] or [ch for ch in allowed if ch.n] or allowed
        best = max(seen, key=q)
        nat = next((ch for ch in allowed if ch.move == natural), None)
        margin = float(cfg.get("margin", 0.05))
        spent = time.time() - t0
        if nat is not None and nat.n and q(nat) >= q(best) - margin:
            return natural, None
        if nat is not None and not nat.n:
            return natural, None          # not enough time to judge it: trust the natural move
        ok = [ch for ch in seen if q(ch) >= q(best) - margin]
        pick = random.choices(ok, weights=[raw.get(ch.move, 0.01) + 1e-3 for ch in ok])[0]
        return pick.move, (f"calc ({self.evals} positions, {spent:.1f}s): {board.san(natural)} "
                           f"{(q(nat) if nat else 0):.0%} -> {board.san(pick.move)} {q(pick):.0%}")
