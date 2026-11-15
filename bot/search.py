"""Light, human-like calculation on top of Maia.

Like a human: look at a few natural candidate moves (Maia's policy), check the opponent's
most likely replies (Maia playing the other side), and judge the resulting positions by
feel (Maia's own win/draw/loss head, not Stockfish). Candidates that this short calculation
refutes are dropped; among the rest the bot chooses naturally, weighted by Maia.
"""
import random


def expected(wdl):
    w, d, _ = wdl
    return (w + 0.5 * d) / 1000.0


class Search:
    def __init__(self, maia):
        self.maia = maia

    def _top(self, board, self_elo, oppo_elo, width):
        """Maia's top moves for the side to move, each with wdl from that side's view."""
        e = self.maia.eng
        with self.maia.lock:
            e.self_elo, e.oppo_elo = int(self_elo), int(oppo_elo)
            e.multipv = max(1, min(20, width))
            moves = " ".join(m.uci() for m in board.move_stack)
            root = board.root().fen()
            base = "position startpos" if root == chess.STARTING_FEN else f"position fen {root}"
            e.cmd_position(base + (f" moves {moves}" if moves else ""))
            _, top = e.score_moves()
        return top

    def value(self, board, depth, elos, widths):
        """Expected score (0..1) for the side to move, searching `depth` more plies."""
        if board.is_checkmate():
            return 0.0
        if board.is_game_over(claim_draw=True):
            return 0.5
        me, opp = elos
        top = self._top(board, me, opp, widths[0])
        if not top:
            return 0.5
        if depth <= 1:
            best = 0.0
            for c in top:
                board.push(c["move"])
                mate = board.is_checkmate()
                board.pop()
                best = max(best, 1.0 if mate else expected(c["wdl"]))
            return best
        best = 0.0
        for c in top:
            board.push(c["move"])
            v = 1.0 - self.value(board, depth - 1, (opp, me), widths[1:] or widths)
            board.pop()
            best = max(best, v)
        return best

    def choose(self, board, maia_cfg, scfg, opp_rating=None):
        """Return (move, candidates with search values, note or None)."""
        bot_elo = int(maia_cfg["elo"])
        fixed = maia_cfg.get("oppo_elo", "opponent")
        opp = int(min(2600, max(600, fixed if isinstance(fixed, (int, float)) else (opp_rating or bot_elo))))
        depth = int(scfg.get("depth", 1))          # plies after our move: 1 = their reply, 2 = +our answer
        widths = [int(w) for w in scfg.get("widths", [3, 2])]
        root = self._top(board, bot_elo, opp, int(scfg.get("candidates", 4)))
        if not root:
            return None, [], None
        if root[0]["policy"] >= float(scfg.get("skip_if_prob", 0.9)):
            return None, root, None   # obvious move: no need to calculate
        for c in root:
            board.push(c["move"])
            if board.is_checkmate():
                c["value"] = 1.0
            elif depth <= 0:
                c["value"] = expected(c["wdl"])
            else:
                c["value"] = 1.0 - self.value(board, depth, (opp, bot_elo), widths)
            board.pop()
        best = max(c["value"] for c in root)
        margin = float(scfg.get("margin", 0.10))     # how much worse than the best line is still OK
        ok = [c for c in root if c["value"] >= best - margin]
        temp = max(0.05, float(maia_cfg["temperature"]))
        pick = random.choices(ok, weights=[c["policy"] ** (1 / temp) for c in ok])[0]
        note = None
        dropped = [c for c in root if c not in ok]
        if dropped and dropped[0] is root[0]:
            note = (f"calc: rejected Maia's first idea {board.san(root[0]['move'])} "
                    f"({root[0]['value']:.2f} vs {best:.2f}), played {board.san(pick['move'])}")
        return pick["move"], root, note
