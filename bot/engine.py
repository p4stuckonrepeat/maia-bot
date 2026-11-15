"""Maia-3 move selection (in-process) plus the optional Stockfish blunder guard."""
import math
import random
import threading
import time

import chess
import chess.engine
from maia3.uci import Maia3UCIEngine, parse_args

from bot.style import STYLES


class Maia:
    """Runs the official Maia-3 inference code inside this process."""

    def __init__(self, model, checkpoint=None):
        self.lock = threading.Lock()
        args = ["--model", model, "--use-uci-history", "--multipv", "1"]
        if checkpoint:
            args += ["--checkpoint", checkpoint]       # e.g. the Tal fine-tuned weights
        self.eng = Maia3UCIEngine(parse_args(args))
        self.eng.ensure_model_loaded()

    def choose(self, board, maia_cfg, opp_rating=None, candidates=1):
        """Return (sampled move, candidates ranked by Maia probability, seconds).
        Each candidate: {"move", "policy", "wdl"}; wdl is Maia's own win/draw/loss estimate."""
        fixed = maia_cfg.get("oppo_elo", "opponent")
        if isinstance(fixed, (int, float)):
            opp_rating = fixed
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

    def choose_styled(self, board, maia_cfg, style_cfg, opp_rating=None):
        """Maia's natural moves, re-weighted by a consistent playing style.
        Only moves inside Maia's own top-p set are eligible; temperature applies as usual."""
        _, top, secs = self.choose(board, maia_cfg, opp_rating, candidates=int(style_cfg.get("candidates", 8)))
        if not top:
            return None, top, secs
        score_fn = STYLES[style_cfg.get("name", "positional")]
        if style_cfg.get("name") == "tal_learned":   # weights fitted on Tal's games decide everything
            from bot.style import tal_learned_logit
            logits = [tal_learned_logit(board, c) for c in top]
            temp = max(0.05, float(maia_cfg["temperature"]))
            mx = max(logits)
            weights = [math.exp((l - mx) / temp) for l in logits]
            return random.choices([c["move"] for c in top], weights=weights)[0], top, secs
        strength = float(style_cfg.get("strength", 1.0))
        temp = max(0.05, float(maia_cfg["temperature"]))
        total = sum(c["policy"] for c in top)
        rel = float(style_cfg.get("min_relative_prob", 0.5))
        pool, mass = [], 0.0
        for c in top:  # Maia's nucleus, and only moves nearly as likely as Maia's top choice
            if pool and c["policy"] < rel * top[0]["policy"]:
                break
            pool.append(c)
            mass += c["policy"] / total
            if mass >= float(maia_cfg["top_p"]):
                break
        def style_of(c):
            w, d, _ = c.get("wdl", (0, 1000, 0))
            try:
                return score_fn(board, c["move"], (w + 0.5 * d) / 1000.0)
            except TypeError:
                return score_fn(board, c["move"])
        weights = [(c["policy"] ** (1 / temp)) * math.exp(strength * style_of(c)) for c in pool]
        return random.choices([c["move"] for c in pool], weights=weights)[0], top, secs


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
            winning = before >= gcfg.get("winning_cp", 150)
            floor = gcfg.get("winning_floor_cp", -50)
            throws_win = winning and after < floor
            collapses = after <= gcfg["lost_cp"] and before - after >= gcfg["min_drop_cp"]
            if not (throws_win or collapses):
                return move, None
            # Blunder: replace it with one of Maia's own alternatives.
            cands = [c["move"] for c in candidates_fn() if c["move"] != move]
            if gcfg.get("replace_with", "softer_mistake") == "softer_mistake":
                # A smaller mistake instead of the blunder: aim for a worse-but-playable position
                # (e.g. winning -> roughly equal), not for the best move.
                target = min(before - float(gcfg.get("softer_drop_cp", 300)),
                             float(gcfg.get("softer_target_cp", 0)))
                target = max(target, gcfg["lost_cp"] + 150, floor if winning else -10000)
                scored = [(m, self.after(board, m, color, depth)) for m in cands]
                ok = [(m, a) for m, a in scored if a > gcfg["lost_cp"] and (not winning or a >= floor)]
                if ok:
                    m, a = min(ok, key=lambda x: abs(x[1] - target))
                    return m, (f"vetoed {board.san(move)} ({before:+}->{after:+}), played a smaller "
                               f"mistake: Maia's {board.san(m)} ({a:+})")
            else:
                for m in cands:
                    a = self.after(board, m, color, depth)
                    if a > gcfg["lost_cp"] and (not winning or a >= floor):
                        return m, (f"vetoed {board.san(move)} ({before:+}->{after:+}), "
                                   f"played Maia's {board.san(m)} ({a:+})")
            best = self.sf.play(board, chess.engine.Limit(depth=depth)).move
            return best, (f"vetoed {board.san(move)} ({before:+}->{after:+}), "
                          f"no safe Maia move, played Stockfish's {board.san(best)}")

    def assist(self, board, move, top, acfg):
        """Quality filter between Maia's own candidate moves: if the chosen move is much worse
        than another natural candidate, pick a natural candidate that is close to the best.
        Return (move, note or None)."""
        color, depth = board.turn, int(acfg.get("depth", 12))
        if acfg.get("only_when_not_top", True) and top and move == top[0]["move"]:
            return move, None  # Maia's own first choice: keep its natural (human) inaccuracies
        cands = [c for c in top if c["policy"] >= float(acfg.get("min_prob", 0.03))]
        if move not in [c["move"] for c in cands]:
            cands.append({"move": move, "policy": 0.01})
        if len(cands) < 2:
            return move, None
        with self.lock:
            scores = {c["move"]: self.after(board, c["move"], color, depth) for c in cands}
        best = max(scores.values())
        limit = float(acfg.get("max_loss_cp", 100))
        if best - scores[move] <= limit:
            return move, None
        ok = [c for c in cands if best - scores[c["move"]] <= limit]
        pick = random.choices([c["move"] for c in ok], weights=[c["policy"] for c in ok])[0]
        return pick, (f"assist: {board.san(move)} ({scores[move]:+}) -> Maia's "
                      f"{board.san(pick)} ({scores[pick]:+})")

    def close(self):
        try:
            self.sf.quit()
        except Exception:
            pass
