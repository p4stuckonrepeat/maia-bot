"""Maia-3 Lichess bot: accepts challenges and plays them with Maia-3 for one session."""
import argparse
import os
import threading
import time

import chess
from bot.book import Book
from bot.engine import Guard, Maia
from bot.lichess import Lichess
from bot.settings import Settings
from bot.timing import think_seconds

STANDARD_FEN = chess.STARTING_FEN


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Bot:
    def __init__(self, settings, api, maia, guard, me, book=None):
        self.settings, self.api, self.maia, self.guard, self.me = settings, api, maia, guard, me
        self.book = book or Book()
        self.lock = threading.Lock()
        self.active = set()     # game ids being played
        self.pending = {}       # challenge id -> accept time (game not started yet)
        self.accepting = True
        self.results = []

    @property
    def cfg(self):
        return self.settings.cfg

    # ---------- challenges ----------
    def decline_reason(self, ch):
        c = self.cfg["challenges"]
        tc = ch.get("timeControl", {})
        if ch.get("variant", {}).get("key") not in c["variants"]:
            return "variant"
        if tc.get("type") == "unlimited":
            if not c.get("accept_unlimited", False):
                return "timeControl"
        elif tc.get("type") != "clock":
            return "timeControl"
        elif tc["limit"] < c["min_base_seconds"]:
            return "tooFast"
        elif tc["limit"] > c["max_base_seconds"] or tc["increment"] > c["max_increment"]:
            return "tooSlow"
        if ch.get("rated") and not c["accept_rated"]:
            return "casual"
        if not ch.get("rated") and not c["accept_casual"]:
            return "rated"
        is_bot = (ch.get("challenger") or {}).get("title") == "BOT"
        if is_bot and not c["accept_bots"]:
            return "noBot"
        if not is_bot and not c["accept_humans"]:
            return "onlyBot"
        with self.lock:
            now = time.time()
            for cid in [k for k, v in self.pending.items() if now - v > 60]:
                del self.pending[cid]
            if not self.accepting or len(self.active) + len(self.pending) >= c["max_games"]:
                return "later"
        return None

    def on_challenge(self, ch):
        challenger = (ch.get("challenger") or {})
        if challenger.get("id") == self.me:
            return  # a challenge we sent
        reason = self.decline_reason(ch)
        who = f"{challenger.get('name')} ({challenger.get('rating')})"
        tc = ch.get("timeControl", {})
        desc = f"{who} {'unlimited' if tc.get('type') == 'unlimited' else str(tc.get('limit', '?')) + '+' + str(tc.get('increment', '?'))} " \
               f"{'rated' if ch.get('rated') else 'casual'}"
        if reason:
            log(f"declined {desc}: {reason}")
            self.api.decline(ch["id"], reason)
        elif self.api.accept(ch["id"]):
            log(f"accepted {desc}")
            with self.lock:
                self.pending[ch["id"]] = time.time()

    # ---------- events ----------
    def event_loop(self):
        while True:
            try:
                for ev in self.api.stream("/api/stream/event"):
                    t = ev.get("type")
                    if t == "challenge":
                        self.on_challenge(ev["challenge"])
                    elif t in ("challengeCanceled", "challengeDeclined"):
                        with self.lock:
                            self.pending.pop(ev["challenge"]["id"], None)
                    elif t == "gameStart":
                        g = ev["game"]
                        gid = g.get("gameId") or g.get("id")
                        with self.lock:
                            self.pending.pop(gid, None)
                            if gid in self.active:
                                continue
                            self.active.add(gid)
                        threading.Thread(target=self.play_game, args=(gid,), daemon=True).start()
            except Exception as e:
                log(f"event stream error: {e}")
            log("event stream closed, reconnecting in 5s")
            time.sleep(5)

    # ---------- games ----------
    def play_game(self, gid):
        color, opp, opp_rating, sent_ply, finished = None, "?", None, -1, False
        vetoes, book_moves, last_opening = 0, 0, None
        failures = 0
        log(f"[{gid}] game started")
        try:
            while not finished and failures < 30:
                try:
                    for ev in self.api.stream(f"/api/bot/game/stream/{gid}"):
                        failures = 0
                        t = ev.get("type")
                        if t == "gameFull":
                            white, black = ev.get("white", {}), ev.get("black", {})
                            color = chess.WHITE if white.get("id") == self.me else chess.BLACK
                            other = black if color == chess.WHITE else white
                            opp = other.get("name") or f"AI level {other.get('aiLevel')}"
                            opp_rating = other.get("rating")
                            fen = ev.get("initialFen", "startpos")
                            if fen not in ("startpos", STANDARD_FEN):
                                log(f"[{gid}] non-standard start position, resigning")
                                self.api._req("POST", f"/api/bot/game/{gid}/resign")
                                return
                            state = ev["state"]
                        elif t == "gameState" and color is not None:
                            state = ev
                        else:
                            continue
                        if state["status"] not in ("created", "started"):
                            self.finish(gid, state, color, opp, opp_rating, vetoes,
                                        last_opening, book_moves)
                            finished = True
                            break
                        moves = state["moves"].split()
                        board = chess.Board()
                        for m in moves:
                            board.push_uci(m)
                        if board.turn == color and len(moves) != sent_ply and not board.is_game_over():
                            t0 = time.time()
                            move, note, opening, top_prob = self.pick_move(gid, board, state, color,
                                                                           opp_rating)
                            mine = "w" if color == chess.WHITE else "b"
                            think = think_seconds(state.get(f"{mine}time"), state.get(f"{mine}inc"),
                                                  len(moves), top_prob, board.legal_moves.count(),
                                                  opening is not None, self.cfg.get("timing", {}))
                            wait = think - (time.time() - t0)
                            if wait > 0:
                                time.sleep(wait)
                            if note:
                                vetoes += 1
                            if opening:
                                book_moves += 1
                                last_opening = opening
                            if self.api.move(gid, move.uci()):
                                sent_ply = len(moves)
                                log(f"[{gid}] {board.fullmove_number}{'.' if color else '...'}"
                                    f"{board.san(move)} ({time.time() - t0:.2f}s)"
                                    + (f"  GUARD: {note}" if note else "")
                                    + (f"  [book: {opening}]" if opening else ""))
                except Exception as e:
                    failures += 1
                    log(f"[{gid}] stream error: {e}")
                if not finished:
                    failures += 1
                    time.sleep(min(10, 1 + failures))
        finally:
            with self.lock:
                self.active.discard(gid)
            self.book.end(gid)
            log(f"[{gid}] game thread ended")

    def pick_move(self, gid, board, state, color, opp_rating):
        """Return (move, guard note or None, opening name if from the book, Maia's top probability)."""
        move, opening = self.book.pick(gid, board, color, self.cfg)
        if move:
            return move, None, opening, 1.0
        mcfg, gcfg, scfg = self.cfg["maia"], self.cfg.get("guard", {}), self.cfg.get("style", {})
        move, top = None, []
        if scfg.get("enabled"):
            move, top, _ = self.maia.choose_styled(board, mcfg, scfg, opp_rating)
        if move is None:
            move, top, _ = self.maia.choose(board, mcfg, opp_rating)
        top_prob = max((c["policy"] for c in top), default=0.5)
        clock = state.get("wtime" if color == chess.WHITE else "btime")
        low_clock = clock is not None and clock < gcfg.get("min_clock_seconds", 15) * 1000
        if self.guard is None or low_clock:
            return move, None, None, top_prob
        notes = []
        acfg = self.cfg.get("assist", {})
        if acfg.get("enabled"):
            if len(top) < 2:
                top = self.maia.choose(board, mcfg, opp_rating, candidates=8)[1]
            move, n = self.guard.assist(board, move, top, acfg)
            if n:
                notes.append(n)
        if gcfg.get("enabled"):
            move, n = self.guard.check(board, move, gcfg,
                                       lambda: self.maia.choose(board, mcfg, opp_rating, candidates=8)[1])
            if n:
                notes.append(n)
        return move, ("; ".join(notes) or None), None, top_prob

    def finish(self, gid, state, color, opp, opp_rating, vetoes=0, opening=None, book_moves=0):
        winner = state.get("winner")
        if winner is None:
            result = "draw" if state["status"] in ("draw", "stalemate") else state["status"]
        else:
            mine = "white" if color == chess.WHITE else "black"
            result = "win" if winner == mine else "loss"
        self.results.append((gid, opp, opp_rating, result, state["status"], vetoes, opening, book_moves))
        log(f"[{gid}] finished vs {opp} ({opp_rating}): {result} ({state['status']}), "
            f"opening: {opening or '-'} ({book_moves} book moves), guard vetoes: {vetoes} "
            f"https://lichess.org/{gid}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--check", action="store_true", help="check token and Maia, do not go online")
    args = p.parse_args()

    settings = Settings()
    cfg = settings.cfg
    api = Lichess(os.environ["LICHESS_TOKEN"])
    acc = api.account()
    me, games_played = acc["id"], acc.get("count", {}).get("all", 0)
    is_bot = acc.get("title") == "BOT"
    log(f"Lichess account: {acc['username']} | BOT: {is_bot} | games played: {games_played}")

    if not is_bot and not args.check:
        if games_played > 0:
            raise SystemExit("This account has already played games, so Lichess will not make it "
                             "a BOT account. Create a fresh account and a new token.")
        ok, msg = api.upgrade_to_bot()
        log(f"Upgrade to BOT account: {'done' if ok else 'FAILED ' + msg}")
        if not ok:
            raise SystemExit(1)

    maia = Maia(cfg["maia"]["model"])
    guard = Guard()
    try:
        run(settings, api, maia, guard, acc, me, args)
    finally:
        guard.close()


def run(settings, api, maia, guard, acc, me, args):
    cfg = settings.cfg
    mv, _, secs = maia.choose(chess.Board(), cfg["maia"])
    log(f"Maia-3 ready ({cfg['maia']['model']}, elo {cfg['maia']['elo']}, "
        f"temp {cfg['maia']['temperature']}, top_p {cfg['maia']['top_p']}): "
        f"first move {mv} in {secs:.2f}s")
    if args.check:
        log("Check finished. Everything works; the bot did not go online.")
        return

    log(f"Blunder guard: {'ON' if cfg.get('guard', {}).get('enabled') else 'OFF'}")
    st = cfg.get("style", {})
    log(f"Style: {st.get('name', 'positional') + ' ' + str(st.get('strength')) if st.get('enabled') else 'OFF'}")
    book = Book()
    log(f"Opening book: {book.loaded()} positions in {len(book.openings)} openings")
    bot = Bot(settings, api, maia, guard, me, book)
    settings.start(log)
    threading.Thread(target=bot.event_loop, daemon=True).start()
    hours = min(args.hours, 5.6)
    end = time.time() + hours * 3600
    stop_accept = end - settings.cfg["session"]["stop_accepting_minutes_before_end"] * 60
    log(f"Online as https://lichess.org/@/{acc['username']} for {hours:g} h. Waiting for challenges...")

    while True:
        time.sleep(5)
        now = time.time()
        if bot.accepting and now > stop_accept:
            bot.accepting = False
            log("Session ending soon: no longer accepting challenges.")
        with bot.lock:
            busy = bool(bot.active)
        if (not bot.accepting and not busy) or now > end - 60:
            break

    w = sum(r[3] == "win" for r in bot.results)
    d = sum(r[3] == "draw" for r in bot.results)
    l = sum(r[3] == "loss" for r in bot.results)
    log(f"Session over. Games: {len(bot.results)} | W {w} D {d} L {l}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"## Bot session\nGames: {len(bot.results)} | W {w} D {d} L {l}\n\n")
            for gid, opp, rating, result, status, vetoes, opening, bm in bot.results:
                f.write(f"- [{result}] vs {opp} ({rating}), {status}, {opening or 'no book'} "
                        f"({bm} book moves), guard vetoes {vetoes}: https://lichess.org/{gid}\n")


if __name__ == "__main__":
    main()
