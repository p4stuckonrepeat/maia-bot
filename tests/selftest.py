"""Offline self-test: challenge filter + game loop against a fake Lichess, real Maia."""
import faulthandler, sys
faulthandler.dump_traceback_later(90, exit=True)
import time
import chess
from bot.main import Bot
from bot.engine import Guard, Maia
from bot.settings import Settings


class FakeApi:
    def __init__(self):
        self.sent, self.declined, self.accepted = [], [], []
        self.moves = []

    def accept(self, cid):
        self.accepted.append(cid); return True

    def decline(self, cid, reason="generic"):
        self.declined.append((cid, reason))

    def move(self, gid, uci):
        self.sent.append(uci); self.moves.append(uci); return True

    def stream(self, path):
        # We are white vs "opp". Opponent replies with its first legal move each turn.
        yield {"type": "gameFull", "white": {"id": "me"}, "black": {"id": "opp", "name": "Opp", "rating": 2000},
               "initialFen": "startpos", "state": {"type": "gameState", "moves": "", "status": "started"}}
        for _ in range(6):
            b = chess.Board()
            for m in self.moves:
                b.push_uci(m)
            reply = next(iter(b.legal_moves)).uci()
            self.moves.append(reply)
            yield {"type": "gameState", "moves": " ".join(self.moves), "status": "started"}
        yield {"type": "gameState", "moves": " ".join(self.moves), "status": "resign", "winner": "white"}


settings = Settings()
cfg = settings.cfg
api = FakeApi()
print("starting Maia", flush=True)
maia = Maia(cfg["maia"]["model"])
print("Maia started, first move:", maia.choose(chess.Board(), cfg["maia"])[0], flush=True)
guard = Guard()

# Guard test 1: 3...Nf6?? allows Qxf7# -> must be vetoed and replaced by a Maia alternative.
b = chess.Board()
for m in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4"]:
    b.push_uci(m)
blunder = chess.Move.from_uci("g8f6")
fn = lambda: maia.choose(b, cfg["maia"], 2000, candidates=8)[1]
move, note = guard.check(b, blunder, cfg["guard"], fn)
print("guard test 1:", b.san(move), "|", note, flush=True)
assert move != blunder and note
# Guard test 2: a normal move (3...g6) must pass untouched.
ok = chess.Move.from_uci("g7g6")
move, note = guard.check(b, ok, cfg["guard"], fn)
print("guard test 2:", b.san(move), "|", note, flush=True)
assert move == ok and note is None
assert b.fen().startswith("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3")   # board left unchanged

# Book test: main openings (weight > 0) win over fallback (weight 0) openings.
from bot.book import Book, key
bk = Book(path="/nonexistent")
start = key(chess.Board())
after_e4 = chess.Board(); after_e4.push_uci("e2e4")
after_d4 = chess.Board(); after_d4.push_uci("d2d4")
bk.openings = {
    "ruy_lopez": {"color": "white", "positions": {start: [["e2e4", 1]]}},
    "english": {"color": "white", "positions": {start: [["c2c4", 1]]}},
    "sicilian": {"color": "black", "positions": {key(after_e4): [["c7c5", 1]]}},
    "caro_kann": {"color": "black", "positions": {key(after_e4): [["c7c6", 1]]}},
    "kings_indian": {"color": "black", "positions": {key(after_d4): [["g8f6", 1]]}},
    "french": {"color": "black", "positions": {key(after_d4): [["e7e6", 1]]}},  # fake, weight 0
}
bcfg = {"openings": {"white": {"ruy_lopez": {"weight": 100}, "english": {"weight": 0}},
                     "black": {"sicilian": {"weight": 100}, "caro_kann": {"weight": 0},
                               "kings_indian": {"weight": 100}, "french": {"weight": 0}}}}
for i in range(50):
    assert bk.pick(f"w{i}", chess.Board(), chess.WHITE, bcfg)[0].uci() == "e2e4"
    assert bk.pick(f"b{i}", after_e4, chess.BLACK, bcfg)[0].uci() == "c7c5"
    assert bk.pick(f"d{i}", after_d4, chess.BLACK, bcfg)[0].uci() == "g8f6"
bcfg["openings"]["black"]["kings_indian"]["weight"] = 0
bcfg["openings"]["black"].pop("french")
assert bk.pick("x", after_d4, chess.BLACK, bcfg)[0].uci() == "g8f6"   # fallback when nothing else fits
after_c4 = chess.Board(); after_c4.push_uci("c2c4")
assert bk.pick("y", after_c4, chess.BLACK, bcfg) == (None, None)       # out of book -> Maia
print("book tests ok", flush=True)

# Style tests: castling is preferred, weakening pawns / hanging pieces are not.
from bot.style import positional_score
b = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
assert positional_score(b, chess.Move.from_uci("e1g1")) > 0
b1 = chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 1 6")
assert positional_score(b1, chess.Move.from_uci("g2g4")) < positional_score(b1, chess.Move.from_uci("c2c3"))
b2 = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
assert positional_score(b2, chess.Move.from_uci("f3g5")) < 0       # hangs the knight to Qd8
styled = [maia.choose_styled(chess.Board(), cfg["maia"], cfg["style"], 2000)[0] for _ in range(20)]
assert all(m in chess.Board().legal_moves for m in styled)
print("style tests ok:", sorted(set(m.uci() for m in styled)), flush=True)

# Timing tests (3+0 game): obvious < hard on average, book fast, time trouble fast.
from bot.timing import think_seconds
import random as _r, statistics as _st
T = cfg["timing"]
obvious = _st.mean(think_seconds(150000, 0, 30, 0.95, 20, False, T, _r) for _ in range(300))
hard = _st.mean(think_seconds(150000, 0, 30, 0.25, 20, False, T, _r) for _ in range(300))
book = max(think_seconds(180000, 0, 2, 1.0, 20, True, T, _r) for _ in range(300))
panic = max(think_seconds(8000, 0, 60, 0.2, 20, False, T, _r) for _ in range(300))
print(f"timing: obvious {obvious:.2f}s, hard {hard:.2f}s, book max {book:.2f}s, <10s max {panic:.2f}s", flush=True)
assert obvious < hard and book <= 2.5 and panic <= 0.96
# clock-independent: same difficulty, very different clocks -> similar time
a = _st.mean(think_seconds(600000, 0, 30, 0.6, 20, False, T, _r) for _ in range(300))
b = _st.mean(think_seconds(120000, 0, 30, 0.6, 20, False, T, _r) for _ in range(300))
print(f'timing normal move: 10 min left {a:.2f}s, 2 min left {b:.2f}s', flush=True)
assert abs(a - b) < 1.0

bot = Bot(settings, api, maia, guard, "me", Book(path="/nonexistent"))

def ch(cid, limit=180, inc=0, variant="standard", rated=True, title=None):
    return {"id": cid, "challenger": {"id": "x", "name": "X", "rating": 1500, "title": title},
            "variant": {"key": variant}, "timeControl": {"type": "clock", "limit": limit, "increment": inc},
            "rated": rated}

assert bot.decline_reason(ch("a", limit=60)) == "tooFast"
assert bot.decline_reason(ch("b", limit=1800)) == "tooSlow"
assert bot.decline_reason(ch("c", variant="chess960")) == "variant"
assert bot.decline_reason(ch("d")) is None
bot.on_challenge(ch("d")); assert api.accepted == ["d"]
assert bot.decline_reason(ch("e")) == "later"   # max_games = 1 while d is pending
bot.pending.clear()

print("filter tests ok", flush=True)
bot.active.add("g1")
bot.play_game("g1")
assert len(api.sent) == 7, api.sent   # first move + one reply to each of 6 opponent moves
assert bot.results[-1][3] == "win"
assert "g1" not in bot.active
b = chess.Board()
for m in api.moves:
    b.push_uci(m)   # every move legal
print("SELFTEST PASSED:", chess.Board().variation_san(b.move_stack))
guard.close()
faulthandler.cancel_dump_traceback_later()
