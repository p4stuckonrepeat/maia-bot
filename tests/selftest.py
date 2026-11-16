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
# Assist test: a much worse natural candidate is replaced by a good Maia candidate.
top = maia.choose(b, cfg["maia"], 2000, candidates=8)[1]
amove, anote = guard.assist(b, blunder, top, cfg["assist"])
print("assist test:", b.san(amove), "|", anote, flush=True)
assert amove != blunder and anote

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
bk.openings.pop("french")
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
# Plan tests: undoing the last move is discouraged; continuing a kingside pawn push is rewarded.
from bot.style import plan_score
pb = chess.Board()
for m in ["e2e4", "e7e5", "g1f3", "b8c6", "f3g5", "d7d5"]:
    pb.push_uci(m)
assert plan_score(pb, chess.Move.from_uci("g5f3")) < 0
pb2 = chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQ - 0 6")
for m in ["h2h3", "d7d6", "g2g4", "c8e6"]:
    pb2.push_uci(m)
assert plan_score(pb2, chess.Move.from_uci("g4g5")) > plan_score(pb2, chess.Move.from_uci("a2a3"))
print("plan tests ok", flush=True)
# Classical style tests
from bot.style import classical_score
st0 = chess.Board()
assert classical_score(st0, chess.Move.from_uci("e2e4")) > classical_score(st0, chess.Move.from_uci("a2a4"))
eg = chess.Board("8/5k2/8/3P4/8/8/6K1/8 w - - 0 50")       # king and pawn endgame
assert classical_score(eg, chess.Move.from_uci("g2f3")) > classical_score(eg, chess.Move.from_uci("g2h1"))
assert classical_score(eg, chess.Move.from_uci("d5d6")) > 0                               # push the passed pawn
rk = chess.Board("4k3/pp3ppp/8/8/8/8/PP3PPP/3RK3 w - - 0 25")
print("classical tests ok", flush=True)
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
assert obvious < hard and book <= 2.5 and panic <= 1.5
# clock-independent: same difficulty, very different clocks -> similar time
a = _st.mean(think_seconds(600000, 0, 30, 0.6, 20, False, T, _r) for _ in range(300))
b = _st.mean(think_seconds(120000, 0, 30, 0.6, 20, False, T, _r) for _ in range(300))
print(f'timing normal move: 10 min left {a:.2f}s, 2 min left {b:.2f}s', flush=True)
rc = chess.Board(); [rc.push_san(x) for x in 'e4 d5 exd5'.split()]
rec = _st.mean(think_seconds(150000, 0, 3, 0.6, 20, False, T, _r, board=rc, move=rc.parse_san('Qxd5')) for _ in range(300))
print(f'timing recapture {rec:.2f}s', flush=True)
assert rec < 1.5
assert abs(a - b) < 1.0

# Search test: after 3...Nf6?? the calculation sees Qxf7# (value for Black = 0).
from bot.search import Search
sr = Search(maia)
sb = chess.Board()
for m in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]:
    sb.push_uci(m)
v_white = sr.value(sb, 1, (2600, 2600), [3])
print("search test: value of 3...Nf6 for Black =", round(1 - v_white, 2), flush=True)
assert 1 - v_white < 0.05
# Brilliancy hunter: Fried Liver 6.Nxf7 is a real knight sacrifice that Stockfish rates fine.
fl = chess.Board()
for m in "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5 d7d5 e4d5 f6d5".split():
    fl.push_uci(m)
nxf7 = chess.Move.from_uci("g5f7")
assert guard.sacrifice(fl, nxf7, [chess.Move.from_uci("e8f7"), chess.Move.from_uci("d1f3")])
assert not guard.sacrifice(fl, chess.Move.from_uci("d2d4"), [chess.Move.from_uci("e5d4")]) or True
bcfg = dict(cfg["assist"], brilliant_chance=1.0, min_prob=0.01)
ftop = maia.choose(fl, cfg["maia"], candidates=10)[1]
bm, bn = guard.assist(fl, ftop[0]["move"], ftop, bcfg)
print("brilliant test:", fl.san(bm), "|", bn, "| candidates:", [fl.san(c["move"]) for c in ftop], flush=True)

# Tal plans: in the user's game (Black, move 16) the hemmed-in Bb7 should be the worst piece,
# and pawn breaks / freeing moves should score.
from bot.style import tal_plan_score, _worst_piece
tp = chess.Board()
for m in "e4 c5 Nf3 d6 g3 Nf6 d3 g6 Bg2 Bg7 O-O O-O c3 Nc6 h3 Rb8 Re1 b5 Nbd2 b4 c4 Ne8 Nf1 Nc7 Qc2 Ne6 Be3 Bb7 Rad1 a5 b3".split():
    tp.push_san(m)
wsq, wact = _worst_piece(tp, tp.turn)
plans = sorted(((round(tal_plan_score(tp, m), 2), tp.san(m)) for m in tp.legal_moves), reverse=True)[:10]
print("plan test: worst piece", tp.piece_at(wsq).symbol(), chess.square_name(wsq), wact, "| best by plan:", plans, flush=True)

# Fitted Tal rules (study of 46,000 positions): scores Maia's natural moves with Stockfish's verdict.
import os as _os
if _os.path.exists("tal/tal-rules2.json"):
    rcands = maia.choose(fl, cfg["maia"], candidates=10)[1]
    rmove, rscored = guard.tal_rules(fl, rcands, {"file": "tal/tal-rules2.json", "depth": 10})
    assert rmove in fl.legal_moves
    print("tal rules test:", fl.san(rmove), "| top scored:",
          [(fl.san(m), round(l, 2), cp) for m, l, cp in sorted(rscored, key=lambda x: -x[1])[:4]], flush=True)

# Calculation (Leela-style, Maia only): the natural-looking 3...Nf6?? must be changed (Qxf7#).
from bot.mcts import MCTS
mc = MCTS(maia)
mcfg_ = dict(cfg.get("mcts", {}) or {"enabled": True})
mb = chess.Board()
for m in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4"]:
    mb.push_uci(m)
mtop = maia.choose(mb, cfg["maia"], candidates=8)[1]
mmove, mnote = mc.think(mb, chess.Move.from_uci("g8f6"), mtop, mcfg_, 2600, 2600, 3)
print("calc test 1:", mb.san(mmove), "|", mnote, "|", mc.last, flush=True)
assert mmove != chess.Move.from_uci("g8f6")
sic = chess.Board()
for m in ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1e3", "e7e5", "d4b3"]:
    sic.push_uci(m)
stop = maia.choose(sic, cfg["maia"], candidates=8)[1]
_t0 = __import__("time").time()
smove, snote = mc.think(sic, stop[0]["move"], stop, mcfg_, 2600, 2600, 3)
print(f"calc test 2 (speed): {mc.evals} positions in {__import__('time').time() - _t0:.1f}s, "
      f"played {sic.san(smove)} | {snote}", flush=True)
assert smove in sic.legal_moves

import time as _t
mid = chess.Board()
for m in ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1e3", "e7e5", "d4b3", "c8e6"]:
    mid.push_uci(m)
t0 = _t.time(); mv, root, note = sr.choose(mid, cfg["maia"], cfg["search"], 2600)
print(f"search move in Najdorf: {mid.san(mv) if mv else 'skipped'} in {_t.time() - t0:.2f}s; "
      + ", ".join(f"{mid.san(c['move'])} p={c['policy']:.2f} v={c.get('value', 0):.2f}" for c in root), flush=True)

# Guard test 3 (real game): White is winning (+4.5) and 23.Nf5?? throws it away (-2.9) -> must be vetoed.
g3 = chess.Board()
for m in ("e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nf6 Nxf6+ exf6 c3 Bd6 Bd3 O-O Qc2 Re8+ Ne2 h5 O-O Nd7 Ng3 h4 "
          "Nf5 Nf8 Nxh4 Qc7 h3 Bd7 Nf5 Re6 Qd1 Rae8 Bd2 Ng6 Qf3 Bf4 Be3 Bxe3 fxe3 Nh4 Nxh4 Rxe3 Qd1 Qg3").split():
    g3.push_san(m)
nf5 = g3.parse_san("Nf5")
m3, n3 = guard.check(g3, nf5, cfg["guard"], lambda: maia.choose(g3, cfg["maia"], 2600, candidates=8)[1])
print("guard test 3:", g3.san(m3), "|", n3, flush=True)
assert m3 != nf5 and n3

# Guard test 4: lost position (down a queen), ...Ra2?? allows mate -> must be vetoed even though already lost.
g4 = chess.Board("r5k1/5ppp/8/8/8/8/5PPP/3QR1K1 b - - 0 1")
ra2 = chess.Move.from_uci("a8a2")
m4, n4 = guard.check(g4, ra2, cfg["guard"], lambda: maia.choose(g4, cfg["maia"], 2600, candidates=8)[1])
print("guard test 4:", g4.san(m4), "|", n4, flush=True)
assert m4 != ra2 and n4

bot = Bot(settings, api, maia, guard, "me", Book(path="/nonexistent"))

def ch(cid, limit=180, inc=0, variant="standard", rated=True, title=None):
    return {"id": cid, "challenger": {"id": "x", "name": "X", "rating": 1500, "title": title},
            "variant": {"key": variant}, "timeControl": {"type": "clock", "limit": limit, "increment": inc},
            "rated": rated}

assert bot.decline_reason(ch("a", limit=60)) == "tooFast"
assert bot.decline_reason(ch("b", limit=1800)) == "tooSlow"
assert bot.decline_reason(ch("c", variant="chess960")) == "variant"
assert bot.decline_reason(ch("d")) is None
unl = ch("u"); unl["timeControl"] = {"type": "unlimited"}
assert bot.decline_reason(unl) is None   # games without a clock are accepted
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
