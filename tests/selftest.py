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

bot = Bot(settings, api, maia, guard, "me")

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
