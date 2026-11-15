"""Quick status: current Lichess ratings, ongoing games, last few games."""
import io
import os
import chess.pgn
import requests

h = {"Authorization": f"Bearer {os.environ['LICHESS_TOKEN']}"}
acc = requests.get("https://lichess.org/api/account", headers=h, timeout=20).json()
me = acc["id"]
lines = [f"## {acc['username']}", "Ratings: " + ", ".join(
    f"{k} {v.get('rating')} ({v.get('games')} games{', provisional' if v.get('prov') else ''})"
    for k, v in acc.get("perfs", {}).items() if k in ("bullet", "blitz", "rapid", "classical"))]
playing = requests.get("https://lichess.org/api/account/playing", headers=h, timeout=20).json().get("nowPlaying", [])
lines.append(f"Playing now: {len(playing)}")
for g in playing:
    o = g.get("opponent", {})
    lines.append(f"- vs {o.get('username')} ({o.get('rating')}) https://lichess.org/{g.get('gameId')}")
ch = requests.get("https://lichess.org/api/challenge", headers=h, timeout=20).json()
for c in ch.get("out", []):
    lines.append(f"Outgoing challenge: {c.get('destUser', {}).get('name')} {c.get('timeControl', {}).get('show')} "
                 f"{'rated' if c.get('rated') else 'casual'} status {c.get('status')}")
bots = [__import__('json').loads(l) for l in requests.get("https://lichess.org/api/bot/online", headers=h,
        params={"nb": 100}, timeout=20).text.splitlines() if l.strip()]
inr = [b for b in bots if 1900 <= ((b.get("perfs", {}).get("blitz") or {}).get("rating") or 0) <= 2700]
lines.append(f"Online bots: {len(bots)}, with blitz 1900-2700: {len(inr)}")
r = requests.get(f"https://lichess.org/api/games/user/{me}", headers={**h, "Accept": "application/x-chess-pgn"},
                 params={"max": int(os.environ.get("STATUS_GAMES", "20"))}, timeout=60)
pgn = io.StringIO(r.text)
lines.append("Last games:")
while True:
    g = chess.pgn.read_game(pgn)
    if g is None:
        break
    hd = g.headers
    white = hd["White"].lower() == me
    opp = hd["Black"] if white else hd["White"]
    oelo = hd.get("BlackElo" if white else "WhiteElo")
    res = hd["Result"]
    outcome = "draw" if res == "1/2-1/2" else ("win" if (res == "1-0") == white else "loss")
    lines.append(f"- {hd.get('UTCTime')} {hd.get('Event')} {hd.get('TimeControl')} vs {opp} ({oelo}): {outcome} {hd.get('Site')}")
text = "\n".join(lines)
print(text)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")
