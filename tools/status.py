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
r = requests.get(f"https://lichess.org/api/games/user/{me}", headers={**h, "Accept": "application/x-chess-pgn"},
                 params={"max": 8}, timeout=60)
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
