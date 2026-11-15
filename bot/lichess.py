"""Minimal client for the official Lichess Bot API."""
import json
import time

import requests

BASE = "https://lichess.org"


class Lichess:
    def __init__(self, token):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "User-Agent": "maia3-bot (github.com/p4stuckonrepeat/maia-bot)"})

    def _req(self, method, path, retries=5, **kw):
        for attempt in range(retries):
            try:
                r = self.s.request(method, BASE + path, timeout=20, **kw)
            except requests.RequestException as e:
                print(f"[api] {method} {path} network error: {e}", flush=True)
                time.sleep(2 + attempt * 3)
                continue
            if r.status_code == 429:  # Lichess asks clients to wait a full minute
                print("[api] rate limited, waiting 60s", flush=True)
                time.sleep(60)
                continue
            if r.status_code >= 500:
                time.sleep(2 + attempt * 3)
                continue
            return r
        return None

    def account(self):
        r = self._req("GET", "/api/account")
        if r is None or r.status_code != 200:
            raise SystemExit(f"Token check failed: {r.status_code if r else 'no response'} "
                             f"{r.text[:200] if r else ''}")
        return r.json()

    def upgrade_to_bot(self):
        r = self._req("POST", "/api/bot/account/upgrade")
        return r is not None and r.status_code == 200, (r.text[:300] if r else "no response")

    def accept(self, cid):
        r = self._req("POST", f"/api/challenge/{cid}/accept")
        return r is not None and r.status_code == 200

    def decline(self, cid, reason="generic"):
        self._req("POST", f"/api/challenge/{cid}/decline", data={"reason": reason})

    def move(self, gid, uci):
        r = self._req("POST", f"/api/bot/game/{gid}/move/{uci}", retries=3)
        if r is None or r.status_code != 200:
            print(f"[game {gid}] move {uci} rejected: {r.status_code if r else 'none'} "
                  f"{r.text[:200] if r else ''}", flush=True)
            return False
        return True

    def online_bots(self, nb=100):
        r = self._req("GET", "/api/bot/online", params={"nb": nb})
        if r is None or r.status_code != 200:
            return []
        import json as _j
        return [_j.loads(l) for l in r.text.splitlines() if l.strip()]

    def challenge(self, user, limit, inc, rated=True):
        r = self._req("POST", f"/api/challenge/{user}", retries=2,
                      data={"rated": str(rated).lower(), "clock.limit": limit, "clock.increment": inc,
                            "color": "random", "variant": "standard"})
        if r is None or r.status_code != 200:
            return None, (r.text[:200] if r is not None else "no response")
        d = r.json()
        return (d.get("challenge") or d).get("id"), None

    def cancel_challenge(self, cid):
        self._req("POST", f"/api/challenge/{cid}/cancel", retries=1)

    def stream(self, path):
        """Yield JSON events from an ndjson stream; returns when the stream closes."""
        with self.s.get(BASE + path, stream=True, timeout=(15, 75)) as r:
            if r.status_code == 429:
                print("[api] rate limited on stream, waiting 60s", flush=True)
                time.sleep(60)
                return
            r.raise_for_status()
            for line in r.iter_lines():
                if line:  # empty lines are keep-alives
                    yield json.loads(line)
