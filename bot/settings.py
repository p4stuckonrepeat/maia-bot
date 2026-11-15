"""Live settings: config.yml is re-read from GitHub every 30 s, so edits apply mid-session."""
import os
import threading
import time

import requests
import yaml


class Settings:
    def __init__(self, path=None, poll_seconds=30):
        path = path or os.environ.get("BOT_CONFIG", "config.yml")
        self.path, self.poll = path, poll_seconds
        self.cfg = yaml.safe_load(open(path))
        self.repo = os.environ.get("GITHUB_REPOSITORY")
        self.token = os.environ.get("GITHUB_TOKEN")
        self.ref = os.environ.get("GITHUB_REF_NAME", "main")

    def start(self, log):
        if not (self.repo and self.token):
            log("Live settings off (not running on GitHub Actions).")
            return
        threading.Thread(target=self._loop, args=(log,), daemon=True).start()
        log("Live settings on: edits to config.yml apply within ~30 s.")

    def _loop(self, log):
        url = f"https://api.github.com/repos/{self.repo}/contents/{self.path}?ref={self.ref}"
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github.raw"}
        while True:
            time.sleep(self.poll)
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code != 200:
                    continue
                new = yaml.safe_load(r.text)
                if not isinstance(new, dict) or new == self.cfg:
                    continue
                old_guard = self.cfg.get("guard", {}).get("enabled")
                self.cfg = new
                g = new.get("guard", {}).get("enabled")
                log(f"Settings updated from GitHub. Blunder guard: {'ON' if g else 'OFF'}"
                    + (" (changed)" if g != old_guard else ""))
            except Exception as e:
                log(f"Settings refresh failed (keeping current): {e}")
