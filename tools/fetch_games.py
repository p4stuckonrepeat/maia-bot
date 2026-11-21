"""Download a player's games on the GitHub runner (chess.com blocks some cloud GPU IPs).

  PLAYER=... python tools/fetch_games.py OUT.pgn.gz
Only the download code of tal.train is used; torch / maia3 are stubbed so they need not be installed.
"""
import gzip
import sys

from unittest import mock

for m in ["torch", "torch.nn", "torch.nn.functional", "torch.utils", "torch.utils.data",
          "maia3", "maia3.dataset", "maia3.uci", "maia3.utils"]:
    sys.modules[m] = mock.MagicMock()
sys.modules["torch"].cuda.is_available.return_value = False
sys.modules["torch"].utils.data.Dataset = object
sys.path.insert(0, ".")

from tal.train import download_games, log  # noqa: E402

text = download_games()
with gzip.open(sys.argv[1], "wt") as f:
    f.write(text)
log(f"saved {text.count('[Event ')} games to {sys.argv[1]}")
