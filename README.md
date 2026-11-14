# maia-bot

A human-like Lichess bot powered by [Maia-3](https://github.com/CSSLab/maia3) (79M model).

## Start / stop
- **Start:** Actions → *Run bot* → Run workflow (mode `play`, hours up to 5.5).
- **Stop:** open the running workflow → *Cancel workflow run*.
- **Test without going online:** same, with mode `check`.

## Settings
Edit `config.yml` (Elo, temperature, accepted time controls, rated/casual, bots/humans).

## Status
- [x] Maia-3 plays challenges on Lichess
- [ ] Opening repertoire
- [ ] Stockfish blunder guard
- [ ] Dashboard and accuracy statistics

License: AGPL-3.0 (same as Maia-3).
