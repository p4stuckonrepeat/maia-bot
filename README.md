# maia-bot

A human-like Lichess bot powered by [Maia-3](https://github.com/CSSLab/maia3) (79M model).

## Start / stop
- **Start:** Actions → *Run bot* → Run workflow (mode `play`, hours up to 5.5).
- **Stop:** open the running workflow → *Cancel workflow run*.
- **Test without going online:** same, with mode `check`.

## Settings
Edit `config.yml` (Elo, temperature, accepted time controls, rated/casual, bots/humans).

## Analysis API
Maia's likely moves for any position, each with a Stockfish eval (for a userscript or shortcut).
- **Setup once:** Settings → Secrets → Actions → new secret `ANALYZE_KEY` (16+ random characters).
- **Start:** Actions → *Analysis API* → Run workflow. The URL (`https://….trycloudflare.com/analyze`) is in the run summary; it changes every run.
- **Call:** `GET /analyze?fen=<FEN>` with header `X-API-Key: <key>` (or `&key=<key>`). Optional `depth` (1–18, default 12) and `n` (1–8 moves, default 5). Evals are from White's side.
- **Local:** `ANALYZE_KEY=<key> python -m bot.api` → `http://localhost:8000/analyze`.

## Status
- [x] Maia-3 plays challenges on Lichess
- [ ] Opening repertoire
- [ ] Stockfish blunder guard
- [ ] Dashboard and accuracy statistics

License: AGPL-3.0 (same as Maia-3).
