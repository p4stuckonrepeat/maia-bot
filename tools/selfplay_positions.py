"""Positions for tactics training: the Tal model plays itself (human-like sampling), and Stockfish
judges its 8 most natural candidates in each recorded position. Stockfish is only used here, for
training labels; the bot never uses it during play.

Env: SP_POSITIONS (per machine), SP_SHARD, SP_DEPTH, TAL_CKPT. Output: selfplay-<shard>.jsonl
"""
import json
import os
import random
import time

import chess
import chess.engine


def main():
    from maia3.uci import Maia3UCIEngine, parse_args
    shard = int(os.environ.get("SP_SHARD", "0"))
    want = int(os.environ.get("SP_POSITIONS", "3000"))
    depth = int(os.environ.get("SP_DEPTH", "10"))
    rng = random.Random(1000 + shard)
    eng = Maia3UCIEngine(parse_args(["--model", "maia3-79m", "--checkpoint", os.environ.get("TAL_CKPT", "tal-79m.pt")]))
    eng.ensure_model_loaded()
    sf = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH", "/usr/games/stockfish"))
    sf.configure({"Threads": 4, "Hash": 256})
    out, n, games, t0 = open(f"selfplay-{shard}.jsonl", "w"), 0, 0, time.time()
    try:
        while n < want:
            b = chess.Board()
            games += 1
            while not b.is_game_over(claim_draw=True) and b.ply() < 140 and n < want:
                eng.self_elo = eng.oppo_elo = 2600
                eng.multipv = 8
                moves = " ".join(m.uci() for m in b.move_stack)
                eng.cmd_position("position startpos" + (f" moves {moves}" if moves else ""))
                _, top = eng.score_moves()
                if not top:
                    break
                if b.ply() >= 16 and rng.random() < 0.5:
                    cands = []
                    for c in top:
                        b.push(c["move"])
                        if b.is_checkmate():
                            cp = 10000
                        elif b.is_game_over():
                            cp = 0
                        else:
                            cp = sf.analyse(b, chess.engine.Limit(depth=depth))["score"].pov(not b.turn).score(mate_score=10000)
                        b.pop()
                        cands.append({"m": c["move"].uci(), "p": round(c["policy"], 5), "cp": cp})
                    out.write(json.dumps({"moves": [m.uci() for m in b.move_stack], "cands": cands,
                                          "best": max(c["cp"] for c in cands)}) + "\n")
                    n += 1
                    if n % 250 == 0:
                        print(f"{n}/{want} positions, {games} games, {time.time() - t0:.0f}s", flush=True)
                # human-like sampling (temperature 1 over the top candidates)
                w = [c["policy"] for c in top]
                b.push(rng.choices([c["move"] for c in top], weights=w)[0])
    finally:
        sf.quit()
        out.close()
    print(f"done: {n} positions from {games} games")


if __name__ == "__main__":
    main()
