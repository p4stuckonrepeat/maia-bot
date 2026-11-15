"""Deep Stockfish look at a position: evaluation before/after a move and the best replies."""
import os
import chess
import chess.engine

moves = os.environ["MOVES"].split()
check = int(os.environ.get("CHECK_PLY", len(moves)))  # index of the move to examine
sf = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
sf.configure({"Threads": os.cpu_count(), "Hash": 1024})
b = chess.Board()
for m in moves[:check]:
    b.push_san(m)
me = b.turn
mv = b.parse_san(moves[check])
for info in sf.analyse(b, chess.engine.Limit(depth=26), multipv=4):
    print(f"before {b.fullmove_number}: best {b.san(info['pv'][0])} {info['score'].pov(me).score(mate_score=10000)/100:+.2f} "
          f"line {b.variation_san(info['pv'][:8])}")
b.push(mv)
for info in sf.analyse(b, chess.engine.Limit(depth=26), multipv=4):
    print(f"after {moves[check]}: reply {b.san(info['pv'][0])} -> {info['score'].pov(me).score(mate_score=10000)/100:+.2f} for the bot "
          f"line {b.variation_san(info['pv'][:8])}")
sf.quit()
