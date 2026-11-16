"""Tal's tactics: which sacrifices and tactical motifs he used, from the study data (tal/study.py).

For every studied Tal move: is it a sacrifice (Stockfish-checked), on which square, with which piece,
with check, next to the enemy king; and tactical motifs (fork, discovered attack, attack on the king's
shelter). Compared with the Tal model's first choice. Writes tal/motifs-report.md and
tal/tal-motifs.json (Tal's favourite sacrifice motifs, used by the bot's sacrifice hunter).
"""
import collections
import gzip
import json
import os

import chess

import tal.study as study
from bot.style import VALUE
from tal.antiblunder import rebuild
from tal.train import download_games, log, tal_games

NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
         chess.QUEEN: "queen", chess.KING: "king"}


def rel(sq, color):
    return chess.square_name(sq if color == chess.WHITE else chess.square_mirror(sq))


def motif_key(board, move):
    """Sacrifice motif as Tal would name it: piece + target square from his side, e.g. 'bishop takes h7 +'."""
    p = board.piece_at(move.from_square)
    cap = board.is_capture(move)
    chk = board.gives_check(move)
    return f"{NAMES[p.piece_type]} {'takes' if cap else 'to'} {rel(move.to_square, board.turn)}{' +' if chk else ''}"


def tactics(board, move):
    """Tactical motifs of a move."""
    me, out = board.turn, set()
    p = board.piece_at(move.from_square)
    b = board.copy(stack=False)
    b.push(move)
    # fork: the moved piece attacks two enemy pieces worth more than it (or undefended), king included
    tgt = []
    for s in b.attacks(move.to_square):
        t = b.piece_at(s)
        if t and t.color != me and (t.piece_type == chess.KING or VALUE[t.piece_type] > VALUE[p.piece_type]
                                    or not b.attackers(not me, s)):
            tgt.append(t.piece_type)
    if len(tgt) >= 2:
        out.add("fork")
    # discovered attack / check: another of our pieces now attacks something it did not before
    for sq in b.pieces(chess.BISHOP, me) | b.pieces(chess.ROOK, me) | b.pieces(chess.QUEEN, me):
        if sq == move.to_square:
            continue
        before = board.attacks(sq)
        new = b.attacks(sq) & ~before
        for s in new:
            t = b.piece_at(s)
            if t and t.color != me and (t.piece_type == chess.KING or VALUE[t.piece_type] >= 3):
                out.add("discovered check" if t.piece_type == chess.KING else "discovered attack")
    # attack on the king's shelter: captures or hits a pawn in front of the enemy king
    k = b.king(not me)
    if k is not None:
        shelter = chess.SquareSet(chess.BB_KING_ATTACKS[k]) & board.pieces(chess.PAWN, not me)
        if move.to_square in shelter:
            out.add("hits the king's pawn shelter")
    if b.is_check():
        out.add("check")
    return out


def main():
    rows = []
    with gzip.open(os.environ.get("STUDY_FILE", "study.jsonl.gz"), "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    by_shard = {}
    for r in rows:
        by_shard.setdefault(r["shard"], []).append(r)
    data = rebuild(by_shard, tal_games(download_games()))
    F = {n: i for i, n in enumerate(study.FEATURES)}
    sac_tal, sac_kind = collections.Counter(), collections.Counter()
    sac_model = collections.Counter()
    tac_tal, tac_model = collections.Counter(), collections.Counter()
    phase_sac = collections.Counter()
    n = 0
    for r, g, ply in data:
        b = chess.Board()
        for m in g["moves"][:ply]:
            b.push_uci(m)
        n += 1
        tal_c = next(c for c in r["cands"] if c["m"] == r["tal"])
        top = max(r["cands"][:r["n_top"]], key=lambda c: c["p"])
        for c, sac_ctr, tac_ctr in ((tal_c, sac_tal, tac_tal), (top, sac_model, tac_model)):
            mv = chess.Move.from_uci(c["m"])
            for t in tactics(b, mv):
                tac_ctr[t] += 1
            kinds = [k for k in ("sound_sac", "spec_sac", "unsound_sac") if c["f"][F[k]] > 0]
            if kinds:
                sac_ctr[motif_key(b, mv)] += 1
                if c is tal_c:
                    sac_kind[kinds[0]] += 1
                    phase_sac[r["phase"]] += 1
    L = [f"# Tal's tactics ({n} positions from his games, Stockfish-checked)", "",
         "## Tactical motifs: Tal vs the Tal model's first choice", "", "| Motif | Tal | Model |", "|---|---|---|"]
    for t in sorted(set(tac_tal) | set(tac_model), key=lambda t: -tac_tal[t]):
        L.append(f"| {t} | {tac_tal[t] / n:.2%} | {tac_model[t] / n:.2%} |")
    L += ["", "## Tal's favourite sacrifices (piece + square, seen from his side)", "",
          "| Sacrifice | Tal | Model's first choice |", "|---|---|---|"]
    fav = [k for k, v in sac_tal.most_common() if v >= int(os.environ.get("MOTIF_MIN", "3"))]
    for k in fav[:30]:
        L.append(f"| {k} | {sac_tal[k]} | {sac_model[k]} |")
    L += ["", f"Sound / speculative / unsound: {sac_kind['sound_sac']} / {sac_kind['spec_sac']} / {sac_kind['unsound_sac']}",
          f"By phase: " + ", ".join(f"{k} {v}" for k, v in phase_sac.most_common())]
    json.dump({"favourite_sacrifices": {k: sac_tal[k] for k in fav}, "positions": n},
              open("tal/tal-motifs.json", "w"), indent=1)
    text = "\n".join(L)
    open("tal/motifs-report.md", "w").write(text + "\n")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")
    log("done")


if __name__ == "__main__":
    main()
