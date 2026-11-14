"""Positional playing style: a consistent tie-breaker between moves Maia finds natural.

Each move gets a small style score. The bot multiplies Maia's probability by
exp(strength * score), so the style only decides between moves Maia already rates
as plausible; it never plays a move Maia considers unnatural.
"""
import chess

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
CENTER = chess.SquareSet(chess.BB_CENTER | chess.BB_C3 | chess.BB_D3 | chess.BB_E3 | chess.BB_F3
                         | chess.BB_C6 | chess.BB_D6 | chess.BB_E6 | chess.BB_F6
                         | chess.BB_C4 | chess.BB_F4 | chess.BB_C5 | chess.BB_F5)


def pawn_weaknesses(board, color):
    files = [chess.square_file(s) for s in board.pieces(chess.PAWN, color)]
    counts = {f: files.count(f) for f in set(files)}
    doubled = sum(c - 1 for c in counts.values())
    isolated = sum(c for f, c in counts.items() if (f - 1) not in counts and (f + 1) not in counts)
    return doubled + isolated


def positional_score(board, move):
    color, piece = board.turn, board.piece_type_at(move.from_square)
    score = 0.0
    captured = board.piece_type_at(move.to_square) or (chess.PAWN if board.is_en_passant(move) else None)

    if board.is_castling(move):
        score += 0.4                                   # king safety first
    if board.fullmove_number <= 8 and piece == chess.QUEEN and not captured:
        score -= 0.3                                   # no early queen adventures
    if piece in (chess.KNIGHT, chess.BISHOP):
        back = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
        if board.fullmove_number <= 12 and chess.BB_SQUARES[move.from_square] & back:
            score += 0.2                               # develop pieces
        if move.to_square in CENTER and move.from_square not in CENTER:
            score += 0.15                              # centralise

    king = board.king(color)
    if piece == chess.PAWN and king is not None and chess.square_rank(king) in (0, 7):
        kf, pf = chess.square_file(king), chess.square_file(move.from_square)
        if (kf >= 5 and pf >= 5) or (kf <= 2 and pf <= 2):
            score -= 0.3                               # keep the king's pawn shield intact

    before = pawn_weaknesses(board, color)
    board.push(move)
    try:
        if pawn_weaknesses(board, color) > before:
            score -= 0.35                              # avoid doubled/isolated pawns
        if piece not in (chess.PAWN, chess.KING):
            gain = VALUE[captured] if captured else 0
            to = move.to_square
            pawn_attack = any(board.piece_type_at(s) == chess.PAWN
                              for s in board.attackers(not color, to))
            if pawn_attack and VALUE[piece] > gain + 1:
                score -= 0.6                           # no speculative piece sacrifices
            elif board.is_attacked_by(not color, to) and not board.is_attacked_by(color, to) \
                    and VALUE[piece] > gain:
                score -= 0.4                           # don't leave pieces hanging
    finally:
        board.pop()
    return score


STYLES = {"positional": positional_score}
