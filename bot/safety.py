"""Engine-free tactical safety, the way strong humans check ("loose pieces drop off").

All of this is static (no search, no engine): microseconds per move.
- hanging(board, color): our pieces (knight or more) that the opponent can win right now
  (attacked by a cheaper piece, or attacked and not defended)
- danger_after(board_after_our_move, color): does the opponent have a cheap shot next move:
  a capture that wins material, or a pawn push / move that attacks two of our pieces, or attacks
  one piece while another is already loose (the ...e5 pattern)
- offered(board, move): is this move a piece sacrifice (piece left where it can be taken for less)
"""
import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def _cheapest(board, color, sq):
    att = board.attackers(color, sq)
    return min((VAL[board.piece_type_at(s)] for s in att), default=None), len(att)


def loose(board, color, sq):
    """Can the opponent win the piece on sq (roughly, static exchange)?"""
    p = board.piece_at(sq)
    if p is None or p.color != color or p.piece_type in (chess.PAWN, chess.KING):
        return False
    low, n_att = _cheapest(board, not color, sq)
    if low is None:
        return False
    _, n_def = _cheapest(board, color, sq)
    return low < VAL[p.piece_type] or n_def == 0 or n_att > n_def


def hanging(board, color):
    return [sq for sq, p in board.piece_map().items() if p.color == color and loose(board, color, sq)]


def danger_after(board, color, max_moves=60):
    """board: position after our move (opponent to move). True if they have a cheap tactical shot."""
    if board.turn == color:
        return False
    if hanging(board, color):
        return True
    n = 0
    for m in board.legal_moves:
        piece = board.piece_at(m.from_square)
        quiet_pawn = piece and piece.piece_type == chess.PAWN and not board.is_capture(m)
        if not (quiet_pawn or board.gives_check(m) or board.is_capture(m)):
            continue
        n += 1
        if n > max_moves:
            break
        board.push(m)
        try:
            hit = [s for s in board.attacks(m.to_square)
                   if board.piece_at(s) and board.piece_at(s).color == color
                   and board.piece_at(s).piece_type != chess.PAWN]
            loose_now = hanging(board, color)
            # fork / double attack, or one attacked piece while another is already loose
            if len(hit) >= 2 and (quiet_pawn or board.is_check()):
                return True
            if quiet_pawn and hit and any(s not in hit for s in loose_now):
                return True
            if quiet_pawn and hit and any(VAL[board.piece_type_at(s)] > 1 for s in hit) and len(loose_now) >= 2:
                return True
        finally:
            board.pop()
    return False


def offered(board, move):
    """Piece sacrifice: a knight or more moves where the opponent can take it, losing us 2+ points."""
    p = board.piece_at(move.from_square)
    if p is None or p.piece_type in (chess.PAWN, chess.KING):
        return False
    gain = VAL[board.piece_type_at(move.to_square)] if board.piece_type_at(move.to_square) else 0
    b = board.copy(stack=False)
    b.push(move)
    low, _ = _cheapest(b, not p.color, move.to_square)
    if low is None:
        return False
    _, n_def = _cheapest(b, p.color, move.to_square)
    lost = VAL[p.piece_type] - (low if n_def else 0)
    return lost - gain >= 2
