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


def zone(square):
    f = chess.square_file(square)
    return 0 if f <= 2 else (1 if f <= 4 else 2)   # queenside / centre / kingside


def plan_score(board, move):
    """Reward moves that continue the bot's own recent moves (one coherent idea)."""
    stack = board.move_stack
    mine = [stack[i] for i in range(len(stack) - 2, -1, -2)][:3]   # our last 3 moves, newest first
    if not mine:
        return 0.0
    color, piece = board.turn, board.piece_type_at(move.from_square)
    score, last = 0.0, mine[0]
    if move.from_square == last.to_square and move.to_square == last.from_square:
        score -= 0.4                                   # don't undo the previous move
    zones = [zone(m.to_square) for m in mine]
    focus = max(set(zones), key=zones.count)
    if zones.count(focus) >= 2 and zone(move.to_square) == focus:
        score += 0.15                                  # keep playing on the same part of the board
    if piece == chess.PAWN:
        prev_pawn_wings = [zone(m.to_square) for m in mine[:2]
                           if board.piece_type_at(m.to_square) == chess.PAWN]
        if zone(move.to_square) in prev_pawn_wings and zone(move.to_square) != 1:
            score += 0.15                              # continue a pawn expansion on that wing
    board.push(move)
    try:
        if board.piece_at(last.to_square) and board.piece_at(last.to_square).color == color \
                and last.to_square in board.attacks(move.to_square):
            score += 0.1                               # support the piece moved last
    finally:
        board.pop()
    return score


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
    return score + plan_score(board, move)


STYLES = {"positional": positional_score}


# ---------------------------------------------------------------------------------------------
# "classical": rules from classical chess theory, applied by game phase.
#   Opening: classical principles (centre, development, castling, no early queen/flank pawns).
#   Middlegame: Nimzowitsch (outposts, open files, 7th rank, blockade, prophylaxis) and
#               Steinitz (keep the bishop pair, trade when ahead, attack only when better).
#   Endgame: Capablanca/Tarrasch (active king, push passed pawns, rooks behind passed pawns).
# ---------------------------------------------------------------------------------------------
CENTER4 = [chess.D4, chess.E4, chess.D5, chess.E5]


def material(board, color):
    return sum(VALUE[p.piece_type] for p in board.piece_map().values() if p.color == color)


def is_passed(board, sq, color):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    for s in board.pieces(chess.PAWN, not color):
        sf, sr = chess.square_file(s), chess.square_rank(s)
        if abs(sf - f) <= 1 and ((sr > r) if color == chess.WHITE else (sr < r)):
            return False
    return True


def phase(board):
    non_pawn = sum(VALUE[p.piece_type] for p in board.piece_map().values()
                   if p.piece_type not in (chess.PAWN, chess.KING))
    if non_pawn <= 26:
        return "endgame"
    return "opening" if board.fullmove_number <= 10 else "middlegame"


def opp_mobility(board):
    """Legal moves the side NOT to move would have (null move), or None when in check."""
    if board.is_check():
        return None
    board.push(chess.Move.null())
    try:
        return board.legal_moves.count()
    finally:
        board.pop()


def classical_score(board, move, exp=None):
    color, piece = board.turn, board.piece_type_at(move.from_square)
    to, frm = move.to_square, move.from_square
    rel_rank = chess.square_rank(to) if color == chess.WHITE else 7 - chess.square_rank(to)
    captured = board.piece_type_at(to) or (chess.PAWN if board.is_en_passant(move) else None)
    ph = phase(board)
    s = 0.0

    if ph == "opening":
        if piece == chess.PAWN and to in CENTER4:
            s += 0.3                                   # occupy the centre
        if board.is_castling(move):
            s += 0.4
        if piece in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(frm) == (0 if color else 7):
            s += 0.25                                  # develop
        stack = board.move_stack
        if len(stack) >= 2 and frm == stack[-2].to_square and not captured:
            s -= 0.2                                   # same piece twice in the opening
        if piece == chess.QUEEN and not captured:
            s -= 0.3                                   # early queen
        if piece == chess.PAWN and chess.square_file(frm) in (0, 1, 6, 7) and not captured:
            s -= 0.2                                   # flank pawns
        if piece == chess.KING and not board.is_castling(move):
            s -= 0.4

    if ph != "endgame":
        # Nimzowitsch: outposts, open files, 7th rank, blockade
        if piece == chess.KNIGHT and 3 <= rel_rank <= 5:
            enemy_pawn_can_hit = any(
                abs(chess.square_file(p) - chess.square_file(to)) == 1 and
                ((chess.square_rank(p) > chess.square_rank(to)) if color == chess.WHITE
                 else (chess.square_rank(p) < chess.square_rank(to)))
                for p in board.pieces(chess.PAWN, not color))
            supported = any(board.piece_type_at(a) == chess.PAWN for a in board.attackers(color, to))
            if not enemy_pawn_can_hit and supported:
                s += 0.3
        if piece == chess.ROOK:
            f = chess.square_file(to)
            own = any(chess.square_file(p) == f for p in board.pieces(chess.PAWN, color))
            enemy = any(chess.square_file(p) == f for p in board.pieces(chess.PAWN, not color))
            if not own and not enemy:
                s += 0.25
            elif not own:
                s += 0.15
            if rel_rank == 6:
                s += 0.25
        for p in board.pieces(chess.PAWN, not color):
            if is_passed(board, p, not color):
                stop = p + 8 if (not color) == chess.WHITE else p - 8   # square in front of it
                if 0 <= stop < 64 and to == stop and piece != chess.PAWN:
                    s += 0.25                          # blockade
        # Steinitz: keep the bishop pair; trade when ahead, not when behind; attack only when better
        if piece == chess.BISHOP and captured == chess.KNIGHT and len(board.pieces(chess.BISHOP, color)) == 2:
            s -= 0.2
        diff = material(board, color) - material(board, not color)
        if captured and VALUE[captured] == VALUE[piece] and piece != chess.PAWN:
            s += 0.2 if diff >= 2 else (-0.2 if diff <= -2 else 0)
        king = board.king(not color)
        if piece == chess.PAWN and king is not None and not captured and rel_rank >= 3 \
                and abs(chess.square_file(to) - chess.square_file(king)) <= 1 and exp is not None:
            s += 0.15 if exp > 0.6 else (-0.2 if exp < 0.5 else 0)
        # prophylaxis: restrict the opponent
        before = opp_mobility(board)
        board.push(move)
        try:
            after = board.legal_moves.count()
        finally:
            board.pop()
        if before is not None and before - after >= 3:
            s += 0.1
    else:
        # Capablanca/Tarrasch endgame rules
        if piece == chess.KING:
            def cdist(sq):
                return max(abs(chess.square_file(sq) - 3.5), abs(chess.square_rank(sq) - 3.5))
            if cdist(to) < cdist(frm):
                s += 0.25                              # activate the king
        if piece == chess.PAWN and is_passed(board, frm, color):
            s += 0.3                                   # passed pawns must be pushed
        if piece == chess.ROOK:
            for p in list(board.pieces(chess.PAWN, color)) + list(board.pieces(chess.PAWN, not color)):
                pc = board.piece_at(p)
                if chess.square_file(p) == chess.square_file(to) and is_passed(board, p, pc.color):
                    behind = (chess.square_rank(to) < chess.square_rank(p)) if pc.color == chess.WHITE \
                        else (chess.square_rank(to) > chess.square_rank(p))
                    if behind:
                        s += 0.2                       # rook behind the passed pawn
                        break

    # always: safety, structure, king shield, plan (from the positional style)
    return s + positional_score(board, move) * 0.8


STYLES["classical"] = classical_score


# ---------------------------------------------------------------------------------------------
# "tal_learned": Tal-style features with weights fitted on Tal's own games (tal/tal-rules.json).
# ---------------------------------------------------------------------------------------------
_TAL = {}


def tal_learned_logit(board, cand, rules_file="tal/tal-rules.json"):
    import json
    import os
    if _TAL.get("_file") != rules_file:
        _TAL.clear()
        path = os.path.join(os.path.dirname(__file__), "..", rules_file)
        if os.path.exists(path):
            _TAL.update(json.load(open(path)))
        else:                                   # rules not learned yet: just follow the model
            _TAL.update({"features": ["logp"], "weights": {"logp": 1.0}})
        _TAL["_file"] = rules_file
    if _TAL["features"] == ["logp"]:
        import math
        return math.log(cand["policy"] + 1e-6)
    from tal.imitate import features
    w, d, _ = cand.get("wdl", (0, 1000, 0))
    f = features(board, cand["move"], cand["policy"], (w + 0.5 * d) / 1000)
    return sum(_TAL["weights"][n] * v for n, v in zip(_TAL["features"], f))


STYLES["tal_learned"] = lambda board, move, exp=None: 0.0   # handled in Maia.choose_styled


# --- Tal plans: classic middlegame theory as a light tie-breaker between Maia's natural moves ---
# 1. Improve your worst piece (Tarrasch / Dvoretsky): a move that frees the least active piece.
# 2. Open the position with pawn breaks (Tal): a pawn move that challenges an enemy pawn, more so
#    in the centre and towards the enemy king, especially when it frees a hemmed-in bishop.
# 3. Bring pieces to the enemy king (Tal): more pressure on the squares around it.

def _activity(board, sq):
    return len(board.attacks(sq))


def _worst_piece(board, color):
    worst, wa = None, 99
    for sq in board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color) | board.pieces(chess.ROOK, color):
        a = _activity(board, sq)
        if a < wa:
            worst, wa = sq, a
    return worst, wa


def _king_pressure(board, color):
    k = board.king(not color)
    if k is None:
        return 0
    zone_sqs = chess.SquareSet(chess.BB_KING_ATTACKS[k] | chess.BB_SQUARES[k])
    return sum(len(board.attackers(color, sq)) for sq in zone_sqs)


def tal_plan_score(board, move):
    me = board.turn
    score = 0.0
    worst, wa = _worst_piece(board, me)
    kp_before = _king_pressure(board, me)
    piece = board.piece_at(move.from_square)
    b = board.copy(stack=False)
    b.push(move)
    # 1. worst piece: moved to a better square, or its lines opened by the move
    if worst is not None and wa <= 4:
        new_sq = move.to_square if move.from_square == worst else worst
        if b.piece_at(new_sq) and b.piece_at(new_sq).color == me:
            gain = _activity(b, new_sq) - wa
            if gain > 0:
                score += min(gain, 6) / 6.0
    # 2. pawn breaks: the pawn now attacks an enemy pawn (centre / king side count more)
    if piece and piece.piece_type == chess.PAWN and not board.is_capture(move):
        hits = [sq for sq in b.attacks(move.to_square) if b.piece_at(sq) and
                b.piece_at(sq).piece_type == chess.PAWN and b.piece_at(sq).color != me]
        if hits:
            f = chess.square_file(move.to_square)
            score += 0.5 + (0.3 if f in (2, 3, 4, 5) else 0)
            k = board.king(not me)
            if k is not None and abs(chess.square_file(k) - f) <= 2:
                score += 0.3
    # 3. king pressure
    gain = _king_pressure(b, me) - kp_before
    if gain > 0:
        score += min(gain, 3) * 0.2
    return score
