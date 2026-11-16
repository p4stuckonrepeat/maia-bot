# Mikhail Tal study (46000 positions from his games, move 9 onwards)

## 1. How accurate Tal was (Stockfish, among the natural candidate moves)

| | Tal | Tal model's first choice |
|---|---|---|
| average loss (centipawns) | 17 | 29 |
| best or within 0.2 pawns | 77% | 71% |
| mistakes (1-2 pawns) | 2.5% | 4.8% |
| blunders (> 2 pawns) | 0.9% | 3.2% |

- When behind (< -0.8): average loss 21 cp, blunders 1.2% (6230 positions)
- When equal: average loss 12 cp, blunders 0.4% (25710 positions)
- When ahead (> +0.8): average loss 24 cp, blunders 1.8% (14060 positions)
- When sharp (3+ captures available): average loss 18 cp, blunders 1.0% (17156 positions)

## 2. What Tal does more or less often than the Tal model's first choice

| Trait | Tal | Model | Ratio |
|---|---|---|---|
| capture | 24.0% | 26.4% | 0.91x |
| check | 6.5% | 7.4% | 0.87x |
| sacrifice | 3.8% | 3.4% | 1.12x |
| king_pressure | 18.1% | 19.2% | 0.94x |
| near_king | 13.1% | 14.1% | 0.93x |
| pawn_storm | 5.3% | 4.8% | 1.11x |
| mobility | 51.7% | 52.7% | 0.98x |
| retreat | 14.9% | 13.8% | 1.08x |
| queen_trade | 0.8% | 0.9% | 0.83x |
| castle | 1.5% | 1.7% | 0.88x |
| develop | 5.7% | 6.3% | 0.91x |
| center_pawn | 4.1% | 4.2% | 0.97x |
| pawn_break | 5.6% | 5.5% | 1.01x |
| safety | 47.1% | 48.2% | 0.98x |
| plan_worst | 11.1% | 11.8% | 0.94x |
| plan_break | 3.6% | 3.3% | 1.09x |
| plan_king | 18.1% | 19.2% | 0.94x |
| threat | 22.4% | 23.6% | 0.95x |
| forcing | 42.7% | 46.2% | 0.92x |
| even_trade | 9.0% | 10.4% | 0.86x |
| opens_king_file | 1.1% | 1.2% | 0.89x |
| complexity | 83.8% | 83.1% | 1.01x |
| sf_loss | 42.7% | 48.2% | 0.89x |
| sound_sac | 0.4% | 0.2% | 1.69x |

## 3. Sacrifices

- Tal sacrificed a piece in 230 of 46000 positions (0.5%, about once every 200 moves).
- Of those: sound (within 0.3 pawns of best) 177, speculative (0.3-2 pawns worse) 35, unsound (2+ pawns worse) 18.
- Sound sacrifice available among natural moves: 406 positions; Tal played one in 43% of them.
- Speculative sacrifice (0.3-2 pawns worse) available: 1320 positions; Tal played one in 2% of them.
- Position after Tal's sacrifices (Stockfish): 10% -226, median +56, 90% +588 cp.
- Position before them: 10% -175, median +102, 90% +588 cp.

## 4. Rules fitted on Tal's games, tested on games the bot never saw

| Rules | Exact Tal move |
|---|---|
| Tal model only | 56.2% |
| + basic Tal traits | 56.2% |
| + plans (worst piece, breaks, king) | 56.2% |
| + tactics (threats, forcing, trades, open files, complexity) | 56.3% |
| + Stockfish (move quality, sound/speculative sacrifices) | 62.3% |
| all rules, separate weights when behind / equal / ahead | 62.3% |

Best: **+ Stockfish (move quality, sound/speculative sacrifices)**. Weights (positive = Tal likes it more than the model does):
```
{
 "logp": 0.859,
 "exp": 0.053,
 "capture": 0.011,
 "check": -0.089,
 "sacrifice": 0.095,
 "king_pressure": 0.011,
 "near_king": -0.015,
 "pawn_storm": 0.004,
 "mobility": 0.037,
 "retreat": -0.029,
 "queen_trade": -0.039,
 "castle": 0.053,
 "develop": -0.018,
 "center_pawn": 0.039,
 "pawn_break": 0.075,
 "safety": 0.116,
 "plan_worst": -0.008,
 "plan_break": -0.004,
 "plan_king": 0.051,
 "threat": 0.06,
 "forcing": -0.018,
 "even_trade": -0.173,
 "opens_king_file": -0.009,
 "complexity": 0.131,
 "sf_loss": -1.303,
 "sound_sac": 0.028,
 "spec_sac": -0.038,
 "unsound_sac": 0.03
}
```
