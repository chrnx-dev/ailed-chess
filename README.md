# AILED — Psyche & Personality System for Chess Engines

**Architecture-agnostic middleware that gives any chess engine an emotional state.**

A continuous *psyche* score (ψ ∈ [−100, +100]) tracks position quality move-by-move and drives a five-stage audio-inspired signal chain that reshapes the engine's move probability distribution — without any search, tree traversal, or model retraining.

Works as a drop-in layer on top of any move predictor that outputs logits or probabilities: Maia2, Stockfish NNUE policy heads, or your own transformer.

> This module is a partial release accompanying the research paper:
> **"Personality and Psyche: Continuous Affective Modulation for Human-Like Chess Engines"**

---

## What It Does

### Psyche System

The `PsycheCalculator` evaluates the board position after every move across five factors:

| Factor | Default Weight | Description |
|---|---|---|
| Material | 10 | Piece value difference (P=1, N/B=3, R=5, Q=9) |
| King safety | 5 | Pawn shield + nearby enemy attackers |
| Pieces attacked | 3 | Own pieces under attack |
| Center control | 2 | Attacks on d4/d5/e4/e5 |
| Mobility | 1 | Legal moves minus opponent's legal moves |

These are combined, tanh-compressed to [−100, +100], and blended into the running psyche state with configurable reactivity, phase-awareness (opening/middlegame/endgame), sacrifice dampening, and daily decay toward neutral.

```python
from ailed.psyche import PsycheCalculator
import chess

calc = PsycheCalculator()          # default config
board = chess.Board()
psyche = 0.0                       # neutral at game start

board.push_san("e4")
psyche = calc.update_psyche(psyche, board)
print(f"ψ = {psyche:.1f}")        # e.g. ψ = 3.2
```

### EQ Signal Chain

`MoveSelector` applies a full audio-inspired signal chain to the move probability distribution:

```
Softmax → Noise gate → Dynamics → Sort + EQ bands → Saturation → Renormalize → Sample
```

1. **Noise gate** — silences moves below probability threshold
2. **Dynamics** — compressor/expander on probability spread
3. **EQ bands** — moves sorted into 5 quality bands, each with gain multiplier
4. **Saturation** — soft-clips dominant moves
5. **Renormalize** — produces valid distribution

All five parameters interpolate across N psyche anchors. At ψ = −70 (stressed) behavior differs dramatically from ψ = +70 (overconfident).

```python
from ailed.uci.move_selector import MoveSelector
from ailed.uci.eq_curve import PRESETS
import torch, chess

selector = MoveSelector(curve=PRESETS["human"])
board = chess.Board()
legal_moves = list(board.legal_moves)
logits = torch.randn(len(legal_moves))

move = selector.select_move(
    legal_moves=legal_moves,
    move_logits=logits,
    base_confidence=0.8,
    psyche=-40.0,
)
```

### Six Built-in Personalities

| Name | Character |
|---|---|
| `human` (default) | Stressed → noisy. Confident → autopilot. |
| `human5_symmetric` | Fine-grained psyche response (5 anchors). |
| `human5_asymmetric` | Wider stress, narrower confidence zone. |
| `classical` | Tight, positional, disciplined. |
| `jazz` | Avoids obvious moves, boosts alternatives. |
| `rock` | Bold, loves extremes (V-shape EQ). |
| `metal` | Chaotic, everything gets a shot. |
| `flat` | Bypass — raw probabilities. |

---

## Installation

Requires Python 3.11+.

```bash
pip install chess pydantic torch
git clone https://github.com/chrnx-dev/ailed-chess
cd ailed-chess
pip install -e .
```

---

## Architecture Agnosticism

Works on top of **any move predictor that outputs logits or probabilities**:

- Maia2 (convert probability dict to log-probs)
- Stockfish NNUE policy head
- Your custom transformer
- Any neural network with move output

The paper demonstrates this across two engines of vastly different scale (60K-game model vs 169M-game Maia2), showing consistent behavioral signatures from psyche regardless of base model.

---

## What Is NOT Included

This repository contains **only** the psyche and personality modules.

Full AILED system (released with paper) includes:
- Move predictor (23.5M param transformer)
- Training pipeline
- Lichess bot deployment
- Web dashboard
- Nightly study loop
- Terminal UI

---

## Citation

```bibtex
@misc{ailed2026,
  title={Personality and Psyche: Continuous Affective Modulation for Human-Like Chess Engines},
  author={[Author]},
  year={2026},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
