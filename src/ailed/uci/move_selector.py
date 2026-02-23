"""Move selector: personality × psyche via audio signal chain.

All transforms are pure probability reshaping — no search, no history.

Signal chain:
  1. Softmax     → raw probabilities
  2. Noise gate  → silence moves below threshold
  3. Dynamics    → compressor / expander on probability spread
  4. Sort + EQ   → per-band gain multipliers
  5. Saturation  → clip dominant moves, redistribute
  6. Renormalize → valid distribution
  7. Sample      → selected move
"""

import torch
import chess
from typing import List, Optional

from ailed.uci.selection_config import SelectionConfig
from ailed.uci.eq_curve import EQCurve, PRESETS, band_ranges


class MoveSelector:
    """Personality-driven move selector.

    Personality (EQ preset) = static character trait.
    Psyche (emotional state) = dynamic modulator.

    Together they control: which moves survive the gate, how spread
    the distribution is (dynamics), which quality bands are favoured
    (EQ), and how much any single move can dominate (saturation).
    """

    def __init__(
        self,
        config: Optional[SelectionConfig] = None,
        curve: Optional[EQCurve] = None,
    ) -> None:
        self.config = config or SelectionConfig()
        self.curve = curve or PRESETS.get(self.config.eq_curve) or PRESETS["human5_symmetric"]

    def select_move(
        self,
        legal_moves: List[chess.Move],
        move_logits: torch.Tensor,
        base_confidence: float,
        psyche: float,
    ) -> chess.Move:
        """Select a move through the full signal chain.

        Args:
            legal_moves: List of legal chess moves.
            move_logits: Raw logits from move predictor for each legal move.
            base_confidence: Confidence in [0, 1]. Wet/dry mix for EQ.
            psyche: Current psyche value in [-100, 100].

        Returns:
            Selected chess move.
        """
        if len(legal_moves) == 0:
            raise ValueError("No legal moves provided")
        if len(legal_moves) != len(move_logits):
            raise ValueError(
                f"Number of legal moves ({len(legal_moves)}) must match "
                f"number of logits ({len(move_logits)})"
            )

        # 1. Softmax → raw probabilities
        probs = torch.softmax(move_logits, dim=0)

        # 2. Noise gate — silence moves below threshold
        gate_threshold = self.curve.get_gate_threshold(psyche)
        if gate_threshold > 0:
            mask = probs >= gate_threshold
            if mask.sum() > 0:  # keep at least something
                probs = probs * mask.float()
                probs = probs / probs.sum()

        # 3. Dynamics — compressor / expander
        dynamics_power = self.curve.get_dynamics_power(psyche)
        if abs(dynamics_power - 1.0) > 1e-6:
            probs = probs ** dynamics_power
            total = probs.sum()
            if total > 0:
                probs = probs / total

        # 4. Sort by probability → 5 quality bands → apply EQ gains
        sorted_indices = torch.argsort(probs, descending=True)
        n = len(sorted_indices)

        gains = self.curve.get_effective_gains(psyche, base_confidence)

        modified_probs = probs.clone()
        ranges = band_ranges(n)
        for band_idx, (start, end) in enumerate(ranges):
            if start < end:
                band_move_indices = sorted_indices[start:end]
                modified_probs[band_move_indices] *= gains[band_idx]

        # 5. Saturation — clip dominant moves
        sat_ceiling = self.curve.get_saturation_ceiling(psyche)
        if sat_ceiling < 1.0:
            modified_probs = modified_probs.clamp(max=sat_ceiling)

        # 6. Renormalize
        total = modified_probs.sum()
        if total > 0:
            modified_probs = modified_probs / total
        else:
            modified_probs = torch.ones_like(modified_probs) / n

        # 7. Sample
        selected_idx = torch.multinomial(modified_probs, num_samples=1).item()
        return legal_moves[selected_idx]
