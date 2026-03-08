"""ailed-chess: human-like chess psyche and personality (EQ) system.

Public API surface for the ailed-chess package.
"""

from ailed_chess.psyche.calculator import PsycheCalculator
from ailed_chess.psyche.config import PsycheConfig
from ailed_chess.uci.move_selector import MoveSelector
from ailed_chess.uci.selection_config import SelectionConfig
from ailed_chess.uci.eq_curve import EQCurve, PRESETS

__all__ = [
    "PsycheCalculator",
    "PsycheConfig",
    "MoveSelector",
    "SelectionConfig",
    "EQCurve",
    "PRESETS",
]
