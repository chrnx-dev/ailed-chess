"""Configuration for psyche calculator.

This module defines the configuration parameters for the psyche calculation system,
including weights for different position evaluation factors and psyche bounds.
"""

from pydantic import BaseModel, ConfigDict, Field


class PsycheConfig(BaseModel):
    """Configuration for psyche calculation.

    Defines weights for different position evaluation factors and psyche system parameters.
    All weights are applied to normalized factor scores to compute psyche delta.
    """

    # Factor weights
    material_weight: float = Field(
        default=10,
        description="Weight for material difference (piece values)"
    )
    king_safety_weight: float = Field(
        default=5,
        description="Weight for king safety evaluation"
    )
    pieces_attacked_weight: float = Field(
        default=3,
        description="Weight for pieces under attack"
    )
    center_control_weight: float = Field(
        default=2,
        description="Weight for center square control"
    )
    mobility_weight: float = Field(
        default=1,
        description="Weight for mobility (legal moves available)"
    )

    # Target-blend formula parameters
    position_scale: float = Field(
        default=50.0,
        gt=0.0,
        description="Controls how much raw eval is needed to reach psyche extremes"
    )
    reactivity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="How quickly psyche moves toward target (0-1, where 0.3 = 30% per move)"
    )

    # Special penalties/bonuses
    check_penalty: float = Field(
        default=-50,
        description="Fixed penalty when current player is in check"
    )

    # Game phase detection
    opening_move_limit: int = Field(
        default=15,
        ge=1,
        description="Fullmove number threshold for opening phase"
    )
    endgame_material_threshold: int = Field(
        default=13,
        ge=0,
        description="Total non-pawn/non-king material (both sides) at or below which is endgame"
    )
    opening_reactivity_factor: float = Field(
        default=0.15,
        gt=0.0,
        le=2.0,
        description="Reactivity multiplier during opening phase (low = psyche persists)"
    )
    endgame_reactivity_factor: float = Field(
        default=0.1,
        gt=0.0,
        le=2.0,
        description="Reactivity multiplier during endgame phase (near-zero = psyche frozen)"
    )

    # Sacrifice detection
    sacrifice_dampening: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How much to dampen material loss when positional compensation exists (0-1)"
    )

    # Resilience
    resilience: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Symmetric resistance to psyche change (0=no resistance, 1=frozen)"
    )

    # Decay settings
    decay_rate: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Daily decay rate toward neutral (0-1, where 0.20 = 20% decay)"
    )

    # Psyche bounds
    min_psyche: float = Field(
        default=-100,
        description="Minimum psyche value (most negative emotional state)"
    )
    max_psyche: float = Field(
        default=100,
        description="Maximum psyche value (most positive emotional state)"
    )

    model_config = ConfigDict(frozen=True)  # Make config immutable
