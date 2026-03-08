"""Configuration for move selection.

This module defines the configuration parameters for the EQ-curve-based
move selection system.
"""

from pydantic import BaseModel, ConfigDict, Field


class SelectionConfig(BaseModel):
    """Configuration for EQ-based move selection.

    The eq_curve parameter selects a named preset that defines per-band
    gain multipliers for the 5 quality bands (best/good/mild/bad/worst).
    Available presets: flat, classical, rock, jazz, metal, human,
    human5_symmetric, human5_asymmetric.
    """

    eq_curve: str = Field(
        default="human5_symmetric",
        description="Named EQ curve preset for move selection personality.",
    )

    model_config = ConfigDict(frozen=True)
