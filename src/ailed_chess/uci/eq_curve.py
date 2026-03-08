"""Equalizer curves with dynamics for psyche-driven move selection.

Moves are sorted by model probability and split into 5 bands
(best / good / mild / bad / worst). Each band gets a gain multiplier
that shapes the selection personality -- like an audio equalizer
applied to the move quality spectrum.

EQ states are defined at N psyche anchors and piecewise-linearly
interpolated.  Works for any number of anchors >= 2.

Full signal chain (all pure probability transforms, no search):
  1. Softmax     -> raw probabilities
  2. Noise gate  -> silence moves below threshold
  3. Dynamics    -> compressor / expander on probability spread
  4. Sort + EQ   -> per-band gain multipliers
  5. Saturation  -> soft-clip dominant moves
  6. Renormalize -> valid distribution
  7. Sample      -> selected move
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


BAND_NAMES = ("best", "good", "mild", "bad", "worst")
NUM_BANDS = len(BAND_NAMES)


def _lerp_anchors(
    anchors: tuple[float, ...],
    values: tuple[float, ...],
    psyche: float,
) -> float:
    """Piecewise-linear interpolation across N anchors."""
    if psyche <= anchors[0]:
        return values[0]
    if psyche >= anchors[-1]:
        return values[-1]
    for i in range(len(anchors) - 1):
        if psyche <= anchors[i + 1]:
            t = (psyche - anchors[i]) / (anchors[i + 1] - anchors[i])
            return values[i] + (values[i + 1] - values[i]) * t
    return values[-1]


class EQCurve(BaseModel):
    """Personality curve: EQ + dynamics + gate + saturation.

    Parameters are defined at N psyche anchors and piecewise-linearly
    interpolated.  Works for any number of anchors >= 2.
    """

    name: str
    anchors: tuple[float, ...]
    bands: tuple[tuple[float, ...], ...]
    dynamics: tuple[float, ...]
    gate: tuple[float, ...]
    saturation: tuple[float, ...]

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_anchors(self) -> "EQCurve":
        n = len(self.anchors)
        if n < 2:
            raise ValueError(f"Need at least 2 anchors, got {n}")
        for i in range(n - 1):
            if self.anchors[i] >= self.anchors[i + 1]:
                raise ValueError(
                    f"Anchors must be sorted ascending: "
                    f"{self.anchors[i]} >= {self.anchors[i + 1]}"
                )
        for label, field in [
            ("bands", self.bands),
            ("dynamics", self.dynamics),
            ("gate", self.gate),
            ("saturation", self.saturation),
        ]:
            if len(field) != n:
                raise ValueError(f"{label} length {len(field)} != anchors length {n}")
        for i, b in enumerate(self.bands):
            if len(b) != NUM_BANDS:
                raise ValueError(f"bands[{i}] has {len(b)} elements, expected {NUM_BANDS}")
        return self

    # -- Public API ----------------------------------------------------------

    def get_effective_gains(
        self, psyche: float, confidence: float = 1.0
    ) -> list[float]:
        """Compute interpolated per-band gains for current psyche.

        Args:
            psyche: Current psyche value in [-100, 100].
            confidence: Outcome confidence in [0, 1]. Acts as wet/dry mix:
                1.0 = full EQ effect, 0.0 = flat (bypass).

        Returns:
            List of 5 effective gain multipliers (best -> worst).
        """
        bands = [
            _lerp_anchors(self.anchors, tuple(b[i] for b in self.bands), psyche)
            for i in range(NUM_BANDS)
        ]
        effective = [1.0 + (g - 1.0) * confidence for g in bands]
        return [max(g, 0.01) for g in effective]

    def get_dynamics_power(self, psyche: float) -> float:
        """Compute dynamics power."""
        return _lerp_anchors(self.anchors, self.dynamics, psyche)

    def get_gate_threshold(self, psyche: float) -> float:
        """Compute noise gate threshold for current psyche."""
        return _lerp_anchors(self.anchors, self.gate, psyche)

    def get_saturation_ceiling(self, psyche: float) -> float:
        """Compute saturation ceiling for current psyche."""
        return _lerp_anchors(self.anchors, self.saturation, psyche)


def band_ranges(n: int, num_bands: int = NUM_BANDS) -> list[tuple[int, int]]:
    """Split *n* sorted move indices into *num_bands* (start, end) ranges.

    Distributes remainder to the first bands so sizes are as equal as possible.
    When n < num_bands the trailing bands are empty (start == end).
    """
    if n <= num_bands:
        ranges = []
        for i in range(num_bands):
            if i < n:
                ranges.append((i, i + 1))
            else:
                ranges.append((n, n))
        return ranges

    base_size = n // num_bands
    remainder = n % num_bands
    ranges = []
    start = 0
    for i in range(num_bands):
        size = base_size + (1 if i < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


# -- Preset personalities ----------------------------------------------------
#
# Each preset is a complete personality: EQ shape, dynamics response,
# gate behavior, and saturation character -- all modulated by psyche.
#
# Personality = static character trait (the preset)
# Psyche = dynamic emotional state (-100 to +100)
# Together = how this character plays under this mood

PRESETS: dict[str, EQCurve] = {
    # Reference -- bypass everything.  Raw model probabilities.
    "flat": EQCurve(
        name="flat",
        anchors=(-100.0, 0.0, 100.0),
        bands=(
            (1.0, 1.0, 1.0, 1.0, 1.0),  # stress
            (1.0, 1.0, 1.0, 1.0, 1.0),  # neutral
            (1.0, 1.0, 1.0, 1.0, 1.0),  # confident
        ),
        dynamics=(1.0, 1.0, 1.0),
        gate=(0.0, 0.0, 0.0),
        saturation=(1.0, 1.0, 1.0),
    ),
    # Classical -- clean, positional, reliable.
    # Tight gate + mild saturation = focused, disciplined play.
    "classical": EQCurve(
        name="classical",
        anchors=(-100.0, 0.0, 100.0),
        bands=(
            (1.0, 1.4, 1.1, 0.8, 0.5),  # stress
            (2.0, 1.5, 1.0, 0.5, 0.2),  # neutral
            (2.5, 1.2, 0.6, 0.3, 0.1),  # confident
        ),
        dynamics=(0.8, 1.0, 1.5),
        gate=(0.01, 0.02, 0.05),
        saturation=(0.50, 0.60, 0.85),
    ),
    # Rock -- bold, loves extremes.  V-shape EQ.
    # Moderate gate + moderate saturation.
    "rock": EQCurve(
        name="rock",
        anchors=(-100.0, 0.0, 100.0),
        bands=(
            (1.4, 0.6, 0.4, 0.8, 1.6),  # stress
            (1.6, 0.8, 0.6, 0.8, 1.4),  # neutral
            (2.0, 0.8, 0.5, 0.6, 1.0),  # confident
        ),
        dynamics=(0.6, 1.0, 1.6),
        gate=(0.005, 0.01, 0.03),
        saturation=(0.35, 0.45, 0.70),
    ),
    # Jazz -- avoids the obvious, boosts interesting alternatives.
    # Open gate + heavy saturation = no single move dominates.
    "jazz": EQCurve(
        name="jazz",
        anchors=(-100.0, 0.0, 100.0),
        bands=(
            (0.5, 1.2, 1.5, 1.4, 0.8),  # stress
            (0.7, 1.8, 1.5, 0.8, 0.3),  # neutral
            (0.9, 1.4, 1.2, 0.8, 0.5),  # confident
        ),
        dynamics=(0.5, 1.0, 1.4),
        gate=(0.001, 0.005, 0.01),
        saturation=(0.25, 0.35, 0.50),
    ),
    # Metal -- chaotic, favours the unexpected.
    # Gate basically off + very heavy saturation = everything gets a shot.
    "metal": EQCurve(
        name="metal",
        anchors=(-100.0, 0.0, 100.0),
        bands=(
            (0.2, 0.4, 0.8, 1.5, 2.5),  # stress
            (0.3, 0.6, 1.0, 1.5, 2.0),  # neutral
            (0.5, 0.8, 1.0, 1.3, 1.6),  # confident
        ),
        dynamics=(0.4, 1.0, 1.3),
        gate=(0.0, 0.001, 0.005),
        saturation=(0.20, 0.30, 0.50),
    ),
    # Human -- the default personality.
    # Stress: open gate (noise leaks), low saturation (no clear choice).
    # Neutral: moderate gate, moderate saturation -- balanced.
    # Confident: tight gate, high saturation -- plays on autopilot.
    "human": EQCurve(
        name="human",
        anchors=(-100.0, 0.0, 100.0),
        bands=(
            (0.8, 1.3, 1.0, 1.4, 1.2),  # stress
            (1.3, 1.2, 1.0, 0.7, 0.5),  # neutral
            (1.0, 1.0, 1.0, 1.0, 1.0),  # confident
        ),
        dynamics=(0.5, 1.0, 2.0),
        gate=(0.005, 0.02, 0.06),
        saturation=(0.30, 0.50, 0.85),
    ),
    # Human 5-anchor symmetric -- same personality as "human" with
    # intermediate anchors at -50 and +50 (midpoints of their neighbors).
    # Gives finer-grained psyche response in the transition zones.
    "human5_symmetric": EQCurve(
        name="human5_symmetric",
        anchors=(-100.0, -50.0, 0.0, 50.0, 100.0),
        bands=(
            (0.8, 1.3, 1.0, 1.4, 1.2),     # -100 panic (= human stress)
            (1.05, 1.25, 1.0, 1.05, 0.85),  # -50  stressed (midpoint)
            (1.3, 1.2, 1.0, 0.7, 0.5),      #   0  neutral (= human neutral)
            (1.15, 1.1, 1.0, 0.85, 0.75),   # +50  confident (midpoint)
            (1.0, 1.0, 1.0, 1.0, 1.0),      # +100 overconfident (= human confident)
        ),
        dynamics=(0.5, 0.75, 1.0, 1.5, 2.0),
        gate=(0.005, 0.0125, 0.02, 0.04, 0.06),
        saturation=(0.30, 0.40, 0.50, 0.675, 0.85),
    ),
    # Human 5-anchor asymmetric -- same band/dynamics/gate/saturation values
    # as human5_symmetric but with asymmetric anchor positions: wider stress
    # zone (60 units) and narrower confident zone (40 units).
    "human5_asymmetric": EQCurve(
        name="human5_asymmetric",
        anchors=(-100.0, -60.0, 0.0, 40.0, 100.0),
        bands=(
            (0.8, 1.3, 1.0, 1.4, 1.2),     # -100 panic
            (1.05, 1.25, 1.0, 1.05, 0.85),  # -60  stressed (midpoint)
            (1.3, 1.2, 1.0, 0.7, 0.5),      #   0  neutral
            (1.15, 1.1, 1.0, 0.85, 0.75),   # +40  confident (midpoint)
            (1.0, 1.0, 1.0, 1.0, 1.0),      # +100 overconfident
        ),
        dynamics=(0.5, 0.75, 1.0, 1.5, 2.0),
        gate=(0.005, 0.0125, 0.02, 0.04, 0.06),
        saturation=(0.30, 0.40, 0.50, 0.675, 0.85),
    ),
}
