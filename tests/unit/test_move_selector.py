"""Tests for personality x psyche move selector (full signal chain)."""

import pytest
import torch
import chess
from ailed_chess.uci.move_selector import MoveSelector
from ailed_chess.uci.selection_config import SelectionConfig
from ailed_chess.uci.eq_curve import EQCurve, PRESETS, band_ranges, _lerp_anchors


# -- _lerp_anchors tests -----------------------------------------------------


class TestLerpAnchors:
    """Test generic piecewise-linear interpolation."""

    def test_3_anchors_at_endpoints(self):
        anchors = (-100.0, 0.0, 100.0)
        values = (0.5, 1.0, 2.0)
        assert abs(_lerp_anchors(anchors, values, -100.0) - 0.5) < 1e-6
        assert abs(_lerp_anchors(anchors, values, 0.0) - 1.0) < 1e-6
        assert abs(_lerp_anchors(anchors, values, 100.0) - 2.0) < 1e-6

    def test_3_anchors_midpoint(self):
        anchors = (-100.0, 0.0, 100.0)
        values = (0.5, 1.0, 2.0)
        assert abs(_lerp_anchors(anchors, values, -50.0) - 0.75) < 1e-6
        assert abs(_lerp_anchors(anchors, values, 50.0) - 1.5) < 1e-6

    def test_5_anchors_at_endpoints(self):
        anchors = (-100.0, -50.0, 0.0, 50.0, 100.0)
        values = (0.2, 0.5, 1.0, 1.5, 2.0)
        assert abs(_lerp_anchors(anchors, values, -100.0) - 0.2) < 1e-6
        assert abs(_lerp_anchors(anchors, values, 100.0) - 2.0) < 1e-6

    def test_5_anchors_at_inner_anchors(self):
        anchors = (-100.0, -50.0, 0.0, 50.0, 100.0)
        values = (0.2, 0.5, 1.0, 1.5, 2.0)
        assert abs(_lerp_anchors(anchors, values, -50.0) - 0.5) < 1e-6
        assert abs(_lerp_anchors(anchors, values, 0.0) - 1.0) < 1e-6
        assert abs(_lerp_anchors(anchors, values, 50.0) - 1.5) < 1e-6

    def test_5_anchors_between_inner(self):
        anchors = (-100.0, -50.0, 0.0, 50.0, 100.0)
        values = (0.2, 0.5, 1.0, 1.5, 2.0)
        assert abs(_lerp_anchors(anchors, values, -25.0) - 0.75) < 1e-6

    def test_clamps_below_min_anchor(self):
        anchors = (-100.0, 0.0, 100.0)
        values = (0.5, 1.0, 2.0)
        assert abs(_lerp_anchors(anchors, values, -200.0) - 0.5) < 1e-6

    def test_clamps_above_max_anchor(self):
        anchors = (-100.0, 0.0, 100.0)
        values = (0.5, 1.0, 2.0)
        assert abs(_lerp_anchors(anchors, values, 200.0) - 2.0) < 1e-6

    def test_asymmetric_anchors(self):
        anchors = (-100.0, -60.0, 0.0, 40.0, 100.0)
        values = (0.2, 0.6, 1.0, 1.4, 2.0)
        assert abs(_lerp_anchors(anchors, values, -60.0) - 0.6) < 1e-6
        assert abs(_lerp_anchors(anchors, values, 40.0) - 1.4) < 1e-6

    def test_2_anchors_works(self):
        anchors = (-100.0, 100.0)
        values = (0.0, 2.0)
        assert abs(_lerp_anchors(anchors, values, 0.0) - 1.0) < 1e-6


# -- EQCurve validation tests ------------------------------------------------


class TestEQCurveValidation:
    """Test EQCurve field validation."""

    def test_unsorted_anchors_rejected(self):
        with pytest.raises(Exception):
            EQCurve(
                name="bad",
                anchors=(100.0, 0.0, -100.0),
                bands=(
                    (1.0, 1.0, 1.0, 1.0, 1.0),
                    (1.0, 1.0, 1.0, 1.0, 1.0),
                    (1.0, 1.0, 1.0, 1.0, 1.0),
                ),
                dynamics=(1.0, 1.0, 1.0),
                gate=(0.0, 0.0, 0.0),
                saturation=(1.0, 1.0, 1.0),
            )

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(Exception):
            EQCurve(
                name="bad",
                anchors=(-100.0, 0.0, 100.0),
                bands=(
                    (1.0, 1.0, 1.0, 1.0, 1.0),
                    (1.0, 1.0, 1.0, 1.0, 1.0),
                ),
                dynamics=(1.0, 1.0, 1.0),
                gate=(0.0, 0.0, 0.0),
                saturation=(1.0, 1.0, 1.0),
            )

    def test_single_anchor_rejected(self):
        with pytest.raises(Exception):
            EQCurve(
                name="bad",
                anchors=(0.0,),
                bands=((1.0, 1.0, 1.0, 1.0, 1.0),),
                dynamics=(1.0,),
                gate=(0.0,),
                saturation=(1.0,),
            )

    def test_wrong_band_size_rejected(self):
        with pytest.raises(Exception):
            EQCurve(
                name="bad",
                anchors=(-100.0, 0.0, 100.0),
                bands=(
                    (1.0, 1.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (1.0, 1.0, 1.0),
                ),
                dynamics=(1.0, 1.0, 1.0),
                gate=(0.0, 0.0, 0.0),
                saturation=(1.0, 1.0, 1.0),
            )

    def test_valid_5_anchor_accepted(self):
        curve = EQCurve(
            name="ok",
            anchors=(-100.0, -50.0, 0.0, 50.0, 100.0),
            bands=(
                (1.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            dynamics=(1.0, 1.0, 1.0, 1.0, 1.0),
            gate=(0.0, 0.0, 0.0, 0.0, 0.0),
            saturation=(1.0, 1.0, 1.0, 1.0, 1.0),
        )
        assert len(curve.anchors) == 5


# -- EQ curve tests -----------------------------------------------------------


class TestEQCurve:
    """Test suite for EQCurve: interpolation, dynamics, gate, saturation."""

    # -- Band interpolation --------------------------------------------------

    def test_flat_gains_at_neutral_psyche(self):
        curve = PRESETS["flat"]
        gains = curve.get_effective_gains(psyche=0.0, confidence=1.0)
        assert all(abs(g - 1.0) < 0.01 for g in gains)

    def test_stress_interpolation_returns_stress_bands(self):
        curve = PRESETS["human"]
        gains = curve.get_effective_gains(psyche=-100.0, confidence=1.0)
        for g, expected in zip(gains, curve.bands[0]):
            assert abs(g - expected) < 0.01

    def test_neutral_interpolation_returns_neutral_bands(self):
        curve = PRESETS["human"]
        gains = curve.get_effective_gains(psyche=0.0, confidence=1.0)
        for g, expected in zip(gains, curve.bands[1]):
            assert abs(g - expected) < 0.01

    def test_confident_interpolation_returns_confident_bands(self):
        curve = PRESETS["human"]
        gains = curve.get_effective_gains(psyche=100.0, confidence=1.0)
        for g, expected in zip(gains, curve.bands[2]):
            assert abs(g - expected) < 0.01

    def test_midpoint_interpolation(self):
        curve = PRESETS["human"]
        gains = curve.get_effective_gains(psyche=-50.0, confidence=1.0)
        for i in range(5):
            expected = (curve.bands[1][i] + curve.bands[0][i]) / 2.0
            assert abs(gains[i] - expected) < 0.01

    def test_human_stress_boosts_bad_bands(self):
        curve = PRESETS["human"]
        gains = curve.get_effective_gains(psyche=-100.0, confidence=1.0)
        assert gains[3] > gains[0]  # bad > best

    def test_human_neutral_boosts_top_bands(self):
        curve = PRESETS["human"]
        gains = curve.get_effective_gains(psyche=0.0, confidence=1.0)
        assert gains[0] > gains[4]  # best > worst

    def test_human_confident_is_flat(self):
        curve = PRESETS["human"]
        gains = curve.get_effective_gains(psyche=100.0, confidence=1.0)
        assert all(abs(g - 1.0) < 0.01 for g in gains)

    def test_confidence_as_wet_dry_mix(self):
        curve = PRESETS["classical"]
        full = curve.get_effective_gains(psyche=0.0, confidence=1.0)
        half = curve.get_effective_gains(psyche=0.0, confidence=0.5)
        zero = curve.get_effective_gains(psyche=0.0, confidence=0.0)

        assert all(abs(g - 1.0) < 0.01 for g in zero)
        for i in range(5):
            expected = 1.0 + (full[i] - 1.0) * 0.5
            assert abs(half[i] - expected) < 0.01

    def test_gains_never_zero(self):
        curve = PRESETS["metal"]
        gains = curve.get_effective_gains(psyche=100.0, confidence=1.0)
        assert all(g > 0 for g in gains)

    def test_all_presets_exist(self):
        for name in ("flat", "classical", "rock", "jazz", "metal", "human",
                      "human5_symmetric", "human5_asymmetric"):
            assert name in PRESETS

    # -- Dynamics ------------------------------------------------------------

    def test_dynamics_power_at_neutral(self):
        curve = PRESETS["human"]
        assert abs(curve.get_dynamics_power(0.0) - 1.0) < 1e-6

    def test_dynamics_power_at_stress(self):
        curve = PRESETS["human"]
        assert abs(curve.get_dynamics_power(-100.0) - curve.dynamics[0]) < 1e-6

    def test_dynamics_power_at_overconfident(self):
        curve = PRESETS["human"]
        assert abs(curve.get_dynamics_power(100.0) - curve.dynamics[2]) < 1e-6

    def test_dynamics_interpolation_halfway(self):
        curve = PRESETS["human"]
        expected = (curve.dynamics[0] + curve.dynamics[1]) / 2.0
        assert abs(curve.get_dynamics_power(-50.0) - expected) < 1e-6

    def test_flat_dynamics_always_unity(self):
        curve = PRESETS["flat"]
        for psyche in [-100, -50, 0, 50, 100]:
            assert abs(curve.get_dynamics_power(float(psyche)) - 1.0) < 1e-6

    def test_expander_is_less_than_one(self):
        curve = PRESETS["human"]
        assert curve.dynamics[0] < 1.0

    def test_compressor_is_greater_than_one(self):
        curve = PRESETS["human"]
        assert curve.dynamics[2] > 1.0

    # -- Noise gate ----------------------------------------------------------

    def test_gate_at_stress(self):
        curve = PRESETS["human"]
        assert abs(curve.get_gate_threshold(-100.0) - curve.gate[0]) < 1e-6

    def test_gate_at_neutral(self):
        curve = PRESETS["human"]
        assert abs(curve.get_gate_threshold(0.0) - curve.gate[1]) < 1e-6

    def test_gate_at_confident(self):
        curve = PRESETS["human"]
        assert abs(curve.get_gate_threshold(100.0) - curve.gate[2]) < 1e-6

    def test_gate_interpolation(self):
        curve = PRESETS["human"]
        expected = (curve.gate[1] + curve.gate[2]) / 2.0
        assert abs(curve.get_gate_threshold(50.0) - expected) < 1e-6

    def test_flat_gate_always_zero(self):
        curve = PRESETS["flat"]
        for psyche in [-100, 0, 100]:
            assert curve.get_gate_threshold(float(psyche)) == 0.0

    def test_stress_gate_lower_than_confident(self):
        """Stressed = open gate, confident = tight gate."""
        curve = PRESETS["human"]
        assert curve.get_gate_threshold(-100.0) < curve.get_gate_threshold(100.0)

    # -- Saturation ----------------------------------------------------------

    def test_saturation_at_stress(self):
        curve = PRESETS["human"]
        assert abs(curve.get_saturation_ceiling(-100.0) - curve.saturation[0]) < 1e-6

    def test_saturation_at_neutral(self):
        curve = PRESETS["human"]
        assert abs(curve.get_saturation_ceiling(0.0) - curve.saturation[1]) < 1e-6

    def test_saturation_at_confident(self):
        curve = PRESETS["human"]
        assert abs(curve.get_saturation_ceiling(100.0) - curve.saturation[2]) < 1e-6

    def test_saturation_interpolation(self):
        curve = PRESETS["human"]
        expected = (curve.saturation[1] + curve.saturation[0]) / 2.0
        assert abs(curve.get_saturation_ceiling(-50.0) - expected) < 1e-6

    def test_flat_saturation_always_one(self):
        curve = PRESETS["flat"]
        for psyche in [-100, 0, 100]:
            assert curve.get_saturation_ceiling(float(psyche)) == 1.0

    def test_stress_saturation_lower_than_confident(self):
        """Stressed = heavy saturation, confident = clean signal."""
        curve = PRESETS["human"]
        assert curve.get_saturation_ceiling(-100.0) < curve.get_saturation_ceiling(100.0)

    # -- Band ranges ---------------------------------------------------------

    def test_band_ranges_even_distribution(self):
        ranges = band_ranges(20)
        sizes = [end - start for start, end in ranges]
        assert sizes == [4, 4, 4, 4, 4]

    def test_band_ranges_uneven(self):
        ranges = band_ranges(7)
        sizes = [end - start for start, end in ranges]
        assert sum(sizes) == 7
        assert sizes == [2, 2, 1, 1, 1]

    def test_band_ranges_fewer_than_bands(self):
        ranges = band_ranges(3)
        sizes = [end - start for start, end in ranges]
        assert sizes == [1, 1, 1, 0, 0]

    def test_band_ranges_single_move(self):
        ranges = band_ranges(1)
        sizes = [end - start for start, end in ranges]
        assert sizes == [1, 0, 0, 0, 0]


# -- Five-anchor preset tests ------------------------------------------------


class TestFiveAnchorPresets:
    """Test 5-anchor human presets."""

    def test_human5_symmetric_exists(self):
        assert "human5_symmetric" in PRESETS

    def test_human5_asymmetric_exists(self):
        assert "human5_asymmetric" in PRESETS

    def test_human5_symmetric_has_5_anchors(self):
        curve = PRESETS["human5_symmetric"]
        assert len(curve.anchors) == 5
        assert curve.anchors == (-100.0, -50.0, 0.0, 50.0, 100.0)

    def test_human5_asymmetric_has_5_anchors(self):
        curve = PRESETS["human5_asymmetric"]
        assert len(curve.anchors) == 5
        assert curve.anchors == (-100.0, -60.0, 0.0, 40.0, 100.0)

    def test_human5_symmetric_endpoints_match_human(self):
        """Extreme anchors (-100, 0, +100) should match the 3-anchor human."""
        h3 = PRESETS["human"]
        h5 = PRESETS["human5_symmetric"]
        for psyche in [-100.0, 0.0, 100.0]:
            g3 = h3.get_effective_gains(psyche, confidence=1.0)
            g5 = h5.get_effective_gains(psyche, confidence=1.0)
            for a, b in zip(g3, g5):
                assert abs(a - b) < 0.01, f"Mismatch at psyche={psyche}: {a} vs {b}"

    def test_human5_symmetric_intermediate_is_midpoint(self):
        """At -50 (an anchor), 5-anchor should equal the auto-derived midpoint."""
        h5 = PRESETS["human5_symmetric"]
        gains_at_minus50 = h5.get_effective_gains(-50.0, confidence=1.0)
        h3 = PRESETS["human"]
        for i in range(5):
            expected = (h3.bands[0][i] + h3.bands[1][i]) / 2.0
            assert abs(gains_at_minus50[i] - expected) < 0.01

    def test_human5_symmetric_dynamics_at_minus50(self):
        h5 = PRESETS["human5_symmetric"]
        h3 = PRESETS["human"]
        expected = (h3.dynamics[0] + h3.dynamics[1]) / 2.0
        assert abs(h5.get_dynamics_power(-50.0) - expected) < 1e-6

    def test_human5_all_psyche_extremes_work(self):
        for preset in ("human5_symmetric", "human5_asymmetric"):
            curve = PRESETS[preset]
            for psyche in [-100.0, -80.0, -50.0, -25.0, 0.0, 25.0, 50.0, 80.0, 100.0]:
                gains = curve.get_effective_gains(psyche, confidence=0.5)
                assert len(gains) == 5
                assert all(g > 0 for g in gains)


# -- SelectionConfig tests ---------------------------------------------------


class TestSelectionConfig:

    def test_default_config(self):
        config = SelectionConfig()
        assert config.eq_curve == "human5_symmetric"

    def test_custom_curve(self):
        config = SelectionConfig(eq_curve="rock")
        assert config.eq_curve == "rock"

    def test_config_is_frozen(self):
        config = SelectionConfig()
        with pytest.raises(Exception):
            config.eq_curve = "jazz"


# -- MoveSelector tests ------------------------------------------------------


class TestMoveSelector:
    """Test suite for full signal chain: gate -> dynamics -> EQ -> saturation."""

    @pytest.fixture
    def selector(self):
        return MoveSelector()

    @pytest.fixture
    def flat_selector(self):
        return MoveSelector(SelectionConfig(eq_curve="flat"))

    def test_initialization_default(self, selector):
        assert selector.config.eq_curve == "human5_symmetric"
        assert selector.curve.name == "human5_symmetric"

    def test_initialization_invalid_curve_falls_back(self):
        selector = MoveSelector(SelectionConfig(eq_curve="nonexistent"))
        assert selector.curve.name == "human5_symmetric"

    def test_select_move_returns_legal(self, selector):
        legal_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("g1f3"),
        ]
        logits = torch.tensor([2.0, 1.0, 0.5])
        move = selector.select_move(legal_moves, logits, 0.6, 0.0)
        assert move in legal_moves

    def test_select_move_deterministic_with_seed(self, selector):
        legal_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("g1f3"),
        ]
        logits = torch.tensor([2.0, 1.0, 0.5])

        torch.manual_seed(42)
        m1 = selector.select_move(legal_moves, logits, 0.6, 0.0)
        torch.manual_seed(42)
        m2 = selector.select_move(legal_moves, logits, 0.6, 0.0)
        assert m1 == m2

    def test_stress_expander_increases_diversity(self):
        sel = MoveSelector(SelectionConfig(eq_curve="human"))
        legal_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("b1c3"),
            chess.Move.from_uci("g2g3"),
        ]
        logits = torch.tensor([3.0, 1.5, 0.5, 0.0, -0.5])

        torch.manual_seed(100)
        neutral_picks = [
            sel.select_move(legal_moves, logits, 0.8, 0.0)
            for _ in range(200)
        ]
        torch.manual_seed(100)
        stress_picks = [
            sel.select_move(legal_moves, logits, 0.8, -80.0)
            for _ in range(200)
        ]

        neutral_unique = len(set(neutral_picks))
        stress_unique = len(set(stress_picks))
        assert stress_unique >= neutral_unique

    def test_compressor_concentrates_picks(self):
        sel = MoveSelector(SelectionConfig(eq_curve="human"))
        legal_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("b1c3"),
            chess.Move.from_uci("g2g3"),
        ]
        logits = torch.tensor([3.0, 1.5, 0.5, 0.0, -0.5])

        torch.manual_seed(42)
        neutral_top = sum(
            1 for _ in range(200)
            if sel.select_move(legal_moves, logits, 0.8, 0.0)
            == chess.Move.from_uci("e2e4")
        )
        torch.manual_seed(42)
        confident_top = sum(
            1 for _ in range(200)
            if sel.select_move(legal_moves, logits, 0.8, 80.0)
            == chess.Move.from_uci("e2e4")
        )

        assert confident_top > neutral_top

    def test_gate_filters_low_probability_moves(self):
        """Tight gate should remove low-prob moves entirely."""
        # Custom curve with very aggressive gate
        curve = EQCurve(
            name="tight_gate",
            anchors=(-100.0, 0.0, 100.0),
            bands=(
                (1.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            dynamics=(1.0, 1.0, 1.0),
            gate=(0.2, 0.2, 0.2),  # 20% threshold -- very tight
            saturation=(1.0, 1.0, 1.0),
        )
        # Inject custom curve
        PRESETS["_test_gate"] = curve
        try:
            sel = MoveSelector(SelectionConfig(eq_curve="_test_gate"))
            legal_moves = [
                chess.Move.from_uci("e2e4"),  # will have high prob
                chess.Move.from_uci("d2d4"),
                chess.Move.from_uci("g1f3"),
                chess.Move.from_uci("b1c3"),
                chess.Move.from_uci("g2g3"),  # will have very low prob
            ]
            # e2e4 dominates, g2g3 is negligible
            logits = torch.tensor([5.0, 2.0, 0.5, -1.0, -3.0])

            torch.manual_seed(42)
            picks = [sel.select_move(legal_moves, logits, 0.5, 0.0) for _ in range(100)]
            # g2g3 should never be picked (gated out)
            assert chess.Move.from_uci("g2g3") not in picks
        finally:
            del PRESETS["_test_gate"]

    def test_saturation_prevents_single_move_dominance(self):
        """Heavy saturation should spread picks even with a dominant move."""
        curve = EQCurve(
            name="heavy_sat",
            anchors=(-100.0, 0.0, 100.0),
            bands=(
                (1.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            dynamics=(1.0, 1.0, 1.0),
            gate=(0.0, 0.0, 0.0),
            saturation=(0.15, 0.15, 0.15),  # 15% ceiling -- very heavy
        )
        PRESETS["_test_sat"] = curve
        try:
            sel = MoveSelector(SelectionConfig(eq_curve="_test_sat"))
            legal_moves = [
                chess.Move.from_uci("e2e4"),
                chess.Move.from_uci("d2d4"),
                chess.Move.from_uci("g1f3"),
                chess.Move.from_uci("b1c3"),
                chess.Move.from_uci("g2g3"),
            ]
            # e2e4 normally dominates (~61% after softmax)
            logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0])

            torch.manual_seed(42)
            picks = [sel.select_move(legal_moves, logits, 0.5, 0.0) for _ in range(200)]
            top_count = sum(1 for m in picks if m == chess.Move.from_uci("e2e4"))
            # With 15% ceiling and 5 moves, top can't dominate
            assert top_count < 100  # well below the ~160+ without saturation
        finally:
            del PRESETS["_test_sat"]

    def test_different_curves_produce_different_distributions(self):
        legal_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("b1c3"),
            chess.Move.from_uci("g2g3"),
        ]
        logits = torch.tensor([3.0, 1.5, 0.5, 0.0, -0.5])

        results = {}
        for curve_name in ("classical", "jazz", "metal"):
            sel = MoveSelector(SelectionConfig(eq_curve=curve_name))
            torch.manual_seed(42)
            picks = [sel.select_move(legal_moves, logits, 0.6, 0.0) for _ in range(200)]
            top_count = sum(1 for m in picks if m == chess.Move.from_uci("e2e4"))
            results[curve_name] = top_count

        assert results["classical"] > results["metal"]

    def test_empty_legal_moves_raises(self, selector):
        with pytest.raises(ValueError, match="No legal moves"):
            selector.select_move([], torch.tensor([]), 0.5, 0.0)

    def test_mismatched_lengths_raises(self, selector):
        moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        logits = torch.tensor([1.0, 0.5, 0.3])
        with pytest.raises(ValueError, match="must match"):
            selector.select_move(moves, logits, 0.5, 0.0)

    def test_single_legal_move(self, selector):
        moves = [chess.Move.from_uci("e2e4")]
        logits = torch.tensor([1.0])
        move = selector.select_move(moves, logits, 0.5, 0.0)
        assert move == chess.Move.from_uci("e2e4")

    def test_all_presets_work(self):
        legal_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("g1f3"),
        ]
        logits = torch.tensor([2.0, 1.0, 0.5])

        for name in PRESETS:
            if name.startswith("_test"):
                continue
            sel = MoveSelector(SelectionConfig(eq_curve=name))
            move = sel.select_move(legal_moves, logits, 0.6, 0.0)
            assert move in legal_moves

    def test_all_presets_all_psyche_extremes(self):
        legal_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("b1c3"),
            chess.Move.from_uci("g2g3"),
        ]
        logits = torch.tensor([3.0, 1.5, 0.5, 0.0, -0.5])

        for name in PRESETS:
            if name.startswith("_test"):
                continue
            sel = MoveSelector(SelectionConfig(eq_curve=name))
            for psyche in [-100.0, -50.0, 0.0, 50.0, 100.0]:
                move = sel.select_move(legal_moves, logits, 0.6, psyche)
                assert move in legal_moves
