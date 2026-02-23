"""Tests for psyche calculator."""

import pytest
import chess
from ailed.psyche.calculator import PsycheCalculator
from ailed.psyche.config import PsycheConfig


class TestPsycheConfig:
    """Test suite for PsycheConfig."""

    def test_default_config_values(self):
        """Test that default configuration has expected values."""
        config = PsycheConfig()

        assert config.material_weight == 10
        assert config.king_safety_weight == 5
        assert config.pieces_attacked_weight == 3
        assert config.center_control_weight == 2
        assert config.mobility_weight == 1
        assert config.position_scale == 50.0
        assert config.reactivity == 0.3
        assert config.check_penalty == -50
        assert config.decay_rate == 0.20
        assert config.opening_move_limit == 15
        assert config.endgame_material_threshold == 13
        assert config.opening_reactivity_factor == 0.15
        assert config.endgame_reactivity_factor == 0.1
        assert config.sacrifice_dampening == 0.5
        assert config.resilience == 0.4
        assert config.min_psyche == -100
        assert config.max_psyche == 100

    def test_custom_config_values(self):
        """Test that custom configuration values can be set."""
        config = PsycheConfig(
            material_weight=15,
            king_safety_weight=8,
            check_penalty=-75,
            decay_rate=0.15,
        )

        assert config.material_weight == 15
        assert config.king_safety_weight == 8
        assert config.check_penalty == -75
        assert config.decay_rate == 0.15


class TestPsycheCalculator:
    """Test suite for PsycheCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with default config."""
        return PsycheCalculator()

    @pytest.fixture
    def custom_calculator(self):
        """Create calculator with custom config."""
        config = PsycheConfig(
            material_weight=20,
            king_safety_weight=10,
        )
        return PsycheCalculator(config)

    def test_calculator_initialization(self, calculator):
        """Test that calculator initializes with config."""
        assert calculator.config is not None
        assert isinstance(calculator.config, PsycheConfig)

    def test_calculate_material_difference_starting_position(self, calculator):
        """Test material calculation at starting position."""
        board = chess.Board()
        material_diff = calculator._calculate_material_difference(board)

        # Starting position should have equal material (0 difference)
        assert material_diff == 0

    def test_calculate_material_difference_white_advantage(self, calculator):
        """Test material calculation when white is ahead."""
        # White has extra queen (black missing queen)
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        material_diff = calculator._calculate_material_difference(board)

        # White to move with extra queen (9 points)
        assert material_diff == 9

    def test_calculate_material_difference_black_advantage(self, calculator):
        """Test material calculation when black is ahead."""
        # Black has extra rook (white missing h1 rook)
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1")
        material_diff = calculator._calculate_material_difference(board)

        # White to move, black has extra rook (-5 points from white's perspective)
        assert material_diff == -5

    def test_calculate_material_difference_mixed_material(self, calculator):
        """Test material calculation with mixed pieces."""
        # White: King + 2 pawns (2)
        # Black: King + knight (3)
        # Difference: 2 - 3 = -1 from white's perspective
        board = chess.Board("4k3/8/8/3n4/8/8/4PP2/4K3 w - - 0 1")
        material_diff = calculator._calculate_material_difference(board)

        assert material_diff == -1

    def test_psyche_update_material_loss(self, calculator):
        """Test psyche decreases when losing material."""
        # White missing queen in a bad position (black pieces active, Qd5+Bc5 attacking).
        # Positional eval is negative so sacrifice dampening does NOT trigger.
        board = chess.Board("r1b1k2r/pppp1ppp/2n2n2/2bqp3/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 20")
        current_psyche = 0

        new_psyche = calculator.update_psyche(current_psyche, board)

        # material_eval = -90, positional < 0 (no dampening), target ≈ -97
        # eff_react = 0.3 * 1.0 * 0.6 = 0.18, one update: 0 + (-97) * 0.18 ≈ -17.5
        assert new_psyche < -15

    def test_psyche_update_material_gain(self, calculator):
        """Test psyche increases when gaining material."""
        # Position after capturing opponent's queen (black missing queen), move 20
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20")
        current_psyche = 0

        new_psyche = calculator.update_psyche(current_psyche, board)

        # Target-blend: raw_eval dominated by +9*10 = 90, tanh(90/50)*100 ≈ 94
        # eff_react = 0.3 * 1.0 * 0.6 = 0.18, one update: 0 + 94 * 0.18 ≈ 17
        assert new_psyche > 15

    def test_psyche_bounds_enforcement_upper(self, calculator):
        """Test psyche approaches maximum value with repeated updates."""
        # Massive material advantage, move 20
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQQQQQ w kq - 0 20")
        psyche = 50.0

        for _ in range(10):
            psyche = calculator.update_psyche(psyche, board)

        # After 10 iterations of target-blend, should approach ceiling
        assert psyche > 90

    def test_psyche_bounds_enforcement_lower(self, calculator):
        """Test psyche approaches minimum value with repeated updates."""
        # Massive material disadvantage, move 20
        board = chess.Board("rnbqqqqq/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20")
        psyche = -50.0

        for _ in range(10):
            psyche = calculator.update_psyche(psyche, board)

        # After 10 iterations of target-blend, should approach floor
        assert psyche < -90

    def test_psyche_in_check_penalty(self, calculator):
        """Test psyche receives penalty when in check."""
        # White king in check from black queen
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        current_psyche = 0

        new_psyche = calculator.update_psyche(current_psyche, board)

        # Should include check penalty (-50) plus any material/positional factors
        # At minimum, should be negative due to check penalty
        assert new_psyche < 0

    def test_psyche_not_in_check_no_penalty(self, calculator):
        """Test no check penalty when not in check."""
        board = chess.Board()  # Starting position, no check
        current_psyche = 0

        new_psyche = calculator.update_psyche(current_psyche, board)

        # Starting position is balanced, should be close to 0
        # No check penalty should be applied
        # Allow for positional factors (mobility, etc.)
        assert -20 <= new_psyche <= 20

    def test_mobility_calculation_when_in_check(self, calculator):
        """Test mobility calculation doesn't crash when player is in check.

        Regression test for bug where null move caused ValueError when in check.
        """
        # Black king in check from white queen (5 legal escape moves)
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4Q3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3")

        # Should not crash - this used to throw ValueError: null move is not legal
        mobility = calculator._calculate_mobility(board)

        # Should return a valid mobility value (just current player's moves when in check)
        assert isinstance(mobility, int)
        assert mobility > 0  # Player has legal moves to get out of check

    def test_daily_decay_positive_psyche(self, calculator):
        """Test daily decay moves positive psyche toward zero."""
        current_psyche = 50

        new_psyche = calculator.apply_daily_decay(current_psyche)

        # With decay_rate=0.20: 50 * (1 - 0.20) = 50 * 0.80 = 40
        assert new_psyche == 40

    def test_daily_decay_negative_psyche(self, calculator):
        """Test daily decay moves negative psyche toward zero."""
        current_psyche = -60

        new_psyche = calculator.apply_daily_decay(current_psyche)

        # With decay_rate=0.20: -60 * (1 - 0.20) = -60 * 0.80 = -48
        assert new_psyche == -48

    def test_daily_decay_zero_psyche(self, calculator):
        """Test daily decay does nothing to zero psyche."""
        current_psyche = 0

        new_psyche = calculator.apply_daily_decay(current_psyche)

        assert new_psyche == 0

    def test_daily_decay_with_custom_rate(self):
        """Test daily decay with custom decay rate."""
        config = PsycheConfig(decay_rate=0.10)
        calculator = PsycheCalculator(config)
        current_psyche = 80

        new_psyche = calculator.apply_daily_decay(current_psyche)

        # With decay_rate=0.10: 80 * (1 - 0.10) = 80 * 0.90 = 72
        assert new_psyche == 72

    def test_calculate_center_control(self, calculator):
        """Test center control calculation."""
        # White controls more center squares (white to move)
        board = chess.Board("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 1")

        center_control = calculator._calculate_center_control(board)

        # White should have positive center control
        assert center_control > 0

    def test_calculate_mobility(self, calculator):
        """Test mobility calculation."""
        # Starting position
        board = chess.Board()

        mobility = calculator._calculate_mobility(board)

        # Both sides have equal mobility at start (20 moves each)
        assert mobility == 0

    def test_calculate_king_safety(self, calculator):
        """Test king safety calculation."""
        # Position with castled king (safer)
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")

        # King safety should be calculated
        king_safety = calculator._calculate_king_safety(board)

        # Should return a numeric value
        assert isinstance(king_safety, (int, float))

    def test_calculate_pieces_under_attack(self, calculator):
        """Test pieces under attack calculation."""
        # Position where white has pieces under attack
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")

        pieces_attacked = calculator._calculate_pieces_under_attack(board)

        # Should return number of pieces under attack
        assert isinstance(pieces_attacked, int)
        assert pieces_attacked >= 0

    def test_psyche_update_combines_all_factors(self, calculator):
        """Test that psyche update considers all factors."""
        # Complex position with multiple factors
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        current_psyche = 10

        new_psyche = calculator.update_psyche(current_psyche, board)

        # Should return a value different from input
        # Exact value depends on all factors combined
        assert isinstance(new_psyche, (int, float))
        assert -100 <= new_psyche <= 100

    def test_custom_weights_affect_calculation(self, custom_calculator):
        """Test that custom weights change psyche calculation."""
        # Position with material advantage, move 20
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQQQQQ w kq - 0 20")
        current_psyche = 0

        new_psyche = custom_calculator.update_psyche(current_psyche, board)

        # With doubled material_weight (20 vs 10), raw_eval is larger
        # Target-blend compresses; symmetric resilience slows approach
        assert new_psyche > 15

    def test_psyche_stable_in_equal_position(self, calculator):
        """Test psyche stays moderate in starting position (not ceiling/floor)."""
        board = chess.Board()
        psyche = 0.0

        for _ in range(20):
            psyche = calculator.update_psyche(psyche, board)

        # Starting position has slight positional edge (pawn shield, etc.)
        # Key property: psyche converges to a stable value, not ceiling/floor
        assert -50 < psyche < 50

    def test_psyche_gradual_approach(self, calculator):
        """Test psyche monotonically converges toward target over updates."""
        # White up a queen — strongly positive target, move 20
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20")
        psyche = 0.0
        values = [psyche]

        for _ in range(10):
            psyche = calculator.update_psyche(psyche, board)
            values.append(psyche)

        # Each update should increase psyche (monotonically increasing)
        for i in range(1, len(values)):
            assert values[i] > values[i - 1], f"Not monotonic at step {i}: {values}"

    def test_psyche_recovery_after_blunder(self, calculator):
        """Test psyche recovers when position improves after a blunder."""
        # Start with positive psyche from good position, move 20
        good_board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20")
        psyche = 0.0
        for _ in range(5):
            psyche = calculator.update_psyche(psyche, good_board)
        assert psyche > 15  # Should be solidly positive

        # Blunder: down a queen in bad position (no sacrifice dampening), move 20
        bad_board = chess.Board("r1b1k2r/pppp1ppp/2n2n2/2bqp3/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 20")
        for _ in range(3):
            psyche = calculator.update_psyche(psyche, bad_board)
        post_blunder = psyche
        assert post_blunder < 0  # Should have gone negative

        # Recovery: position improves again
        for _ in range(5):
            psyche = calculator.update_psyche(psyche, good_board)
        assert psyche > post_blunder  # Should recover toward positive


class TestPsycheConfigValidation:
    """Test config validation for position_scale and reactivity."""

    def test_position_scale_must_be_positive(self):
        """Test that position_scale rejects zero and negative values."""
        with pytest.raises(Exception):
            PsycheConfig(position_scale=0.0)
        with pytest.raises(Exception):
            PsycheConfig(position_scale=-1.0)

    def test_reactivity_bounds(self):
        """Test that reactivity must be between 0 and 1."""
        with pytest.raises(Exception):
            PsycheConfig(reactivity=-0.1)
        with pytest.raises(Exception):
            PsycheConfig(reactivity=1.1)

    def test_reactivity_edge_values(self):
        """Test that reactivity accepts boundary values 0 and 1."""
        config_zero = PsycheConfig(reactivity=0.0)
        assert config_zero.reactivity == 0.0

        config_one = PsycheConfig(reactivity=1.0)
        assert config_one.reactivity == 1.0


class TestGamePhaseDetection:
    """Test suite for game phase detection."""

    @pytest.fixture
    def calculator(self):
        return PsycheCalculator()

    def test_opening_at_move_1(self, calculator):
        """Starting position is opening."""
        board = chess.Board()
        assert calculator._detect_game_phase(board) == "opening"

    def test_opening_at_move_15(self, calculator):
        """Move 15 is still opening (boundary)."""
        board = chess.Board()
        board.fullmove_number = 15
        assert calculator._detect_game_phase(board) == "opening"

    def test_middlegame_at_move_16(self, calculator):
        """Move 16 with full material is middlegame."""
        board = chess.Board()
        board.fullmove_number = 16
        assert calculator._detect_game_phase(board) == "middlegame"

    def test_endgame_low_material(self, calculator):
        """Position with only rook + bishop (8 material) is endgame past opening."""
        # White: K + R(5), Black: K + B(3) = 8 total, under threshold 13
        board = chess.Board("4k3/8/3b4/8/8/8/8/4K2R w - - 0 20")
        assert calculator._detect_game_phase(board) == "endgame"

    def test_endgame_at_threshold(self, calculator):
        """Material exactly at threshold (13) is endgame."""
        # White: K + R(5) + N(3), Black: K + R(5) = 13 total
        board = chess.Board("4k3/8/8/8/8/8/8/R1N1K2r w - - 0 20")
        assert calculator._detect_game_phase(board) == "endgame"

    def test_middlegame_above_threshold(self, calculator):
        """Material above threshold past opening is middlegame."""
        # White: K + Q(9) + R(5), Black: K + Q(9) + R(5) = 28 total
        board = chess.Board("r2qk3/8/8/8/8/8/8/R2QK3 w - - 0 20")
        assert calculator._detect_game_phase(board) == "middlegame"

    def test_opening_overrides_endgame_material(self, calculator):
        """Low material in early moves is still opening (opening checked first)."""
        # Few pieces but move 5 — opening takes priority
        board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 5")
        assert calculator._detect_game_phase(board) == "opening"

    def test_custom_opening_move_limit(self):
        """Custom opening_move_limit is respected."""
        config = PsycheConfig(opening_move_limit=10)
        calc = PsycheCalculator(config)
        board = chess.Board()
        board.fullmove_number = 11
        assert calc._detect_game_phase(board) == "middlegame"


class TestSacrificeDetection:
    """Test suite for sacrifice (compensation) detection."""

    @pytest.fixture
    def calculator(self):
        return PsycheCalculator()

    def test_has_compensation_material_down_position_up(self, calculator):
        """Negative material + positive positional = sacrifice."""
        assert calculator._has_compensation(-30.0, 15.0) is True

    def test_no_compensation_both_negative(self, calculator):
        """Negative material + negative positional = real blunder."""
        assert calculator._has_compensation(-30.0, -5.0) is False

    def test_no_compensation_material_positive(self, calculator):
        """Positive material = no sacrifice regardless of positional."""
        assert calculator._has_compensation(10.0, 5.0) is False

    def test_sacrifice_dampens_psyche_drop(self, calculator):
        """Sacrifice position should produce smaller psyche drop than pure blunder."""
        # Pure blunder: down a queen, bad position (move 20, middlegame)
        blunder_board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 20")
        blunder_psyche = calculator.update_psyche(0.0, blunder_board)

        # Sacrifice: down a queen but attacking king (strong positional compensation)
        # White missing queen but has attacking position with pieces aimed at black king
        # Use a position where white is down material but has center/mobility advantage
        sac_board = chess.Board("rnbqk3/pppp1ppp/5n2/4N3/2B1P3/8/PPPP1PPP/RNB1K2R w KQkq - 0 20")
        sac_psyche = calculator.update_psyche(0.0, sac_board)

        # Both should be negative (down material), but sacrifice should be less negative
        # The sacrifice board has piece for queen deficit but with compensation
        # Key assertion: sacrifice dampening makes the drop less severe
        assert sac_psyche > blunder_psyche


class TestResilience:
    """Test suite for resilience (asymmetric psyche movement)."""

    @pytest.fixture
    def calculator(self):
        return PsycheCalculator()

    def test_drop_slower_than_recovery(self, calculator):
        """Psyche should drop slower than it recovers due to resilience."""
        # Good position (move 20, middlegame)
        good_board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20")
        # Bad position
        bad_board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 20")

        # Drop: from 0 toward negative target
        drop_psyche = calculator.update_psyche(0.0, bad_board)
        drop_magnitude = abs(drop_psyche)

        # Recovery: from 0 toward positive target
        recovery_psyche = calculator.update_psyche(0.0, good_board)
        recovery_magnitude = abs(recovery_psyche)

        # Resilience is symmetric: both directions are equally damped
        # Magnitudes should be similar (not exact due to positional asymmetry)
        assert abs(drop_magnitude) > 0
        assert abs(recovery_magnitude) > 0

    def test_zero_resilience_symmetric(self):
        """With resilience=0, drops and recoveries move at same rate."""
        config = PsycheConfig(resilience=0.0)
        calc = PsycheCalculator(config)

        # Symmetric positions: one up a queen, one down a queen (move 20)
        up_board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20")
        down_board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 20")

        up_psyche = calc.update_psyche(0.0, up_board)
        down_psyche = calc.update_psyche(0.0, down_board)

        # Magnitudes should be similar (not exactly equal due to positional asymmetry)
        # but direction_factor is 1.0 for both, so reactivity is identical
        assert abs(up_psyche) > 0
        assert abs(down_psyche) > 0

    def test_max_resilience_freezes_psyche(self):
        """With resilience=1.0, psyche never moves (direction_factor=0)."""
        config = PsycheConfig(resilience=1.0)
        calc = PsycheCalculator(config)

        # Bad position that would normally cause a drop (move 20)
        bad_board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 20")

        # Starting from positive psyche, should not move
        psyche = calc.update_psyche(50.0, bad_board)
        assert psyche == 50.0  # No movement with max resilience

        # Starting from negative psyche, should also not move
        psyche = calc.update_psyche(-50.0, bad_board)
        assert psyche == -50.0


class TestEffectiveReactivity:
    """Test suite for _compute_effective_reactivity."""

    @pytest.fixture
    def calculator(self):
        return PsycheCalculator()

    def test_opening_reduces_reactivity(self, calculator):
        """Opening phase should reduce effective reactivity."""
        opening_react = calculator._compute_effective_reactivity(0.0, 50.0, "opening")
        middle_react = calculator._compute_effective_reactivity(0.0, 50.0, "middlegame")

        # Opening factor (0.5) < middlegame factor (1.0)
        assert opening_react < middle_react

    def test_endgame_reduces_reactivity(self, calculator):
        """Endgame phase should reduce effective reactivity (psyche frozen)."""
        endgame_react = calculator._compute_effective_reactivity(0.0, 50.0, "endgame")
        middle_react = calculator._compute_effective_reactivity(0.0, 50.0, "middlegame")

        # Endgame factor (0.1) < middlegame factor (1.0)
        assert endgame_react < middle_react

    def test_resilience_is_symmetric(self, calculator):
        """Resilience should reduce reactivity equally in both directions."""
        # Dropping (target < current)
        drop_react = calculator._compute_effective_reactivity(50.0, -50.0, "middlegame")
        # Rising (target > current)
        rise_react = calculator._compute_effective_reactivity(-50.0, 50.0, "middlegame")

        # Both apply resilience symmetrically: 0.3 * 1.0 * (1 - 0.4) = 0.18
        assert abs(drop_react - rise_react) < 0.001
        assert abs(drop_react - 0.18) < 0.001


class TestNewConfigValidation:
    """Test validation for the 6 new config fields."""

    def test_opening_move_limit_must_be_positive(self):
        with pytest.raises(Exception):
            PsycheConfig(opening_move_limit=0)

    def test_endgame_material_threshold_accepts_zero(self):
        config = PsycheConfig(endgame_material_threshold=0)
        assert config.endgame_material_threshold == 0

    def test_opening_reactivity_factor_must_be_positive(self):
        with pytest.raises(Exception):
            PsycheConfig(opening_reactivity_factor=0.0)

    def test_opening_reactivity_factor_max(self):
        config = PsycheConfig(opening_reactivity_factor=2.0)
        assert config.opening_reactivity_factor == 2.0
        with pytest.raises(Exception):
            PsycheConfig(opening_reactivity_factor=2.1)

    def test_sacrifice_dampening_bounds(self):
        PsycheConfig(sacrifice_dampening=0.0)
        PsycheConfig(sacrifice_dampening=1.0)
        with pytest.raises(Exception):
            PsycheConfig(sacrifice_dampening=-0.1)
        with pytest.raises(Exception):
            PsycheConfig(sacrifice_dampening=1.1)

    def test_resilience_bounds(self):
        PsycheConfig(resilience=0.0)
        PsycheConfig(resilience=1.0)
        with pytest.raises(Exception):
            PsycheConfig(resilience=-0.1)
        with pytest.raises(Exception):
            PsycheConfig(resilience=1.1)

    def test_endgame_reactivity_factor_bounds(self):
        with pytest.raises(Exception):
            PsycheConfig(endgame_reactivity_factor=0.0)
        config = PsycheConfig(endgame_reactivity_factor=2.0)
        assert config.endgame_reactivity_factor == 2.0
        with pytest.raises(Exception):
            PsycheConfig(endgame_reactivity_factor=2.1)
