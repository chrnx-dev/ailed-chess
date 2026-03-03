"""Psyche calculator for emotional state tracking.

This module implements the psyche calculation system that evaluates chess positions
and updates the AI's emotional state based on position factors like material,
king safety, mobility, center control, and pieces under attack.
"""

import math

import chess
from typing import Optional

from ailed.psyche.config import PsycheConfig


class PsycheCalculator:
    """Calculate psyche (emotional state) from chess position evaluation.

    The calculator evaluates various position factors and combines them using
    configurable weights to produce a psyche delta. The psyche score represents
    the AI's emotional state ranging from -100 (most negative) to +100 (most positive).

    Position factors evaluated:
    - Material difference (piece values)
    - King safety (pawn shield + nearby attackers)
    - Mobility (legal moves available)
    - Center control (control of d4, d5, e4, e5)
    - Pieces under attack

    The calculator is stateless - psyche values are passed in and returned.
    """

    # Piece values for material calculation (standard chess values)
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,  # King has no material value
    }

    # Center squares for center control calculation
    CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]

    def __init__(self, config: Optional[PsycheConfig] = None):
        """Initialize psyche calculator.

        Args:
            config: Configuration for psyche calculation. If None, uses default config.
        """
        self.config = config or PsycheConfig()

    def _calculate_material_difference(self, board: chess.Board) -> int:
        """Calculate material difference from current player's perspective.

        Uses standard piece values: pawn=1, knight/bishop=3, rook=5, queen=9.

        Args:
            board: Current board position.

        Returns:
            Material difference (positive = current player ahead, negative = behind).
        """
        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = self.PIECE_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        # Return from current player's perspective
        if board.turn == chess.WHITE:
            return white_material - black_material
        else:
            return black_material - white_material

    def _calculate_king_safety(self, board: chess.Board) -> float:
        """Calculate king safety from current player's perspective.

        Evaluates pawn shield and nearby enemy attackers.

        Args:
            board: Current board position.

        Returns:
            King safety score (positive = safer, negative = more danger).
        """
        current_color = board.turn
        king_square = board.king(current_color)

        if king_square is None:
            return 0  # No king (shouldn't happen in valid position)

        safety_score = 0

        # Evaluate pawn shield (pawns near king)
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)

        # Check for pawn shield in front of king
        for file_offset in [-1, 0, 1]:
            shield_file = king_file + file_offset
            if 0 <= shield_file <= 7:
                # Check rank in front of king (depends on color)
                if current_color == chess.WHITE:
                    shield_rank = king_rank + 1
                else:
                    shield_rank = king_rank - 1

                if 0 <= shield_rank <= 7:
                    shield_square = chess.square(shield_file, shield_rank)
                    piece = board.piece_at(shield_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == current_color:
                        safety_score += 1  # Pawn shield bonus

        # Count enemy attackers near king
        enemy_attackers = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != current_color:
                # Check if this enemy piece attacks squares near king
                attacks = board.attacks(square)
                # Check king square and adjacent squares
                for file_offset in [-1, 0, 1]:
                    for rank_offset in [-1, 0, 1]:
                        nearby_file = king_file + file_offset
                        nearby_rank = king_rank + rank_offset
                        if 0 <= nearby_file <= 7 and 0 <= nearby_rank <= 7:
                            nearby_square = chess.square(nearby_file, nearby_rank)
                            if nearby_square in attacks:
                                enemy_attackers += 1
                                break  # Count this piece only once

        safety_score -= enemy_attackers

        return safety_score

    def _calculate_mobility(self, board: chess.Board) -> int:
        """Calculate mobility difference from current player's perspective.

        Mobility is the number of legal moves available.

        Args:
            board: Current board position.

        Returns:
            Mobility difference (positive = more mobile, negative = less mobile).
        """
        # Count legal moves for current player
        current_mobility = board.legal_moves.count()

        # Cannot use null move if in check (null moves are illegal in check)
        # In this case, we can't easily calculate opponent mobility without
        # making an actual move, so we return just the current mobility
        if board.is_check():
            return current_mobility

        # Count legal moves for opponent (temporarily switch sides)
        board.push(chess.Move.null())  # Make a null move to switch sides
        opponent_mobility = board.legal_moves.count()
        board.pop()  # Undo null move

        return current_mobility - opponent_mobility

    def _calculate_center_control(self, board: chess.Board) -> int:
        """Calculate center control from current player's perspective.

        Counts pieces and pawns controlling the center squares (d4, d5, e4, e5).

        Args:
            board: Current board position.

        Returns:
            Center control difference (positive = more control, negative = less control).
        """
        current_color = board.turn
        current_control = 0
        opponent_control = 0

        for center_square in self.CENTER_SQUARES:
            # Count attackers of this center square
            attackers = board.attackers(current_color, center_square)
            current_control += len(attackers)

            opponent_attackers = board.attackers(not current_color, center_square)
            opponent_control += len(opponent_attackers)

        return current_control - opponent_control

    def _calculate_pieces_under_attack(self, board: chess.Board) -> int:
        """Calculate number of current player's pieces under attack.

        Args:
            board: Current board position.

        Returns:
            Number of current player's pieces that are attacked by opponent.
        """
        current_color = board.turn
        pieces_attacked = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == current_color:
                # Check if opponent attacks this square
                if board.is_attacked_by(not current_color, square):
                    pieces_attacked += 1

        return pieces_attacked

    # Starting non-pawn, non-king material for both sides combined
    # 4N(3) + 4B(3) + 4R(5) + 2Q(9) = 12 + 12 + 20 + 18 = 62
    STARTING_NONPAWN_MATERIAL = 62

    def _detect_game_phase(self, board: chess.Board) -> str:
        """Detect current game phase from board state.

        Opening ends when EITHER the move limit is exceeded OR significant
        non-pawn material has been exchanged. This prevents damped opening
        reactivity from masking rapid early collapses.

        Args:
            board: Current board position.

        Returns:
            One of "opening", "middlegame", or "endgame".
        """
        # Count total non-pawn, non-king material for both sides
        total_material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type not in (chess.PAWN, chess.KING):
                total_material += self.PIECE_VALUES[piece.piece_type]

        if total_material <= self.config.endgame_material_threshold:
            return "endgame"

        material_lost = self.STARTING_NONPAWN_MATERIAL - total_material
        if (
            board.fullmove_number <= self.config.opening_move_limit
            and material_lost < self.config.opening_material_loss_threshold
        ):
            return "opening"

        return "middlegame"

    def _has_compensation(self, material_eval: float, positional_eval: float) -> bool:
        """Detect if a material deficit has positional compensation (sacrifice).

        A sacrifice is indicated when material is negative but positional factors
        are positive — the player gave up material for an attack or initiative.

        Args:
            material_eval: Weighted material evaluation (negative = down material).
            positional_eval: Sum of all non-material positional factors.

        Returns:
            True if sacrifice with compensation detected.
        """
        return material_eval < 0 and positional_eval > 0

    def _compute_effective_reactivity(
        self, current_psyche: float, target_psyche: float, phase: str
    ) -> float:
        """Compute effective reactivity based on game phase and resilience.

        Phase factors scale reactivity (opening dampens, endgame amplifies).
        Resilience makes psyche drops slower than recovery.

        Args:
            current_psyche: Current psyche value.
            target_psyche: Target psyche from position evaluation.
            phase: Game phase ("opening", "middlegame", "endgame").

        Returns:
            Effective reactivity value.
        """
        phase_factors = {
            "opening": self.config.opening_reactivity_factor,
            "middlegame": 1.0,
            "endgame": self.config.endgame_reactivity_factor,
        }
        phase_factor = phase_factors[phase]

        # Resilience applies symmetrically — psyche resists change in either direction
        direction_factor = 1.0 - self.config.resilience

        return self.config.reactivity * phase_factor * direction_factor

    def update_psyche(self, current_psyche: float, board: chess.Board) -> float:
        """Update psyche based on current board position.

        Evaluates all position factors and combines them using configured weights
        to produce a psyche delta. Applies sacrifice dampening, game-phase-aware
        reactivity, and resilience before clamping to bounds.

        Args:
            current_psyche: Current psyche value.
            board: Current board position to evaluate.

        Returns:
            New psyche value after update, clamped to [min_psyche, max_psyche].
        """
        # Calculate all position factors
        material = self._calculate_material_difference(board)
        king_safety = self._calculate_king_safety(board)
        mobility = self._calculate_mobility(board)
        center_control = self._calculate_center_control(board)
        pieces_attacked = self._calculate_pieces_under_attack(board)

        # Check penalty
        check_bonus = self.config.check_penalty if board.is_check() else 0

        # Split into material and positional evaluations
        material_eval = material * self.config.material_weight
        positional_eval = (
            king_safety * self.config.king_safety_weight +
            -pieces_attacked * self.config.pieces_attacked_weight +
            center_control * self.config.center_control_weight +
            mobility * self.config.mobility_weight +
            check_bonus
        )

        # Sacrifice dampening: reduce material penalty when positional compensation exists
        if self._has_compensation(material_eval, positional_eval):
            material_eval *= (1.0 - self.config.sacrifice_dampening)

        raw_eval = material_eval + positional_eval

        # Target-blend with tanh compression
        target_psyche = math.tanh(raw_eval / self.config.position_scale) * 100.0

        # Phase-aware reactivity with resilience
        phase = self._detect_game_phase(board)
        effective_reactivity = self._compute_effective_reactivity(
            current_psyche, target_psyche, phase
        )

        new_psyche = current_psyche + (target_psyche - current_psyche) * effective_reactivity
        new_psyche = max(self.config.min_psyche, min(self.config.max_psyche, new_psyche))

        return new_psyche

    def apply_daily_decay(self, current_psyche: float) -> float:
        """Apply daily decay to move psyche toward neutral (0).

        Decay is percentage-based: new_psyche = current_psyche * (1 - decay_rate)

        Args:
            current_psyche: Current psyche value.

        Returns:
            New psyche value after decay.
        """
        return current_psyche * (1 - self.config.decay_rate)
