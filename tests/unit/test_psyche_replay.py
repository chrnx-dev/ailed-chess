"""Tests for psyche replay side-normalized metrics."""

from __future__ import annotations

from ailed_chess.psyche.analysis import compute_side_normalized_psyche


def test_compute_side_normalized_psyche_reports_expected_deltas() -> None:
    # Arrange / Act
    result = compute_side_normalized_psyche(
        white_mean_psyche=-40.0,
        black_mean_psyche=-60.0,
        overall_mean_psyche=-50.0,
    )

    # Assert
    assert result["overall_vs_white_baseline"] == -10.0
    assert result["overall_vs_black_baseline"] == 10.0
    assert result["white_black_gap"] == 20.0
