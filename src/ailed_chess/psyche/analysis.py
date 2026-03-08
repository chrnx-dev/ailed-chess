"""Analysis helpers for psyche experiment reporting."""

from __future__ import annotations


def compute_side_normalized_psyche(
    *,
    white_mean_psyche: float,
    black_mean_psyche: float,
    overall_mean_psyche: float,
) -> dict[str, float]:
    """Return side-normalized psyche deltas against each side baseline."""

    return {
        "white_baseline": round(white_mean_psyche, 2),
        "black_baseline": round(black_mean_psyche, 2),
        "overall_mean": round(overall_mean_psyche, 2),
        "overall_vs_white_baseline": round(overall_mean_psyche - white_mean_psyche, 2),
        "overall_vs_black_baseline": round(overall_mean_psyche - black_mean_psyche, 2),
        "white_black_gap": round(white_mean_psyche - black_mean_psyche, 2),
    }
