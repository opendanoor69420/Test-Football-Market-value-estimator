"""
pr_calibration.py
------------------
Dedicated calibration module for the Football Player Market Value Estimator.
"""

import numpy as np

PR_BRACKET_LIMITS = {
    (1, 3): {"max_val": 45_000_000.0, "target_band": (5_000_000, 45_000_000)},
    (4, 4): {"transition_band": (30_000_000, 60_000_000)},
    (5, 6): {"target_band": (40_000_000, 70_000_000)},
    (7, 8): {"target_band": (60_000_000, 90_000_000)},
    (9, 10): {"min_val": 100_000_000.0, "target_band": (100_000_000, 220_000_000)}
}

CLUB_LEVEL_MULTIPLIERS = {
    1: 1.12,  # Champions League level
    2: 1.06,  # Europa League level
    3: 1.02,  # Conference League level
    4: 1.00,  # Mid-table
    5: 0.95,  # Low-mid table
    6: 0.90   # Relegation-level teams
}

def calibrate_valuation(ml_prediction_eur: float, pr_score: int, club_competitive_level: int, performance_score: float = 0.5) -> dict:
    pr_score = int(np.clip(pr_score, 1, 10))
    club_competitive_level = int(np.clip(club_competitive_level, 1, 6))
    performance_score = float(np.clip(performance_score, 0.0, 1.0))

    base_val = ml_prediction_eur
    pr_adjustment_note = ""

    if pr_score <= 3:
        ceiling = 45_000_000.0
        calibrated_pr_val = 5_000_000.0 + (performance_score * (ceiling - 5_000_000.0) * 0.7)
        calibrated_pr_val = min(calibrated_pr_val, ceiling)
        pr_adjustment_note = f"PR {pr_score} (Restricted to Hard Maximum €45M)"
    elif pr_score == 4:
        low_b, high_b = 30_000_000.0, 60_000_000.0
        calibrated_pr_val = low_b + (performance_score * (high_b - low_b))
        calibrated_pr_val = 0.5 * calibrated_pr_val + 0.5 * np.clip(base_val, low_b, high_b)
        pr_adjustment_note = "PR 4 (Transition Bracket €30M–€60M)"
    elif 5 <= pr_score <= 6:
        low_b, high_b = 40_000_000.0, 70_000_000.0
        scaled_perf = low_b + (performance_score * (high_b - low_b))
        calibrated_pr_val = 0.6 * base_val + 0.4 * scaled_perf
        calibrated_pr_val = np.clip(calibrated_pr_val, low_b, high_b)
        pr_adjustment_note = f"PR {pr_score} (Target Band €40M–€70M)"
    elif 7 <= pr_score <= 8:
        low_b, high_b = 60_000_000.0, 90_000_000.0
        scaled_perf = low_b + (performance_score * (high_b - low_b))
        calibrated_pr_val = 0.6 * base_val + 0.4 * scaled_perf
        calibrated_pr_val = np.clip(calibrated_pr_val, low_b, high_b)
        pr_adjustment_note = f"PR {pr_score} (Target Band €60M–€90M)"
    else:
        floor_val = 100_000_000.0
        elite_scaling = floor_val + (performance_score * 80_000_000.0) + (max(0, base_val - floor_val) * 0.5)
        calibrated_pr_val = max(floor_val, elite_scaling)
        pr_adjustment_note = f"PR {pr_score} (World-Class Minimum €100M+ Floor)"

    club_multiplier = CLUB_LEVEL_MULTIPLIERS.get(club_competitive_level, 1.00)
    club_adjusted_val = calibrated_pr_val * club_multiplier

    if pr_score <= 3:
        final_val = min(club_adjusted_val, 45_000_000.0)
    elif pr_score >= 9:
        final_val = max(club_adjusted_val, 100_000_000.0)
    else:
        final_val = club_adjusted_val

    final_val = float(np.clip(final_val, 500_000.0, 300_000_000.0))

    return {
        "original_ml_prediction": round(base_val, 2),
        "pr_score": pr_score,
        "pr_note": pr_adjustment_note,
        "club_competitive_level": club_competitive_level,
        "club_multiplier": club_multiplier,
        "calibrated_market_value_eur": round(final_val, 2)
    }
