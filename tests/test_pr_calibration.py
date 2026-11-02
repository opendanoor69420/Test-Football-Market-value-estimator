"""
tests/test_pr_calibration.py
-----------------------------
Automated unit tests validating PR calibration rules, hard ceilings, floors,
club level multipliers, boundary conditions, and non-negative finite outputs.
"""

import pytest
from pr_calibration import calibrate_valuation

def test_scenario_1_pr2_ucl_ceiling():
    """Scenario 1: PR 2, excellent statistics, Champions League-level club -> max €45M."""
    res = calibrate_valuation(
        ml_prediction_eur=90_000_000.0,
        pr_score=2,
        club_competitive_level=1,
        performance_score=0.95
    )
    assert res["calibrated_market_value_eur"] <= 45_000_000.0
    assert res["pr_score"] == 2

def test_scenario_2_pr5_mid_table():
    """Scenario 2: PR 5, average statistics, mid-table club -> within €40M-€70M."""
    res = calibrate_valuation(
        ml_prediction_eur=50_000_000.0,
        pr_score=5,
        club_competitive_level=4,
        performance_score=0.50
    )
    assert 40_000_000.0 <= res["calibrated_market_value_eur"] <= 70_000_000.0

def test_scenario_3_pr7_uel():
    """Scenario 3: PR 7, excellent statistics, Europa League-level club -> within €60M-€90M."""
    res = calibrate_valuation(
        ml_prediction_eur=75_000_000.0,
        pr_score=7,
        club_competitive_level=2,
        performance_score=0.85
    )
    assert 60_000_000.0 <= res["calibrated_market_value_eur"] <= 90_000_000.0

def test_scenario_4_pr9_mid_table_floor():
    """Scenario 4: PR 9, average statistics, mid-table club -> minimum €100M."""
    res = calibrate_valuation(
        ml_prediction_eur=60_000_000.0,
        pr_score=9,
        club_competitive_level=4,
        performance_score=0.30
    )
    assert res["calibrated_market_value_eur"] >= 100_000_000.0

def test_scenario_5_pr10_ucl_elite():
    """Scenario 5: PR 10, excellent statistics, Champions League club -> considerably higher than €100M."""
    res = calibrate_valuation(
        ml_prediction_eur=150_000_000.0,
        pr_score=10,
        club_competitive_level=1,
        performance_score=0.99
    )
    assert res["calibrated_market_value_eur"] > 120_000_000.0

def test_scenario_6_pr3_ucl_ceiling():
    """Scenario 6: PR 3, excellent statistics, Champions League-level club -> must not exceed €45M."""
    res = calibrate_valuation(
        ml_prediction_eur=120_000_000.0,
        pr_score=3,
        club_competitive_level=1,
        performance_score=1.0
    )
    assert res["calibrated_market_value_eur"] <= 45_000_000.0

def test_pr_slider_sensitivity():
    """Ensure changing PR slider produces meaningful valuation changes for identical stats."""
    res_pr4 = calibrate_valuation(50_000_000.0, pr_score=4, club_competitive_level=3, performance_score=0.6)
    res_pr8 = calibrate_valuation(50_000_000.0, pr_score=8, club_competitive_level=3, performance_score=0.6)
    assert res_pr8["calibrated_market_value_eur'] > res_pr4["calibrated_market_value_eur"]

def test_non_negative_and_finite():
    """Ensure predictions are always finite and non-negative."""
    res = calibrate_valuation(0.0, pr_score=1, club_competitive_level=6, performance_score=0.0)
    assert res["calibrated_market_value_eur"] >= 0.0
    assert not np.isnan(res["calibrated_market_value_eur"])
