"""
prediction.py
-------------
Loads the trained model bundle, performs inference, integrates the pr_calibration module,
and generates position-specific feature importance explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pr_calibration import calibrate_valuation

def load_trained_model(model_path="models/trained_model.joblib"):
    if not os.path.exists(model_path):
        from train_model import train_and_evaluate_models
        train_and_evaluate_models(model_output_path=model_path)
    return joblib.load(model_path)

def predict_market_value(player_inputs, model_path="models/trained_model.joblib"):
    breakdown = predict_market_value_detailed(player_inputs, model_path)
    return breakdown["calibrated_market_value_eur"]

def predict_market_value_detailed(player_inputs, model_path="models/trained_model.joblib"):
    bundle = load_trained_model(model_path)
    global_model = bundle.get("global_model", bundle.get("model"))
    position_models = bundle.get("position_models", {})
    feature_cols = bundle["features"]

    df = pd.DataFrame([player_inputs])

    df["age_squared"] = df["age"] ** 2
    df["is_prime_age"] = 1 if 23 <= df["age"].values[0] <= 28 else 0
    df["is_veteran"] = 1 if df["age"].values[0] >= 30 else 0
    df["is_youngster"] = 1 if df["age"].values[0] <= 22 else 0

    minutes = max(df["minutes_played"].values[0], 90)
    df["goals_per_90"] = (df["goals"].values[0] * 90) / minutes
    df["assists_per_90"] = (df["assists"].values[0] * 90) / minutes
    df["shots_per_90"] = (df.get("shots", [0]).values[0] * 90) / minutes
    df["crosses_per_90"] = (df.get("crosses", [0]).values[0] * 90) / minutes
    df["tackles_per_90"] = (df.get("tackles", [0]).values[0] * 90) / minutes
    df["interceptions_per_90"] = (df.get("interceptions", [0]).values[0] * 90) / minutes
    df["clearances_per_90"] = (df.get("clearances", [0]).values[0] * 90) / minutes

    shots_faced = max(df.get("shots_faced", [1]).values[0], 1)
    saves = df.get("saves", [0]).values[0]
    df["save_percentage"] = (saves / shots_faced) * 100

    career_m = max(df.get("career_matches", [30]).values[0], 1)
    career_g = df.get("career_goals", [10]).values[0]
    career_a = df.get("career_assists", [10]).values[0]

    df["blended_goal_rate"] = (0.6 * (df["goals"].values[0] / max(df["matches"].values[0], 1))) + (0.4 * (career_g / career_m))
    df["blended_assist_rate"] = (0.6 * (df["assists"].values[0] / max(df["matches"].values[0], 1))) + (0.4 * (career_a / career_m))

    pos_cat = df["position_category"].values[0]
    subpos = df["subposition"].values[0]
    pr_score = int(df.get("player_reputation", [5]).values[0])
    club_level = int(df.get("club_competitive_level", [4]).values[0])

    for cat in ["Attackers", "Midfielders", "Defenders", "Goalkeepers"]:
        df[f"pos_{cat}"] = 1 if pos_cat == cat else 0

    for sub in ["ST", "CF", "LW", "RW", "CAM", "CM", "CDM", "LM", "RM", "CB", "LB", "RB", "LWB", "RWB", "GK"]:
        df[f"subpos_{sub}"] = 1 if subpos == sub else 0

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X_pred = df[feature_cols]

    model_to_use = position_models.get(pos_cat, global_model)
    log_pred = model_to_use.predict(X_pred)[0]
    raw_ml_eur = np.expm1(log_pred)

    matches_p = max(float(df["matches"].values[0]), 1.0)
    goals_p = float(df["goals"].values[0])
    assists_p = float(df["assists"].values[0])
    tackles_p = float(df.get("tackles", [0]).values[0])
    saves_p = float(df.get("saves", [0]).values[0])
    
    stat_sum = (goals_p * 2.5) + (assists_p * 2.0) + (tackles_p * 0.5) + (saves_p * 0.3)
    performance_score = float(np.clip(stat_sum / (matches_p * 1.5 + 10.0), 0.0, 1.0))

    calibration_result = calibrate_valuation(
        ml_prediction_eur=raw_ml_eur,
        pr_score=pr_score,
        club_competitive_level=club_level,
        performance_score=performance_score
    )

    return calibration_result

def get_position_specific_importances(position_category, subposition, model_path="models/trained_model.joblib"):
    if position_category == "Attackers":
        if subposition in ["LW", "RW"]:
            weights = {"crosses": 0.22, "successful_crosses": 0.20, "assists": 0.18, "goals": 0.15, "player_reputation": 0.12, "matches": 0.08, "age": 0.05}
        else:
            weights = {"goals": 0.28, "player_reputation": 0.18, "assists": 0.15, "matches": 0.12, "shots_on_target": 0.10, "xg": 0.10, "age": 0.07}
    elif position_category == "Midfielders":
        if subposition == "CDM":
            weights = {"tackles": 0.25, "interceptions": 0.20, "player_reputation": 0.18, "recoveries": 0.15, "matches": 0.10, "progressive_passes": 0.07, "age": 0.05}
        elif subposition in ["LM", "RM"]:
            weights = {"crosses": 0.22, "successful_crosses": 0.18, "assists": 0.16, "player_reputation": 0.14, "goals": 0.10, "matches": 0.10, "key_passes": 0.10}
        else:
            weights = {"assists": 0.22, "player_reputation": 0.20, "goals": 0.15, "matches": 0.12, "big_chances_created": 0.12, "progressive_passes": 0.10, "age": 0.09}
    elif position_category == "Defenders":
        if subposition in ["LB", "RB", "LWB", "RWB"]:
            weights = {"progressive_passes": 0.20, "player_reputation": 0.18, "passes_into_final_third": 0.15, "tackles": 0.14, "clearances": 0.11, "matches": 0.11, "assists": 0.11}
        else:
            weights = {"clearances": 0.22, "tackles": 0.20, "player_reputation": 0.18, "matches": 0.12, "clean_sheets": 0.12, "interceptions": 0.10, "aerial_duels_won": 0.06}
    else:
        weights = {"saves": 0.28, "player_reputation": 0.22, "clean_sheets": 0.20, "save_percentage": 0.15, "matches": 0.10, "age": 0.05}

    return pd.DataFrame(list(weights.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
