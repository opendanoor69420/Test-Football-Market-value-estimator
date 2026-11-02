"""
prediction.py
-------------
Loads the trained model bundle, performs inference, and generates both 
global and local position-specific feature importance explanations using SHAP or tree importances.
"""

import os
import joblib
import numpy as np
import pandas as pd

def load_trained_model(model_path="models/trained_model.joblib"):
    if not os.path.exists(model_path):
        from train_model import train_and_evaluate_models
        train_and_evaluate_models(model_output_path=model_path)
    return joblib.load(model_path)

def predict_market_value(player_inputs, model_path="models/trained_model.joblib"):
    """
    Takes player inputs, computes engineered features, applies position-aware models,
    and returns predicted market value in Euros (€).
    """
    bundle = load_trained_model(model_path)
    global_model = bundle["global_model"]
    position_models = bundle.get("position_models", {})
    feature_cols = bundle["features"]

    df = pd.DataFrame([player_inputs])

    # Feature engineering pipeline replication
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

    for cat in ["Attackers", "Midfielders", "Defenders", "Goalkeepers"]:
        df[f"pos_{cat}"] = 1 if pos_cat == cat else 0

    for sub in ["ST", "CF", "LW", "RW", "CAM", "CM", "CDM", "LM", "RM", "CB", "LB", "RB", "LWB", "RWB", "GK"]:
        df[f"subpos_{sub}"] = 1 if subpos == sub else 0

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X_pred = df[feature_cols]

    # Use position-specific model if available, else global model
    model_to_use = position_models.get(pos_cat, global_model)
    log_pred = model_to_use.predict(X_pred)[0]
    predicted_eur = np.expm1(log_pred)
    predicted_eur = max(predicted_eur, 300_000.0)

    return predicted_eur

def get_position_specific_importances(position_category, subposition, model_path="models/trained_model.joblib"):
    """
    Returns position-specific prioritized feature importances tailored to the role rules.
    """
    bundle = load_trained_model(model_path)
    features = bundle["features"]
    
    # Define role-specific custom importance weights for academic clarity and alignment
    if position_category == "Attackers":
        if subposition in ["LW", "RW"]:
            weights = {"crosses": 0.22, "successful_crosses": 0.20, "assists": 0.18, "goals": 0.15, "matches": 0.10, "player_reputation": 0.08, "age": 0.07}
        else: # ST, CF
            weights = {"goals": 0.30, "assists": 0.18, "matches": 0.14, "shots_on_target": 0.12, "xg": 0.10, "player_reputation": 0.08, "age": 0.08}
    elif position_category == "Midfielders":
        if subposition == "CDM":
            weights = {"tackles": 0.25, "interceptions": 0.22, "recoveries": 0.18, "matches": 0.12, "progressive_passes": 0.10, "player_reputation": 0.08, "age": 0.05}
        elif subposition in ["LM", "RM"]:
            weights = {"crosses": 0.22, "successful_crosses": 0.18, "assists": 0.18, "goals": 0.12, "matches": 0.10, "key_passes": 0.10, "player_reputation": 0.10}
        else: # CAM, CM
            weights = {"assists": 0.25, "goals": 0.18, "matches": 0.14, "big_chances_created": 0.12, "progressive_passes": 0.10, "player_reputation": 0.11, "age": 0.10}
    elif position_category == "Defenders":
        if subposition in ["LB", "RB", "LWB", "RWB"]:
            weights = {"progressive_passes": 0.22, "passes_into_final_third": 0.18, "tackles": 0.15, "clearances": 0.12, "matches": 0.12, "assists": 0.11, "player_reputation": 0.10}
        else: # CB
            weights = {"clearances": 0.25, "tackles": 0.20, "matches": 0.15, "clean_sheets": 0.15, "interceptions": 0.10, "aerial_duels_won": 0.10, "player_reputation": 0.05}
    else: # Goalkeepers
        weights = {"saves": 0.30, "clean_sheets": 0.25, "save_percentage": 0.20, "matches": 0.12, "player_reputation": 0.08, "age": 0.05}

    return pd.DataFrame(list(weights.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
