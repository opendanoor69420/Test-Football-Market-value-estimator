"""
prediction.py
-------------
Loads the trained model bundle and performs inference on user inputs,
handling feature consistency and preventing unrealistic predictions.
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
    Takes a dictionary of raw player inputs, computes engineered features,
    and returns the predicted market value in Euros (€).
    """
    bundle = load_trained_model(model_path)
    model = bundle["model"]
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

    shots_faced = max(df.get("shots_faced", [1]).values[0], 1)
    saves = df.get("saves", [0]).values[0]
    df["save_percentage"] = (saves / shots_faced) * 100

    career_m = max(df.get("career_matches", [30]).values[0], 1)
    career_g = df.get("career_goals", [10]).values[0]
    career_a = df.get("career_assists", [10]).values[0]

    df["blended_goal_rate"] = (0.6 * (df["goals"].values[0] / max(df["matches"].values[0], 1))) + (0.4 * (career_g / career_m))
    df["blended_assist_rate"] = (0.6 * (df["assists"].values[0] / max(df["matches"].values[0], 1))) + (0.4 * (career_a / career_m))

    pos_cat = df["position_category"].values[0]
    for cat in ["Attackers", "Midfielders", "Defenders", "Goalkeepers"]:
        df[f"pos_{cat}"] = 1 if pos_cat == cat else 0

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X_pred = df[feature_cols]

    log_pred = model.predict(X_pred)[0]
    predicted_eur = np.expm1(log_pred)

    predicted_eur = max(predicted_eur, 300_000.0)

    return predicted_eur
