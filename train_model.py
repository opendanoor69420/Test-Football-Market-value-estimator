"""
train_model.py
--------------
Trains position-aware regression models (stratified by position category: Attackers, Midfielders, Defenders, Goalkeepers)
plus a global ensemble, evaluating MAE, RMSE, and R² Score, and saving robust model bundles.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_and_preprocess_data
from feature_engineering import engineer_features

def train_and_evaluate_models(data_path="data/raw/players.csv", model_output_path="models/trained_model.joblib"):
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    df = load_and_preprocess_data(data_path)
    df = engineer_features(df)

    # Comprehensive feature set
    features = [
        "age", "age_squared", "club_competitive_level", "player_reputation",
        "matches", "goals", "assists", "xg", "xa", "shots", "shots_on_target", 
        "touches_in_box", "big_chances_created", "crosses", "successful_crosses",
        "minutes_played", "career_matches", "career_goals", "career_assists", 
        "clean_sheets", "tackles", "interceptions", "blocks", "clearances", 
        "aerial_duels_won", "progressive_passes", "passes_into_final_third",
        "saves", "shots_faced", "goals_conceded", "psxg_plus_minus",
        "goals_per_90", "assists_per_90", "shots_per_90", "crosses_per_90",
        "tackles_per_90", "interceptions_per_90", "clearances_per_90",
        "save_percentage", "blended_goal_rate", "blended_assist_rate",
        "is_prime_age", "is_veteran", "is_youngster"
    ]

    pos_cols = [c for c in df.columns if c.startswith("pos_") or c.startswith("subpos_")]
    features.extend(pos_cols)

    X = df[features]
    y = df["log_market_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Position-Aware Models per Category + Global Model
    position_models = {}
    categories = ["Attackers", "Midfielders", "Defenders", "Goalkeepers"]

    print("\n--- Training Position-Aware Stratified Models ---")
    for cat in categories:
        cat_mask = df.loc[X_train.index, f"pos_{cat}"] == 1
        if cat_mask.sum() > 10:
            X_cat_train = X_train[cat_mask]
            y_cat_train = y_train[cat_mask]
            
            cat_model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
            cat_model.fit(X_cat_train, y_cat_train)
            position_models[cat] = cat_model
            print(f"Trained specialized model for [{cat}] with {cat_mask.sum()} training samples.")

    # Train Global Ensemble Model (Random Forest & XGBoost)
    global_model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    global_model.fit(X_train, y_train)

    preds_log = global_model.predict(X_test)
    preds_eur = np.expm1(preds_log)
    y_test_eur = np.expm1(y_test)

    mae = mean_absolute_error(y_test_eur, preds_eur)
    rmse = np.sqrt(mean_squared_error(y_test_eur, preds_eur))
    r2 = r2_score(y_test_eur, preds_eur)

    print(f"\n[Global Random Forest Evaluation]")
    print(f"  MAE:  €{mae:,.2f}")
    print(f"  RMSE: €{rmse:,.2f}")
    print(f"  R²:   {r2:.4f}\n")

    # Save model bundle
    model_bundle = {
        "global_model": global_model,
        "position_models": position_models,
        "features": features,
        "model_name": "Position-Aware Random Forest Ensemble"
    }
    joblib.dump(model_bundle, model_output_path)
    print(f"Trained model bundle successfully saved to {model_output_path}")

    return global_model, features

if __name__ == "__main__":
    train_and_evaluate_models()
