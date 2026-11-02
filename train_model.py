"""
train_model.py
--------------
Trains multiple regression algorithms (Linear Regression baseline, Random Forest, XGBoost),
evaluates them using MAE, RMSE, and R² Score, and saves the best performing model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_and_preprocess_data
from feature_engineering import engineer_features

def train_and_evaluate_models(data_path="data/raw/players.csv", model_output_path="models/trained_model.joblib"):
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Load and engineer features
    df = load_and_preprocess_data(data_path)
    df = engineer_features(df)

    # Define feature set for training
    features = [
        "age", "age_squared", "club_tier", "matches", "goals", "assists", 
        "xg", "xa", "shots", "shots_on_target", "touches_in_box", "minutes_played",
        "career_matches", "career_goals", "career_assists", "clean_sheets", 
        "tackles", "interceptions", "blocks", "saves", "shots_faced",
        "goals_per_90", "assists_per_90", "shots_per_90", "save_percentage",
        "blended_goal_rate", "blended_assist_rate",
        "is_prime_age", "is_veteran", "is_youngster"
    ]

    # Include position dummy columns if present
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    features.extend(pos_cols)

    X = df[features]
    y = df["log_market_value"] # Train on log-transformed target

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Models to Compare
    models = {
        "Ridge Baseline": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    best_model_name = None
    best_model = None
    best_r2 = -float("inf")

    print("\n--- Model Evaluation Results ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds_log = model.predict(X_test)
        
        # Convert back from log scale for evaluation
        preds_eur = np.expm1(preds_log)
        y_test_eur = np.expm1(y_test)

        mae = mean_absolute_error(y_test_eur, preds_eur)
        rmse = np.sqrt(mean_squared_error(y_test_eur, preds_eur))
        r2 = r2_score(y_test_eur, preds_eur)

        print(f"[{name}]")
        print(f"  MAE:  €{mae:,.2f}")
        print(f"  RMSE: €{rmse:,.2f}")
        print(f"  R²:   {r2:.4f}\n")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    print(f"Best Selected Model: {best_model_name} with R² Score: {best_r2:.4f}")

    # Save best model and feature list bundle
    model_bundle = {
        "model": best_model,
        "features": features,
        "model_name": best_model_name
    }
    joblib.dump(model_bundle, model_output_path)
    print(f"Trained model bundle successfully saved to {model_output_path}")

    return best_model, features

if __name__ == "__main__":
    train_and_evaluate_models()
