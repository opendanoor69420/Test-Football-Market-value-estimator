"""
feature_engineering.py
----------------------
Constructs position-aware features and metrics:
- Role-specific interaction features (e.g. winger crossing efficiency, CDM ball-winning index)
- Career vs current season performance metrics
- Age group and peak-age scaling
- Club competitive level encoding
- Player Current Reputation (PR) feature inclusion
"""

import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Takes cleaned DataFrame and constructs robust features for position-aware modeling.
    """
    df = df.copy()

    # 1. Age Factor & Peak Indicators
    df["age_squared"] = df["age"] ** 2
    df["is_prime_age"] = df["age"].apply(lambda x: 1 if 23 <= x <= 28 else 0)
    df["is_veteran"] = df["age"].apply(lambda x: 1 if x >= 30 else 0)
    df["is_youngster"] = df["age"].apply(lambda x: 1 if x <= 22 else 0)

    # 2. Per-90 Minute Adjustment Rates
    minutes = df["minutes_played"].replace(0, 90)
    df["goals_per_90"] = (df["goals"] * 90) / minutes
    df["assists_per_90"] = (df["assists"] * 90) / minutes
    df["shots_per_90"] = (df["shots"] * 90) / minutes
    df["crosses_per_90"] = (df.get("crosses", pd.Series([0]*len(df))) * 90) / minutes
    df["tackles_per_90"] = (df.get("tackles", pd.Series([0]*len(df))) * 90) / minutes
    df["interceptions_per_90"] = (df.get("interceptions", pd.Series([0]*len(df))) * 90) / minutes
    df["clearances_per_90"] = (df.get("clearances", pd.Series([0]*len(df))) * 90) / minutes

    # 3. Goalkeeper specific metrics
    shots_faced = df["shots_faced"].replace(0, 1)
    df["save_percentage"] = (df["saves"] / shots_faced) * 100

    # 4. Career vs Current Performance Weighting (Recency weighted)
    career_m = df["career_matches"].replace(0, 1)
    career_g_per_match = df["career_goals"] / career_m
    current_g_per_match = df["goals"] / df["matches"].replace(0, 1)
    df["blended_goal_rate"] = (0.6 * current_g_per_match) + (0.4 * career_g_per_match)

    career_a_per_match = df["career_assists"] / career_m
    current_a_per_match = df["assists"] / df["matches"].replace(0, 1)
    df["blended_assist_rate"] = (0.6 * current_a_per_match) + (0.4 * career_a_per_match)

    # 5. Position One-Hot Encoding & Subposition flags
    position_dummies = pd.get_dummies(df["position_category"], prefix="pos")
    df = pd.concat([df, position_dummies], axis=1)

    # Specific subposition flags for position-aware weighting
    for sub in ["ST", "CF", "LW", "RW", "CAM", "CM", "CDM", "LM", "RM", "CB", "LB", "RB", "LWB", "RWB", "GK"]:
        df[f"subpos_{sub}"] = df["subposition"].apply(lambda x: 1 if x == sub else 0)

    # 6. Target Log Transformation
    df["log_market_value"] = np.log1p(df["market_value_eur"])

    print("Position-aware feature engineering completed successfully.")
    return df

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    df = load_and_preprocess_data()
    df_feat = engineer_features(df)
    print(df_feat[['name', 'subposition', 'club_competitive_level', 'player_reputation', 'log_market_value']].head())
