"""
feature_engineering.py
----------------------
Constructs advanced features:
- Career vs current season performance metrics (recency weighting)
- Age group and peak-age scaling
- Club reputation tier encoding
- Position-specific efficiency rates (e.g., goals per 90 mins, save percentage)
"""

import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Takes the cleaned DataFrame and constructs robust features for Machine Learning modeling.
    """
    df = df.copy()

    # 1. Age Factor & Peak Indicators
    # Peak football age is generally 24-28. Create distance from peak and age groups.
    df["age_squared"] = df["age"] ** 2
    df["is_prime_age"] = df["age"].apply(lambda x: 1 if 23 <= x <= 28 else 0)
    df["is_veteran"] = df["age"].apply(lambda x: 1 if x >= 30 else 0)
    df["is_youngster"] = df["age"].apply(lambda x: 1 if x <= 22 else 0)

    # 2. Per-90 Minute Adjustment Rates (Normalizing stats by playing time)
    # Avoid division by zero
    minutes = df["minutes_played"].replace(0, 90)
    df["goals_per_90"] = (df["goals"] * 90) / minutes
    df["assists_per_90"] = (df["assists"] * 90) / minutes
    df["shots_per_90"] = (df["shots"] * 90) / minutes

    # 3. Goalkeeper specific metrics
    shots_faced = df["shots_faced"].replace(0, 1)
    df["save_percentage"] = (df["saves"] / shots_faced) * 100

    # 4. Career vs Current Performance Weighting
    # Combine current stats and career consistency without double counting
    # Recency weight: 60% current season performance rate, 40% career rate
    career_m = df["career_matches"].replace(0, 1)
    career_g_per_match = df["career_goals"] / career_m
    current_g_per_match = df["goals"] / df["matches"].replace(0, 1)
    df["blended_goal_rate"] = (0.6 * current_g_per_match) + (0.4 * career_g_per_match)

    career_a_per_match = df["career_assists"] / career_m
    current_a_per_match = df["assists"] / df["matches"].replace(0, 1)
    df["blended_assist_rate"] = (0.6 * current_a_per_match) + (0.4 * career_a_per_match)

    # 5. Position One-Hot Encoding
    position_dummies = pd.get_dummies(df["position_category"], prefix="pos")
    df = pd.concat([df, position_dummies], axis=1)

    # 6. Target Log Transformation (Market values in football are highly skewed)
    df["log_market_value"] = np.log1p(df["market_value_eur"])

    print("Feature engineering completed successfully.")
    return df

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    df = load_and_preprocess_data()
    df_feat = engineer_features(df)
    print(df_feat[['name', 'age', 'club_tier', 'goals_per_90', 'log_market_value']].head())
