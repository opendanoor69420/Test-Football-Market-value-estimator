"""
data_preprocessing.py
---------------------
Handles data cleaning, missing value imputation, duplicate removal, 
type conversions, and robust data validation.
"""

import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(file_path="data/raw/players.csv"):
    """
    Loads raw CSV data, cleans duplicates, imputes missing statistics intelligently,
    and returns a standardized Pandas DataFrame ready for feature engineering.
    """
    if not os.path.exists(file_path):
        from data_collection import generate_sample_player_dataset
        generate_sample_player_dataset(file_path)

    df = pd.read_csv(file_path)

    # 1. Remove duplicate players
    df.drop_duplicates(subset=["name", "club", "position_category"], inplace=True)

    # 2. Handle missing or NaN values intelligently by position category
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df.groupby("position_category")[col].transform("median"))
            df[col] = df[col].fillna(df[col].median())

    # Ensure non-negative constraints for numerical variables
    for col in ["matches", "goals", "assists", "age", "market_value_eur", "minutes_played", "player_reputation", "club_competitive_level"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    print(f"Data preprocessing complete. Cleaned dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    df_clean = load_and_preprocess_data()
    print(df_clean.head(3))
