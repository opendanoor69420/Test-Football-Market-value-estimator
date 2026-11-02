"""
app.py
------
Interactive Streamlit Web Application for the Football Player Market Value Estimator.
Features dynamic position-specific inputs, clean football-themed UI, and prediction explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from prediction import predict_market_value, load_trained_model
from data_preprocessing import load_and_preprocess_data

# Page Configuration
st.set_page_config(
    page_title="Football Player Market Value Estimator",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚽ Football Player Market Value Estimator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI & Machine Learning Powered Transfer Valuation System</p>', unsafe_allow_html=True)

# Sidebar for Club & Player Identity
st.sidebar.header("📋 Player Profile Information")
player_name = st.sidebar.text_input("Player Name", value="Marcus Rashford")
age = st.sidebar.slider("Age", 16, 40, 24)

club = st.sidebar.selectbox("Club", [
    "Real Madrid", "Barcelona", "Manchester United", "Manchester City", 
    "Liverpool", "Bayern Munich", "Paris Saint-Germain", "Arsenal", 
    "Chelsea", "Tottenham", "Juventus", "AC Milan", "Brighton", "Aston Villa", "Wolves"
])

club_tier_map = {
    "Real Madrid": 5, "Barcelona": 5, "Manchester City": 5, "Bayern Munich": 5,
    "Paris Saint-Germain": 5, "Liverpool": 5, "Arsenal": 4, "Manchester United": 4,
    "Chelsea": 4, "Tottenham": 4, "Juventus": 4, "AC Milan": 4,
    "Brighton": 3, "Aston Villa": 3, "Wolves": 2
}
club_tier = club_tier_map.get(club, 3)

position_category = st.sidebar.selectbox("Position Category", ["Attackers", "Midfielders", "Defenders", "Goalkeepers"])

# Dynamic Subpositions
subpos_options = {
    "Attackers": ["ST", "CF", "LW", "RW"],
    "Midfielders": ["CAM", "CM", "CDM", "LM", "RM"],
    "Defenders": ["CB", "LB", "RB", "LWB", "RWB"],
    "Goalkeepers": ["GK"]
}
subposition = st.sidebar.selectbox("Specific Role", subpos_options[position_category])

st.sidebar.markdown("---")
st.sidebar.header("📊 Performance Statistics")

# Dynamic Statistics Input Based on Position Category
matches = st.sidebar.slider("Matches Played", 1, 38, 30)
minutes_played = matches * st.sidebar.slider("Avg Minutes per Match", 45, 90, 85)

# Initialize stats dictionary
stats = {
    "matches": matches,
    "minutes_played": minutes_played,
    "goals": 0, "assists": 0, "xg": 0.0, "xa": 0.0,
    "shots": 0, "shots_on_target": 0, "touches_in_box": 0,
    "clean_sheets": 0, "tackles": 0, "interceptions": 0, "blocks": 0,
    "saves": 0, "shots_faced": 0,
    "career_matches": matches + 100,
    "career_goals": 20,
    "career_assists": 15
}

if position_category == "Attackers":
    stats["goals"] = st.sidebar.number_input("Goals", 0, 50, 18)
    stats["assists"] = st.sidebar.number_input("Assists", 0, 30, 7)
    stats["xg"] = st.sidebar.number_input("Expected Goals (xG)", 0.0, 50.0, 15.2)
    stats["xa"] = st.sidebar.number_input("Expected Assists (xA)", 0.0, 30.0, 5.4)
    stats["shots"] = st.sidebar.number_input("Total Shots", 0, 150, 85)
    stats["shots_on_target"] = st.sidebar.number_input("Shots on Target", 0, 80, 42)
    stats["touches_in_box"] = st.sidebar.number_input("Touches in Penalty Box", 0, 300, 125)
    stats["career_goals"] = stats["goals"] * 3
    stats["career_assists"] = stats["assists"] * 3

elif position_category == "Midfielders":
    stats["goals"] = st.sidebar.number_input("Goals", 0, 30, 6)
    stats["assists"] = st.sidebar.number_input("Assists", 0, 30, 10)
    stats["xg"] = st.sidebar.number_input("Expected Goals (xG)", 0.0, 20.0, 5.1)
    stats["xa"] = st.sidebar.number_input("Expected Assists (xA)", 0.0, 25.0, 8.8)
    stats["tackles"] = st.sidebar.number_input("Tackles Won", 0, 100, 45)
    stats["interceptions"] = st.sidebar.number_input("Interceptions", 0, 80, 30)
    stats["career_goals"] = stats["goals"] * 4
    stats["career_assists"] = stats["assists"] * 3

elif position_category == "Defenders":
    stats["goals"] = st.sidebar.number_input("Goals", 0, 15, 3)
    stats["assists"] = st.sidebar.number_input("Assists", 0, 20, 4)
    stats["clean_sheets"] = st.sidebar.number_input("Clean Sheets", 0, 25, 14)
    stats["tackles"] = st.sidebar.number_input("Tackles Won", 0, 120, 65)
    stats["interceptions"] = st.sidebar.number_input("Interceptions", 0, 100, 55)
    stats["blocks"] = st.sidebar.number_input("Blocks", 0, 80, 35)
    stats["career_goals"] = stats["goals"] * 4

elif position_category == "Goalkeepers":
    stats["clean_sheets"] = st.sidebar.number_input("Clean Sheets", 0, 25, 15)
    stats["shots_faced"] = st.sidebar.number_input("Shots Faced", 0, 200, 110)
    stats["saves"] = st.sidebar.number_input("Total Saves", 0, 180, 88)

# Construct full input payload
player_payload = {
    "name": player_name,
    "age": age,
    "club": club,
    "club_tier": club_tier,
    "position_category": position_category,
    "subposition": subposition,
    **stats
}

# Main Panel layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📌 Selected Player Summary")
    st.write(f"**Player Name:** {player_name}")
    st.write(f"**Club:** {club} (Tier {club_tier}/5)")
    st.write(f"**Position:** {subposition} ({position_category})")
    st.write(f"**Age:** {age} years old")
    st.write(f"**Matches Played:** {matches}")

    if st.button("🚀 Estimate Market Value", type="primary", use_container_width=True):
        predicted_val = predict_market_value(player_payload)
        st.session_state["predicted_val"] = predicted_val
        st.session_state["evaluated_player"] = player_name

with col2:
    st.subheader("💰 Valuation Result")
    if "predicted_val" in st.session_state:
        val = st.session_state["predicted_val"]
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Estimated Transfer Value</h3>
                <h1 style="color: #2563EB;">€{val:,.2f}</h1>
                <p><b>{st.session_state.get('evaluated_player', player_name)}</b> | Age: {age} | {subposition}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("👈 Enter player details in the sidebar and click **Estimate Market Value**.")

# Additional Analytics Section
st.markdown("---")
st.subheader("📈 Model Insights & Factors")

try:
    bundle = load_trained_model()
    model = bundle["model"]
    features = bundle["features"]
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
        feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False).head(8)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(feat_imp_df["Feature"][::-1], feat_imp_df["Importance"][::-1], color="#2563EB")
        ax.set_xlabel("Relative Importance")
        ax.set_title("Top Features Influencing Market Valuations")
        st.pyplot(fig)
except Exception as e:
    st.warning("Feature importance graph requires trained tree-based model.")

# Dataset Reference Section
if st.checkbox("Show Sample Dataset Preview (~330 Players)"):
    df_preview = load_and_preprocess_data()
    st.dataframe(df_preview[['name', 'position_category', 'subposition', 'age', 'club', 'market_value_eur']].head(10))
