"""
app.py
------
Interactive Streamlit Web Application for the Football Player Market Value Estimator.
Features club competitive levels, Player Reputation (PR) scores (1-10), dynamic statistics inputs,
and comparable player insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from prediction import predict_market_value, load_trained_model
from data_preprocessing import load_and_preprocess_data

st.set_page_config(
    page_title="Football Player Market Value Estimator",
    page_icon="⚽",
    layout="wide"
)

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
st.markdown('<p class="sub-header">AI & Machine Learning Powered Transfer Valuation System (660 Players Database)</p>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("📋 Player Profile Information")
player_name = st.sidebar.text_input("Player Name", value="Kaoru Mitoma")
age = st.sidebar.slider("Age", 16, 40, 26)

club = st.sidebar.text_input("Club Name", value="Brighton & Hove Albion")

# Club Competitive Level Selection
club_level_desc = {
    "1: Champions League level": 1,
    "2: Europa League level": 2,
    "3: Conference League level": 3,
    "4: Mid-table": 4,
    "5: Low-mid table (survival-focused)": 5,
    "6: Relegation-level teams": 6
}
selected_club_level_label = st.sidebar.selectbox("Club Competitive Level", list(club_level_desc.keys()), index=3)
club_competitive_level = club_level_desc[selected_club_level_label]

# Player Current Reputation (PR) score 1-10
st.sidebar.markdown("---")
st.sidebar.header("⭐ Player Reputation (PR)")
st.sidebar.markdown("""
**PR Scale (1–10):**
* **1:** Unknown / Academy
* **2–3:** Low reputation
* **4:** Below-average
* **5:** Average professional
* **6:** Established good player
* **7:** Highly regarded
* **8:** Top-level player
* **9:** World-class
* **10:** Top 5 in the world
""")
player_reputation = st.sidebar.slider("Player Current Reputation (PR)", 1, 10, 7)

position_category = st.sidebar.selectbox("Position Category", ["Attackers", "Midfielders", "Defenders", "Goalkeepers"])

subpos_options = {
    "Attackers": ["ST", "CF", "LW", "RW"],
    "Midfielders": ["CAM", "CM", "CDM", "LM", "RM"],
    "Defenders": ["CB", "LB", "RB", "LWB", "RWB"],
    "Goalkeepers": ["GK"]
}
subposition = st.sidebar.selectbox("Specific Role", subpos_options[position_category])

st.sidebar.markdown("---")
st.sidebar.header("📊 Performance Statistics")

matches = st.sidebar.slider("Matches Played", 1, 38, 25)
minutes_played = matches * st.sidebar.slider("Avg Minutes per Match", 45, 90, 80)

stats = {
    "matches": matches,
    "minutes_played": minutes_played,
    "goals": 0, "assists": 0, "xg": 0.0, "xa": 0.0,
    "shots": 0, "shots_on_target": 0, "touches_in_box": 0,
    "clean_sheets": 0, "tackles": 0, "interceptions": 0, "blocks": 0,
    "saves": 0, "shots_faced": 0,
    "career_matches": matches + 120,
    "career_goals": 25,
    "career_assists": 20
}

if position_category == "Attackers":
    stats["goals"] = st.sidebar.number_input("Goals", 0, 50, 3)
    stats["assists"] = st.sidebar.number_input("Assists", 0, 30, 1)
    stats["xg"] = st.sidebar.number_input("Expected Goals (xG)", 0.0, 50.0, 4.2)
    stats["xa"] = st.sidebar.number_input("Expected Assists (xA)", 0.0, 30.0, 2.1)
    stats["shots"] = st.sidebar.number_input("Total Shots", 0, 150, 35)
    stats["shots_on_target"] = st.sidebar.number_input("Shots on Target", 0, 80, 14)
    stats["touches_in_box"] = st.sidebar.number_input("Touches in Penalty Box", 0, 300, 85)

elif position_category == "Midfielders":
    stats["goals"] = st.sidebar.number_input("Goals", 0, 30, 4)
    stats["assists"] = st.sidebar.number_input("Assists", 0, 30, 6)
    stats["xg"] = st.sidebar.number_input("Expected Goals (xG)", 0.0, 20.0, 3.8)
    stats["xa"] = st.sidebar.number_input("Expected Assists (xA)", 0.0, 25.0, 5.2)
    stats["tackles"] = st.sidebar.number_input("Tackles Won", 0, 100, 32)
    stats["interceptions"] = st.sidebar.number_input("Interceptions", 0, 80, 22)

elif position_category == "Defenders":
    stats["goals"] = st.sidebar.number_input("Goals", 0, 15, 2)
    stats["assists"] = st.sidebar.number_input("Assists", 0, 20, 3)
    stats["clean_sheets"] = st.sidebar.number_input("Clean Sheets", 0, 25, 10)
    stats["tackles"] = st.sidebar.number_input("Tackles Won", 0, 120, 55)
    stats["interceptions"] = st.sidebar.number_input("Interceptions", 0, 100, 42)
    stats["blocks"] = st.sidebar.number_input("Blocks", 0, 80, 28)

elif position_category == "Goalkeepers":
    stats["clean_sheets"] = st.sidebar.number_input("Clean Sheets", 0, 25, 12)
    stats["shots_faced"] = st.sidebar.number_input("Shots Faced", 0, 200, 95)
    stats["saves"] = st.sidebar.number_input("Total Saves", 0, 180, 74)

player_payload = {
    "name": player_name,
    "age": age,
    "club": club,
    "club_competitive_level": club_competitive_level,
    "player_reputation": player_reputation,
    "position_category": position_category,
    "subposition": subposition,
    **stats
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📌 Selected Player Summary")
    st.write(f"**Player Name:** {player_name}")
    st.write(f"**Club:** {club} ({selected_club_level_label})")
    st.write(f"**Position:** {subposition} ({position_category})")
    st.write(f"**Age:** {age} years old")
    st.write(f"**Player Reputation (PR):** {player_reputation}/10")
    st.write(f"**Matches Played:** {matches}")

    if st.button("🚀 Estimate Market Value", type="primary", use_container_width=True):
        predicted_val = predict_market_value(player_payload)
        st.session_state["predicted_val"] = predicted_val
        st.session_state["evaluated_player"] = player_name
        st.session_state["evaluated_pr"] = player_reputation

with col2:
    st.subheader("💰 Valuation Result")
    if "predicted_val" in st.session_state:
        val = st.session_state["predicted_val"]
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Estimated Transfer Value</h3>
                <h1 style="color: #2563EB;">€{val:,.2f}</h1>
                <p><b>{st.session_state.get('evaluated_player', player_name)}</b> | Age: {age} | PR: {st.session_state.get('evaluated_pr', player_reputation)}/10 | {subposition}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.info("💡 **Model Explanation:** The estimation balances current season statistics, age curve positioning, Player Reputation (PR), and club competitive tier without letting club prestige or reputation disproportionately override actual on-field performance.")
    else:
        st.info("👈 Enter player details in the sidebar and click **Estimate Market Value**.")

st.markdown("---")
st.subheader("📈 Model Feature Importance")

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
    st.warning("Feature importance graph requires tree-based model.")

if st.checkbox("Show Sample Dataset Preview (660 Players)"):
    df_preview = load_and_preprocess_data()
    st.dataframe(df_preview[['name', 'position_category', 'subposition', 'age', 'club', 'club_competitive_level', 'player_reputation', 'market_value_eur']].head(10))
