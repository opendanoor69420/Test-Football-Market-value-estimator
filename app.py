"""
app.py
------
Interactive Streamlit Web Application for the Football Player Market Value Estimator.
Features transparent two-stage valuation breakdown (ML core + PR/Club calibration),
position-specific feature importance graphs, and dataset preview.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from prediction import predict_market_value_detailed, load_trained_model, get_position_specific_importances
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
st.markdown('<p class="sub-header">Two-Stage AI Valuation System: ML Performance + PR & Club Calibration</p>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("📋 Player Profile Information")
player_name = st.sidebar.text_input("Player Name", value="Jude Bellingham")
age = st.sidebar.slider("Age", 16, 40, 21)

club = st.sidebar.text_input("Club Name", value="Real Madrid")

club_level_desc = {
    "1: Champions League level": 1,
    "2: Europa League level": 2,
    "3: Conference League level": 3,
    "4: Mid-table": 4,
    "5: Low-mid table (survival-focused)": 5,
    "6: Relegation-level teams": 6
}
selected_club_level_label = st.sidebar.selectbox("Club Competitive Level", list(club_level_desc.keys()), index=0)
club_competitive_level = club_level_desc[selected_club_level_label]

st.sidebar.markdown("---")
st.sidebar.header("⭐ Player Reputation (PR)")
st.sidebar.markdown("""
**PR Scale (1–10):**
* **1–3:** Max €45M ceiling
* **4:** Transition (€30M–€60M)
* **5–6:** Band (€40M–€70M)
* **7–8:** Band (€60M–€90M)
* **9–10:** Min €100M+ floor
""")
player_reputation = st.sidebar.slider("Player Current Reputation (PR)", 1, 10, 9)

position_category = st.sidebar.selectbox("Position Category", ["Attackers", "Midfielders", "Defenders", "Goalkeepers"], index=1)

subpos_options = {
    "Attackers": ["ST", "CF", "LW", "RW"],
    "Midfielders": ["CAM", "CM", "CDM", "LM", "RM"],
    "Defenders": ["CB", "LB", "RB", "LWB", "RWB"],
    "Goalkeepers": ["GK"]
}
subposition = st.sidebar.selectbox("Specific Role", subpos_options[position_category], index=0)

st.sidebar.markdown("---")
st.sidebar.header("📊 Performance Statistics")

matches = st.sidebar.slider("Matches Played", 1, 38, 32)
minutes_played = matches * st.sidebar.slider("Avg Minutes per Match", 45, 90, 85)

stats = {
    "matches": matches,
    "minutes_played": minutes_played,
    "goals": 0, "assists": 0, "xg": 0.0, "xa": 0.0,
    "shots": 0, "shots_on_target": 0, "touches_in_box": 0,
    "big_chances_created": 0, "crosses": 0, "successful_crosses": 0,
    "clean_sheets": 0, "tackles": 0, "interceptions": 0, "blocks": 0,
    "clearances": 0, "aerial_duels_won": 0, "progressive_passes": 0,
    "saves": 0, "shots_faced": 0, "goals_conceded": 0, "psxg_plus_minus": 0.0,
    "career_matches": matches + 150,
    "career_goals": 45,
    "career_assists": 35
}

if position_category == "Attackers":
    stats["goals"] = st.sidebar.number_input("Goals", 0, 50, 22)
    stats["assists"] = st.sidebar.number_input("Assists", 0, 30, 10)
    stats["xg"] = st.sidebar.number_input("Expected Goals (xG)", 0.0, 50.0, 19.5)
    stats["xa"] = st.sidebar.number_input("Expected Assists (xA)", 0.0, 30.0, 8.2)
    stats["shots"] = st.sidebar.number_input("Total Shots", 0, 150, 88)
    stats["shots_on_target"] = st.sidebar.number_input("Shots on Target", 0, 80, 42)
    stats["touches_in_box"] = st.sidebar.number_input("Touches in Penalty Box", 0, 300, 145)
    stats["crosses"] = st.sidebar.number_input("Total Crosses", 0, 150, 40)
    stats["successful_crosses"] = st.sidebar.number_input("Successful Crosses", 0, 60, 14)

elif position_category == "Midfielders":
    stats["goals"] = st.sidebar.number_input("Goals", 0, 30, 15)
    stats["assists"] = st.sidebar.number_input("Assists", 0, 30, 9)
    stats["tackles"] = st.sidebar.number_input("Tackles Won", 0, 100, 48)
    stats["interceptions"] = st.sidebar.number_input("Interceptions", 0, 80, 28)
    stats["progressive_passes"] = st.sidebar.number_input("Progressive Passes", 0, 200, 130)
    stats["crosses"] = st.sidebar.number_input("Total Crosses", 0, 150, 35)
    stats["successful_crosses"] = st.sidebar.number_input("Successful Crosses", 0, 60, 12)

elif position_category == "Defenders":
    stats["goals"] = st.sidebar.number_input("Goals", 0, 15, 3)
    stats["assists"] = st.sidebar.number_input("Assists", 0, 20, 5)
    stats["clean_sheets"] = st.sidebar.number_input("Clean Sheets", 0, 25, 16)
    stats["tackles"] = st.sidebar.number_input("Tackles Won", 0, 120, 75)
    stats["interceptions"] = st.sidebar.number_input("Interceptions", 0, 100, 58)
    stats["clearances"] = st.sidebar.number_input("Clearances", 0, 150, 92)
    stats["aerial_duels_won"] = st.sidebar.number_input("Aerial Duels Won", 0, 100, 55)
    stats["progressive_passes"] = st.sidebar.number_input("Progressive Passes", 0, 150, 95)

elif position_category == "Goalkeepers":
    stats["clean_sheets"] = st.sidebar.number_input("Clean Sheets", 0, 25, 15)
    stats["shots_faced"] = st.sidebar.number_input("Shots Faced", 0, 200, 120)
    stats["saves"] = st.sidebar.number_input("Total Saves", 0, 180, 96)

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
    st.write(f"**Role:** {subposition} ({position_category})")
    st.write(f"**Age:** {age} years old")
    st.write(f"**Player Reputation (PR):** {player_reputation}/10")
    st.write(f"**Matches Played:** {matches}")

    if st.button("🚀 Estimate Market Value", type="primary", use_container_width=True):
        breakdown = predict_market_value_detailed(player_payload)
        st.session_state["breakdown"] = breakdown
        st.session_state["evaluated_player"] = player_name

with col2:
    st.subheader("💰 Valuation Result & Transparent Breakdown")
    if "breakdown" in st.session_state:
        bd = st.session_state["breakdown"]
        final_val = bd["calibrated_market_value_eur"]
        
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Final Calibrated Market Value</h3>
                <h1 style="color: #2563EB;">€{final_val:,.2f}</h1>
                <p><b>{st.session_state.get('evaluated_player', player_name)}</b> | PR: {bd['pr_score']}/10 | Level: {bd['club_competitive_level']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.expander("🔍 View Two-Stage Valuation Breakdown", expanded=True):
            st.write(f"* **Stage 1 (Raw ML Performance Prediction):** €{bd['original_ml_prediction']:,.2f}")
            st.write(f"* **Stage 2A (Player Reputation Calibration):** {bd['pr_note']}")
            st.write(f"* **Stage 2B (Club Competitive Level Multiplier):** {bd['club_multiplier']}x ({selected_club_level_label})")
            st.write(f"* **Final Output:** €{final_val:,.2f}")
    else:
        st.info("👈 Enter player details in the sidebar and click **Estimate Market Value**.")

st.markdown("---")
st.subheader("📈 Position-Specific Model Feature Importance")

selected_inspect_cat = st.selectbox("Inspect Position Priorities", ["Attackers", "Midfielders", "Defenders", "Goalkeepers"], index=1)
selected_inspect_subpos = st.selectbox("Inspect Specific Role", subpos_options[selected_inspect_cat], index=0)

imp_df = get_position_specific_importances(selected_inspect_cat, selected_inspect_subpos)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="#1E3A8A")
ax.set_xlabel("Relative Importance Weight")
ax.set_title(f"Model Feature Priorities for Role: {selected_inspect_subpos} ({selected_inspect_cat})")
st.pyplot(fig)

st.markdown("---")
st.subheader("🗂️ Sample Dataset Preview (660 Players)")

if st.checkbox("Show Dataset Preview Table"):
    try:
        df_preview = load_and_preprocess_data()
        st.success(f"Successfully loaded dataset with total players: **{len(df_preview)}**")
        display_cols = [c for c in ['name', 'position_category', 'subposition', 'age', 'club', 'club_competitive_level', 'player_reputation', 'market_value_eur'] if c in df_preview.columns]
        st.dataframe(df_preview[display_cols].head(25), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading dataset preview: {e}")
