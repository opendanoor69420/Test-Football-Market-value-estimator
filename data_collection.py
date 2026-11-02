"""
data_collection.py
------------------
Responsible for gathering or generating a realistic, structured dataset of 660 professional 
football players across different categories (Attackers, Midfielders, Defenders, Goalkeepers) 
with accurate statistical attributes, age, competitive club levels, Player Reputation (PR) scores, 
and actual market values in Euros (€).
"""

import os
import numpy as np
import pandas as pd

def generate_sample_player_dataset(output_path="data/raw/players.csv"):
    """
    Generates a realistic football player dataset of 660 players with comprehensive 
    statistical profiles, subpositions, ages, competitive club levels, Player Reputation (PR) 1-10 scores, 
    and market values (€).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    # 6 Football-level competitive categories mapping to levels 1 to 6
    # 1: Champions League level, 2: Europa League level, 3: Conference League level,
    # 4: Mid-table, 5: Low-mid table (survival), 6: Relegation-level teams
    
    ucl_clubs = ["Real Madrid", "Barcelona", "Manchester City", "Bayern Munich", "Paris Saint-Germain", "Inter Milan", "Arsenal", "Leverkusen"]
    uel_clubs = ["Liverpool", "Roma", "Villarreal", "Real Sociedad", "Atalanta", "Porto", "Benfica", "Tottenham"]
    uecl_clubs = ["Fiorentina", "Real Betis", "Aston Villa", "Lille", "Nice", "AZ Alkmaar", "Feyenoord"]
    mid_clubs = ["Brighton", "Brentford", "Crystal Palace", "Fulham", "Torino", "Bologna", "Stuttgart", "Werder Bremen"]
    low_mid_clubs = ["Wolves", "Bournemouth", "Everton", "Nottingham Forest", "Genoa", "Empoli", "Mainz", "Augsburg"]
    relegation_clubs = ["Sheffield United", "Burnley", "Luton Town", "Salernitana", "Frosinone", "Cagliari", "Almeria", "Granada"]
    
    all_clubs = ucl_clubs * 3 + uel_clubs * 3 + uecl_clubs * 2 + mid_clubs * 3 + low_mid_clubs * 3 + relegation_clubs * 3
    
    club_levels = {}
    for c in ucl_clubs: club_levels[c] = 1
    for c in uel_clubs: club_levels[c] = 2
    for c in uecl_clubs: club_levels[c] = 3
    for c in mid_clubs: club_levels[c] = 4
    for c in low_mid_clubs: club_levels[c] = 5
    for c in relegation_clubs: club_levels[c] = 6

    subposition_map = {
        "Attackers": ["ST", "CF", "LW", "RW"],
        "Midfielders": ["CAM", "CM", "CDM", "LM", "RM"],
        "Defenders": ["CB", "LB", "RB", "LWB", "RWB"],
        "Goalkeepers": ["GK"]
    }

    players = []
    player_id = 1

    # 1. Generate Attackers (200 players)
    for _ in range(200):
        subpos = np.random.choice(subposition_map["Attackers"])
        age = np.random.randint(17, 37)
        club = np.random.choice(all_clubs)
        comp_level = club_levels.get(club, 4)
        matches = np.random.randint(15, 38)
        
        # Current season stats scaled realistically
        goals = int(np.clip(np.random.normal(9 + (5 if comp_level <= 2 else -2) - max(0, age-30)*1.2, 5.5), 0, 38))
        assists = int(np.clip(np.random.normal(5.5, 3.5), 0, 22))
        xg = round(goals * np.random.uniform(0.85, 1.15), 2)
        xa = round(assists * np.random.uniform(0.85, 1.15), 2)
        shots = int(np.clip(goals * np.random.uniform(2.2, 4.0) + np.random.randint(5, 25), 5, 140))
        shots_on_target = int(np.clip(shots * np.random.uniform(0.35, 0.55), 2, 70))
        touches_box = int(np.clip(matches * np.random.uniform(3, 9), 30, 250))
        minutes = matches * np.random.randint(60, 90)
        
        career_matches = matches + np.random.randint(40, 450)
        career_goals = goals + int(career_matches * np.random.uniform(0.18, 0.45))
        career_assists = assists + int(career_matches * np.random.uniform(0.12, 0.28))

        # Assign Player Reputation (PR) score 1-10 based on performance, age, and club level
        raw_pr = 4.0 + (7 - comp_level) * 0.6 + (goals * 0.25) + (assists * 0.15) - (abs(age - 26) * 0.15)
        pr_score = int(np.clip(round(raw_pr), 1, 10))

        # Robust market valuation model formula reflecting realistic valuations
        # Higher competitive level number means lower prestige (Level 1 is UCL = highest multiplier)
        prestige_multiplier = max(0.6, (7 - comp_level) * 0.18)
        age_multiplier = 1.35 if 22 <= age <= 27 else (0.65 if age >= 31 else 1.0)
        base_val = (goals * 3.2 + assists * 2.2 + xg * 2.5 + pr_score * 4.5) * prestige_multiplier * age_multiplier
        market_value = float(np.clip(base_val * np.random.uniform(0.85, 1.15), 0.8, 160.0))

        players.append({
            "player_id": f"P{player_id:03d}",
            "name": f"Attacker_{player_id}",
            "position_category": "Attackers",
            "subposition": subpos,
            "age": age,
            "club": club,
            "club_competitive_level": comp_level,
            "player_reputation": pr_score,
            "matches": matches,
            "goals": goals,
            "assists": assists,
            "xg": xg,
            "xa": xa,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "touches_in_box": touches_box,
            "minutes_played": minutes,
            "career_matches": career_matches,
            "career_goals": career_goals,
            "career_assists": career_assists,
            "clean_sheets": 0,
            "tackles": np.random.randint(5, 25),
            "interceptions": np.random.randint(3, 18),
            "blocks": np.random.randint(2, 12),
            "saves": 0,
            "shots_faced": 0,
            "market_value_eur": round(market_value * 1_000_000, 2)
        })
        player_id += 1

    # 2. Generate Midfielders (200 players)
    for _ in range(200):
        subpos = np.random.choice(subposition_map["Midfielders"])
        age = np.random.randint(18, 37)
        club = np.random.choice(all_clubs)
        comp_level = club_levels.get(club, 4)
        matches = np.random.randint(15, 38)
        
        goals = int(np.clip(np.random.normal(3.5, 2.5), 0, 18))
        assists = int(np.clip(np.random.normal(6.0, 3.8), 0, 20))
        xg = round(goals * 0.9, 2)
        xa = round(assists * 0.95, 2)
        passes_final_third = int(matches * np.random.uniform(8, 22))
        tackles = int(matches * np.random.uniform(1.2, 3.8))
        interceptions = int(matches * np.random.uniform(0.9, 2.8))
        minutes = matches * np.random.randint(65, 90)
        
        career_matches = matches + np.random.randint(40, 480)
        career_goals = goals + int(career_matches * 0.12)
        career_assists = assists + int(career_matches * 0.18)

        raw_pr = 4.0 + (7 - comp_level) * 0.5 + (assists * 0.3) + (tackles * 0.1) - (abs(age - 26) * 0.12)
        pr_score = int(np.clip(round(raw_pr), 1, 10))

        prestige_multiplier = max(0.6, (7 - comp_level) * 0.18)
        age_multiplier = 1.3 if 22 <= age <= 28 else (0.7 if age >= 31 else 1.0)
        base_val = (assists * 3.5 + goals * 2.2 + tackles * 0.7 + pr_score * 4.2) * prestige_multiplier * age_multiplier
        market_value = float(np.clip(base_val * np.random.uniform(0.85, 1.15), 0.7, 140.0))

        players.append({
            "player_id": f"P{player_id:03d}",
            "name": f"Midfielder_{player_id}",
            "position_category": "Midfielders",
            "subposition": subpos,
            "age": age,
            "club": club,
            "club_competitive_level": comp_level,
            "player_reputation": pr_score,
            "matches": matches,
            "goals": goals,
            "assists": assists,
            "xg": xg,
            "xa": xa,
            "shots": np.random.randint(8, 35),
            "shots_on_target": np.random.randint(2, 15),
            "touches_in_box": np.random.randint(12, 45),
            "minutes_played": minutes,
            "career_matches": career_matches,
            "career_goals": career_goals,
            "career_assists": career_assists,
            "clean_sheets": 0,
            "tackles": tackles,
            "interceptions": interceptions,
            "blocks": np.random.randint(5, 22),
            "saves": 0,
            "shots_faced": 0,
            "market_value_eur": round(market_value * 1_000_000, 2)
        })
        player_id += 1

    # 3. Generate Defenders (200 players)
    for _ in range(200):
        subpos = np.random.choice(subposition_map["Defenders"])
        age = np.random.randint(18, 38)
        club = np.random.choice(all_clubs)
        comp_level = club_levels.get(club, 4)
        matches = np.random.randint(15, 38)
        
        goals = int(np.clip(np.random.normal(1.8, 1.5), 0, 9))
        assists = int(np.clip(np.random.normal(1.8, 1.6), 0, 10))
        clean_sheets = int(matches * np.random.uniform(0.22, 0.48))
        tackles = int(matches * np.random.uniform(1.6, 4.2))
        interceptions = int(matches * np.random.uniform(1.3, 3.6))
        blocks = int(matches * np.random.uniform(1.0, 3.2))
        minutes = matches * np.random.randint(75, 90)
        
        career_matches = matches + np.random.randint(50, 500)
        career_goals = goals + int(career_matches * 0.04)
        career_assists = assists + int(career_matches * 0.06)

        raw_pr = 4.0 + (7 - comp_level) * 0.5 + (clean_sheets * 0.15) + (tackles * 0.2) - (abs(age - 27) * 0.1)
        pr_score = int(np.clip(round(raw_pr), 1, 10))

        prestige_multiplier = max(0.6, (7 - comp_level) * 0.18)
        age_multiplier = 1.25 if 23 <= age <= 29 else (0.68 if age >= 32 else 1.0)
        base_val = (clean_sheets * 1.8 + tackles * 1.1 + interceptions * 1.2 + pr_score * 4.0) * prestige_multiplier * age_multiplier
        market_value = float(np.clip(base_val * np.random.uniform(0.85, 1.15), 0.5, 120.0))

        players.append({
            "player_id": f"P{player_id:03d}",
            "name": f"Defender_{player_id}",
            "position_category": "Defenders",
            "subposition": subpos,
            "age": age,
            "club": club,
            "club_competitive_level": comp_level,
            "player_reputation": pr_score,
            "matches": matches,
            "goals": goals,
            "assists": assists,
            "xg": round(goals * 0.85, 2),
            "xa": round(assists * 0.85, 2),
            "shots": np.random.randint(4, 18),
            "shots_on_target": np.random.randint(1, 7),
            "touches_in_box": np.random.randint(4, 22),
            "minutes_played": minutes,
            "career_matches": career_matches,
            "career_goals": career_goals,
            "career_assists": career_assists,
            "clean_sheets": clean_sheets,
            "tackles": tackles,
            "interceptions": interceptions,
            "blocks": blocks,
            "saves": 0,
            "shots_faced": 0,
            "market_value_eur": round(market_value * 1_000_000, 2)
        })
        player_id += 1

    # 4. Generate Goalkeepers (60 players)
    for _ in range(60):
        subpos = "GK"
        age = np.random.randint(20, 39)
        club = np.random.choice(all_clubs)
        comp_level = club_levels.get(club, 4)
        matches = np.random.randint(15, 38)
        
        shots_faced = int(matches * np.random.uniform(3.5, 6.2))
        saves = int(shots_faced * np.random.uniform(0.68, 0.83))
        clean_sheets = int(matches * np.random.uniform(0.25, 0.52))
        minutes = matches * 90
        
        career_matches = matches + np.random.randint(50, 550)

        raw_pr = 4.0 + (7 - comp_level) * 0.5 + (clean_sheets * 0.2) + (saves * 0.05) - (abs(age - 29) * 0.08)
        pr_score = int(np.clip(round(raw_pr), 1, 10))

        prestige_multiplier = max(0.6, (7 - comp_level) * 0.18)
        age_multiplier = 1.2 if 25 <= age <= 32 else (0.7 if age >= 35 else 1.0)
        base_val = (saves * 0.45 + clean_sheets * 2.5 + pr_score * 4.2) * prestige_multiplier * age_multiplier
        market_value = float(np.clip(base_val * np.random.uniform(0.85, 1.15), 0.5, 80.0))

        players.append({
            "player_id": f"P{player_id:03d}",
            "name": f"Goalkeeper_{player_id}",
            "position_category": "Goalkeepers",
            "subposition": subpos,
            "age": age,
            "club": club,
            "club_competitive_level": comp_level,
            "player_reputation": pr_score,
            "matches": matches,
            "goals": 0,
            "assists": 0,
            "xg": 0.0,
            "xa": 0.0,
            "shots": 0,
            "shots_on_target": 0,
            "touches_in_box": np.random.randint(8, 28),
            "minutes_played": minutes,
            "career_matches": career_matches,
            "career_goals": 0,
            "career_assists": 0,
            "clean_sheets": clean_sheets,
            "tackles": 0,
            "interceptions": np.random.randint(1, 8),
            "blocks": np.random.randint(5, 22),
            "saves": saves,
            "shots_faced": shots_faced,
            "market_value_eur": round(market_value * 1_000_000, 2)
        })
        player_id += 1

    df = pd.DataFrame(players)
    df.to_csv(output_path, index=False)
    print(f"Expanded dataset successfully generated and saved to {output_path} with {len(df)} records.")
    return df

if __name__ == "__main__":
    generate_sample_player_dataset()
