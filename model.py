# model.py
"""
Predicting Final League Positions Using Betting Odds & Simulation
Author: Victoria Friss de Kereki
Last updated: 07/03/2026
"""

from datetime import datetime, timedelta
import os
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# -----------------------------
# --- ENVIRONMENT / API KEYS ---
# -----------------------------
load_dotenv("API_KEY.env")

ODDS_API_KEY = os.getenv("ODDS_DATA_API_KEY")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

if ODDS_API_KEY is None:
    raise ValueError("ODDS_DATA_API_KEY not found in API_KEY.env")
if FOOTBALL_API_KEY is None:
    raise ValueError("FOOTBALL_DATA_API_KEY not found in API_KEY.env")

# -----------------------------
# --- CONSTANTS ---
# -----------------------------
YEAR = 2025
SEASONS = [2025, 2024]
TODAY = datetime.utcnow().date()
END_OF_SEASON = TODAY + timedelta(days=365)

LEAGUES_SCRAPE = {
    "ENG.1": "premierleague_england",
    "ITA.1": "seriea_italy",
    "ESP.1": "laliga_spain",
    "GER.1": "bundesliga_germany",
    "FRA.1": "ligue1_france",
}

LEAGUES_ODDS = {
    "soccer_epl": "odds_premierleague_england",
    "soccer_italy_serie_a": "odds_seriea_italy",
    "soccer_spain_la_liga": "odds_laliga_spain",
    "soccer_germany_bundesliga": "odds_bundesliga_germany",
    "soccer_france_ligue_one": "odds_ligue1_france",
}

COMPETITIONS_FOOTBALL = {
    "PL": "premierleague_england",
    "SA": "seriea_italy",
    "PD": "laliga_spain",
    "BL1": "bundesliga_germany",
    "FL1": "ligue1_france",
}

TEAM_NAME_MAPPING = {
    "AAS Roma": "AS Roma",
    "OComo": "Como",
    "B04Bayer Leverkusen": "Bayer Leverkusen",
    "M05Mainz": "Mainz",
    "NLyon": "Lyon",
    "LLille": "Lille",
    "ENice": "Nice",
    "ZMetz": "Metz",
}

MAPPINGS = {
    # Premier League
    "premierleague_england": {
        "Aston Villa FC": "Aston Villa",
        "Brighton and Hove Albion": "Brighton & Hove Albion",
        "Bournemouth": "AFC Bournemouth",
        "Leeds United FC": "Leeds United",
        "Newcastle United FC": "Newcastle United",
        "Crystal Palace FC": "Crystal Palace",
        "Chelsea FC": "Chelsea",
        "Arsenal FC": "Arsenal",
        "Everton FC": "Everton",
        "Burnley FC": "Burnley",
        "Sunderland AFC": "Sunderland",
        "West Ham United FC": "West Ham United",
        "Manchester City FC": "Manchester City",
        "Manchester United FC": "Manchester United",
        "Fulham FC": "Fulham",
        "Liverpool FC": "Liverpool",
        "Brentford FC": "Brentford",
        "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
        "Nottingham Forest FC": "Nottingham Forest",
        "Tottenham Hotspur FC": "Tottenham Hotspur",
    },
    # Serie A
    "seriea_italy": {
        "US Sassuolo Calcio": "Sassuolo",
        "Cagliari Calcio": "Cagliari",
        "Atalanta BC": "Atalanta",
        "SS Lazio": "Lazio",
        "Genoa CFC": "Genoa",
        "Udinese Calcio": "Udinese",
        "FC Internazionale Milano": "Internazionale",
        "Torino FC": "Torino",
        "AC Pisa 1909": "Pisa",
        "ACF Fiorentina": "Fiorentina",
        "AS Roma": "AS Roma",
        "Juventus FC": "Juventus",
        "Como 1907": "Como",
        "US Cremonese": "Cremonese",
        "Bologna FC 1909": "Bologna",
        "Parma Calcio 1913": "Parma",
        "Hellas Verona FC": "Hellas Verona",
        "SSC Napoli": "Napoli",
        "US Lecce": "Lecce",
        "Inter Milan": "Internazionale",
    },
    # La Liga
    "laliga_spain": {
        "Club Atlético de Madrid": "Atlético Madrid",
        "Rayo Vallecano de Madrid": "Rayo Vallecano",
        "Valencia CF": "Valencia",
        "Deportivo Alavés": "Alavés",
        "CA Osasuna": "Osasuna",
        "RCD Espanyol de Barcelona": "Espanyol",
        "Getafe CF": "Getafe",
        "Real Sociedad de Fútbol": "Real Sociedad",
        "Levante UD": "Levante",
        "Real Betis Balompié": "Real Betis",
        "RCD Mallorca": "Mallorca",
        "Girona FC": "Girona",
        "Villarreal CF": "Villarreal",
        "FC Barcelona": "Barcelona",
        "Elche CF": "Elche",
        "Sevilla FC": "Sevilla",
        "Real Madrid CF": "Real Madrid",
        "RC Celta de Vigo": "Celta Vigo",
        "Athletic Bilbao": "Athletic Club",
    },
    # Bundesliga
    "bundesliga_germany": {
        "1. FC Köln": "FC Cologne",
        "TSG 1899 Hoffenheim": "TSG Hoffenheim",
        "1. FSV Mainz 05": "Mainz",
        "SV Werder Bremen": "Werder Bremen",
        "Hamburger SV": "Hamburg SV",
        "Bayer 04 Leverkusen": "Bayer Leverkusen",
        "FC St. Pauli 1910": "St. Pauli",
        "FC Bayern München": "Bayern Munich",
        "1. FC Heidenheim": "1. FC Heidenheim 1846",
        "Union Berlin": "1. FC Union Berlin",
        "Borussia Monchengladbach": "Borussia Mönchengladbach",
        "Augsburg": "FC Augsburg",
    },
    # Ligue 1
    "ligue1_france": {
        "Racing Club de Lens": "Lens",
        "OGC Nice": "Nice",
        "FC Metz": "Metz",
        "Angers SCO": "Angers",
        "Stade Brestois 29": "Brest",
        "Olympique Lyonnais": "Lyon",
        "Paris Saint-Germain FC": "Paris Saint-Germain",
        "AS Monaco FC": "AS Monaco",
        "Lille OSC": "Lille",
        "Toulouse FC": "Toulouse",
        "FC Nantes": "Nantes",
        "RC Strasbourg Alsace": "Strasbourg",
        "Olympique de Marseille": "Marseille",
        "Stade Rennais FC 1901": "Stade Rennais",
        "Auxerre": "AJ Auxerre",
    },
}

# -----------------------------
# --- FUNCTIONS: ODDS & FIXTURES ---
# -----------------------------

def flatten_odds(data):
    """Flatten API JSON odds data into DataFrame."""
    rows = []
    for match in data:
        match_id = match["id"]
        home = match["home_team"]
        away = match["away_team"]
        time = match["commence_time"]
        for book in match["bookmakers"]:
            h2h = next((m for m in book["markets"] if m["key"]=="h2h"), None)
            if not h2h:
                continue
            outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
            rows.append({
                "match_id": match_id,
                "commence_time": time,
                "home_team": home,
                "away_team": away,
                "bookmaker": book["title"],
                "home_odds": outcomes.get(home),
                "draw_odds": outcomes.get("Draw"),
                "away_odds": outcomes.get(away),
            })
    return pd.DataFrame(rows)

def bookmaker_implied_probs(df):
    """Convert odds to implied probabilities and average across bookmakers."""
    df = df.assign(
        p_home_raw=1/df["home_odds"],
        p_draw_raw=1/df["draw_odds"],
        p_away_raw=1/df["away_odds"]
    )
    total = df["p_home_raw"] + df["p_draw_raw"] + df["p_away_raw"]
    df = df.assign(
        p_home_book=df["p_home_raw"]/total,
        p_draw_book=df["p_draw_raw"]/total,
        p_away_book=df["p_away_raw"]/total
    )
    return df.groupby(["home_team","away_team"], as_index=False).agg(
        p_home_book=("p_home_book","mean"),
        p_draw_book=("p_draw_book","mean"),
        p_away_book=("p_away_book","mean")
    )

# -----------------------------
# --- FUNCTIONS: POISSON MODEL ---
# -----------------------------

def compute_team_strengths(past_df):
    """Compute attack & defense strength from past matches."""
    teams = pd.concat([past_df['homeTeam'], past_df['awayTeam']]).unique()
    strength = []
    for team in teams:
        home_games = past_df[past_df['homeTeam']==team]
        away_games = past_df[past_df['awayTeam']==team]
        attack = (home_games['homeGoals'].sum()+away_games['awayGoals'].sum()) / (len(home_games)+len(away_games))
        defense = (home_games['awayGoals'].sum()+away_games['homeGoals'].sum()) / (len(home_games)+len(away_games))
        strength.append({'team':team,'attack_strength':attack,'defense_strength':defense})
    return pd.DataFrame(strength).set_index('team')

def poisson_probs(home_attack, home_defense, away_attack, away_defense, max_goals=5):
    """Compute match outcome probabilities using Poisson."""
    lambda_home = home_attack*away_defense
    lambda_away = away_attack*home_defense
    prob_home = [np.exp(-lambda_home)*lambda_home**i/np.math.factorial(i) for i in range(max_goals+1)]
    prob_away = [np.exp(-lambda_away)*lambda_away**i/np.math.factorial(i) for i in range(max_goals+1)]
    p_home, p_draw, p_away = 0,0,0
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            prob = prob_home[i]*prob_away[j]
            if i>j:
                p_home+=prob
            elif i==j:
                p_draw+=prob
            else:
                p_away+=prob
    return p_home, p_draw, p_away

# -----------------------------
# --- FINAL PROBABILITIES MERGE ---
# -----------------------------

def merge_model_and_odds(df_model, df_odds):
    """Merge model probabilities with bookmaker odds to create final probabilities."""
    df_odds = df_odds.drop_duplicates(subset=["home_team","away_team"], keep="first")
    df_final = df_model.merge(df_odds, left_on=["homeTeam","awayTeam"], right_on=["home_team","away_team"], how="left")
    df_final["p_home_final"] = np.where(df_final["p_home_book"].notna(), df_final["p_home_book"], df_final["p_home_win"])
    df_final["p_draw_final"] = np.where(df_final["p_draw_book"].notna(), df_final["p_draw_book"], df_final["p_draw"])
    df_final["p_away_final"] = np.where(df_final["p_away_book"].notna(), df_final["p_away_book"], df_final["p_away_win"])
    prob_cols = ["p_home_final","p_draw_final","p_away_final"]
    df_final[prob_cols] = df_final[prob_cols].div(df_final[prob_cols].sum(axis=1), axis=0)
    return df_final[["utcDate","homeTeam","awayTeam"]+prob_cols]

# -----------------------------
# --- SIMULATION USING FINAL PROBS ---
# -----------------------------

def simulate_season_with_final_probs(fixtures_df, n_sim=1000):
    """Simulate a season using final probabilities (model+odds)."""
    teams = pd.unique(fixtures_df[['homeTeam','awayTeam']].values.ravel())
    results = {team:[] for team in teams}
    prob_cols = ["p_home_final","p_draw_final","p_away_final"]
    for _ in range(n_sim):
        points = {team:0 for team in teams}
        for _, row in fixtures_df.iterrows():
            outcome = np.random.choice(['home','draw','away'], p=row[prob_cols])
            if outcome=='home': points[row['homeTeam']]+=3
            elif outcome=='away': points[row['awayTeam']]+=3
            else:
                points[row['homeTeam']]+=1
                points[row['awayTeam']]+=1
        for team in teams:
            results[team].append(points[team])
    avg_points = {team: np.mean(results[team]) for team in teams}
    final_table = pd.DataFrame({'team':teams,'avg_points':[avg_points[t] for t in teams]})
    final_table.sort_values('avg_points',ascending=False,inplace=True)
    final_table['predicted_position']=range(1,len(teams)+1)
    return final_table, results