# ===============================
# IMPORTS
# ===============================

import numpy as np
import pandas as pd
import requests
from datetime import datetime
from scipy.stats import poisson


# ===============================
# GLOBAL SETTINGS
# ===============================

year = 2025

leagues = [
    "premierleague_england",
    "seriea_italy",
    "laliga_spain",
    "bundesliga_germany",
    "ligue1_france"
]


# ===============================
# TEAM NAME FIXES
# ===============================

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


def clean_team_names(df, column="team"):
    df = df.copy()
    df[column] = df[column].replace(TEAM_NAME_MAPPING)
    return df


# ===============================
# GET CURRENT LEAGUE TABLES
# ===============================

def get_league_tables():

    league_codes = {
        "ENG.1": "premierleague_england",
        "ITA.1": "seriea_italy",
        "ESP.1": "laliga_spain",
        "GER.1": "bundesliga_germany",
        "FRA.1": "ligue1_france",
    }

    tables = {}

    for code, name in league_codes.items():

        url = f"https://www.espn.com/soccer/standings/_/league/{code}/season/{year}"
        html_tables = pd.read_html(url)

        teams_raw = html_tables[0]
        stats = html_tables[1]

        teams = pd.DataFrame()
        teams["position"] = teams_raw.iloc[:, 0].str.extract(r"^(\d+)").astype(int)

        teams["team"] = (
            teams_raw.iloc[:, 0]
            .str.replace(r"^\d+", "", regex=True)
            .str.replace(r"^[A-Z]{2,3}", "", regex=True)
            .str.strip()
        )

        stats.columns = ["gp", "w", "d", "l", "gf", "ga", "gd", "pts"]
        stats = stats.apply(lambda c: c.astype(str).str.replace("+", "").astype(int))

        df = pd.concat([teams, stats], axis=1)
        df = clean_team_names(df)

        tables[name] = df

    return tables


# ===============================
# MATCH SIMULATION
# ===============================

def simulate_once(fixtures, table):
    """
    Simulate remaining matches once and return final table
    """

    table_sim = table.copy()

    points = dict(zip(table_sim["team"], table_sim["pts"]))

    for _, row in fixtures.iterrows():

        home = row["homeTeam"]
        away = row["awayTeam"]

        probs = [
            row["p_home_final"],
            row["p_draw_final"],
            row["p_away_final"]
        ]

        outcome = np.random.choice(["H", "D", "A"], p=probs)

        if outcome == "H":
            points[home] += 3

        elif outcome == "D":
            points[home] += 1
            points[away] += 1

        else:
            points[away] += 3

    result_df = table_sim.copy()
    result_df["pts"] = result_df["team"].map(points)

    result_df = result_df.sort_values(
        ["pts", "gd"],
        ascending=[False, False]
    )

    result_df["position"] = np.arange(1, len(result_df) + 1)

    return result_df


# ===============================
# MONTE CARLO SIMULATIONS
# ===============================

def run_simulations(fixtures, table, n_sim=10000):

    position_counts = {
        team: np.zeros(len(table)) for team in table["team"]
    }

    for i in range(n_sim):

        final_table = simulate_once(fixtures, table)

        for _, row in final_table.iterrows():
            position_counts[row["team"]][row["position"] - 1] += 1

        if (i + 1) % 1000 == 0:
            print(f"{i+1}/{n_sim} simulations completed")

    pos_df = pd.DataFrame(
        position_counts,
        index=np.arange(1, len(table) + 1)
    )

    pos_df_t = pos_df.T

    pos_df_pct = pos_df_t.div(pos_df_t.sum(axis=1), axis=0) * 100

    return pos_df_pct.round(2)


# ===============================
# MAIN PREDICTION FUNCTION
# ===============================

def generate_predictions(df_simulation_all):

    results = {}

    tables = get_league_tables()

    for league in leagues:

        print(f"\n=== {league.replace('_',' ').title()} ===")

        fixtures = df_simulation_all[league].copy()
        table = tables[league].copy()

        result = run_simulations(fixtures, table)

        results[league] = result

        print(f"Finished simulations for {league}")

    return results