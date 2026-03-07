# dataset_probabilities.py

import pandas as pd
import numpy as np
from scipy.stats import poisson

# === 1. HELPERS ===

def get_team_columns(df):
    if {"homeTeam", "awayTeam"}.issubset(df.columns):
        return "homeTeam", "awayTeam"
    if {"home_team", "away_team"}.issubset(df.columns):
        return "home_team", "away_team"
    raise KeyError("No home/away team columns found")

def extract_teams(df):
    home_col, away_col = get_team_columns(df)
    return set(df[home_col]).union(set(df[away_col]))

def filter_current_season(past_matches, season_start=pd.Timestamp("2025-08-01")):
    df = past_matches.copy()
    if "utcDate" not in df.columns:
        raise ValueError("Expected column 'utcDate' not found in past matches")
    df["utcDate"] = pd.to_datetime(df["utcDate"], utc=True).dt.tz_localize(None)
    return df[df["utcDate"] >= season_start]

def season_fixtures(past_matches, future_matches):
    return pd.concat(
        [past_matches[["homeTeam", "awayTeam"]], future_matches[["homeTeam", "awayTeam"]]],
        ignore_index=True,
    )

def find_missing_reverse_fixture(team, opponent, fixtures):
    team_home = ((fixtures.homeTeam == team) & (fixtures.awayTeam == opponent)).any()
    team_away = ((fixtures.homeTeam == opponent) & (fixtures.awayTeam == team)).any()
    if team_home and not team_away:
        return opponent, team
    if team_away and not team_home:
        return team, opponent
    return None

def match_probabilities_league(
    home,
    away,
    attack,
    defense,
    league_avg_scored,
    home_advantage,
    max_goals=6,
):
    exp_home = np.exp(
        np.log(league_avg_scored) + np.log(attack[home]) + np.log(defense[away]) + home_advantage
    )
    exp_away = np.exp(
        np.log(league_avg_scored) + np.log(attack[away]) + np.log(defense[home])
    )

    p_home = poisson.pmf(range(max_goals + 1), exp_home)
    p_away = poisson.pmf(range(max_goals + 1), exp_away)

    p_win = p_draw = p_loss = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = p_home[i] * p_away[j]
            if i > j:
                p_win += prob
            elif i == j:
                p_draw += prob
            else:
                p_loss += prob
    return p_win, p_draw, p_loss

# === 2. COMPUTE FINAL PROBABILITIES FUNCTION ===

def compute_final_probabilities(leagues, past_matches_dict, fixtures_dict, betting_odds_dict):
    df_final_probabilities_all = {}
    home_advantage_by_league = {}
    
    for league in leagues:
        # 2a. Weight past matches
        df_all = past_matches_dict[league].copy()
        df_all["date"] = pd.to_datetime(df_all["utcDate"])
        df_all = df_all.sort_values("date").reset_index(drop=True)
        df_all["weight"] = np.linspace(1, 2, len(df_all))
        past_matches_dict[league + "_weighted"] = df_all

        # 2b. Home advantage
        home_adv = df_all["homeGoals"].mean() - df_all["awayGoals"].mean()
        home_advantage_by_league[league] = home_adv

        # 2c. Team attack/defense
        teams = pd.unique(df_all[["homeTeam", "awayTeam"]].values.ravel("K"))
        attack = pd.Series(1.0, index=teams)
        defense = pd.Series(1.0, index=teams)
        team_stats = {}

        for team in teams:
            home_games = df_all[df_all["homeTeam"] == team]
            away_games = df_all[df_all["awayTeam"] == team]

            goals_scored = (
                (home_games["homeGoals"] * home_games["weight"]).sum() +
                (away_games["awayGoals"] * away_games["weight"]).sum()
            )
            goals_against = (
                (home_games["awayGoals"] * home_games["weight"]).sum() +
                (away_games["homeGoals"] * away_games["weight"]).sum()
            )
            matches = home_games["weight"].sum() + away_games["weight"].sum()
            team_stats[team] = {"scored": goals_scored / matches, "against": goals_against / matches}

        league_avg_scored = (df_all["homeGoals"].mean() + df_all["awayGoals"].mean()) / 2
        for team in teams:
            attack[team] = team_stats[team]["scored"] / league_avg_scored
            defense[team] = team_stats[team]["against"] / league_avg_scored

        # 2d. Compute Poisson probabilities for future matches
        df_future = fixtures_dict[league]
        results = []
        for _, row in df_future.iterrows():
            home = row["homeTeam"]
            away = row["awayTeam"]
            p_win, p_draw, p_loss = match_probabilities_league(
                home, away, attack, defense, league_avg_scored, home_adv
            )
            results.append({
                "utcDate": row.get("utcDate", pd.NaT),
                "homeTeam": home,
                "awayTeam": away,
                "p_home_win": p_win,
                "p_draw": p_draw,
                "p_away_win": p_loss,
            })
        df_probabilities = pd.DataFrame(results)

        # 2e. Combine with betting odds if available
        df_book = betting_odds_dict[league]
        df_book = df_book.drop_duplicates(subset=["home_team", "away_team"], keep="first")
        df_final = df_probabilities.merge(
            df_book,
            left_on=["homeTeam", "awayTeam"],
            right_on=["home_team", "away_team"],
            how="left"
        )

        for col_model, col_book, col_final in [
            ("p_home_win", "p_home_book", "p_home_final"),
            ("p_draw", "p_draw_book", "p_draw_final"),
            ("p_away_win", "p_away_book", "p_away_final")
        ]:
            df_final[col_final] = np.where(
                df_final[col_book].notna(),
                df_final[col_book],
                df_final[col_model]
            )

        prob_cols = ["p_home_final", "p_draw_final", "p_away_final"]
        df_final[prob_cols] = df_final[prob_cols].div(df_final[prob_cols].sum(axis=1), axis=0)

        df_final_probabilities_all[league] = df_final[[
            "utcDate", "homeTeam", "awayTeam", "p_home_final", "p_draw_final", "p_away_final"
        ]]
    
    return df_final_probabilities_all

# === 3. OPTIONAL MAIN BLOCK FOR DIRECT RUNNING ===

if __name__ == "__main__":
    leagues = [
        "premierleague_england",
        "seriea_italy",
        "laliga_spain",
        "bundesliga_germany",
        "ligue1_france",
    ]

    # Expect your past_matches, fixtures, betting_odds to be loaded into globals()
    past_matches_all = {league: globals()[f"past_matches_{league}_all"] for league in leagues}
    fixtures_all = {league: globals()[f"fixtures_{league}"] for league in leagues}
    betting_odds_all = {league: globals()[f"betting_odds_{league}"] for league in leagues}

    df_simulation_all = compute_final_probabilities(leagues, past_matches_all, fixtures_all, betting_odds_all)

    for league, df in df_simulation_all.items():
        print(f"\n=== {league.replace('_', ' ').title()} ===")
        print(df.head(3))
        print(f"Number of matches: {len(df)}")