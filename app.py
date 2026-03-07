# -------------------------------
# 1️⃣ IMPORTS
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------
# 2️⃣ HELPER FUNCTIONS FOR STYLING

def zero_style(val):
    if val < 1:
        return "background-color: white !important;"
    return ""

def color_scale(val, mid=0.14, max_val=0.75):
    if val >= max_val:
        return 1.0
    elif val <= mid:
        return val / mid * 0.5
    else:
        return 0.5 + (val - mid) / (max_val - mid) * 0.5

def style_probabilities_table(df):
    """
    Apply gradient styling to probability columns while keeping POS/TEAM/GP/PTS nice.
    """
    prob_cols = df.columns.difference(["POS", "TEAM", "GP", "PTS"])
    vmax = max(df[prob_cols].max().max(), 1)

    styled = df.style
    for col in prob_cols:
        col_data = df[col].apply(lambda x: color_scale(x))
        styled = styled.background_gradient(cmap="Greens", vmin=0, vmax=vmax, gmap=col_data, subset=[col])

    styled = styled.applymap(zero_style, subset=prob_cols)
    styled = styled.format({col: "{:.2f}%" for col in prob_cols}) \
                   .set_properties(subset=["POS","GP","PTS"], **{"text-align":"center","font-weight":"600"}) \
                   .set_properties(subset=["TEAM"], **{"text-align":"left","font-weight":"600"}) \
                   .set_properties(subset=prob_cols, **{"text-align":"center","font-weight":"500"}) \
                   .hide(axis="index")
    return styled

# -------------------------------
# 3️⃣ STREAMLIT APP

st.set_page_config(page_title="Football League Simulation", layout="wide")
st.title("⚽ Football League Monte Carlo Simulation")

# --- Load precomputed results ---
st.info("⏳ Loading precomputed simulation results...")

try:
    with open("data/precomputed_pos_counts.pkl", "rb") as f:
        position_distribution_all = pickle.load(f)
    with open("data/precomputed_pos_pct.pkl", "rb") as f:
        position_distribution_pct_all = pickle.load(f)
    st.success("✅ Precomputed results loaded.")
except Exception as e:
    st.error(f"❌ Failed to load precomputed results: {e}")
    st.stop()

# --- Select League ---
leagues = list(position_distribution_pct_all.keys())
league = st.selectbox("Select League", leagues)

# -------------------------------
# 4️⃣ DISPLAY SELECTED LEAGUE

# Convert to DataFrame for display
pos_pct_df = position_distribution_pct_all[league].copy()
pos_pct_df = pos_pct_df.reset_index()

# Fix MultiIndex if present
if isinstance(pos_pct_df.columns, pd.MultiIndex):
    pos_pct_df.columns = [str(c) for c in pos_pct_df.columns]

# Create columns POS, TEAM, GP, PTS if missing in index
if "POS" not in pos_pct_df.columns:
    pos_pct_df["POS"] = pos_pct_df.index + 1
if "TEAM" not in pos_pct_df.columns:
    pos_pct_df["TEAM"] = pos_pct_df.index.astype(str)
if "GP" not in pos_pct_df.columns:
    pos_pct_df["GP"] = 0
if "PTS" not in pos_pct_df.columns:
    pos_pct_df["PTS"] = 0

# Ensure types
pos_pct_df["TEAM"] = pos_pct_df["TEAM"].astype(str)
pos_pct_df["POS"] = pos_pct_df["POS"].astype(int)
pos_pct_df["GP"] = pos_pct_df["GP"].astype(int)
pos_pct_df["PTS"] = pos_pct_df["PTS"].astype(int)

st.header(f"🏆 {league.replace('_',' ').title()} Simulation Results")
st.write("Styled probabilities table for league positions:")

# Style and display
styled_table = style_probabilities_table(pos_pct_df)
st.dataframe(styled_table)