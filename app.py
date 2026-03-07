# app.py

# -------------------------------
# 1️⃣ IMPORTS
import streamlit as st
import importlib.util
import sys
import pandas as pd
import pickle

# -------------------------------
# 2️⃣ HELPER: dynamic module import
def import_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# -------------------------------
# 3️⃣ IMPORT REQUIRED MODULES (just for constants/functions)
dataset_processing = import_module_from_path("dataset_processing", "2_dataset_processing.py")
dataset_simulation = import_module_from_path("dataset_simulation", "4_simulations.py")

# -------------------------------
# 4️⃣ STREAMLIT APP
st.set_page_config(page_title="Football League Simulation", layout="wide")
st.title("⚽ Football League Monte Carlo Simulation")

# -------------------------------
# 5️⃣ Load precomputed results
st.info("⏳ Loading precomputed simulation results...")
with open("data/precomputed_pos_counts.pkl", "rb") as f:
    position_distribution_all = pickle.load(f)

with open("data/precomputed_pos_pct.pkl", "rb") as f:
    position_distribution_pct_all = pickle.load(f)

st.success("✅ Precomputed results loaded.")

# -------------------------------
# 6️⃣ Sidebar league selection
league = st.sidebar.selectbox(
    "Select League",
    dataset_processing.leagues
)

# --- 7️⃣ Display selected league ---
st.header(f"🏆 {league.replace('_', ' ').title()} Simulation Results")

# Get the precomputed percentage table
pos_pct = position_distribution_pct_all.get(league)

# Build a basic table for meta info (GP, PTS)
table = pd.DataFrame({
    "team": pos_pct.index,
    "pts": 0,   # placeholder
    "gd": 0,
    "gp": 0,
    "position": range(1, len(pos_pct)+1)
})

# Style the table using your simulation module
styled_table = dataset_simulation.style_position_table(pos_pct, table)

# Convert the styled table to HTML and render safely in Streamlit
st.write("Styled probabilities table for league positions:")
st.markdown(styled_table.to_html(), unsafe_allow_html=True)