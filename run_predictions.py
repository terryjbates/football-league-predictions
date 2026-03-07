import os
from model import generate_predictions

print("Running daily football predictions...")

# df_simulation_all must be created earlier in your pipeline
results = generate_predictions(df_simulation_all)

os.makedirs("predictions", exist_ok=True)

for league, df in results.items():

    path = f"predictions/{league}.csv"
    df.to_csv(path)

    print(f"Saved {path}")

print("All predictions saved successfully.")