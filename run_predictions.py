from model import generate_predictions
import os

print("Running daily football predictions...")

results = generate_predictions()

os.makedirs("predictions", exist_ok=True)

for league, df in results.items():

    file_path = f"predictions/{league}.csv"
    df.to_csv(file_path)

    print(f"Saved {file_path}")

print("All predictions saved successfully.")