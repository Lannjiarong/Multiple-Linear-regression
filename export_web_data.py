import os
import json
import pandas as pd

# Load flood dataset and 50 startups
df_flood = pd.read_csv('data/flood_prediction_train.csv')
df_startups = pd.read_csv('data/50_Startups.csv')

# Prepare JSON structures
flood_sample = df_flood.head(200).to_dict(orient='records')
flood_cols = [c for c in df_flood.columns if c != 'id']

startups_sample = df_startups.to_dict(orient='records')

feature_selection_flood = pd.read_csv('data/feature_selection_history.csv').to_dict(orient='records')
feature_selection_startups = pd.read_csv('data/50_startups_selection_history.csv').to_dict(orient='records')

os.makedirs('web', exist_ok=True)
js_content = f"""// Preloaded datasets for offline/instant evaluation
const PRELOADED_DATA = {{
  flood: {{
    name: "Kaggle Flood Prediction (20 Features)",
    target: "FloodProbability",
    features: {json.dumps([c for c in flood_cols if c != 'FloodProbability'])},
    rows: {json.dumps(flood_sample)},
    featureSelection: {json.dumps(feature_selection_flood)}
  }},
  startups: {{
    name: "50 Startups (5 Features)",
    target: "Profit",
    features: ["R&D Spend", "Administration", "Marketing Spend", "State"],
    rows: {json.dumps(startups_sample)},
    featureSelection: {json.dumps(feature_selection_startups)}
  }}
}};
"""

with open('web/preloaded_data.js', 'w', encoding='utf-8') as f:
    f.write(js_content)
print("Created web/preloaded_data.js successfully!")
