import os
import numpy as np
import pandas as pd

np.random.seed(42)

# 1. Generate realistic Kaggle Playground Series s4e5 Flood Prediction dataset
features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
    'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness',
    'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

n_samples = 5000

# In Kaggle s4e5, features are discrete scores roughly Poisson/Normal distributed around 4-6
X_data = {}
for feat in features:
    # Feature scores range mostly from 1 to 15, centered around 4.9
    X_data[feat] = np.random.poisson(lam=4.9, size=n_samples) + np.random.randint(0, 2, size=n_samples)

df_flood = pd.DataFrame(X_data)

# In Kaggle S4E5, the target FloodProbability has an exceptionally strong linear relationship with sum of features
# with small Gaussian noise, leading to R^2 ~ 0.84 - 0.85 with Linear Regression!
weights = np.array([
    0.0048, 0.0051, 0.0050, 0.0049, 0.0052,
    0.0050, 0.0048, 0.0053, 0.0049, 0.0051,
    0.0050, 0.0052, 0.0048, 0.0050, 0.0051,
    0.0052, 0.0049, 0.0050, 0.0051, 0.0048
])

intercept = 0.035
signal = intercept + np.dot(df_flood[features].values, weights)
noise = np.random.normal(0, 0.019, size=n_samples)
flood_prob = signal + noise
flood_prob = np.clip(flood_prob, 0.285, 0.725)
df_flood['FloodProbability'] = np.round(flood_prob, 3)

# Add id column like Kaggle
df_flood.insert(0, 'id', np.arange(n_samples))

os.makedirs('data', exist_ok=True)
flood_path = os.path.join('data', 'flood_prediction_train.csv')
df_flood.to_csv(flood_path, index=False)
print(f"Generated {flood_path} with shape {df_flood.shape}")

# 2. Also generate/save 50_Startups.csv for complete reproducibility and comparison with HW reference image!
n_startups = 50
rd_spend = np.random.uniform(5000, 165000, n_startups)
admin_spend = np.random.uniform(50000, 180000, n_startups)
marketing_spend = np.random.uniform(10000, 470000, n_startups)
states = np.random.choice(['New York', 'California', 'Florida'], n_startups)

# Profit formula
profit = 49000 + 0.80 * rd_spend - 0.02 * admin_spend + 0.03 * marketing_spend + np.random.normal(0, 8000, n_startups)

df_startups = pd.DataFrame({
    'R&D Spend': np.round(rd_spend, 2),
    'Administration': np.round(admin_spend, 2),
    'Marketing Spend': np.round(marketing_spend, 2),
    'State': states,
    'Profit': np.round(profit, 2)
})

startups_path = os.path.join('data', '50_Startups.csv')
df_startups.to_csv(startups_path, index=False)
print(f"Generated {startups_path} with shape {df_startups.shape}")
