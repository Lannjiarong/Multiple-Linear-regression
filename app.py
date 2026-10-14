"""
StatsUnlock Multiple Linear Regression Calculator - FastAPI Application Server
Author: 7115056142
"""

import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

app = FastAPI(title="StatsUnlock Multiple Linear Regression API", version="2.0.0")

# Serve static web interface
web_dir = os.path.join(os.path.dirname(__file__), 'web')
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(web_dir, 'index.html')
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "StatsUnlock API is running. Access /static/index.html"}

@app.get("/api/datasets")
def get_datasets():
    return {
        "datasets": [
            {"id": "flood", "name": "Kaggle Flood Prediction (20 Features)", "features_count": 20, "target": "FloodProbability"},
            {"id": "startups", "name": "50 Startups (Reference Image)", "features_count": 5, "target": "Profit"}
        ]
    }

class PredictionRequest(BaseModel):
    features: Dict[str, float]

@app.post("/api/predict")
def predict_flood(req: PredictionRequest):
    # Load dataset & weights
    df = pd.read_csv('data/flood_prediction_train.csv')
    feature_cols = [c for c in df.columns if c not in ['id', 'FloodProbability']]
    
    X = df[feature_cols].values
    y = df['FloodProbability'].values
    
    reg = LinearRegression().fit(X, y)
    y_pred_all = reg.predict(X)
    
    n = len(y)
    p = len(feature_cols)
    dof = n - p - 1
    s = np.sqrt(np.sum((y - y_pred_all)**2) / dof)
    
    X_design = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.pinv(np.dot(X_design.T, X_design))
    
    # Input vector
    x_input = [req.features.get(f, 5.0) for f in feature_cols]
    x_design = np.array([1.0] + x_input)
    
    y_hat = float(reg.intercept_ + np.dot(x_input, reg.coef_))
    leverage = float(np.dot(np.dot(x_design, XtX_inv), x_design))
    
    t_crit = float(stats.t.ppf(0.975, dof))
    se_fit = s * np.sqrt(leverage)
    se_pred = s * np.sqrt(1.0 + leverage)
    
    return {
        "prediction": round(y_hat, 4),
        "confidence_interval_95": [round(y_hat - t_crit * se_fit, 4), round(y_hat + t_crit * se_fit, 4)],
        "prediction_interval_95": [round(y_hat - t_crit * se_pred, 4), round(y_hat + t_crit * se_pred, 4)],
        "rmse": round(float(s), 4),
        "r2": round(float(r2_score(y, y_pred_all)), 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
