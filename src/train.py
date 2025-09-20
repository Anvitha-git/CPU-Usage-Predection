import pandas as pd
import json
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_FILE = "data/processed/train.csv"
MODEL_FILE = "model.pkl"
METRICS_FILE = "metrics.json"

FEATURES = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes']
TARGET = 'cpu_usage'

def main():
    # load data
    df = pd.read_csv(TRAIN_FILE)

    # one-hot encode controller_kind
    df = pd.get_dummies(df, columns=['controller_kind'], drop_first=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    # save model
    joblib.dump(model, MODEL_FILE)

    # evaluate on training set (baseline metrics)
    preds = model.predict(X)
    
    from math import sqrt

    metrics = {
        "mae": mean_absolute_error(y, preds),
        "rmse": sqrt(mean_squared_error(y, preds)),  # manual RMSE
        "r2": r2_score(y, preds),
        }


    # save metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Model trained and saved to {MODEL_FILE}")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
