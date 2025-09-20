import pandas as pd
import json
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

TEST_FILE = "data/processed/test.csv"
MODEL_FILE = "model.pkl"
EVAL_FILE = "eval.json"
FEATURES = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes']

def main():
    # Load test data
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"{TEST_FILE} not found.")
    df = pd.read_csv(TEST_FILE)

    # One-hot encode controller_kind
    df = pd.get_dummies(df, columns=['controller_kind'], drop_first=False)

    # Load trained model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"{MODEL_FILE} not found.")
    model = joblib.load(MODEL_FILE)

    # Separate target
    if 'cpu_usage' not in df.columns:
        raise ValueError("Target column 'cpu_usage' not found in test data.")
    y = df['cpu_usage']
    X = df.drop(columns=['cpu_usage'])

    # Ensure test set has the same columns as the model expects
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in X.columns:
            X[col] = 0  # add missing column with zeros
    X = X[model_features]  # reorder columns to match training

    # Make predictions
    y_pred = model.predict(X)

    # Compute metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # Save evaluation results
    eval_results = {"MAE": mae, "RMSE": rmse, "R2": r2}
    with open(EVAL_FILE, "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Evaluation complete. Results saved to {EVAL_FILE}")

if __name__ == "__main__":
    main()
