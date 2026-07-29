"""
train_model.py

Trains and compares multiple regression models for salary prediction:
    - Linear Regression (baseline)
    - Random Forest Regressor
    - XGBoost Regressor

Picks the best-performing model (by R2 score on the test set) and saves it,
along with the one-hot encoded column list, to the models/ folder.

Usage:
    python train_model.py
    python train_model.py --csv data/salary_data.csv   (default)
"""

import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

REQUIRED_COLUMNS = ["experience", "education_level", "job_role", "location", "industry", "salary"]


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"{name:22s}  MAE: {mae:>12,.0f}   RMSE: {rmse:>12,.0f}   R2: {r2:.4f}")
    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/salary_data.csv")
    args = parser.parse_args()

    df = load_data(args.csv)

    X = pd.get_dummies(df.drop("salary", axis=1))
    y = df["salary"]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)),
        ("XGBoost", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)),
    ]

    print("\nModel comparison on held-out test set:")
    print("-" * 70)
    results = []
    for name, model in candidates:
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))
    print("-" * 70)

    best = max(results, key=lambda r: r["r2"])
    print(f"\nBest model: {best['name']}  (R2 = {best['r2']:.4f})")

    joblib.dump(best["model"], "models/salary_model.pkl")
    joblib.dump(feature_columns, "models/model_columns.pkl")
    joblib.dump(best["name"], "models/model_name.pkl")
    print("Saved: models/salary_model.pkl, models/model_columns.pkl")


if __name__ == "__main__":
    main()
