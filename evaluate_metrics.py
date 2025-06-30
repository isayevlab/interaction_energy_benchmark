import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
import argparse

def evaluate_metrics(df: pd.DataFrame) -> dict:
    y_true = df['ref_energy_int']
    y_pred = df['pred_energy_int']

    r2 = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted vs reference interaction energies")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the csv file containing predictions")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"File not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    metrics = evaluate_metrics(df)

    print(f"\nEvaluation Results for: {os.path.basename(args.csv_path)}")
    print(f"R²   : {metrics['R2']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"MAE  : {metrics['MAE']:.4f}")

if __name__ == "__main__":
    main()
