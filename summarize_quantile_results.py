# -*- coding: utf-8 -*-
"""Summarize quantile ERA5 results from all sites."""
from pathlib import Path
import pandas as pd


def main():
    files = sorted(Path("results").glob("quantile_era5_f*_new/quantile_era5_f*_day.csv"))
    if not files:
        files = sorted(Path("results").glob("quantile_era5_f*/quantile_era5_f*_day.csv"))
    if not files:
        raise FileNotFoundError("No quantile_era5_f*_day.csv files found. Please run train_quantile_era5.py first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_dir = Path("results/paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_all = out_dir / "table_all_sites_quantile_era5.csv"
    df.to_csv(out_all, index=False, encoding="utf-8-sig")

    mean = df.groupby("horizon_min").agg({
        "RMSE": "mean",
        "MAE": "mean",
        "R2": "mean",
        "Pinball_q50": "mean",
        "PICP_raw": "mean",
        "PINAW_raw": "mean",
        "PICP_cal": "mean",
        "PINAW_cal": "mean",
        "AvgWidth_cal": "mean",
        "n_test": "sum",
    }).reset_index()

    out_mean = out_dir / "table_quantile_era5_mean_by_horizon.csv"
    mean.to_csv(out_mean, index=False, encoding="utf-8-sig")

    print("[OK] saved:", out_all)
    print("[OK] saved:", out_mean)
    print(mean.to_string(index=False))


if __name__ == "__main__":
    main()
