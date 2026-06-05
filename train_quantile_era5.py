# -*- coding: utf-8 -*-
"""
Train LightGBM quantile models with ERA5 features for PV ultra-short-term forecasting.

Usage:
    python src/train_quantile_era5.py --processed_csv data/processed/processed_all_stations_era5_timeseries_new.csv --user f6 --filter day --out_dir results/quantile_era5_f6_new

Outputs:
    quantile_era5_<user>_<filter>.csv
    quantile_predictions_<user>_<filter>.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_baseline import WEATHER_COLS, TIME_COLS, add_supervised_features, chrono_split


def pinball_loss(y, pred, q: float) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_metrics(y, low, high, norm: float) -> dict:
    y = np.asarray(y)
    lo = np.minimum(low, high)
    hi = np.maximum(low, high)
    return {
        "PICP": float(np.mean((y >= lo) & (y <= hi))),
        "PINAW": float(np.mean(hi - lo) / max(norm, 1e-9)),
        "AvgWidth": float(np.mean(hi - lo)),
    }


def conformal_delta(y_cal, low_cal, high_cal, alpha: float = 0.2) -> float:
    y = np.asarray(y_cal)
    lo = np.minimum(low_cal, high_cal)
    hi = np.maximum(low_cal, high_cal)
    scores = np.maximum(np.maximum(lo - y, y - hi), 0)
    n = len(scores)
    if n == 0:
        return 0.0
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def calc_point_metrics(y, pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
    }


def get_era5_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith("era5_") and c not in {"era5_grid_lat", "era5_grid_lon"}
    ]


def run_quantile(df: pd.DataFrame, user: str, filter_mode: str):
    rows = []
    pred_rows = []

    df_user = df[df["user_id"] == user].copy()
    if df_user.empty:
        raise ValueError(f"No rows for user={user}")

    cap = float(df_user["capacity_kw"].dropna().iloc[0])
    era5_cols = get_era5_cols(df_user)
    print(f"[INFO] ERA5 feature columns: {era5_cols}")

    for h in [1, 2, 4]:
        horizon_min = h * 15
        featdf = add_supervised_features(df_user, h)
        power_cols = [c for c in featdf.columns if c.startswith(("lag_", "roll_", "delta_"))]
        cols = power_cols + TIME_COLS + WEATHER_COLS + era5_cols

        required = ["dt", "target"] + cols
        sample = featdf.dropna(subset=required).copy()

        if filter_mode == "day":
            sample = sample[(sample["irradiance"] > 0) | (sample["target"] > 0)].copy()

        train, val, test = chrono_split(sample, train_frac=0.70, val_frac=0.15)
        print(f"[INFO] {user} {horizon_min}min: train={len(train)}, val={len(val)}, test={len(test)}, features={len(cols)}")

        pred_val = {}
        pred_test = {}

        for q in [0.1, 0.5, 0.9]:
            model = LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=240,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=4,
                verbose=-1,
            )
            model.fit(train[cols], train["target"])
            pred_val[q] = np.clip(model.predict(val[cols]), 0, cap)
            pred_test[q] = np.clip(model.predict(test[cols]), 0, cap)

        y_val = val["target"].values
        y_test = test["target"].values
        low = pred_test[0.1]
        med = pred_test[0.5]
        high = pred_test[0.9]

        raw = interval_metrics(y_test, low, high, cap)
        delta = conformal_delta(y_val, pred_val[0.1], pred_val[0.9], alpha=0.2)
        low_cal = np.clip(low - delta, 0, cap)
        high_cal = np.clip(high + delta, 0, cap)
        cal = interval_metrics(y_test, low_cal, high_cal, cap)
        point = calc_point_metrics(y_test, med)

        rows.append({
            "user": user,
            "filter": filter_mode,
            "horizon_min": horizon_min,
            "feature_set": "power_weather_era5",
            "model": "LightGBM-Quantile",
            **point,
            "Pinball_q10": pinball_loss(y_test, low, 0.1),
            "Pinball_q50": pinball_loss(y_test, med, 0.5),
            "Pinball_q90": pinball_loss(y_test, high, 0.9),
            "PICP_raw": raw["PICP"],
            "PINAW_raw": raw["PINAW"],
            "AvgWidth_raw": raw["AvgWidth"],
            "conformal_delta_kw": delta,
            "PICP_cal": cal["PICP"],
            "PINAW_cal": cal["PINAW"],
            "AvgWidth_cal": cal["AvgWidth"],
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_features": len(cols),
        })

        tmp = pd.DataFrame({
            "user": user,
            "dt": test["dt"].values,
            "horizon_min": horizon_min,
            "y_true": y_test,
            "q10": low,
            "q50": med,
            "q90": high,
            "q10_cal": low_cal,
            "q90_cal": high_cal,
        })
        pred_rows.append(tmp)

    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_csv", type=Path, required=True)
    parser.add_argument("--user", default="f6")
    parser.add_argument("--filter", choices=["all", "day"], default="day")
    parser.add_argument("--out_dir", type=Path, default=Path("results/quantile_era5"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.processed_csv, parse_dates=["dt"])
    res, preds = run_quantile(df, args.user, args.filter)

    out_metrics = args.out_dir / f"quantile_era5_{args.user}_{args.filter}.csv"
    out_preds = args.out_dir / f"quantile_predictions_{args.user}_{args.filter}.csv"
    res.to_csv(out_metrics, index=False, encoding="utf-8-sig")
    preds.to_csv(out_preds, index=False, encoding="utf-8-sig")

    print(res.to_string(index=False))
    print(f"saved: {out_metrics}")
    print(f"saved: {out_preds}")


if __name__ == "__main__":
    main()
