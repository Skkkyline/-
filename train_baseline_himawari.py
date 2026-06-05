# -*- coding: utf-8 -*-
"""
Train point prediction models with Himawari feature sets.

Use this after you have a reasonably long Himawari feature period, e.g. 1-2 months.
For the current 3-day sample, use --only_himawari only as a pipeline check, not as final paper result.

Example:
    python src/train_baseline_himawari.py ^
      --processed_csv data/processed/processed_all_stations_era5_himawari_202303_202304.csv ^
      --user f6 --filter day --only_himawari ^
      --out_dir results/himawari_f6
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    from train_baseline import WEATHER_COLS, TIME_COLS, add_supervised_features, chrono_split
except Exception:
    WEATHER_COLS = [
        "pressure_pa", "humidity", "cloud", "wind10", "wind10_dir", "temp_k",
        "irradiance", "precip", "wind100", "wind100_dir", "temp_c",
    ]
    TIME_COLS = ["sin_hour", "cos_hour", "sin_doy", "cos_doy", "month", "hour", "is_day_weather"]

    def add_supervised_features(df_user: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
        df = df_user.sort_values("dt").copy()
        ycol = "power_kw_clean"
        lags = [horizon_steps, horizon_steps + 1, horizon_steps + 2, horizon_steps + 3,
                horizon_steps + 4, horizon_steps + 8, horizon_steps + 12,
                horizon_steps + 24, horizon_steps + 96]
        for lag in lags:
            df[f"lag_{lag}"] = df[ycol].shift(lag)
        shifted = df[ycol].shift(horizon_steps)
        for w in [4, 8, 12, 24, 96]:
            minp = max(2, w // 2)
            df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=minp).mean()
            df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=minp).std()
            df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=minp).max()
            df[f"roll_min_{w}"] = shifted.rolling(w, min_periods=minp).min()
        df[f"delta_h{horizon_steps}"] = df[ycol].shift(horizon_steps) - df[ycol].shift(horizon_steps + 1)
        df["target"] = df[ycol]
        return df

    def chrono_split(df: pd.DataFrame, train_frac: float = 0.75, val_frac: float = 0.10):
        df = df.sort_values("dt").copy()
        n = len(df)
        i1 = int(n * train_frac)
        i2 = int(n * (train_frac + val_frac))
        return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def calc_metrics(y, pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
        "n": int(len(y)),
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dt" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "dt"})
    if "user_id" not in df.columns and "user" in df.columns:
        df = df.rename(columns={"user": "user_id"})
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def run_one(df: pd.DataFrame, user: str, filter_mode: str, only_himawari: bool) -> pd.DataFrame:
    rows = []
    df = normalize_columns(df)
    df_user = df[df["user_id"] == user].copy()
    if df_user.empty:
        raise ValueError(f"No rows for user={user}")
    cap = float(df_user["capacity_kw"].dropna().iloc[0])

    era5_cols = [c for c in df_user.columns if c.startswith("era5_") and c not in {"era5_grid_lat", "era5_grid_lon"}]
    sat_cols = [c for c in df_user.columns if c.startswith("sat_b13_") or c in ["sat_age_min"]]
    sat_cols = [c for c in sat_cols if c not in {"sat_utc_date", "sat_utc_time"}]

    for h in [1, 2, 4]:
        featdf = add_supervised_features(df_user, h)
        power_cols = [c for c in featdf.columns if c.startswith(("lag_", "roll_", "delta_"))]
        sample = featdf.dropna(subset=["target"] + power_cols).copy()
        if filter_mode == "day":
            sample = sample[(sample["irradiance"] > 0) | (sample["target"] > 0)]
        if only_himawari:
            if "has_himawari" in sample.columns:
                sample = sample[sample["has_himawari"] == True]
            sample = sample.dropna(subset=sat_cols)
        if len(sample) < 80:
            print(f"[WARN] user={user} horizon={h*15}: only {len(sample)} rows after filtering. This is too small for final training.")

        train, val, test = chrono_split(sample)
        if len(test) == 0 or len(train) == 0:
            print(f"[WARN] skip horizon={h*15}, insufficient train/test rows")
            continue

        pred_col = f"lag_{h}"
        rows.append({
            "user": user, "filter": filter_mode, "horizon_min": h * 15,
            "feature_set": "persistence", "model": "Persistence",
            **calc_metrics(test["target"], test[pred_col])
        })

        feature_sets = {
            "power_weather": power_cols + TIME_COLS + WEATHER_COLS,
        }
        if era5_cols:
            feature_sets["power_weather_era5"] = power_cols + TIME_COLS + WEATHER_COLS + era5_cols
        if sat_cols:
            feature_sets["power_weather_himawari"] = power_cols + TIME_COLS + WEATHER_COLS + sat_cols
        if era5_cols and sat_cols:
            feature_sets["power_weather_era5_himawari"] = power_cols + TIME_COLS + WEATHER_COLS + era5_cols + sat_cols

        for feature_set, cols in feature_sets.items():
            cols = [c for c in cols if c in sample.columns]
            train_all = pd.concat([train, val], axis=0).dropna(subset=cols + ["target"])
            test2 = test.dropna(subset=cols + ["target"])
            if len(train_all) < 50 or len(test2) < 10:
                print(f"[WARN] skip {feature_set} horizon={h*15}: train={len(train_all)}, test={len(test2)}")
                continue
            model = XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=4,
                reg_lambda=2.0,
                min_child_weight=2,
                eval_metric="rmse",
            )
            model.fit(train_all[cols], train_all["target"], verbose=False)
            pred = np.clip(model.predict(test2[cols]), 0, cap)
            rows.append({
                "user": user, "filter": filter_mode, "horizon_min": h * 15,
                "feature_set": feature_set, "model": "XGBoost",
                **calc_metrics(test2["target"], pred)
            })

    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_csv", required=True)
    p.add_argument("--user", default="f6")
    p.add_argument("--filter", choices=["all", "day"], default="day")
    p.add_argument("--only_himawari", action="store_true", help="Use only rows with satellite features. For fair E3/E4/E5 comparison.")
    p.add_argument("--out_dir", default="results/himawari_baseline")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.processed_csv)
    res = run_one(df, args.user, args.filter, args.only_himawari)
    out = out_dir / f"baseline_himawari_{args.user}_{args.filter}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print("[OK] saved:", out)


if __name__ == "__main__":
    main()
