# -*- coding: utf-8 -*-
"""
Causal ultra-short-term PV forecasting with ERA5 and Himawari features.

This script uses the strict protocol:
    features available at prediction time T  ->  target power at T + horizon

This avoids using Himawari/ERA5 variables from the target time as predictors.
Use it for final Himawari ablation experiments.

Example:
    python src/train_baseline_himawari_causal.py ^
        --processed_csv data/processed/processed_all_stations_era5_himawari_202303_202304.csv ^
        --user f6 --filter day --require_himawari ^
        --out_dir results/himawari_causal_f6
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


WEATHER_COLS = [
    "pressure_pa", "humidity", "cloud", "wind10", "wind10_dir", "temp_k",
    "irradiance", "precip", "wind100", "wind100_dir", "temp_c",
]
TIME_COLS = ["sin_hour", "cos_hour", "sin_doy", "cos_doy", "month", "hour", "is_day_weather"]


# These features use information at prediction time T only.
POWER_LAGS = [0, 1, 2, 3, 4, 8, 12, 24, 96]
ROLL_WINDOWS = [4, 8, 12, 24, 96]


def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "user" in df.columns and "user_id" not in df.columns:
        df = df.rename(columns={"user": "user_id"})
    if "datetime" in df.columns and "dt" not in df.columns:
        df = df.rename(columns={"datetime": "dt"})
    if "dt" not in df.columns:
        raise KeyError("Cannot find time column. Expected dt or datetime.")
    if "user_id" not in df.columns:
        raise KeyError("Cannot find user column. Expected user_id or user.")
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def select_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def get_era5_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith("era5_") and c not in {"era5_grid_lat", "era5_grid_lon"}
    ]


def get_himawari_cols(df: pd.DataFrame) -> list[str]:
    sat_cols = []
    for c in df.columns:
        if c.startswith("sat_b13_"):
            if c in {"sat_b13_valid_ratio"}:
                continue
            sat_cols.append(c)
    # Satellite age is useful because 15-min samples reuse hourly satellite frames.
    if "sat_age_min" in df.columns:
        sat_cols.append("sat_age_min")
    return list(dict.fromkeys(sat_cols))


def prepare_satellite_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill only physically reasonable missing satellite-derived fields.

    Frame-difference features are undefined for the first frame of each day/site.
    For modeling, setting these to zero means "no previous-frame change available".
    Other satellite features should remain NaN and will be filtered if required.
    """
    df = df.copy()
    for c in df.columns:
        if c.startswith("sat_b13_") and ("diff" in c):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def add_causal_supervised_features(df_user: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    """Create samples at prediction time T with target at T+horizon.

    Important:
        - target = power(T+h)
        - power lags, weather, ERA5, Himawari are all from T or earlier
    """
    df = df_user.sort_values("dt").copy()
    ycol = "power_kw_clean" if "power_kw_clean" in df.columns else "power_kw"
    if ycol not in df.columns:
        raise KeyError("Cannot find power column power_kw_clean or power_kw.")

    # Target future power.
    df["target"] = df[ycol].shift(-horizon_steps)
    df["target_dt"] = df["dt"].shift(-horizon_steps)

    # Current and past power lags.
    for lag in POWER_LAGS:
        df[f"p_lag_{lag}"] = df[ycol].shift(lag)

    # Rolling statistics using current and previous power only.
    for w in ROLL_WINDOWS:
        minp = max(2, w // 2)
        r = df[ycol].rolling(w, min_periods=minp)
        df[f"p_roll_mean_{w}"] = r.mean()
        df[f"p_roll_std_{w}"] = r.std()
        df[f"p_roll_max_{w}"] = r.max()
        df[f"p_roll_min_{w}"] = r.min()

    df["p_delta_1"] = df[ycol] - df[ycol].shift(1)
    df[f"p_delta_h{horizon_steps}"] = df[ycol] - df[ycol].shift(horizon_steps)

    return df


def chrono_split(df: pd.DataFrame, train_frac: float = 0.75, val_frac: float = 0.10):
    df = df.sort_values("dt").copy()
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def calc_metrics(y, pred, cap=None):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    out = {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "n": int(len(y)),
    }
    if cap is not None and cap > 0:
        out["nRMSE_pct"] = out["RMSE"] / cap * 100.0
        out["nMAE_pct"] = out["MAE"] / cap * 100.0
    return out


def fit_xgb(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], cap: float | None):
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
    model.fit(train[cols], train["target"], verbose=False)
    pred = model.predict(test[cols])
    pred = np.clip(pred, 0, cap if cap is not None else None)
    return pred


def run_one(
    df: pd.DataFrame,
    user: str,
    filter_mode: str,
    require_himawari: bool = True,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    rows = []
    df = normalize_columns(df)
    df = prepare_satellite_missing_values(df)

    df_user = df[df["user_id"].astype(str) == str(user)].copy()
    if df_user.empty:
        raise ValueError(f"No rows for user={user}")

    cap = float(df_user["capacity_kw"].dropna().iloc[0]) if "capacity_kw" in df_user.columns else None
    era5_cols = get_era5_cols(df_user)
    sat_cols = get_himawari_cols(df_user)
    if require_himawari and not sat_cols:
        raise ValueError("No Himawari feature columns found. Expected sat_b13_* columns.")

    if horizons is None:
        horizons = [15, 30, 60]

    for horizon_min in horizons:
        if horizon_min % 15 != 0:
            raise ValueError("horizon must be a multiple of 15 minutes")
        h = horizon_min // 15
        featdf = add_causal_supervised_features(df_user, h)

        power_cols = [c for c in featdf.columns if c.startswith("p_lag_") or c.startswith("p_roll_") or c.startswith("p_delta_")]
        time_cols = select_cols(featdf, TIME_COLS)
        weather_cols = select_cols(featdf, WEATHER_COLS)

        required = ["target"] + power_cols
        if require_himawari:
            if "has_himawari" in featdf.columns:
                featdf = featdf[featdf["has_himawari"] == True].copy()
            required += sat_cols

        sample = featdf.dropna(subset=required).copy()

        if filter_mode == "day":
            current_day = sample["irradiance"] > 0 if "irradiance" in sample.columns else sample["p_lag_0"] > 0
            target_day = sample["target"] > 0
            sample = sample[current_day | target_day].copy()
        elif filter_mode == "target_day":
            sample = sample[sample["target"] > 0].copy()

        if len(sample) < 120:
            warnings.warn(
                f"{user}, horizon={horizon_min} min has only {len(sample)} rows. "
                "This is OK for smoke tests, but too small for paper-level results."
            )

        if len(sample) < 30:
            warnings.warn(f"Skip horizon={horizon_min}: not enough samples after filtering.")
            continue

        train, val, test = chrono_split(sample)
        train_all = pd.concat([train, val], axis=0)
        if len(test) < 5 or len(train_all) < 20:
            warnings.warn(f"Skip horizon={horizon_min}: train={len(train_all)}, test={len(test)}")
            continue

        # Persistence baseline: P(T+h) ≈ P(T)
        rows.append({
            "user": user,
            "filter": filter_mode,
            "protocol": "causal_T_to_TplusH",
            "horizon_min": horizon_min,
            "feature_set": "persistence",
            "model": "Persistence",
            "n_features": 1,
            **calc_metrics(test["target"], test["p_lag_0"], cap=cap),
        })

        feature_sets = {
            "power_only": power_cols + time_cols,
            "power_weather": power_cols + time_cols + weather_cols,
            "power_weather_era5": power_cols + time_cols + weather_cols + era5_cols,
            "power_weather_himawari": power_cols + time_cols + weather_cols + sat_cols,
            "power_weather_era5_himawari": power_cols + time_cols + weather_cols + era5_cols + sat_cols,
        }

        for feature_set, cols in feature_sets.items():
            cols = list(dict.fromkeys(select_cols(sample, cols)))
            train2 = train_all.dropna(subset=cols + ["target"])
            test2 = test.dropna(subset=cols + ["target"])
            if len(train2) < 20 or len(test2) < 5:
                warnings.warn(f"Skip {feature_set}, horizon={horizon_min}: train={len(train2)}, test={len(test2)}")
                continue
            pred = fit_xgb(train2, test2, cols, cap)
            rows.append({
                "user": user,
                "filter": filter_mode,
                "protocol": "causal_T_to_TplusH",
                "horizon_min": horizon_min,
                "feature_set": feature_set,
                "model": "XGBoost",
                "n_features": len(cols),
                **calc_metrics(test2["target"], pred, cap=cap),
            })

    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_csv", type=Path, required=True)
    p.add_argument("--user", default="f6")
    p.add_argument("--filter", choices=["all", "day", "target_day"], default="day")
    p.add_argument("--require_himawari", action="store_true", help="Use only rows with Himawari features; recommended for fair Himawari ablation.")
    p.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 60])
    p.add_argument("--out_dir", type=Path, default=Path("results/himawari_causal"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv_smart(args.processed_csv)
    res = run_one(
        df=df,
        user=args.user,
        filter_mode=args.filter,
        require_himawari=args.require_himawari,
        horizons=args.horizons,
    )
    out = args.out_dir / f"baseline_himawari_causal_{args.user}_{args.filter}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
