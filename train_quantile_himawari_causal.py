# -*- coding: utf-8 -*-
"""
Causal quantile forecasting with ERA5 + Himawari features.

Protocol:
    Features available at prediction time T -> target PV power at T + horizon.

This avoids using satellite imagery from the target time as predictors.

Example:
    python src/train_quantile_himawari_causal.py ^
        --processed_csv data/processed/processed_all_stations_era5_himawari_202303_202304_b13_full.csv ^
        --user f6 --filter day --require_himawari ^
        --out_dir results/quantile_himawari_causal_f6_new
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_HGBR = True
except Exception:
    HAS_HGBR = False
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


WEATHER_COLS = [
    "pressure_pa", "humidity", "cloud", "wind10", "wind10_dir", "temp_k",
    "irradiance", "precip", "wind100", "wind100_dir", "temp_c",
]
TIME_COLS = ["sin_hour", "cos_hour", "sin_doy", "cos_doy", "month", "hour", "is_day_weather"]
POWER_LAGS = [0, 1, 2, 3, 4, 8, 12, 24, 96]
ROLL_WINDOWS = [4, 8, 12, 24, 96]
QUANTILES = [0.1, 0.5, 0.9]


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
    if "sat_age_min" in df.columns:
        sat_cols.append("sat_age_min")
    return list(dict.fromkeys(sat_cols))


def prepare_satellite_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c.startswith("sat_b13_") and ("diff" in c):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def add_causal_supervised_features(df_user: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    df = df_user.sort_values("dt").copy()
    ycol = "power_kw_clean" if "power_kw_clean" in df.columns else "power_kw"
    if ycol not in df.columns:
        raise KeyError("Cannot find power column power_kw_clean or power_kw.")

    df["target"] = df[ycol].shift(-horizon_steps)
    df["target_dt"] = df["dt"].shift(-horizon_steps)
    if "power_pu" in df.columns:
        df["target_pu"] = df["power_pu"].shift(-horizon_steps)

    for lag in POWER_LAGS:
        df[f"p_lag_{lag}"] = df[ycol].shift(lag)
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


def chrono_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    df = df.sort_values("dt").copy()
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def pinball_loss(y, pred, q: float) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    e = y - pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def conformal_qhat(y_val, q10_val, q90_val, alpha: float = 0.20) -> float:
    y_val = np.asarray(y_val, dtype=float)
    q10_val = np.asarray(q10_val, dtype=float)
    q90_val = np.asarray(q90_val, dtype=float)
    scores = np.maximum(q10_val - y_val, y_val - q90_val)
    scores = np.maximum(scores, 0.0)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    # Finite-sample conformal quantile. Equivalent to quantile level ceil((n+1)*(1-alpha))/n.
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def calc_interval_metrics(y, q10, q50, q90, q10_cal, q90_cal, cap: float | None) -> dict:
    y = np.asarray(y, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q90 = np.asarray(q90, dtype=float)
    q10_cal = np.asarray(q10_cal, dtype=float)
    q90_cal = np.asarray(q90_cal, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y, q50)))
    mae = float(mean_absolute_error(y, q50))
    r2 = float(r2_score(y, q50)) if len(y) > 1 else np.nan
    picp_raw = float(np.mean((y >= q10) & (y <= q90)))
    picp_cal = float(np.mean((y >= q10_cal) & (y <= q90_cal)))
    avg_width_raw = float(np.mean(q90 - q10))
    avg_width_cal = float(np.mean(q90_cal - q10_cal))

    out = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pinball_q10": pinball_loss(y, q10, 0.1),
        "Pinball_q50": pinball_loss(y, q50, 0.5),
        "Pinball_q90": pinball_loss(y, q90, 0.9),
        "PICP_raw": picp_raw,
        "PICP_cal": picp_cal,
        "AvgWidth_raw": avg_width_raw,
        "AvgWidth_cal": avg_width_cal,
        "n": int(len(y)),
    }
    if cap is not None and cap > 0:
        out["nRMSE_pct"] = rmse / cap * 100.0
        out["nMAE_pct"] = mae / cap * 100.0
        out["PINAW_raw"] = avg_width_raw / cap
        out["PINAW_cal"] = avg_width_cal / cap
        out["nAvgWidth_raw_pct"] = avg_width_raw / cap * 100.0
        out["nAvgWidth_cal_pct"] = avg_width_cal / cap * 100.0
    return out


def make_quantile_model(q: float, seed: int = 42):
    if HAS_LGBM:
        return LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=2,
            verbose=-1,
        )
    if HAS_HGBR:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=seed,
        )
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=q,
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=seed,
    )


def fit_predict_quantiles(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cols: list[str], cap: float | None):
    preds_val = {}
    preds_test = {}
    for q in QUANTILES:
        model = make_quantile_model(q)
        model.fit(train[cols], train["target"])
        preds_val[q] = model.predict(val[cols])
        preds_test[q] = model.predict(test[cols])

    val_stack = np.vstack([preds_val[0.1], preds_val[0.5], preds_val[0.9]]).T
    test_stack = np.vstack([preds_test[0.1], preds_test[0.5], preds_test[0.9]]).T

    # Enforce non-crossing quantiles by sorting per sample.
    val_sorted = np.sort(val_stack, axis=1)
    test_sorted = np.sort(test_stack, axis=1)
    v10, v50, v90 = val_sorted[:, 0], val_sorted[:, 1], val_sorted[:, 2]
    t10, t50, t90 = test_sorted[:, 0], test_sorted[:, 1], test_sorted[:, 2]

    if cap is not None:
        v10 = np.clip(v10, 0, cap); v50 = np.clip(v50, 0, cap); v90 = np.clip(v90, 0, cap)
        t10 = np.clip(t10, 0, cap); t50 = np.clip(t50, 0, cap); t90 = np.clip(t90, 0, cap)
    else:
        v10 = np.clip(v10, 0, None); v50 = np.clip(v50, 0, None); v90 = np.clip(v90, 0, None)
        t10 = np.clip(t10, 0, None); t50 = np.clip(t50, 0, None); t90 = np.clip(t90, 0, None)

    qhat = conformal_qhat(val["target"].to_numpy(), v10, v90, alpha=0.20)
    t10_cal = np.clip(t10 - qhat, 0, cap if cap is not None else None)
    t90_cal = np.clip(t90 + qhat, 0, cap if cap is not None else None)

    return {
        "q10": t10,
        "q50": t50,
        "q90": t90,
        "q10_cal": t10_cal,
        "q90_cal": t90_cal,
        "qhat": qhat,
    }


def run_one(df: pd.DataFrame, user: str, filter_mode: str, require_himawari: bool, horizons: list[int], feature_sets_requested: list[str]):
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

    metric_rows = []
    pred_rows = []

    for horizon_min in horizons:
        if horizon_min % 15 != 0:
            raise ValueError("horizon must be a multiple of 15 minutes")
        h = horizon_min // 15
        featdf = add_causal_supervised_features(df_user, h)
        power_cols = [c for c in featdf.columns if c.startswith("p_lag_") or c.startswith("p_roll_") or c.startswith("p_delta_")]
        time_cols = select_cols(featdf, TIME_COLS)
        weather_cols = select_cols(featdf, WEATHER_COLS)

        if require_himawari and "has_himawari" in featdf.columns:
            featdf = featdf[featdf["has_himawari"] == True].copy()

        sample = featdf.dropna(subset=["target"] + power_cols).copy()

        if filter_mode == "day":
            current_day = sample["irradiance"] > 0 if "irradiance" in sample.columns else sample["p_lag_0"] > 0
            target_day = sample["target"] > 0
            sample = sample[current_day | target_day].copy()
        elif filter_mode == "target_day":
            sample = sample[sample["target"] > 0].copy()

        feature_sets = {
            "power_weather_era5": power_cols + time_cols + weather_cols + era5_cols,
            "power_weather_himawari": power_cols + time_cols + weather_cols + sat_cols,
            "power_weather_era5_himawari": power_cols + time_cols + weather_cols + era5_cols + sat_cols,
        }

        for feature_set in feature_sets_requested:
            if feature_set not in feature_sets:
                raise ValueError(f"Unknown feature_set={feature_set}. Available: {list(feature_sets)}")
            cols = list(dict.fromkeys(select_cols(sample, feature_sets[feature_set])))
            req = cols + ["target"]
            data = sample.dropna(subset=req).copy()
            if len(data) < 180:
                warnings.warn(f"{user} horizon={horizon_min} {feature_set}: only {len(data)} rows.")
            if len(data) < 60:
                warnings.warn(f"Skip {feature_set}, horizon={horizon_min}: not enough rows.")
                continue

            train, val, test = chrono_split(data)
            if len(train) < 40 or len(val) < 10 or len(test) < 10:
                warnings.warn(f"Skip {feature_set}, horizon={horizon_min}: train={len(train)}, val={len(val)}, test={len(test)}")
                continue

            preds = fit_predict_quantiles(train, val, test, cols, cap)
            metrics = calc_interval_metrics(
                test["target"].to_numpy(), preds["q10"], preds["q50"], preds["q90"],
                preds["q10_cal"], preds["q90_cal"], cap=cap,
            )
            metric_rows.append({
                "user": user,
                "filter": filter_mode,
                "protocol": "causal_T_to_TplusH",
                "horizon_min": horizon_min,
                "feature_set": feature_set,
                "model": "LightGBMQuantile" if HAS_LGBM else "GBRQuantile",
                "n_features": len(cols),
                "qhat_conformal": preds["qhat"],
                **metrics,
            })

            out = test[["dt", "target_dt", "target"]].copy()
            out.insert(0, "user", user)
            out.insert(1, "horizon_min", horizon_min)
            out.insert(2, "feature_set", feature_set)
            if "capacity_kw" in test.columns:
                out["capacity_kw"] = test["capacity_kw"].to_numpy()
            out["q10"] = preds["q10"]
            out["q50"] = preds["q50"]
            out["q90"] = preds["q90"]
            out["q10_cal"] = preds["q10_cal"]
            out["q90_cal"] = preds["q90_cal"]
            out["covered_raw"] = (out["target"] >= out["q10"]) & (out["target"] <= out["q90"])
            out["covered_cal"] = (out["target"] >= out["q10_cal"]) & (out["target"] <= out["q90_cal"])
            pred_rows.append(out)

    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_csv", type=Path, required=True)
    p.add_argument("--user", default="f6")
    p.add_argument("--filter", choices=["all", "day", "target_day"], default="day")
    p.add_argument("--require_himawari", action="store_true")
    p.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 60])
    p.add_argument("--feature_sets", nargs="+", default=["power_weather_era5", "power_weather_era5_himawari"])
    p.add_argument("--out_dir", type=Path, default=Path("results/quantile_himawari_causal"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv_smart(args.processed_csv)
    metrics, preds = run_one(
        df=df,
        user=args.user,
        filter_mode=args.filter,
        require_himawari=args.require_himawari,
        horizons=args.horizons,
        feature_sets_requested=args.feature_sets,
    )
    out_metrics = args.out_dir / f"quantile_himawari_causal_{args.user}_{args.filter}.csv"
    out_preds = args.out_dir / f"quantile_predictions_himawari_causal_{args.user}_{args.filter}.csv"
    metrics.to_csv(out_metrics, index=False, encoding="utf-8-sig")
    preds.to_csv(out_preds, index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))
    print("saved metrics:", out_metrics)
    print("saved predictions:", out_preds)


if __name__ == "__main__":
    main()
