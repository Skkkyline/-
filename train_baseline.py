import argparse
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
        "R2": float(r2_score(y, pred)),
        "n": int(len(y)),
    }


def run_one(df: pd.DataFrame, user: str, filter_mode: str):
    rows = []
    df_user = df[df["user_id"] == user].copy()
    cap = float(df_user["capacity_kw"].dropna().iloc[0])

    for h in [1, 2, 4]:
        featdf = add_supervised_features(df_user, h)
        power_cols = [c for c in featdf.columns if c.startswith(("lag_", "roll_", "delta_"))]
        sample = featdf.dropna(subset=["target"] + power_cols).copy()

        if filter_mode == "day":
            sample = sample[(sample["irradiance"] > 0) | (sample["target"] > 0)]

        train, val, test = chrono_split(sample)
        pred_col = f"lag_{h}"
        rows.append({
            "user": user, "filter": filter_mode, "horizon_min": h * 15,
            "feature_set": "persistence", "model": "Persistence",
            **calc_metrics(test["target"], test[pred_col])
        })

        for feature_set, cols in {
            "power_only": power_cols + TIME_COLS,
            "power_weather": power_cols + TIME_COLS + WEATHER_COLS,
        }.items():
            train_all = pd.concat([train, val], axis=0)
            model = XGBRegressor(
                n_estimators=450,
                max_depth=4,
                learning_rate=0.04,
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
            pred = np.clip(model.predict(test[cols]), 0, cap)
            rows.append({
                "user": user, "filter": filter_mode, "horizon_min": h * 15,
                "feature_set": feature_set, "model": "XGBoost",
                **calc_metrics(test["target"], pred)
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_csv", type=Path, default=Path("data/processed/processed_all_stations.csv"))
    parser.add_argument("--user", default="f6")
    parser.add_argument("--filter", choices=["all", "day"], default="day")
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.processed_csv, parse_dates=["dt"])
    res = run_one(df, args.user, args.filter)
    out = args.out_dir / f"baseline_{args.user}_{args.filter}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
