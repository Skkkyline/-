import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_baseline import WEATHER_COLS, TIME_COLS, add_supervised_features, chrono_split


def pinball_loss(y, pred, q):
    y = np.asarray(y)
    pred = np.asarray(pred)
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_metrics(y, low, high, norm):
    y = np.asarray(y)
    lo = np.minimum(low, high)
    hi = np.maximum(low, high)
    return {
        "PICP": float(np.mean((y >= lo) & (y <= hi))),
        "PINAW": float(np.mean(hi - lo) / max(norm, 1e-9)),
        "AvgWidth": float(np.mean(hi - lo)),
    }


def conformal_delta(y_cal, low_cal, high_cal, alpha=0.2):
    y = np.asarray(y_cal)
    lo = np.minimum(low_cal, high_cal)
    hi = np.maximum(low_cal, high_cal)
    scores = np.maximum(np.maximum(lo - y, y - hi), 0)
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def run_quantile(df: pd.DataFrame, user: str, filter_mode: str):
    rows = []
    df_user = df[df["user_id"] == user].copy()
    cap = float(df_user["capacity_kw"].dropna().iloc[0])

    for h in [1, 2, 4]:
        featdf = add_supervised_features(df_user, h)
        power_cols = [c for c in featdf.columns if c.startswith(("lag_", "roll_", "delta_"))]
        cols = power_cols + TIME_COLS + WEATHER_COLS
        sample = featdf.dropna(subset=["target"] + power_cols).copy()

        if filter_mode == "day":
            sample = sample[(sample["irradiance"] > 0) | (sample["target"] > 0)]

        train, val, test = chrono_split(sample, train_frac=0.70, val_frac=0.15)
        pred_val = {}
        pred_test = {}

        for q in [0.1, 0.5, 0.9]:
            model = LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=180,
                learning_rate=0.05,
                num_leaves=24,
                min_child_samples=50,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=4,
                verbose=-1,
            )
            model.fit(train[cols], train["target"])
            pred_val[q] = np.clip(model.predict(val[cols]), 0, cap)
            pred_test[q] = np.clip(model.predict(test[cols]), 0, cap)

        y_test = test["target"].values
        low = pred_test[0.1]
        med = pred_test[0.5]
        high = pred_test[0.9]

        raw = interval_metrics(y_test, low, high, cap)
        delta = conformal_delta(val["target"].values, pred_val[0.1], pred_val[0.9], alpha=0.2)
        cal = interval_metrics(y_test, low - delta, high + delta, cap)

        rows.append({
            "user": user,
            "filter": filter_mode,
            "horizon_min": h * 15,
            "model": "LightGBM-Quantile",
            "RMSE": float(np.sqrt(mean_squared_error(y_test, med))),
            "MAE": float(mean_absolute_error(y_test, med)),
            "R2": float(r2_score(y_test, med)),
            "Pinball_q50": pinball_loss(y_test, med, 0.5),
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
    res = run_quantile(df, args.user, args.filter)
    out = args.out_dir / f"quantile_{args.user}_{args.filter}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
