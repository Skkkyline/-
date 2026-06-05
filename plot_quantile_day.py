# -*- coding: utf-8 -*-
"""Plot one typical day of quantile predictions."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD. If omitted, choose the day with largest daylight fluctuation.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv, parse_dates=["dt"])
    df = df[df["horizon_min"] == args.horizon].copy()
    df["date"] = df["dt"].dt.date.astype(str)

    if args.date is None:
        day_stats = df.groupby("date")["y_true"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
        date = day_stats.index[0]
    else:
        date = args.date

    d = df[df["date"] == date].sort_values("dt").copy()
    if d.empty:
        raise ValueError(f"No rows for date={date}")

    out = Path(args.out) if args.out else Path(f"results/fig_quantile_{args.horizon}min_{date}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(d["dt"], d["y_true"], label="Actual")
    plt.plot(d["dt"], d["q50"], label="q50 prediction")
    plt.fill_between(d["dt"], d["q10_cal"], d["q90_cal"], alpha=0.25, label="Calibrated 80% interval")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.title(f"Quantile forecast interval, horizon={args.horizon} min, date={date}")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"[OK] saved: {out}")


if __name__ == "__main__":
    main()
