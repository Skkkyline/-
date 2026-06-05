import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_csv", type=Path, default=Path("data/processed/processed_all_stations.csv"))
    parser.add_argument("--user", default="f6")
    parser.add_argument("--date", default="2023-03-17")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.processed_csv, parse_dates=["dt"])
    date = pd.to_datetime(args.date).date()
    sub = df[(df["user_id"] == args.user) & (df["dt"].dt.date == date)].copy()
    if sub.empty:
        raise ValueError("No data found for the requested user/date")

    plt.figure(figsize=(10, 4.8))
    plt.plot(sub["dt"], sub["power_kw_clean"], label="Actual power")
    plt.plot(sub["dt"], sub["irradiance"] / max(sub["irradiance"].max(), 1) * sub["capacity_kw"].iloc[0], label="Scaled irradiance")
    plt.title(f"{args.user} daily case: {args.date}")
    plt.xlabel("Time")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    out = args.out_dir / f"daily_case_{args.user}_{args.date}.png"
    plt.savefig(out, dpi=200)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
