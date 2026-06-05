# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd


def main():
    path = Path("data/processed/himawari_features_202303_202304_b13_full.csv")
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date.astype(str)
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    df["user"] = df["user"].astype(str)

    print("=" * 80)
    print("Basic info")
    print("=" * 80)
    print("rows:", len(df))
    print("users:", sorted(df["user"].unique()))
    print("date range:", df["date"].min(), "to", df["date"].max())
    print("time range:", df["time"].min(), "to", df["time"].max())

    print("\nRows per user:")
    print(df.groupby("user").size())

    print("\nRows per date:")
    per_date = df.groupby("date").size()
    print(per_date)

    # Expected dates.
    expected_dates = pd.date_range("2023-03-01", "2023-04-30", freq="D").strftime("%Y-%m-%d").tolist()
    got_dates = set(df["date"].unique())
    missing_dates = [d for d in expected_dates if d not in got_dates]

    print("\nMissing dates:")
    if missing_dates:
        print(missing_dates)
    else:
        print("None")

    # Expected per day: 9 users * 17 time slots = 153 rows.
    low_days = per_date[per_date < 153]
    print("\nDates with rows < 153:")
    if len(low_days):
        print(low_days)
    else:
        print("None")

    # Expected time slots.
    expected_times = pd.date_range("08:00", "16:00", freq="30min").strftime("%H:%M").tolist()

    print("\nMissing time slots by date:")
    any_missing = False
    for d in expected_dates:
        sub = df[df["date"] == d]
        if sub.empty:
            continue
        got_times = set(sub["time"].unique())
        miss = [t for t in expected_times if t not in got_times]
        if miss:
            any_missing = True
            print(d, miss)
    if not any_missing:
        print("None")

    print("\nFeature NaN summary:")
    sat_cols = [c for c in df.columns if c.startswith("sat_")]
    nan_rate = df[sat_cols].isna().mean().sort_values(ascending=False)
    print(nan_rate[nan_rate > 0].head(20))


if __name__ == "__main__":
    main()