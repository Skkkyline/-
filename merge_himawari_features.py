# -*- coding: utf-8 -*-
"""
Merge Himawari features into processed PV + ERA5 samples.

Usage:
    python src/merge_himawari_features.py --processed_csv data/processed/processed_all_stations_era5_timeseries_new.csv --himawari_csv data/processed/himawari_features_sample.csv --out_csv data/processed/processed_all_stations_era5_himawari_sample.csv --tolerance_min 75
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def detect_user_col(df: pd.DataFrame) -> str:
    for c in ["user", "user_id", "光伏用户编号"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "user" in c.lower() or "用户" in c:
            return c
    raise KeyError(f"Cannot detect user column from {list(df.columns)}")


def detect_time_col(df: pd.DataFrame) -> str:
    for c in ["dt", "datetime", "time", "timestamp"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower() or "时间" in c or "日期" in c:
            return c
    raise KeyError(f"Cannot detect datetime column from {list(df.columns)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed_csv", default="data/processed/processed_all_stations_era5_timeseries_new.csv")
    p.add_argument("--himawari_csv", default="data/processed/himawari_features_sample.csv")
    p.add_argument("--out_csv", default="data/processed/processed_all_stations_era5_himawari_sample.csv")
    p.add_argument("--tolerance_min", type=int, default=75)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pv = pd.read_csv(args.processed_csv)
    sat = pd.read_csv(args.himawari_csv)

    pv_user_col = detect_user_col(pv)
    pv_time_col = detect_time_col(pv)
    sat_user_col = detect_user_col(sat)
    sat_time_col = detect_time_col(sat)

    if pv_user_col != "user":
        pv = pv.rename(columns={pv_user_col: "user"})
    if pv_time_col != "datetime":
        pv = pv.rename(columns={pv_time_col: "datetime"})
    if sat_user_col != "user":
        sat = sat.rename(columns={sat_user_col: "user"})
    if sat_time_col != "datetime":
        sat = sat.rename(columns={sat_time_col: "datetime"})

    pv["datetime"] = pd.to_datetime(pv["datetime"])
    sat["datetime"] = pd.to_datetime(sat["datetime"])

    sat_cols = [c for c in sat.columns if c.startswith("sat_")]
    sat = sat[["user", "datetime"] + sat_cols].copy()

    out_parts = []
    for user, gpv in pv.groupby("user", sort=True):
        gsat = sat[sat["user"] == user].copy()
        gpv = gpv.sort_values("datetime")
        gsat = gsat.sort_values("datetime")
        if gsat.empty:
            tmp = gpv.copy()
            for c in sat_cols:
                tmp[c] = pd.NA
            out_parts.append(tmp)
            continue
        merged = pd.merge_asof(
            gpv,
            gsat.drop(columns=["user"]),
            on="datetime",
            direction="backward",
            tolerance=pd.Timedelta(minutes=args.tolerance_min),
        )
        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True)
    sat_feature_cols = [c for c in out.columns if c.startswith("sat_b")]
    out["has_himawari"] = out[sat_feature_cols].notna().any(axis=1) if sat_feature_cols else False

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("[OK] saved:", out_path)
    print("[OK] rows:", len(out), "cols:", len(out.columns))
    print("[OK] has_himawari counts:")
    print(out["has_himawari"].value_counts(dropna=False).to_string())
    if sat_feature_cols:
        print("[OK] first satellite feature columns:", sat_feature_cols[:20])


if __name__ == "__main__":
    main()
