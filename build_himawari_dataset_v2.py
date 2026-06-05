# -*- coding: utf-8 -*-
"""
Build a PV + ERA5 + Himawari dataset from processed PV samples and Himawari feature CSV.

This script is a safer replacement / extension for merge_himawari_features.py.
It keeps the matched satellite timestamp, computes satellite age, and adds derived
cloud-proxy features from B13 brightness temperature.

Example:
    python src/build_himawari_dataset_v2.py ^
      --processed_csv data/processed/processed_all_stations_era5_timeseries_new.csv ^
      --himawari_csv data/processed/himawari_features_sample.csv ^
      --out_csv data/processed/processed_all_stations_era5_himawari_sample_v2.csv ^
      --matched_csv data/processed/processed_himawari_matched_only_sample_v2.csv ^
      --tolerance_min 75
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def detect_user_col(df: pd.DataFrame) -> str:
    for c in ["user_id", "user", "光伏用户编号"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "user" in c.lower() or "用户" in str(c):
            return c
    raise KeyError(f"Cannot detect user column from {list(df.columns)}")


def detect_time_col(df: pd.DataFrame) -> str:
    for c in ["dt", "datetime", "time", "timestamp"]:
        if c in df.columns:
            return c
    for c in df.columns:
        s = str(c).lower()
        if "time" in s or "date" in s or "时间" in str(c) or "日期" in str(c):
            return c
    raise KeyError(f"Cannot detect time column from {list(df.columns)}")


def recompute_b13_diffs(sat: pd.DataFrame, max_gap_min: float = 90.0) -> pd.DataFrame:
    """Recompute B13 temporal difference features safely.

    The original sample may compute diffs across non-contiguous dates. This function resets
    diffs at the first record of each site/day and whenever the time gap is too large.
    """
    sat = sat.copy()
    if "sat_datetime" not in sat.columns:
        return sat
    sat["_sat_date"] = pd.to_datetime(sat["sat_datetime"]).dt.date.astype(str)
    sat = sat.sort_values(["user_id", "sat_datetime"]).copy()
    gap_min = sat.groupby("user_id")["sat_datetime"].diff().dt.total_seconds() / 60.0

    diff_bases = {
        "sat_b13_mean": "sat_b13_mean_diff",
        "sat_b13_center": "sat_b13_center_diff",
        "sat_b13_center_minus_mean": "sat_b13_center_minus_mean_diff",
        "sat_b13_grad_mean": "sat_b13_grad_mean_diff",
    }
    for base, diff_col in diff_bases.items():
        if base not in sat.columns:
            continue
        sat[diff_col] = sat.groupby(["user_id", "_sat_date"])[base].diff()
        sat.loc[gap_min > max_gap_min, diff_col] = np.nan
        sat[diff_col + "_abs"] = sat[diff_col].abs()
    sat = sat.drop(columns=["_sat_date"], errors="ignore")
    return sat


def add_b13_derived_features(sat: pd.DataFrame) -> pd.DataFrame:
    """Add cloud-proxy features from B13 brightness temperature."""
    sat = sat.copy()
    if "sat_b13_mean" in sat.columns:
        # B13 is thermal infrared brightness temperature. Cold values usually indicate high/cold clouds;
        # the negative mean can be a rough cloudiness proxy after normalization/model learning.
        sat["sat_b13_inv_mean"] = -sat["sat_b13_mean"]
        sat["sat_b13_coldness_mean_273"] = np.maximum(0.0, 273.15 - sat["sat_b13_mean"])
        sat["sat_b13_coldness_mean_263"] = np.maximum(0.0, 263.15 - sat["sat_b13_mean"])
    if "sat_b13_center" in sat.columns:
        sat["sat_b13_inv_center"] = -sat["sat_b13_center"]
        sat["sat_b13_coldness_center_273"] = np.maximum(0.0, 273.15 - sat["sat_b13_center"])
        sat["sat_b13_coldness_center_263"] = np.maximum(0.0, 263.15 - sat["sat_b13_center"])
    for c in [
        "sat_b13_mean_diff",
        "sat_b13_center_diff",
        "sat_b13_center_minus_mean_diff",
        "sat_b13_grad_mean_diff",
    ]:
        if c in sat.columns:
            sat[c + "_abs"] = sat[c].abs()
    return sat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed_csv", default="data/processed/processed_all_stations_era5_timeseries_new.csv")
    p.add_argument("--himawari_csv", default="data/processed/himawari_features_sample.csv")
    p.add_argument("--out_csv", default="data/processed/processed_all_stations_era5_himawari_sample_v2.csv")
    p.add_argument("--matched_csv", default="data/processed/processed_himawari_matched_only_sample_v2.csv")
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

    if pv_user_col != "user_id":
        pv = pv.rename(columns={pv_user_col: "user_id"})
    if pv_time_col != "dt":
        pv = pv.rename(columns={pv_time_col: "dt"})
    if sat_user_col != "user_id":
        sat = sat.rename(columns={sat_user_col: "user_id"})
    if sat_time_col != "sat_datetime":
        sat = sat.rename(columns={sat_time_col: "sat_datetime"})

    pv["dt"] = pd.to_datetime(pv["dt"])
    sat["sat_datetime"] = pd.to_datetime(sat["sat_datetime"])

    sat = recompute_b13_diffs(sat)
    sat = add_b13_derived_features(sat)

    sat_cols = [c for c in sat.columns if c.startswith("sat_")]
    keep_sat = ["user_id", "sat_datetime"] + [c for c in sat_cols if c != "sat_datetime"]
    sat = sat[keep_sat].copy()

    out_parts = []
    for user, gpv in pv.groupby("user_id", sort=True):
        gsat = sat[sat["user_id"] == user].copy()
        gpv = gpv.sort_values("dt")
        gsat = gsat.sort_values("sat_datetime")
        if gsat.empty:
            tmp = gpv.copy()
            tmp["sat_datetime"] = pd.NaT
            for c in [x for x in sat.columns if x not in ["user_id", "sat_datetime"]]:
                tmp[c] = np.nan
            out_parts.append(tmp)
            continue

        merged = pd.merge_asof(
            gpv,
            gsat.drop(columns=["user_id"]),
            left_on="dt",
            right_on="sat_datetime",
            direction="backward",
            tolerance=pd.Timedelta(minutes=args.tolerance_min),
        )
        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True)
    sat_feature_cols = [c for c in out.columns if c.startswith("sat_b")]
    out["has_himawari"] = out[sat_feature_cols].notna().any(axis=1) if sat_feature_cols else False
    out["sat_age_min"] = (out["dt"] - out["sat_datetime"]).dt.total_seconds() / 60.0

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    matched = out[out["has_himawari"]].copy()
    matched_path = Path(args.matched_csv)
    matched_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(matched_path, index=False, encoding="utf-8-sig")

    print("[OK] saved full merged:", out_path)
    print("[OK] saved matched only:", matched_path)
    print("[OK] full rows:", len(out), "matched rows:", len(matched))
    print("[OK] has_himawari counts:")
    print(out["has_himawari"].value_counts(dropna=False).to_string())
    if not matched.empty:
        print("[OK] satellite age minutes summary:")
        print(matched["sat_age_min"].describe().to_string())
        print("[OK] matched rows by date:")
        print(matched.groupby(matched["dt"].dt.date).size().to_string())
        print("[OK] satellite feature columns:", sat_feature_cols[:40])


if __name__ == "__main__":
    main()
