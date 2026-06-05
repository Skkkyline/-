# -*- coding: utf-8 -*-
"""
Analyze the Himawari sample dataset before model training.

It creates:
  - coverage summary by site/date
  - correlation between satellite features and normalized PV power
  - correlation between satellite features and absolute power ramp

Example:
    python src/analyze_himawari_sample.py ^
      --merged_csv data/processed/processed_himawari_matched_only_sample_v2.csv ^
      --out_dir results/himawari_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def detect_power_col(df: pd.DataFrame) -> str:
    for c in ["power_kw_clean", "power_kw", "power", "actual_power", "实际功率"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "power" in c.lower() or "功率" in str(c):
            return c
    raise KeyError("Cannot detect power column")


def detect_capacity_col(df: pd.DataFrame) -> str | None:
    for c in ["capacity_kw", "装机容量(kW)", "capacity"]:
        if c in df.columns:
            return c
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--merged_csv", default="data/processed/processed_himawari_matched_only_sample_v2.csv")
    p.add_argument("--out_dir", default="results/himawari_analysis")
    return p.parse_args()


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.concat([a, b], axis=1).dropna()
    if len(tmp) < 8:
        return np.nan
    if tmp.iloc[:, 0].std() == 0 or tmp.iloc[:, 1].std() == 0:
        return np.nan
    return float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged_csv)
    if "dt" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "dt"})
    if "user_id" not in df.columns and "user" in df.columns:
        df = df.rename(columns={"user": "user_id"})
    df["dt"] = pd.to_datetime(df["dt"])
    df["date"] = df["dt"].dt.date.astype(str)

    power_col = detect_power_col(df)
    cap_col = detect_capacity_col(df)
    if cap_col:
        df["power_pu"] = df[power_col] / df[cap_col]
    else:
        df["power_pu"] = df.groupby("user_id")[power_col].transform(lambda s: s / s.max())

    df = df.sort_values(["user_id", "dt"]).copy()
    df["power_pu_ramp"] = df.groupby("user_id")["power_pu"].diff()
    df["power_pu_ramp_abs"] = df["power_pu_ramp"].abs()
    df["power_pu_future60"] = df.groupby("user_id")["power_pu"].shift(-4)
    df["power_pu_future60_ramp"] = df["power_pu_future60"] - df["power_pu"]
    df["power_pu_future60_ramp_abs"] = df["power_pu_future60_ramp"].abs()

    coverage = df.groupby(["date", "user_id"]).agg(
        rows=("dt", "count"),
        start=("dt", "min"),
        end=("dt", "max"),
        mean_sat_age_min=("sat_age_min", "mean") if "sat_age_min" in df.columns else ("dt", "count"),
        max_power_pu=("power_pu", "max"),
        mean_power_pu=("power_pu", "mean"),
    ).reset_index()
    coverage_path = out_dir / "himawari_sample_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    sat_features = [c for c in df.columns if c.startswith("sat_b13_")]
    # Focus on features that are usually meaningful for cloud / fluctuation.
    preferred_order = [
        "sat_b13_mean", "sat_b13_inv_mean", "sat_b13_std", "sat_b13_grad_mean",
        "sat_b13_cold_ratio_273", "sat_b13_cold_ratio_263",
        "sat_b13_center", "sat_b13_center_minus_mean",
        "sat_b13_mean_diff", "sat_b13_mean_diff_abs",
        "sat_b13_center_diff", "sat_b13_center_diff_abs",
        "sat_b13_coldness_mean_273", "sat_b13_coldness_center_273",
    ]
    sat_features = [c for c in preferred_order if c in df.columns] + [c for c in sat_features if c not in preferred_order]

    rows = []
    for feat in sat_features:
        rows.append({
            "feature": feat,
            "corr_with_power_pu": safe_corr(df[feat], df["power_pu"]),
            "corr_with_abs_ramp": safe_corr(df[feat], df["power_pu_ramp_abs"]),
            "corr_with_future60_abs_ramp": safe_corr(df[feat], df["power_pu_future60_ramp_abs"]),
            "min": float(df[feat].min(skipna=True)),
            "max": float(df[feat].max(skipna=True)),
            "mean": float(df[feat].mean(skipna=True)),
            "std": float(df[feat].std(skipna=True)),
            "non_null": int(df[feat].notna().sum()),
        })
    corr = pd.DataFrame(rows)
    corr_path = out_dir / "himawari_feature_correlations.csv"
    corr.to_csv(corr_path, index=False, encoding="utf-8-sig")

    print("[OK] saved:", coverage_path)
    print("[OK] saved:", corr_path)
    print("\nCoverage head:")
    print(coverage.head(20).to_string(index=False))
    print("\nTop correlations with future 60-min absolute ramp:")
    print(corr.reindex(corr["corr_with_future60_abs_ramp"].abs().sort_values(ascending=False).index).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
