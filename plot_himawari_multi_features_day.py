# -*- coding: utf-8 -*-
"""
Plot normalized PV power and multiple normalized Himawari features on one day.

Example:
    python src/plot_himawari_multi_features_day.py ^
      --merged_csv data/processed/processed_himawari_matched_only_sample_v2.csv ^
      --site f6 --date 2023-04-15 ^
      --features sat_b13_inv_mean sat_b13_std sat_b13_grad_mean sat_b13_cold_ratio_273
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def detect_power_col(df):
    for c in ["power_kw_clean", "power_kw", "power", "actual_power", "实际功率"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "power" in c.lower() or "功率" in str(c):
            return c
    raise KeyError("Cannot detect power column")


def detect_capacity_col(df):
    for c in ["capacity_kw", "装机容量(kW)", "capacity"]:
        if c in df.columns:
            return c
    return None


def norm01(s):
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return s * 0
    return (s - mn) / (mx - mn)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_csv", default="data/processed/processed_himawari_matched_only_sample_v2.csv")
    p.add_argument("--site", default="f6")
    p.add_argument("--date", required=True)
    p.add_argument("--features", nargs="+", default=["sat_b13_inv_mean", "sat_b13_std", "sat_b13_grad_mean", "sat_b13_cold_ratio_273"])
    p.add_argument("--out_dir", default="results/figures_himawari")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.merged_csv)
    if "dt" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "dt"})
    if "user_id" not in df.columns and "user" in df.columns:
        df = df.rename(columns={"user": "user_id"})
    df["dt"] = pd.to_datetime(df["dt"])
    day = pd.to_datetime(args.date).date()
    sub = df[(df["user_id"] == args.site) & (df["dt"].dt.date == day)].copy()
    if sub.empty:
        raise ValueError(f"No data for site={args.site}, date={args.date}")

    power_col = detect_power_col(sub)
    cap_col = detect_capacity_col(sub)
    if cap_col:
        sub["power_pu"] = sub[power_col] / sub[cap_col]
    else:
        sub["power_pu"] = sub[power_col] / sub[power_col].max()

    available = [f for f in args.features if f in sub.columns]
    missing = [f for f in args.features if f not in sub.columns]
    if missing:
        print("[WARN] missing features:", missing)
    if not available:
        raise ValueError("No requested feature is available.")

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    ax.plot(sub["dt"], sub["power_pu"], linewidth=2.2, label="归一化实测功率")
    for feat in available:
        ax.step(sub["dt"], norm01(sub[feat]), where="post", linewidth=1.6, alpha=0.9, label=f"{feat} 归一化")
    ax.set_title(f"{args.site} 场站 Himawari 多特征与光伏功率对比（{args.date}）")
    ax.set_xlabel("时间")
    ax.set_ylabel("归一化值")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"图_{args.site}_{args.date}_Himawari多特征与功率对比.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("[OK] saved:", out_png)


if __name__ == "__main__":
    main()
