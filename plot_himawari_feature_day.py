# -*- coding: utf-8 -*-
"""
Plot normalized PV power and Himawari B13 features for one station/day.

Usage:
    python src/plot_himawari_feature_day.py --merged_csv data/processed/processed_all_stations_era5_himawari_sample.csv --site f6 --date 2023-04-15 --out_dir results/figures_himawari
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def detect_power_col(df):
    for c in ["power", "power_kw", "actual_power", "实际功率"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "power" in c.lower() or "功率" in c:
            return c
    raise KeyError("Cannot detect power column")


def detect_capacity_col(df):
    for c in ["capacity_kw", "装机容量(kW)", "capacity"]:
        if c in df.columns:
            return c
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_csv", default="data/processed/processed_all_stations_era5_himawari_sample.csv")
    p.add_argument("--site", default="f6")
    p.add_argument("--date", required=True)
    p.add_argument("--feature", default="sat_b13_mean", help="Satellite feature to plot")
    p.add_argument("--out_dir", default="results/figures_himawari")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.merged_csv)
    if "datetime" not in df.columns and "dt" in df.columns:
        df = df.rename(columns={"dt": "datetime"})
    if "user" not in df.columns and "user_id" in df.columns:
        df = df.rename(columns={"user_id": "user"})

    df["datetime"] = pd.to_datetime(df["datetime"])
    day = pd.to_datetime(args.date).date()
    sub = df[(df["user"] == args.site) & (df["datetime"].dt.date == day)].copy()
    if sub.empty:
        raise ValueError(f"No data for site={args.site}, date={args.date}")

    power_col = detect_power_col(sub)
    cap_col = detect_capacity_col(sub)
    if cap_col is not None:
        sub["power_pu"] = sub[power_col] / sub[cap_col]
    else:
        sub["power_pu"] = sub[power_col] / sub[power_col].max()

    if args.feature not in sub.columns:
        raise KeyError(f"Feature not found: {args.feature}. Available sat cols: {[c for c in sub.columns if c.startswith('sat_')][:50]}")

    # Normalize satellite feature to 0-1 for plotting on same axis.
    sat = sub[args.feature]
    sub["sat_norm"] = (sat - sat.min()) / (sat.max() - sat.min()) if sat.max() != sat.min() else sat * 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"图_{args.site}_{args.date}_{args.feature}_与功率对比.png"

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    ax.plot(sub["datetime"], sub["power_pu"], label="归一化实测功率")
    ax.step(sub["datetime"], sub["sat_norm"], where="post", label=f"{args.feature} 归一化")
    ax.set_title(f"{args.site} 场站 Himawari 特征与光伏功率对比（{args.date}）")
    ax.set_xlabel("时间")
    ax.set_ylabel("归一化值")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("[OK] saved:", out_png)


if __name__ == "__main__":
    main()
