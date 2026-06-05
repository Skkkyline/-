# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--site", default=None)
    ap.add_argument("--feature_set", default="power_weather_era5_himawari")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--date", default=None, help="Target date, e.g. 2023-04-15. If omitted, pick the most volatile date.")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--out_dir", default="results/figures_himawari_quantile")
    args = ap.parse_args()

    setup_chinese_font()
    df = pd.read_csv(args.pred_csv, parse_dates=["dt", "target_dt"])
    if args.site and "user" in df.columns:
        df = df[df["user"].astype(str) == str(args.site)].copy()
    df = df[(df["feature_set"] == args.feature_set) & (df["horizon_min"] == args.horizon)].copy()
    if df.empty:
        raise ValueError("No matching rows for selected feature_set/horizon/site.")

    df["date"] = df["target_dt"].dt.strftime("%Y-%m-%d")
    if args.date is None:
        vol = df.groupby("date")["target"].agg(lambda s: s.diff().abs().sum()).sort_values(ascending=False)
        date = vol.index[0]
    else:
        date = args.date
    day = df[df["date"] == date].sort_values("target_dt").copy()
    if day.empty:
        raise ValueError(f"No rows for date={date}")

    ycols = ["target", "q50", "q10_cal", "q90_cal"]
    ylabel = "功率 / kW"
    if args.normalize:
        cap = float(day["capacity_kw"].dropna().iloc[0]) if "capacity_kw" in day.columns else day["target"].max()
        for c in ycols:
            day[c] = day[c] / cap
        ylabel = "归一化功率 / p.u."

    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    ax.plot(day["target_dt"], day["target"], label="实测功率", linewidth=2.0)
    ax.plot(day["target_dt"], day["q50"], label="q50 中位数预测", linewidth=2.0)
    ax.fill_between(day["target_dt"], day["q10_cal"], day["q90_cal"], alpha=0.25, label="校准后 80% 预测区间")
    ax.set_title(f"{args.site or ''} 场站 {args.horizon} min ERA5+Himawari 区间预测结果（{date}）")
    ax.set_xlabel("时间")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_tag = "_归一化" if args.normalize else ""
    site_tag = args.site or "site"
    out = out_dir / f"图_{site_tag}_{args.horizon}min_{date}_ERA5_Himawari区间预测{norm_tag}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("[OK] saved:", out)


if __name__ == "__main__":
    main()
