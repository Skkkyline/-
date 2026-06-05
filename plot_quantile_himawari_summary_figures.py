# -*- coding: utf-8 -*-
"""
plot_quantile_himawari_summary_figures.py

根据 ERA5 与 ERA5+Himawari 因果分位数预测汇总表，生成论文用中文图：
1. q50 nRMSE 对比
2. 校准前/后 PICP 对比
3. 校准后 PINAW 对比
4. ERA5+Himawari 相比 ERA5 的 q50 nRMSE 改善率

默认读取：
    results/paper_tables/table_quantile_himawari_causal_mean_by_horizon.csv
    results/paper_tables/table_quantile_himawari_causal_improvement_vs_era5.csv

如果你使用我检查后生成的文件，也可以手动指定：
    --mean_csv table_quantile_himawari_causal_mean_by_horizon_checked.csv
    --improve_csv table_quantile_himawari_causal_improvement_vs_era5_checked.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def setup_matplotlib():
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 120
    matplotlib.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["axes.labelsize"] = 13
    matplotlib.rcParams["xtick.labelsize"] = 11
    matplotlib.rcParams["ytick.labelsize"] = 11
    matplotlib.rcParams["legend.fontsize"] = 10
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.30
    matplotlib.rcParams["grid.linestyle"] = "--"


LABELS = {
    "power_weather_era5": "历史功率+地面气象+ERA5",
    "power_weather_era5_himawari": "历史功率+地面气象+ERA5+Himawari",
}
HORIZONS = [15, 30, 60]


def find_csv(root: Path, rel: str) -> Path:
    p = root / rel
    if p.exists():
        return p
    candidates = sorted((root / "results" / "paper_tables").glob(Path(rel).name.replace(".csv", "*.csv")))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(p)


def save(fig, out_dir: Path, name: str, no_pdf: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, bbox_inches="tight")
    print("[OK] saved:", png)
    if not no_pdf:
        pdf = out_dir / f"{name}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print("[OK] saved:", pdf)


def plot_nrmse(mean: pd.DataFrame, out_dir: Path, no_pdf: bool):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(HORIZONS))
    width = 0.32
    for i, fs in enumerate(["power_weather_era5", "power_weather_era5_himawari"]):
        vals = mean[mean["feature_set"] == fs].set_index("horizon_min").reindex(HORIZONS)["nRMSE_pct"].values
        bars = ax.bar(x + (i - 0.5) * width, vals, width=width, label=LABELS[fs], edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("ERA5 与 ERA5+Himawari 分位数模型 q50 nRMSE 对比")
    ax.set_xlabel("预测步长")
    ax.set_ylabel("q50 nRMSE / %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} min" for h in HORIZONS])
    ax.legend()
    ax.set_ylim(0, max(mean["nRMSE_pct"].max() * 1.2, 12))
    fig.tight_layout()
    save(fig, out_dir, "图_分位数预测_q50_nRMSE对比", no_pdf)
    plt.close(fig)


def plot_picp(mean: pd.DataFrame, out_dir: Path, no_pdf: bool):
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    x = np.arange(len(HORIZONS))
    styles = [
        ("power_weather_era5", "PICP_raw", "ERA5 原始 PICP", "o", "-"),
        ("power_weather_era5", "PICP_cal", "ERA5 校准后 PICP", "s", "-"),
        ("power_weather_era5_himawari", "PICP_raw", "ERA5+Himawari 原始 PICP", "o", "--"),
        ("power_weather_era5_himawari", "PICP_cal", "ERA5+Himawari 校准后 PICP", "s", "--"),
    ]
    for fs, col, label, marker, ls in styles:
        vals = mean[mean["feature_set"] == fs].set_index("horizon_min").reindex(HORIZONS)[col].values
        ax.plot(x, vals, marker=marker, linestyle=ls, linewidth=2, label=label)
    ax.axhline(0.80, linestyle=":", linewidth=2, label="理论覆盖率 0.80")
    ax.set_title("分位数预测区间覆盖率 PICP 对比")
    ax.set_xlabel("预测步长")
    ax.set_ylabel("PICP")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} min" for h in HORIZONS])
    ax.set_ylim(0.50, 0.86)
    ax.legend(ncol=2)
    fig.tight_layout()
    save(fig, out_dir, "图_分位数预测_PICP对比", no_pdf)
    plt.close(fig)


def plot_pinaw(mean: pd.DataFrame, out_dir: Path, no_pdf: bool):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(HORIZONS))
    for fs in ["power_weather_era5", "power_weather_era5_himawari"]:
        vals = mean[mean["feature_set"] == fs].set_index("horizon_min").reindex(HORIZONS)["PINAW_cal"].values
        ax.plot(x, vals, marker="o", linewidth=2.2, label=LABELS[fs])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("校准后预测区间归一化宽度 PINAW 对比")
    ax.set_xlabel("预测步长")
    ax.set_ylabel("PINAW")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} min" for h in HORIZONS])
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, "图_分位数预测_PINAW对比", no_pdf)
    plt.close(fig)


def plot_improvement(imp: pd.DataFrame, out_dir: Path, no_pdf: bool):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(len(HORIZONS))
    vals = imp.set_index("horizon_min").reindex(HORIZONS)["nRMSE_improve_pct"].values
    ax.plot(x, vals, marker="o", linewidth=2.4)
    ax.axhline(0, linestyle="--", linewidth=1.2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.08 if v >= 0 else -0.12), f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax.set_title("ERA5+Himawari 相比 ERA5 的 q50 nRMSE 改善率")
    ax.set_xlabel("预测步长")
    ax.set_ylabel("nRMSE 改善率 / %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} min" for h in HORIZONS])
    ymin, ymax = min(vals.min(), 0), max(vals.max(), 0)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    fig.tight_layout()
    save(fig, out_dir, "图_分位数预测_Himawari_nRMSE改善率", no_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--mean_csv", default=None)
    parser.add_argument("--improve_csv", default=None)
    parser.add_argument("--out_dir", default="results/figures_himawari_quantile_summary")
    parser.add_argument("--no_pdf", action="store_true")
    args = parser.parse_args()

    setup_matplotlib()
    root = Path(args.root).resolve()
    mean_csv = Path(args.mean_csv).resolve() if args.mean_csv else find_csv(root, "results/paper_tables/table_quantile_himawari_causal_mean_by_horizon.csv")
    improve_csv = Path(args.improve_csv).resolve() if args.improve_csv else find_csv(root, "results/paper_tables/table_quantile_himawari_causal_improvement_vs_era5.csv")
    out_dir = root / args.out_dir

    mean = pd.read_csv(mean_csv)
    imp = pd.read_csv(improve_csv)
    print("[INFO] mean_csv:", mean_csv)
    print("[INFO] improve_csv:", improve_csv)
    print("[INFO] out_dir:", out_dir)

    plot_nrmse(mean, out_dir, args.no_pdf)
    plot_picp(mean, out_dir, args.no_pdf)
    plot_pinaw(mean, out_dir, args.no_pdf)
    plot_improvement(imp, out_dir, args.no_pdf)

    print("[DONE] Himawari 分位数预测中文论文图已生成。")


if __name__ == "__main__":
    main()
