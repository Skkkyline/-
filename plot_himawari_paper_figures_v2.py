# -*- coding: utf-8 -*-
"""
plot_himawari_paper_figures_v2.py

用于重新绘制 Himawari 因果消融实验的三张论文图：
1. 融合 Himawari 特征前后的九站点平均 nRMSE 对比
2. 融合 Himawari 后相对 ERA5 组合的 nRMSE 改善率
3. ERA5+Himawari 相对 ERA5 的站点-步长 nRMSE 改善率热力图

默认读取：
    results/paper_tables/table_himawari_causal_mean_by_horizon_fixed.csv
    results/paper_tables/table_himawari_causal_improvement_vs_era5_fixed.csv
    results/paper_tables/table_himawari_site_improvement_vs_era5_fixed.csv

默认输出：
    results/figures_himawari_paper_v2/

运行示例：
    python src/plot_himawari_paper_figures_v2.py --root .

如果你的脚本放在项目根目录，也可以：
    python plot_himawari_paper_figures_v2.py --root .
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# -----------------------------
# 1. 全局绘图风格
# -----------------------------
def setup_matplotlib() -> None:
    """
    设置中文字体和基本绘图参数。
    Windows 常用：Microsoft YaHei, SimHei
    macOS 常用：Arial Unicode MS, PingFang SC
    Linux 可尝试：Noto Sans CJK SC, WenQuanYi Micro Hei
    """
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 120
    matplotlib.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["axes.titlesize"] = 17
    matplotlib.rcParams["axes.labelsize"] = 14
    matplotlib.rcParams["xtick.labelsize"] = 12
    matplotlib.rcParams["ytick.labelsize"] = 12
    matplotlib.rcParams["legend.fontsize"] = 11
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.30
    matplotlib.rcParams["grid.linestyle"] = "--"


FEATURE_LABELS = {
    "persistence": "持续性基线",
    "power_only": "历史功率",
    "power_weather": "历史功率+地面气象",
    "power_weather_era5": "历史功率+地面气象+ERA5",
    "power_weather_himawari": "历史功率+地面气象+Himawari",
    "power_weather_era5_himawari": "历史功率+地面气象+ERA5+Himawari",
}

# 为了图例更短，柱状图可以用这个短标签版本。
FEATURE_LABELS_SHORT = {
    "persistence": "持续性",
    "power_only": "历史功率",
    "power_weather": "历史+气象",
    "power_weather_era5": "历史+气象+ERA5",
    "power_weather_himawari": "历史+气象+Himawari",
    "power_weather_era5_himawari": "历史+气象+ERA5+Himawari",
}

FEATURE_ORDER = [
    "persistence",
    "power_only",
    "power_weather",
    "power_weather_era5",
    "power_weather_himawari",
    "power_weather_era5_himawari",
]

HORIZON_ORDER = [15, 30, 60]


# -----------------------------
# 2. 工具函数
# -----------------------------
def find_csv(root: Path, default_rel: str) -> Path:
    """
    优先读取默认路径。
    如果不存在，兼容用户从 ChatGPT 下载后文件名带 (1) 的情况。
    """
    p = root / default_rel
    if p.exists():
        return p

    parent = p.parent
    stem = p.stem
    suffix = p.suffix

    candidates = sorted(parent.glob(f"{stem}*{suffix}")) if parent.exists() else []
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"没有找到文件：{p}\n"
        f"也没有在 {parent} 下找到 {stem}*{suffix}\n"
        "请确认 fixed 汇总表已经生成，并位于 results/paper_tables/ 下。"
    )


def save_figure(fig: plt.Figure, out_dir: Path, filename: str, save_pdf: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{filename}.png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"[OK] saved: {png_path}")

    if save_pdf:
        pdf_path = out_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[OK] saved: {pdf_path}")


def format_signed(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}"


# -----------------------------
# 3. 图 1：nRMSE 消融柱状图
# -----------------------------
def plot_mean_nrmse_ablation(mean_csv: Path, out_dir: Path, save_pdf: bool = True) -> None:
    df = pd.read_csv(mean_csv)

    required = {"horizon_min", "feature_set", "nRMSE_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{mean_csv} 缺少列：{missing}")

    df = df[df["feature_set"].isin(FEATURE_ORDER)].copy()
    df["feature_set"] = pd.Categorical(df["feature_set"], categories=FEATURE_ORDER, ordered=True)
    df["horizon_min"] = pd.Categorical(df["horizon_min"], categories=HORIZON_ORDER, ordered=True)
    df = df.sort_values(["horizon_min", "feature_set"])

    pivot = df.pivot(index="horizon_min", columns="feature_set", values="nRMSE_pct").loc[HORIZON_ORDER, FEATURE_ORDER]

    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    x = np.arange(len(HORIZON_ORDER))
    n_features = len(FEATURE_ORDER)
    width = 0.12
    offsets = (np.arange(n_features) - (n_features - 1) / 2) * width

    # 使用 Matplotlib 默认颜色循环，避免颜色过于刺眼。
    for j, feat in enumerate(FEATURE_ORDER):
        vals = pivot[feat].values.astype(float)
        bars = ax.bar(
            x + offsets[j],
            vals,
            width=width,
            label=FEATURE_LABELS_SHORT.get(feat, feat),
            edgecolor="white",
            linewidth=0.5,
        )

        # 只给最优组合和 ERA5 组合标注数值，避免图太乱。
        if feat in ["power_weather_era5", "power_weather_era5_himawari"]:
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0,
                )

    ax.set_title("融合 Himawari 特征前后的九站点平均 nRMSE 对比")
    ax.set_xlabel("预测步长")
    ax.set_ylabel("nRMSE / %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} min" for h in HORIZON_ORDER])
    ax.set_ylim(0, max(13.8, np.nanmax(pivot.values) * 1.15))
    ax.grid(axis="y", alpha=0.30)
    ax.grid(axis="x", visible=False)

    # 图例放在图外上方，避免遮挡柱子。
    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        borderaxespad=0.0,
    )

    note = "注：数值为九站点平均归一化 RMSE；柱上数字标注 ERA5 组合与 ERA5+Himawari 组合。"
    ax.text(
        0.5,
        -0.18,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    save_figure(fig, out_dir, "图_Himawari因果消融_nRMSE对比_论文版", save_pdf=save_pdf)
    plt.close(fig)


# -----------------------------
# 4. 图 2：相对 ERA5 的 nRMSE 改善率
# -----------------------------
def plot_improvement_vs_era5(improve_csv: Path, out_dir: Path, save_pdf: bool = True) -> None:
    df = pd.read_csv(improve_csv)

    required = {"horizon_min", "feature_set", "nrmse_improve_vs_era5_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{improve_csv} 缺少列：{missing}")

    show_features = [
        "power_weather_himawari",
        "power_weather_era5_himawari",
    ]

    labels = {
        "power_weather_himawari": "地面气象+Himawari 相对 ERA5",
        "power_weather_era5_himawari": "地面气象+ERA5+Himawari 相对 ERA5",
    }

    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    xs = np.arange(len(HORIZON_ORDER))
    for feat in show_features:
        sub = df[df["feature_set"] == feat].copy()
        sub = sub.set_index("horizon_min").reindex(HORIZON_ORDER)
        vals = sub["nrmse_improve_vs_era5_pct"].astype(float).values

        ax.plot(
            xs,
            vals,
            marker="o",
            linewidth=2.4,
            markersize=7,
            label=labels.get(feat, feat),
        )

        for x, y in zip(xs, vals):
            ax.text(
                x,
                y + (0.10 if y >= 0 else -0.16),
                f"{y:+.2f}%",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=10,
            )

    ax.axhline(0, linestyle="--", linewidth=1.3)
    ax.set_title("融合 Himawari 后相对 ERA5 组合的 nRMSE 改善率")
    ax.set_xlabel("预测步长")
    ax.set_ylabel("nRMSE 改善率 / %")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{h} min" for h in HORIZON_ORDER])

    # 自动留出上下边距。
    yvals = df[df["feature_set"].isin(show_features)]["nrmse_improve_vs_era5_pct"].astype(float).values
    ymin = min(np.nanmin(yvals), 0)
    ymax = max(np.nanmax(yvals), 0)
    pad = max(0.5, (ymax - ymin) * 0.25)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.30)

    note = (
        "注：正值表示相比“历史功率+地面气象+ERA5”组合，nRMSE 降低；"
        "负值表示误差增大。"
    )
    ax.text(
        0.5,
        -0.20,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    save_figure(fig, out_dir, "图_Himawari相对ERA5_nRMSE改善率_论文版", save_pdf=save_pdf)
    plt.close(fig)


# -----------------------------
# 5. 图 3：站点-步长热力图
# -----------------------------
def plot_site_heatmap(site_csv: Path, out_dir: Path, save_pdf: bool = True) -> None:
    df = pd.read_csv(site_csv)

    required = {"user", "horizon_min", "feature_set", "nrmse_improve_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{site_csv} 缺少列：{missing}")

    df = df[df["feature_set"] == "power_weather_era5_himawari"].copy()

    # 自然排序 f1-f9
    df["site_num"] = df["user"].astype(str).str.extract(r"f(\d+)").astype(int)
    site_order = df.sort_values("site_num")["user"].drop_duplicates().tolist()

    pivot = (
        df.pivot(index="user", columns="horizon_min", values="nrmse_improve_pct")
        .reindex(index=site_order, columns=HORIZON_ORDER)
    )

    data = pivot.values.astype(float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("热力图没有有效数值。")

    vmax = max(abs(np.nanmin(finite)), abs(np.nanmax(finite)))
    # 防止所有值都接近 0 时 norm 出错
    vmax = max(vmax, 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.8, 6.3))
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_title("ERA5+Himawari 相对 ERA5 的站点 nRMSE 改善率")
    ax.set_xlabel("预测步长")
    ax.set_ylabel("站点")
    ax.set_xticks(np.arange(len(HORIZON_ORDER)))
    ax.set_xticklabels([f"{h} min" for h in HORIZON_ORDER])
    ax.set_yticks(np.arange(len(site_order)))
    ax.set_yticklabels(site_order)

    # 单元格标注
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                # 大正负值用白字，其余黑字。
                color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=10, color=color)
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=10, color="gray")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("nRMSE 改善率 / %")

    # 画网格线，增强表格感
    ax.set_xticks(np.arange(-0.5, len(HORIZON_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(site_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    note = "注：正值表示融合 Himawari 后相比 ERA5 组合误差降低；负值表示误差增大。"
    ax.text(
        0.5,
        -0.12,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    save_figure(fig, out_dir, "图_Himawari站点_nRMSE改善率热力图_论文版", save_pdf=save_pdf)
    plt.close(fig)


# -----------------------------
# 6. 主函数
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="项目根目录，默认当前目录。")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/figures_himawari_paper_v2",
        help="输出图像目录。",
    )
    parser.add_argument("--no_pdf", action="store_true", help="只保存 PNG，不保存 PDF。")

    parser.add_argument(
        "--mean_csv",
        type=str,
        default=None,
        help="可选：手动指定 table_himawari_causal_mean_by_horizon_fixed.csv 路径。",
    )
    parser.add_argument(
        "--improve_csv",
        type=str,
        default=None,
        help="可选：手动指定 table_himawari_causal_improvement_vs_era5_fixed.csv 路径。",
    )
    parser.add_argument(
        "--site_csv",
        type=str,
        default=None,
        help="可选：手动指定 table_himawari_site_improvement_vs_era5_fixed.csv 路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    save_pdf = not args.no_pdf

    mean_csv = Path(args.mean_csv).resolve() if args.mean_csv else find_csv(
        root, "results/paper_tables/table_himawari_causal_mean_by_horizon_fixed.csv"
    )
    improve_csv = Path(args.improve_csv).resolve() if args.improve_csv else find_csv(
        root, "results/paper_tables/table_himawari_causal_improvement_vs_era5_fixed.csv"
    )
    site_csv = Path(args.site_csv).resolve() if args.site_csv else find_csv(
        root, "results/paper_tables/table_himawari_site_improvement_vs_era5_fixed.csv"
    )

    print("[INFO] mean_csv   =", mean_csv)
    print("[INFO] improve_csv=", improve_csv)
    print("[INFO] site_csv   =", site_csv)
    print("[INFO] out_dir    =", out_dir)

    plot_mean_nrmse_ablation(mean_csv, out_dir, save_pdf=save_pdf)
    plot_improvement_vs_era5(improve_csv, out_dir, save_pdf=save_pdf)
    plot_site_heatmap(site_csv, out_dir, save_pdf=save_pdf)

    print("\n[DONE] 三张论文版 Himawari 图已生成。")


if __name__ == "__main__":
    main()
