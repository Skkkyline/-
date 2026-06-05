# -*- coding: utf-8 -*-
"""
generate_missing_thesis_figures.py

用途：
    生成毕业论文中还缺少的“方法类/流程类/数据类”图片，配合已有实验结果图，
    帮助满足“图不少于 25 张、图类型不少于 5 类”的要求。

推荐放置位置：
    pv_thesis_rebuild/src/generate_missing_thesis_figures.py

运行：
    cd 你的 pv_thesis_rebuild 项目根目录
    .\.venv\Scripts\python.exe src\generate_missing_thesis_figures.py --root .

输出：
    results/final_paper/figures_generated/

说明：
    本脚本不会替代你的实验结果图。它主要生成：
    - 系统架构图
    - 数据处理流程图
    - 因果预测样本构造图
    - 多源特征融合拓扑图
    - Himawari patch 特征构造图
    - 分位数预测与 conformal 校准流程图
    - 站点空间分布图
    - 装机容量柱状图
    - 功率日内平均曲线图
    - 样本覆盖热力图
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def setup_matplotlib() -> None:
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
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["axes.labelsize"] = 13
    matplotlib.rcParams["xtick.labelsize"] = 11
    matplotlib.rcParams["ytick.labelsize"] = 11
    matplotlib.rcParams["legend.fontsize"] = 10
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.25
    matplotlib.rcParams["grid.linestyle"] = "--"


def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    print("[OK]", out_dir / f"{name}.png")
    plt.close(fig)


def add_box(ax, xy, text, w=1.8, h=0.55):
    x, y = xy
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.4)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)
    return rect


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.2)
    ax.add_patch(arrow)


def fig_overall_framework(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axis_off()
    ax.set_title("多源气象遥感融合的光伏功率超短期概率预测总体框架")

    # Input layer
    boxes = {
        "power": add_box(ax, (0.2, 4.5), "历史功率\nP(t-k)", 1.6, 0.7),
        "meteo": add_box(ax, (2.1, 4.5), "地面气象\nDataFountain", 1.8, 0.7),
        "era5": add_box(ax, (4.3, 4.5), "ERA5 再分析\n背景气象", 1.8, 0.7),
        "him": add_box(ax, (6.5, 4.5), "Himawari-9\nB13 云图", 1.8, 0.7),
    }
    fuse = add_box(ax, (3.1, 3.1), "时间对齐与特征融合\nT 时刻及历史信息", 2.8, 0.75)
    point = add_box(ax, (1.3, 1.7), "XGBoost / GBDT\n点预测", 2.1, 0.75)
    quant = add_box(ax, (5.4, 1.7), "LightGBM Quantile\nq10/q50/q90", 2.4, 0.75)
    calib = add_box(ax, (5.4, 0.55), "Conformal Calibration\n校准预测区间", 2.4, 0.75)
    out1 = add_box(ax, (1.3, 0.55), "P(t+15/30/60)\n点预测结果", 2.1, 0.75)
    out2 = add_box(ax, (8.2, 0.55), "PICP / PINAW\n不确定性评价", 2.0, 0.75)

    for b in boxes.values():
        add_arrow(ax, (b.get_x() + b.get_width() / 2, b.get_y()), (4.5, 3.85))
    add_arrow(ax, (4.0, 3.1), (2.35, 2.45))
    add_arrow(ax, (5.0, 3.1), (6.6, 2.45))
    add_arrow(ax, (2.35, 1.7), (2.35, 1.3))
    add_arrow(ax, (6.6, 1.7), (6.6, 1.3))
    add_arrow(ax, (7.8, 0.95), (8.2, 0.95))

    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5.6)
    save(fig, out_dir, "图01_总体技术路线图")


def fig_data_preprocessing_flow(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_axis_off()
    ax.set_title("多源数据预处理与样本构建流程")

    steps = [
        "原始功率\np1-p96",
        "地面气象\n15 min",
        "ERA5\n小时级",
        "Himawari\n30 min",
        "时间统一\n北京时间",
        "缺失/异常处理",
        "因果对齐\n仅用 T 及以前",
        "15/30/60 min\n监督样本",
        "训练/验证/测试\n时间顺序划分",
    ]

    xs = np.linspace(0.2, 10.0, len(steps))
    y = 2.2
    for i, (x, text) in enumerate(zip(xs, steps)):
        add_box(ax, (x, y), text, 1.05, 0.75)
        if i < len(steps) - 1:
            add_arrow(ax, (x + 1.05, y + 0.38), (xs[i + 1], y + 0.38))

    ax.set_xlim(0, 11.2)
    ax.set_ylim(0.5, 4.0)
    save(fig, out_dir, "图02_多源数据预处理流程图")


def fig_causal_sample(out_dir: Path):
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_axis_off()
    ax.set_title("严格因果预测样本构造示意图")

    times = ["T-180", "T-120", "T-60", "T-30", "T", "T+15", "T+30", "T+60"]
    xs = np.linspace(0.8, 10.5, len(times))
    y = 2.2

    ax.plot(xs, [y] * len(xs), marker="o")
    for x, t in zip(xs, times):
        ax.text(x, y - 0.35, t, ha="center", va="top")

    ax.text(3.1, 3.15, "可用输入窗口：历史功率、地面气象、ERA5、Himawari", ha="center", fontsize=11)
    ax.text(8.7, 3.15, "预测目标", ha="center", fontsize=11)

    ax.annotate("", xy=(xs[4], 2.75), xytext=(xs[0], 2.75), arrowprops=dict(arrowstyle="<->", linewidth=1.2))
    for idx in [5, 6, 7]:
        ax.annotate("", xy=(xs[idx], 1.55), xytext=(xs[4], 1.55), arrowprops=dict(arrowstyle="->", linewidth=1.2))
        ax.text((xs[4] + xs[idx]) / 2, 1.25, f"h={times[idx][2:]} min", ha="center", fontsize=10)

    ax.text(5.5, 0.7, "原则：预测 T+h 时不得使用 T+h 时刻的卫星图像或真实未来气象。", ha="center", fontsize=11)
    ax.set_xlim(0, 11.3)
    ax.set_ylim(0.3, 3.8)
    save(fig, out_dir, "图03_因果预测样本构造图")


def fig_multisource_topology(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_axis_off()
    ax.set_title("多源特征融合拓扑结构图")

    center = add_box(ax, (4.2, 2.7), "融合特征表\nX(T)", 2.0, 0.8)
    nodes = [
        ((0.8, 4.6), "历史功率\n滞后/滑动统计"),
        ((4.1, 4.6), "时间特征\n小时/月份/季节"),
        ((7.4, 4.6), "地面气象\n温度/辐照/风速"),
        ((0.8, 1.0), "ERA5\n温度/风/辐射"),
        ((4.1, 1.0), "Himawari B13\n云图 patch 特征"),
        ((7.4, 1.0), "站点信息\n容量/经纬度"),
    ]

    for xy, text in nodes:
        b = add_box(ax, xy, text, 2.0, 0.8)
        add_arrow(ax, (xy[0] + 1.0, xy[1] + 0.4), (5.2, 3.1))

    add_arrow(ax, (6.2, 3.1), (8.7, 3.1))
    add_box(ax, (8.7, 2.7), "预测模型\n点预测/概率预测", 2.0, 0.8)

    ax.set_xlim(0, 11.2)
    ax.set_ylim(0.4, 5.6)
    save(fig, out_dir, "图04_多源特征融合拓扑结构图")


def fig_himawari_patch(out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_title("Himawari-9 B13 站点邻域 patch 特征构造示意图")

    grid = np.arange(100).reshape(10, 10)
    ax.imshow(grid)
    ax.set_xticks([])
    ax.set_yticks([])

    # patch box
    rect = Rectangle((2.5, 2.5), 5, 5, fill=False, linewidth=2.2)
    ax.add_patch(rect)
    ax.plot(5, 5, marker="x", markersize=10)
    ax.text(5.2, 5.2, "光伏站点", fontsize=11)
    ax.text(5, 1.0, "裁剪站点周围 patch", ha="center", fontsize=12)
    ax.text(
        5,
        9.6,
        "提取：mean / std / center / gradient / cold_ratio / diff",
        ha="center",
        fontsize=10,
    )
    save(fig, out_dir, "图05_Himawari_patch特征构造示意图")


def fig_quantile_conformal_flow(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_axis_off()
    ax.set_title("分位数预测与 Conformal 区间校准流程")

    steps = [
        "训练集\n拟合 q10/q50/q90",
        "验证集\n计算区间残差",
        "分位数校准量\nqhat",
        "测试集\n扩展区间",
        "评价\nPICP / PINAW",
    ]
    xs = np.linspace(0.5, 9.8, len(steps))
    for i, (x, text) in enumerate(zip(xs, steps)):
        add_box(ax, (x, 2.0), text, 1.55, 0.85)
        if i < len(steps) - 1:
            add_arrow(ax, (x + 1.55, 2.42), (xs[i + 1], 2.42))

    ax.text(5.0, 0.95, "校准后区间：[q10 - qhat, q90 + qhat]", ha="center", fontsize=12)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0.5, 3.8)
    save(fig, out_dir, "图06_分位数预测与Conformal校准流程图")


def get_info_csv(root: Path) -> Path | None:
    raw = root / "data" / "raw"
    if not raw.exists():
        return None
    candidates = sorted(raw.glob("*基本信息*.csv"))
    return candidates[0] if candidates else None


def standardize_info(info: pd.DataFrame) -> pd.DataFrame:
    cols = list(info.columns)

    def find_col(keys):
        for k in keys:
            for c in cols:
                if k in str(c):
                    return c
        return None

    id_col = find_col(["光伏用户编号", "用户编号", "user", "id", "编号", "光伏用户名称"])
    lon_col = find_col(["经度", "longitude", "lon"])
    lat_col = find_col(["纬度", "latitude", "lat"])
    cap_col = find_col(["装机容量", "capacity"])

    if id_col is None or lon_col is None or lat_col is None:
        raise ValueError(f"无法识别基本信息表列名：{cols}")

    out = pd.DataFrame()
    out["user"] = info[id_col].astype(str).str.extract(r"(f\d+)", expand=False)
    out["longitude"] = pd.to_numeric(info[lon_col], errors="coerce")
    out["latitude"] = pd.to_numeric(info[lat_col], errors="coerce")
    if cap_col is not None:
        out["capacity_kw"] = pd.to_numeric(info[cap_col], errors="coerce")
    else:
        out["capacity_kw"] = np.nan

    out = out.dropna(subset=["user", "longitude", "latitude"]).copy()
    out["site_num"] = out["user"].str.extract(r"f(\d+)").astype(int)
    return out.sort_values("site_num")


def fig_station_map_and_capacity(root: Path, out_dir: Path):
    info_csv = get_info_csv(root)
    if info_csv is None:
        print("[SKIP] 未找到基本信息 CSV，跳过站点空间分布图和装机容量图。")
        return

    info = standardize_info(read_csv_smart(info_csv))

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    sizes = 80
    if info["capacity_kw"].notna().any():
        sizes = 50 + 180 * (info["capacity_kw"] / info["capacity_kw"].max())
    ax.scatter(info["longitude"], info["latitude"], s=sizes)
    for _, row in info.iterrows():
        ax.text(row["longitude"], row["latitude"], row["user"], fontsize=10, ha="left", va="bottom")
    ax.set_title("分布式光伏场站空间分布")
    ax.set_xlabel("经度 / °E")
    ax.set_ylabel("纬度 / °N")
    save(fig, out_dir, "图07_光伏场站空间分布图")

    if info["capacity_kw"].notna().any():
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        ax.bar(info["user"], info["capacity_kw"])
        ax.set_title("九个光伏场站装机容量对比")
        ax.set_xlabel("站点")
        ax.set_ylabel("装机容量 / kW")
        for i, v in enumerate(info["capacity_kw"]):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        save(fig, out_dir, "图08_九站点装机容量柱状图")


def get_processed_csv(root: Path) -> Path | None:
    candidates = [
        root / "data" / "processed" / "processed_all_stations_era5_himawari_202303_202304_b13_full.csv",
        root / "data" / "processed" / "processed_all_stations_era5_timeseries_new.csv",
        root / "data" / "processed" / "processed_all_stations.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def infer_columns(df: pd.DataFrame):
    user_col = "user" if "user" in df.columns else "user_id" if "user_id" in df.columns else None
    dt_col = "dt" if "dt" in df.columns else "datetime" if "datetime" in df.columns else None
    power_col = "power_kw" if "power_kw" in df.columns else "power" if "power" in df.columns else None
    cap_col = "capacity_kw" if "capacity_kw" in df.columns else None
    return user_col, dt_col, power_col, cap_col


def fig_daily_profiles_and_coverage(root: Path, out_dir: Path):
    p = get_processed_csv(root)
    if p is None:
        print("[SKIP] 未找到 processed CSV，跳过功率曲线和覆盖热力图。")
        return

    # 只读必要列，失败则全读
    df = read_csv_smart(p)
    user_col, dt_col, power_col, cap_col = infer_columns(df)
    if user_col is None or dt_col is None or power_col is None:
        print("[SKIP] processed CSV 列名无法识别，跳过功率曲线和覆盖热力图。")
        return

    df[dt_col] = pd.to_datetime(df[dt_col])
    df["time"] = df[dt_col].dt.strftime("%H:%M")
    df["date"] = df[dt_col].dt.date.astype(str)
    df["hour_float"] = df[dt_col].dt.hour + df[dt_col].dt.minute / 60

    if cap_col and cap_col in df.columns:
        df["power_pu_tmp"] = df[power_col] / df[cap_col]
    elif "power_pu" in df.columns:
        df["power_pu_tmp"] = df["power_pu"]
    else:
        df["power_pu_tmp"] = df[power_col] / df.groupby(user_col)[power_col].transform("max")

    # 平均日内曲线
    prof = df[df["hour_float"].between(5, 19)].groupby(["time"])["power_pu_tmp"].mean().reset_index()
    prof["hour_float"] = pd.to_datetime(prof["time"]).dt.hour + pd.to_datetime(prof["time"]).dt.minute / 60
    prof = prof.sort_values("hour_float")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prof["hour_float"], prof["power_pu_tmp"], linewidth=2)
    ax.set_title("九站点平均日内归一化功率曲线")
    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("归一化功率 / p.u.")
    save(fig, out_dir, "图09_九站点平均日内功率曲线")

    # 覆盖热力图：按站点-月份统计有效功率比例
    df["month"] = df[dt_col].dt.to_period("M").astype(str)
    cov = df.groupby([user_col, "month"])[power_col].apply(lambda s: s.notna().mean()).reset_index(name="coverage")
    pivot = cov.pivot(index=user_col, columns="month", values="coverage").sort_index()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("各站点功率样本有效率热力图")
    ax.set_xlabel("月份")
    ax.set_ylabel("站点")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("有效率")
    save(fig, out_dir, "图10_各站点功率样本有效率热力图")


def fig_himawari_feature_category(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_axis_off()
    ax.set_title("Himawari B13 云图统计特征体系")

    groups = [
        ("亮温水平", "mean / center / p10 / p90"),
        ("空间异质性", "std / min / max"),
        ("云边界纹理", "gradient mean"),
        ("冷云比例", "cold ratio 273K / 263K"),
        ("短时演变", "frame diff / abs diff"),
    ]

    xs = np.linspace(0.5, 8.5, len(groups))
    for x, (title, desc) in zip(xs, groups):
        add_box(ax, (x, 2.0), f"{title}\n{desc}", 1.45, 1.0)

    ax.text(5, 0.8, "特征目标：描述站点周边云覆盖、云边界、空间不均匀性与短时变化", ha="center", fontsize=11)
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0.3, 4.0)
    save(fig, out_dir, "图11_Himawari_B13特征体系图")


def fig_evaluation_metrics(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.set_axis_off()
    ax.set_title("点预测与区间预测评价指标体系")

    left = add_box(ax, (1.2, 2.6), "点预测指标\nRMSE / MAE / R² / nRMSE", 2.6, 0.9)
    right = add_box(ax, (6.4, 2.6), "区间预测指标\nPICP / PINAW", 2.6, 0.9)
    mid = add_box(ax, (3.9, 1.1), "综合分析\n精度 - 可靠性 - 紧致性", 2.7, 0.9)

    add_arrow(ax, (2.5, 2.6), (4.6, 2.0))
    add_arrow(ax, (7.7, 2.6), (5.9, 2.0))

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0.5, 4.0)
    save(fig, out_dir, "图12_评价指标体系图")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="项目根目录")
    parser.add_argument("--out_dir", default="results/final_paper/figures_generated", help="输出目录")
    args = parser.parse_args()

    setup_matplotlib()
    root = Path(args.root).resolve()
    out_dir = root / args.out_dir

    fig_overall_framework(out_dir)
    fig_data_preprocessing_flow(out_dir)
    fig_causal_sample(out_dir)
    fig_multisource_topology(out_dir)
    fig_himawari_patch(out_dir)
    fig_quantile_conformal_flow(out_dir)
    fig_station_map_and_capacity(root, out_dir)
    fig_daily_profiles_and_coverage(root, out_dir)
    fig_himawari_feature_category(out_dir)
    fig_evaluation_metrics(out_dir)

    print("\n[DONE] 缺失论文图已生成。输出目录：", out_dir)


if __name__ == "__main__":
    main()
