# -*- coding: utf-8 -*-
"""
中文论文图生成脚本：光伏功率超短期预测 + ERA5 + 分位数不确定性

推荐放置位置：pv_thesis_rebuild/src/plot_chinese_figures.py
在项目根目录运行，例如：

1) 生成点预测消融柱状图、不确定性指标图：
   python src/plot_chinese_figures.py --root . --summary

2) 生成某个站点某天的区间预测图：
   python src/plot_chinese_figures.py --root . --interval --site f6 --horizon 60 --date 2023-04-08

3) 批量生成 f1-f9 的 60 min 区间预测图。若不指定日期，脚本会自动挑选测试集中波动较强的一天：
   python src/plot_chinese_figures.py --root . --interval-all --horizon 60

注意：
- 本脚本需要读取 results/paper_tables/*.csv 和 results/quantile_era5_f*_new/quantile_predictions_f*_day.csv。
- 如果你把表格 CSV 放在别的位置，可用 --tables_dir 指定。
- 中文字体优先使用 Microsoft YaHei / SimHei，适合 Windows；Linux 下可安装 Noto Sans CJK。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

FEATURE_CN = {
    "persistence": "持续性基线",
    "power_only": "历史功率",
    "power_weather": "历史功率+地面气象",
    "power_weather_era5": "历史功率+地面气象+ERA5",
}


def setup_chinese_font() -> None:
    """Set Chinese font for matplotlib, robust on Windows/PyCharm."""
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        print("[WARN] 未找到常见中文字体。若中文显示为方框，请在 Windows 使用 Microsoft YaHei/SimHei，或安装 Noto Sans CJK。")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 300


def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def ensure_out(root: Path) -> Path:
    out_dir = root / "results" / "figures_zh"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_baseline_nrmse(root: Path, tables_dir: Path) -> None:
    """Plot nRMSE comparison for feature ablation."""
    path = tables_dir / "table_all_sites_baseline_era5_normalized.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    df = read_csv_smart(path)
    agg = (
        df.groupby(["horizon_min", "feature_set"], as_index=False)
        .agg(nRMSE_pct=("nRMSE_pct", "mean"), nMAE_pct=("nMAE_pct", "mean"), R2=("R2", "mean"))
    )
    order = ["persistence", "power_only", "power_weather", "power_weather_era5"]
    horizons = [15, 30, 60]
    x = np.arange(len(horizons))
    width = 0.18

    plt.figure(figsize=(9, 5))
    for i, fs in enumerate(order):
        values = []
        for h in horizons:
            row = agg[(agg["horizon_min"] == h) & (agg["feature_set"] == fs)]
            values.append(float(row["nRMSE_pct"].iloc[0]) if len(row) else np.nan)
        plt.bar(x + (i - 1.5) * width, values, width=width, label=FEATURE_CN.get(fs, fs))

    plt.xticks(x, [f"{h} min" for h in horizons])
    plt.ylabel("nRMSE / %")
    plt.xlabel("预测步长")
    plt.title("不同特征组合下的九站点平均 nRMSE 对比")
    plt.legend(frameon=True, fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    out = ensure_out(root) / "图_点预测消融_nRMSE对比.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("[OK] saved", out)


def plot_era5_improvement(root: Path, tables_dir: Path) -> None:
    """Plot ERA5 improvement rate."""
    path = tables_dir / "table_era5_improvement_mean_by_horizon.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    df = read_csv_smart(path)
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["horizon_min"], df["rmse_improve_pct"], marker="o", label="RMSE 改善率")
    plt.plot(df["horizon_min"], df["mae_improve_pct"], marker="s", label="MAE 改善率")
    plt.axhline(0, linewidth=1, linestyle="--")
    plt.xticks(df["horizon_min"], [f"{int(x)} min" for x in df["horizon_min"]])
    plt.ylabel("改善率 / %")
    plt.xlabel("预测步长")
    plt.title("引入 ERA5 后的误差改善率")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()
    out = ensure_out(root) / "图_ERA5误差改善率.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("[OK] saved", out)


def plot_quantile_metrics(root: Path, tables_dir: Path) -> None:
    """Plot PICP and PINAW before/after calibration."""
    path = tables_dir / "table_quantile_era5_mean_by_horizon.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    df = read_csv_smart(path)
    x = df["horizon_min"]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, df["PICP_raw"], marker="o", label="原始 PICP")
    plt.plot(x, df["PICP_cal"], marker="s", label="校准后 PICP")
    plt.axhline(0.80, linewidth=1, linestyle="--", label="理论覆盖率 0.80")
    plt.xticks(x, [f"{int(v)} min" for v in x])
    plt.ylim(0.55, 0.85)
    plt.ylabel("PICP")
    plt.xlabel("预测步长")
    plt.title("预测区间覆盖率随预测步长变化")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()
    out = ensure_out(root) / "图_不确定性_PICP对比.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("[OK] saved", out)

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, df["PINAW_raw"], marker="o", label="原始 PINAW")
    plt.plot(x, df["PINAW_cal"], marker="s", label="校准后 PINAW")
    plt.xticks(x, [f"{int(v)} min" for v in x])
    plt.ylabel("PINAW")
    plt.xlabel("预测步长")
    plt.title("预测区间归一化宽度随预测步长变化")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()
    out = ensure_out(root) / "图_不确定性_PINAW对比.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("[OK] saved", out)


def locate_prediction_file(root: Path, site: str) -> Path:
    candidates = sorted(root.glob(f"results/quantile_era5_{site}*/quantile_predictions_{site}_day.csv"))
    if candidates:
        return candidates[0]
    candidates = sorted(root.glob(f"**/quantile_predictions_{site}_day.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"没有找到 {site} 的 quantile_predictions 文件。请确认已运行 train_quantile_era5.py，"
        f"例如 results/quantile_era5_{site}_new/quantile_predictions_{site}_day.csv"
    )


def infer_columns(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()
    def pick(cands):
        for c in cands:
            if c in cols:
                return c
        raise KeyError(f"无法识别列，候选={cands}，现有列={cols}")
    return {
        "time": pick(["dt", "datetime", "time", "target_dt"]),
        "horizon": pick(["horizon_min", "horizon", "lead_min"]),
        "actual": pick(["y_true", "actual", "target", "power_true", "power_kw"]),
        "q50": pick(["q50", "pred_q50", "q50_pred", "y_pred_q50"]),
        "lower": pick(["q10_cal", "lower_cal", "q10", "pred_q10", "lower"]),
        "upper": pick(["q90_cal", "upper_cal", "q90", "pred_q90", "upper"]),
    }


def choose_volatile_date(df: pd.DataFrame, time_col: str, actual_col: str) -> str:
    tmp = df.copy()
    tmp["date"] = tmp[time_col].dt.date.astype(str)
    score = tmp.groupby("date")[actual_col].apply(lambda s: s.diff().abs().sum())
    return score.sort_values(ascending=False).index[0]


def plot_interval_day(root: Path, site: str, horizon: int, date: str | None = None, normalize: bool = False) -> None:
    pred_path = locate_prediction_file(root, site)
    df = read_csv_smart(pred_path)
    c = infer_columns(df)
    df[c["time"]] = pd.to_datetime(df[c["time"]])
    df = df[df[c["horizon"]] == horizon].copy()
    if df.empty:
        raise ValueError(f"{pred_path} 中没有 horizon={horizon} 的数据")

    if date is None:
        date = choose_volatile_date(df, c["time"], c["actual"])
    day = df[df[c["time"]].dt.date.astype(str) == date].sort_values(c["time"]).copy()
    if day.empty:
        raise ValueError(f"{pred_path} 中没有日期 {date} 的数据")

    scale = 1.0
    ylabel = "功率 / kW"
    if normalize:
        info_path = root / "data" / "raw" / "A榜-训练集_分布式光伏发电预测_基本信息.csv"
        if info_path.exists():
            info = read_csv_smart(info_path)
            info["站点编号"] = info["光伏用户编号"].astype(str).str.extract(r"(f\d+)", expand=False)
            cap = float(info.loc[info["站点编号"] == site, "装机容量(kW)"].iloc[0])
            scale = cap
            ylabel = "归一化功率 / p.u."

    plt.figure(figsize=(10, 5))
    x = day[c["time"]]
    actual = day[c["actual"]] / scale
    q50 = day[c["q50"]] / scale
    lower = day[c["lower"]] / scale
    upper = day[c["upper"]] / scale

    plt.plot(x, actual, linewidth=1.8, label="实测功率")
    plt.plot(x, q50, linewidth=1.8, label="q50 中位数预测")
    plt.fill_between(x, lower, upper, alpha=0.25, label="校准后 80% 预测区间")
    plt.xlabel("时间")
    plt.ylabel(ylabel)
    plt.title(f"{site} 场站 {horizon} min 光伏功率区间预测结果（{date}）")
    plt.legend(loc="best")
    plt.grid(linestyle="--", alpha=0.30)
    plt.tight_layout()
    suffix = "归一化" if normalize else "功率"
    out = ensure_out(root) / f"图_{site}_{horizon}min_{date}_{suffix}区间预测.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("[OK] saved", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="项目根目录，默认当前目录")
    parser.add_argument("--tables_dir", type=str, default=None, help="表格目录，默认 results/paper_tables")
    parser.add_argument("--summary", action="store_true", help="生成点预测与不确定性汇总图")
    parser.add_argument("--interval", action="store_true", help="生成单个站点某一天区间预测图")
    parser.add_argument("--interval-all", action="store_true", help="批量生成 f1-f9 区间预测图")
    parser.add_argument("--site", type=str, default="f6", help="站点编号，例如 f6")
    parser.add_argument("--horizon", type=int, default=60, help="预测步长：15/30/60")
    parser.add_argument("--date", type=str, default=None, help="日期，例如 2023-04-08；不填则自动选强波动日")
    parser.add_argument("--normalize", action="store_true", help="区间图是否按装机容量归一化")
    args = parser.parse_args()

    setup_chinese_font()
    root = Path(args.root).resolve()
    tables_dir = Path(args.tables_dir).resolve() if args.tables_dir else root / "results" / "paper_tables"

    if args.summary:
        plot_baseline_nrmse(root, tables_dir)
        plot_era5_improvement(root, tables_dir)
        plot_quantile_metrics(root, tables_dir)

    if args.interval:
        plot_interval_day(root, args.site, args.horizon, args.date, args.normalize)

    if args.interval_all:
        for site in [f"f{i}" for i in range(1, 10)]:
            try:
                plot_interval_day(root, site, args.horizon, None, args.normalize)
            except Exception as e:
                print(f"[WARN] {site} 绘图失败：{e}")

    if not (args.summary or args.interval or args.interval_all):
        print("没有指定绘图任务。示例：python src/plot_chinese_figures.py --root . --summary")


if __name__ == "__main__":
    main()
