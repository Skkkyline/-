# -*- coding: utf-8 -*-
"""Summarize causal Himawari ablation results across sites."""
from pathlib import Path
import pandas as pd


def main():
    files = sorted(Path("results").glob("himawari_causal_f*_*/baseline_himawari_causal_f*_day.csv"))
    files += sorted(Path("results").glob("himawari_causal_f*/baseline_himawari_causal_f*_day.csv"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No baseline_himawari_causal_f*_day.csv found under results/.")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = str(f)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    out_dir = Path("results/paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "table_all_sites_himawari_causal_ablation.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    mean = all_df.groupby(["horizon_min", "feature_set"], as_index=False).agg(
        RMSE=("RMSE", "mean"),
        MAE=("MAE", "mean"),
        R2=("R2", "mean"),
        nRMSE_pct=("nRMSE_pct", "mean"),
        nMAE_pct=("nMAE_pct", "mean"),
        n=("n", "mean"),
        sites=("user", "nunique"),
    )
    mean_path = out_dir / "table_himawari_causal_mean_by_horizon.csv"
    mean.to_csv(mean_path, index=False, encoding="utf-8-sig")

    # Improvement table relative to power_weather_era5.
    rows = []
    for horizon, g in mean.groupby("horizon_min"):
        base = g[g["feature_set"] == "power_weather_era5"]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in g.iterrows():
            if r["feature_set"] == "power_weather_era5":
                continue
            rows.append({
                "horizon_min": horizon,
                "compare_to": "power_weather_era5",
                "feature_set": r["feature_set"],
                "rmse_improve_pct": (b["RMSE"] - r["RMSE"]) / b["RMSE"] * 100,
                "mae_improve_pct": (b["MAE"] - r["MAE"]) / b["MAE"] * 100,
                "nrmse_improve_pct": (b["nRMSE_pct"] - r["nRMSE_pct"]) / b["nRMSE_pct"] * 100,
            })
    imp = pd.DataFrame(rows)
    imp_path = out_dir / "table_himawari_causal_improvement_vs_era5.csv"
    imp.to_csv(imp_path, index=False, encoding="utf-8-sig")

    print("saved:", all_path)
    print("saved:", mean_path)
    print("saved:", imp_path)
    print("\nMean results:")
    print(mean.to_string(index=False))


if __name__ == "__main__":
    main()
