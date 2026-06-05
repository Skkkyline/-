# -*- coding: utf-8 -*-
"""
Summarize Himawari causal baseline results safely.

This fixed version avoids duplicate/cumulative CSVs caused by broad globbing.
It only reads one baseline file per site from its corresponding output folder:
    results/himawari_causal_f1_new/baseline_himawari_causal_f1_day.csv
    ...
    results/himawari_causal_f9_new/baseline_himawari_causal_f9_day.csv

Outputs:
    results/paper_tables/table_all_sites_himawari_causal_ablation_fixed.csv
    results/paper_tables/table_himawari_causal_mean_by_horizon_fixed.csv
    results/paper_tables/table_himawari_causal_improvement_vs_era5_fixed.csv
    results/paper_tables/table_himawari_site_improvement_vs_era5_fixed.csv
"""
from pathlib import Path
import pandas as pd

FEATURE_ORDER = [
    "persistence",
    "power_only",
    "power_weather",
    "power_weather_era5",
    "power_weather_himawari",
    "power_weather_era5_himawari",
]


def main():
    results_dir = Path("results")
    out_dir = results_dir / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    missing = []
    for i in range(1, 10):
        user = f"f{i}"
        path = results_dir / f"himawari_causal_{user}_new" / f"baseline_himawari_causal_{user}_day.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        # Safety check: keep only rows for the expected user.
        if "user" in df.columns:
            df = df[df["user"].astype(str).str.lower() == user]
        df["source_file"] = str(path)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No valid baseline_himawari_causal_f*_day.csv files were found.")

    if missing:
        print("[WARN] Missing files:")
        for p in missing:
            print("  ", p)

    all_df = pd.concat(dfs, ignore_index=True)

    key_cols = ["user", "horizon_min", "feature_set"]
    dup = all_df.duplicated(key_cols).sum()
    if dup:
        print(f"[WARN] Found {dup} duplicate rows by {key_cols}; keeping first.")
        all_df = all_df.drop_duplicates(key_cols, keep="first")

    all_df["feature_set"] = pd.Categorical(all_df["feature_set"], categories=FEATURE_ORDER, ordered=True)
    all_df = all_df.sort_values(["user", "horizon_min", "feature_set"]).reset_index(drop=True)

    all_path = out_dir / "table_all_sites_himawari_causal_ablation_fixed.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    mean_df = all_df.groupby(["horizon_min", "feature_set"], observed=False).agg(
        RMSE=("RMSE", "mean"),
        MAE=("MAE", "mean"),
        R2=("R2", "mean"),
        nRMSE_pct=("nRMSE_pct", "mean"),
        nMAE_pct=("nMAE_pct", "mean"),
        n=("n", "mean"),
        sites=("user", "nunique"),
    ).reset_index()
    mean_df["feature_set"] = pd.Categorical(mean_df["feature_set"], categories=FEATURE_ORDER, ordered=True)
    mean_df = mean_df.sort_values(["horizon_min", "feature_set"]).reset_index(drop=True)

    mean_path = out_dir / "table_himawari_causal_mean_by_horizon_fixed.csv"
    mean_df.to_csv(mean_path, index=False, encoding="utf-8-sig")

    # Improvement vs ERA5 combination. Positive means the compared feature_set is better than power_weather_era5.
    rows = []
    for horizon, g in mean_df.groupby("horizon_min"):
        base = g[g["feature_set"].astype(str) == "power_weather_era5"].iloc[0]
        for _, row in g.iterrows():
            fs = str(row["feature_set"])
            if fs == "power_weather_era5":
                continue
            rows.append({
                "horizon_min": horizon,
                "baseline": "power_weather_era5",
                "feature_set": fs,
                "rmse_improve_vs_era5_pct": (base["RMSE"] - row["RMSE"]) / base["RMSE"] * 100,
                "mae_improve_vs_era5_pct": (base["MAE"] - row["MAE"]) / base["MAE"] * 100,
                "nrmse_improve_vs_era5_pct": (base["nRMSE_pct"] - row["nRMSE_pct"]) / base["nRMSE_pct"] * 100,
                "nmae_improve_vs_era5_pct": (base["nMAE_pct"] - row["nMAE_pct"]) / base["nMAE_pct"] * 100,
            })
    imp_df = pd.DataFrame(rows)
    imp_path = out_dir / "table_himawari_causal_improvement_vs_era5_fixed.csv"
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    # Site-level improvement for Himawari feature sets vs ERA5.
    site_rows = []
    for (user, horizon), g in all_df.groupby(["user", "horizon_min"]):
        gg = g.set_index(g["feature_set"].astype(str))
        if "power_weather_era5" not in gg.index:
            continue
        base = gg.loc["power_weather_era5"]
        for fs in ["power_weather_himawari", "power_weather_era5_himawari"]:
            if fs not in gg.index:
                continue
            row = gg.loc[fs]
            site_rows.append({
                "user": user,
                "horizon_min": horizon,
                "feature_set": fs,
                "rmse_improve_pct": (base["RMSE"] - row["RMSE"]) / base["RMSE"] * 100,
                "mae_improve_pct": (base["MAE"] - row["MAE"]) / base["MAE"] * 100,
                "nrmse_improve_pct": (base["nRMSE_pct"] - row["nRMSE_pct"]) / base["nRMSE_pct"] * 100,
                "nmae_improve_pct": (base["nMAE_pct"] - row["nMAE_pct"]) / base["nMAE_pct"] * 100,
                "era5_nRMSE_pct": base["nRMSE_pct"],
                "this_nRMSE_pct": row["nRMSE_pct"],
            })
    site_imp_df = pd.DataFrame(site_rows)
    site_imp_path = out_dir / "table_himawari_site_improvement_vs_era5_fixed.csv"
    site_imp_df.to_csv(site_imp_path, index=False, encoding="utf-8-sig")

    print("[OK] saved:", all_path)
    print("[OK] saved:", mean_path)
    print("[OK] saved:", imp_path)
    print("[OK] saved:", site_imp_path)
    print("\nMean by horizon:")
    print(mean_df.to_string(index=False))
    print("\nImprovement vs power_weather_era5, positive means improvement:")
    print(imp_df.to_string(index=False))


if __name__ == "__main__":
    main()
