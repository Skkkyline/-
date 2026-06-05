# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd


def main():
    root = Path("results")
    files = sorted(root.glob("quantile_himawari_causal_f*_new/quantile_himawari_causal_f*_day.csv"))
    if not files:
        files = sorted(root.glob("**/quantile_himawari_causal_f*_day.csv"))
    # Avoid accidental duplicate files by keeping the newest/default directory per user if duplicate content exists.
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = str(f)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No quantile_himawari_causal_f*_day.csv files found under results.")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["user", "horizon_min", "feature_set"], keep="last")

    out_dir = Path("results/paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "table_all_sites_quantile_himawari_causal.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    numeric_cols = [
        "RMSE", "MAE", "R2", "nRMSE_pct", "nMAE_pct",
        "PICP_raw", "PICP_cal", "PINAW_raw", "PINAW_cal",
        "AvgWidth_raw", "AvgWidth_cal", "nAvgWidth_raw_pct", "nAvgWidth_cal_pct",
        "Pinball_q10", "Pinball_q50", "Pinball_q90", "qhat_conformal", "n",
    ]
    numeric_cols = [c for c in numeric_cols if c in all_df.columns]
    mean_df = all_df.groupby(["horizon_min", "feature_set"], as_index=False).agg(
        **{c: (c, "mean") for c in numeric_cols},
        sites=("user", "nunique"),
    )
    mean_path = out_dir / "table_quantile_himawari_causal_mean_by_horizon.csv"
    mean_df.to_csv(mean_path, index=False, encoding="utf-8-sig")

    rows = []
    for h, g in mean_df.groupby("horizon_min"):
        base = g[g["feature_set"] == "power_weather_era5"]
        fused = g[g["feature_set"] == "power_weather_era5_himawari"]
        if len(base) == 0 or len(fused) == 0:
            continue
        b = base.iloc[0]
        f = fused.iloc[0]
        rows.append({
            "horizon_min": h,
            "baseline": "power_weather_era5",
            "feature_set": "power_weather_era5_himawari",
            "rmse_improve_pct": (b["RMSE"] - f["RMSE"]) / b["RMSE"] * 100,
            "mae_improve_pct": (b["MAE"] - f["MAE"]) / b["MAE"] * 100,
            "nrmse_improve_pct": (b["nRMSE_pct"] - f["nRMSE_pct"]) / b["nRMSE_pct"] * 100 if "nRMSE_pct" in b else None,
            "nmae_improve_pct": (b["nMAE_pct"] - f["nMAE_pct"]) / b["nMAE_pct"] * 100 if "nMAE_pct" in b else None,
            "picp_raw_delta": f.get("PICP_raw", float("nan")) - b.get("PICP_raw", float("nan")),
            "picp_cal_delta": f.get("PICP_cal", float("nan")) - b.get("PICP_cal", float("nan")),
            "pinaw_cal_delta": f.get("PINAW_cal", float("nan")) - b.get("PINAW_cal", float("nan")),
        })
    imp_df = pd.DataFrame(rows)
    imp_path = out_dir / "table_quantile_himawari_causal_improvement_vs_era5.csv"
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    print("[OK] saved:", all_path)
    print("[OK] saved:", mean_path)
    print("[OK] saved:", imp_path)
    print("\nMean by horizon:")
    print(mean_df.to_string(index=False))
    print("\nImprovement vs ERA5:")
    print(imp_df.to_string(index=False))


if __name__ == "__main__":
    main()
