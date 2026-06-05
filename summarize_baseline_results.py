# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd


def main():
    results_dir = Path("results")
    files = sorted(results_dir.glob("era5_timeseries_f*_new/baseline_era5_f*_day.csv"))

    if not files:
        raise FileNotFoundError("没有找到 baseline_era5_f*_day.csv，请先跑 f1-f9 baseline。")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    out_dir = Path("results/paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "table_all_sites_baseline_era5.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    # 只保留 power_weather 与 power_weather_era5，计算 ERA5 改善率
    rows = []
    for (user, horizon), g in all_df.groupby(["user", "horizon_min"]):
        g = g.set_index("feature_set")

        if "power_weather" not in g.index or "power_weather_era5" not in g.index:
            continue

        base = g.loc["power_weather"]
        era5 = g.loc["power_weather_era5"]

        rows.append({
            "user": user,
            "horizon_min": horizon,
            "rmse_power_weather": base["RMSE"],
            "rmse_power_weather_era5": era5["RMSE"],
            "rmse_improve_pct": (base["RMSE"] - era5["RMSE"]) / base["RMSE"] * 100,
            "mae_power_weather": base["MAE"],
            "mae_power_weather_era5": era5["MAE"],
            "mae_improve_pct": (base["MAE"] - era5["MAE"]) / base["MAE"] * 100,
            "r2_power_weather": base["R2"],
            "r2_power_weather_era5": era5["R2"],
        })

    improve_df = pd.DataFrame(rows)
    improve_path = out_dir / "table_era5_improvement_all_sites.csv"
    improve_df.to_csv(improve_path, index=False, encoding="utf-8-sig")

    # 按预测步长求平均
    mean_df = improve_df.groupby("horizon_min").agg({
        "rmse_improve_pct": "mean",
        "mae_improve_pct": "mean",
        "r2_power_weather": "mean",
        "r2_power_weather_era5": "mean",
    }).reset_index()

    mean_path = out_dir / "table_era5_improvement_mean_by_horizon.csv"
    mean_df.to_csv(mean_path, index=False, encoding="utf-8-sig")

    print("[OK] saved:", all_path)
    print("[OK] saved:", improve_path)
    print("[OK] saved:", mean_path)
    print("\nERA5 mean improvement by horizon:")
    print(mean_df.to_string(index=False))


if __name__ == "__main__":
    main()