# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd


def read_csv_smart(path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def main():
    info_path = Path("data/raw/A榜-训练集_分布式光伏发电预测_基本信息.csv")
    out_dir = Path("results/paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    info = read_csv_smart(info_path)
    info["user"] = info["光伏用户编号"].astype(str).str.extract(r"(f\d+)", expand=False)
    info = info[["user", "装机容量(kW)"]].rename(columns={"装机容量(kW)": "capacity_kw"})

    files = [
        "table_all_sites_baseline_era5.csv",
        "table_all_sites_quantile_era5.csv",
    ]

    for name in files:
        path = out_dir / name
        if not path.exists():
            print(f"[SKIP] not found: {path}")
            continue

        df = pd.read_csv(path)
        df = df.merge(info, on="user", how="left")

        df["nRMSE_pct"] = df["RMSE"] / df["capacity_kw"] * 100
        df["nMAE_pct"] = df["MAE"] / df["capacity_kw"] * 100

        if "AvgWidth_cal" in df.columns:
            df["nAvgWidth_cal_pct"] = df["AvgWidth_cal"] / df["capacity_kw"] * 100

        out_path = out_dir / name.replace(".csv", "_normalized.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()