import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILES = {
    "info": "A榜-训练集_分布式光伏发电预测_基本信息.csv",
    "power": "A榜-训练集_分布式光伏发电预测_实际功率数据.csv",
    "weather": "A榜-训练集_分布式光伏发电预测_气象变量数据.csv",
}

RENAME = {
    "光伏用户编号": "user_id",
    "装机容量(kW)": "capacity_kw",
    "经度": "lon",
    "纬度": "lat",
    "气压(Pa）": "pressure_pa",
    "相对湿度（%）": "humidity",
    "云量": "cloud",
    "10米风速（10m/s）": "wind10",
    "10米风向（°)": "wind10_dir",
    "温度（K）": "temp_k",
    "辐照强度（J/m2）": "irradiance",
    "降水（m）": "precip",
    "100m风速（100m/s）": "wind100",
    "100m风向（°)": "wind100_dir",
}


def read_csv_gbk(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="gbk")


def power_wide_to_long(power_wide: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    pcols = [c for c in power_wide.columns if re.fullmatch(r"p\d+", str(c))]
    long = power_wide.melt(
        id_vars=["光伏用户编号", "综合倍率", "时间"],
        value_vars=pcols,
        var_name="slot",
        value_name="power_raw",
    )
    long["slot_num"] = long["slot"].str.extract(r"(\d+)").astype(int)
    long["date"] = pd.to_datetime(long["时间"]).dt.normalize()
    long["dt"] = long["date"] + pd.to_timedelta((long["slot_num"] - 1) * 15, unit="min")
    long["power_kw"] = long["power_raw"] * long["综合倍率"]
    long = long.merge(info[["光伏用户编号", "装机容量(kW)", "经度", "纬度"]], on="光伏用户编号", how="left")
    long["power_kw_clean"] = long["power_kw"].clip(lower=0)
    long["power_kw_clean"] = np.minimum(long["power_kw_clean"], long["装机容量(kW)"])
    long["power_pu"] = long["power_kw_clean"] / long["装机容量(kW)"]
    return long[["光伏用户编号", "dt", "power_kw", "power_kw_clean", "power_pu", "装机容量(kW)", "经度", "纬度"]]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_c"] = df["temp_k"] - 273.15
    df["month"] = df["dt"].dt.month
    df["dayofyear"] = df["dt"].dt.dayofyear
    df["hour"] = df["dt"].dt.hour + df["dt"].dt.minute / 60
    df["minute_of_day"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute
    df["sin_hour"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["cos_hour"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["is_day_weather"] = (df["irradiance"] > 0).astype(int)
    return df


def build_processed(data_dir: Path) -> pd.DataFrame:
    info = read_csv_gbk(data_dir / RAW_FILES["info"])
    power = read_csv_gbk(data_dir / RAW_FILES["power"])
    weather = read_csv_gbk(data_dir / RAW_FILES["weather"])

    weather["dt"] = pd.to_datetime(weather["时间"])
    weather = weather.drop(columns=["时间"]).rename(columns=RENAME)
    info_en = info.rename(columns=RENAME)
    power_long = power_wide_to_long(power, info).rename(columns=RENAME)

    df = weather.merge(power_long, on=["user_id", "dt"], how="left")
    df = df.merge(info_en[["user_id", "capacity_kw", "lon", "lat"]], on="user_id", how="left", suffixes=("", "_info"))
    for col in ["capacity_kw", "lon", "lat"]:
        if f"{col}_info" in df.columns:
            df[col] = df[col].fillna(df[f"{col}_info"])
            df = df.drop(columns=[f"{col}_info"])
    df["power_pu"] = df["power_kw_clean"] / df["capacity_kw"]
    df = add_time_features(df)
    return df.sort_values(["user_id", "dt"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = build_processed(args.data_dir)
    out = args.out_dir / "processed_all_stations.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved: {out}")
    print(df.shape)


if __name__ == "__main__":
    main()
