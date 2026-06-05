"""Merge ERA5 NetCDF variables into the processed DataFountain PV sample table.

Input:
  data/processed/processed_all_stations.csv produced by prepare_data.py
  data/era5/*.nc downloaded by download_era5_monthly.py or CDS page

Output:
  data/processed/processed_all_stations_era5.csv

Design choice:
  ERA5 is hourly and UTC. DataFountain is treated as Beijing time.
  To avoid look-ahead leakage, each 15-min PV sample at local time T uses the latest
  ERA5 hourly record with time <= T, via merge_asof(direction='backward').
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import xarray as xr

VAR_RENAME: Dict[str, str] = {
    # CDS NetCDF short names
    "t2m": "era5_t2m_k",
    "d2m": "era5_d2m_k",
    "sp": "era5_sp_pa",
    "msl": "era5_msl_pa",
    "u10": "era5_u10_ms",
    "v10": "era5_v10_ms",
    "tcc": "era5_tcc",
    "lcc": "era5_lcc",
    "mcc": "era5_mcc",
    "hcc": "era5_hcc",
    "tcwv": "era5_tcwv_kgm2",
    "ssrd": "era5_ssrd_jm2",
    "tp": "era5_tp_m",
    # possible long names in some converted files
    "2m_temperature": "era5_t2m_k",
    "2m_dewpoint_temperature": "era5_d2m_k",
    "surface_pressure": "era5_sp_pa",
    "mean_sea_level_pressure": "era5_msl_pa",
    "10m_u_component_of_wind": "era5_u10_ms",
    "10m_v_component_of_wind": "era5_v10_ms",
    "total_cloud_cover": "era5_tcc",
    "low_cloud_cover": "era5_lcc",
    "medium_cloud_cover": "era5_mcc",
    "high_cloud_cover": "era5_hcc",
    "total_column_water_vapour": "era5_tcwv_kgm2",
    "surface_solar_radiation_downwards": "era5_ssrd_jm2",
    "total_precipitation": "era5_tp_m",
}


def find_coord(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError(f"Cannot find any coordinate from {list(candidates)}. Available coords: {list(ds.coords)}")


def normalize_ds(ds: xr.Dataset) -> xr.Dataset:
    # ERA5 files may include an expver dimension. Keep the first non-null value if present.
    if "expver" in ds.dims:
        ds = ds.isel(expver=0)
    time_name = find_coord(ds, ["valid_time", "time"])
    lat_name = find_coord(ds, ["latitude", "lat"])
    lon_name = find_coord(ds, ["longitude", "lon"])
    rename = {}
    if time_name != "time":
        rename[time_name] = "time"
    if lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name != "longitude":
        rename[lon_name] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def open_era5_dir(era5_dir: Path) -> xr.Dataset:
    files: List[Path] = sorted(era5_dir.glob("*.nc")) + sorted(era5_dir.glob("*.nc4"))
    if not files:
        raise FileNotFoundError(f"No .nc/.nc4 files found in {era5_dir}")

    datasets = []
    for f in files:
        print(f"[OPEN] {f}")
        ds = xr.open_dataset(f)
        ds = normalize_ds(ds)
        datasets.append(ds)
    ds_all = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    ds_all = ds_all.sortby("time")
    _, index = np.unique(ds_all["time"].values, return_index=True)
    ds_all = ds_all.isel(time=np.sort(index))
    return ds_all


def select_station_hourly(ds: xr.Dataset, station_id: str, lon: float, lat: float) -> pd.DataFrame:
    era_lon_values = ds["longitude"].values
    lon_select = lon
    if np.nanmax(era_lon_values) > 180 and lon < 0:
        lon_select = lon % 360

    sub = ds.sel(latitude=lat, longitude=lon_select, method="nearest")
    data_vars = [v for v in sub.data_vars if v in VAR_RENAME]
    if not data_vars:
        raise ValueError(f"No supported ERA5 variables found. Data variables are: {list(sub.data_vars)}")

    df = sub[data_vars].to_dataframe().reset_index()
    keep = ["time"] + data_vars
    df = df[keep].rename(columns=VAR_RENAME)
    df["user_id"] = station_id
    df["dt_utc"] = pd.to_datetime(df["time"], utc=False)
    df["dt"] = df["dt_utc"] + pd.Timedelta(hours=8)  # Beijing time
    df = df.drop(columns=["time", "dt_utc"])

    if "era5_t2m_k" in df:
        df["era5_t2m_c"] = df["era5_t2m_k"] - 273.15
    if "era5_d2m_k" in df:
        df["era5_d2m_c"] = df["era5_d2m_k"] - 273.15
    if "era5_u10_ms" in df and "era5_v10_ms" in df:
        df["era5_wind10_ms"] = np.sqrt(df["era5_u10_ms"] ** 2 + df["era5_v10_ms"] ** 2)
        df["era5_wind10_dir_deg"] = (np.degrees(np.arctan2(-df["u10" if "u10" in df else "era5_u10_ms"], -df["v10" if "v10" in df else "era5_v10_ms"])) + 360) % 360
    if "era5_ssrd_jm2" in df:
        df["era5_ssrd_wm2_1h"] = df["era5_ssrd_jm2"] / 3600.0
    return df.sort_values(["user_id", "dt"])


def build_station_era5_table(ds: xr.Dataset, stations: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in stations.itertuples(index=False):
        print(f"[EXTRACT] {row.user_id}: lon={row.lon:.6f}, lat={row.lat:.6f}")
        frames.append(select_station_hourly(ds, row.user_id, row.lon, row.lat))
    return pd.concat(frames, ignore_index=True).sort_values(["user_id", "dt"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_csv", type=Path, default=Path("data/processed/processed_all_stations.csv"))
    parser.add_argument("--era5_dir", type=Path, default=Path("data/era5"))
    parser.add_argument("--out_csv", type=Path, default=Path("data/processed/processed_all_stations_era5.csv"))
    parser.add_argument("--tolerance_minutes", type=int, default=90)
    args = parser.parse_args()

    processed = pd.read_csv(args.processed_csv, parse_dates=["dt"])
    stations = processed[["user_id", "lon", "lat"]].drop_duplicates().sort_values("user_id")

    ds = open_era5_dir(args.era5_dir)
    era5_hourly = build_station_era5_table(ds, stations)

    merged_frames = []
    tolerance = pd.Timedelta(minutes=args.tolerance_minutes)
    for uid, left in processed.groupby("user_id", sort=False):
        right = era5_hourly[era5_hourly["user_id"] == uid].sort_values("dt")
        left = left.sort_values("dt")
        merged = pd.merge_asof(
            left,
            right.drop(columns=["user_id"]),
            on="dt",
            direction="backward",
            tolerance=tolerance,
        )
        merged_frames.append(merged)

    out = pd.concat(merged_frames, ignore_index=True).sort_values(["user_id", "dt"])
    era_cols = [c for c in out.columns if c.startswith("era5_")]
    print("[SUMMARY] ERA5 columns:", era_cols)
    print("[SUMMARY] Missing ratio:")
    print(out[era_cols].isna().mean().sort_values(ascending=False).to_string())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {args.out_csv} shape={out.shape}")


if __name__ == "__main__":
    main()
