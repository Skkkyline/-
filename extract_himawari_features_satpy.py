# -*- coding: utf-8 -*-
"""
Extract station-level Himawari AHI patch features using Satpy.

Input structure expected:
    data/himawari/raw_selective/YYYY/MM/DD/HHMM/HS_H09_YYYYMMDD_HHMM_B13_FLDK_R20_S0110.DAT.bz2

Output:
    data/processed/himawari_features_sample.csv

Usage:
    python src/extract_himawari_features_satpy.py --raw_dir data/himawari/raw_selective --info_csv data/raw/A榜-训练集_分布式光伏发电预测_基本信息.csv --bands B13 --out_csv data/processed/himawari_features_sample.csv
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PATTERN = re.compile(
    r"HS_H(?P<sat>\d{2})_(?P<date>\d{8})_(?P<time>\d{4})_(?P<band>B\d{2})_FLDK_(?P<res>R\d{2})_S(?P<seg>\d{2})(?P<nseg>\d{2})\.DAT(?:\.bz2)?$",
    re.IGNORECASE,
)


def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def load_sites(info_csv: Path) -> pd.DataFrame:
    df = read_csv_smart(info_csv)
    cols = list(df.columns)

    def pick(keys):
        for key in keys:
            for col in cols:
                if key.lower() in str(col).lower():
                    return col
        raise KeyError(f"Cannot find column with keys={keys}, columns={cols}")

    site_col = pick(["光伏用户编号", "用户编号", "user_id", "id", "编号", "光伏用户名称", "用户名称"])
    lon_col = pick(["经度", "longitude", "lon"])
    lat_col = pick(["纬度", "latitude", "lat"])
    cap_col = None
    try:
        cap_col = pick(["装机容量", "capacity"])
    except Exception:
        pass

    out = pd.DataFrame()
    out["user"] = df[site_col].astype(str).str.extract(r"(f\d+)", expand=False)
    out["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    out["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    if cap_col is not None:
        out["capacity_kw"] = pd.to_numeric(df[cap_col], errors="coerce")
    else:
        out["capacity_kw"] = np.nan
    out = out.dropna(subset=["user", "longitude", "latitude"]).copy()
    out["site_num"] = out["user"].str.extract(r"f(\d+)").astype(int)
    out = out.sort_values("site_num").drop(columns=["site_num"]).reset_index(drop=True)
    return out


def find_groups(raw_dir: Path, bands: list[str]) -> dict[tuple[str, str, str], list[str]]:
    bands = {b.upper() for b in bands}
    files = list(raw_dir.rglob("*.DAT")) + list(raw_dir.rglob("*.DAT.bz2"))
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    expected_nseg: dict[tuple[str, str, str], int] = {}

    for f in files:
        m = PATTERN.search(f.name)
        if not m:
            continue
        d = m.groupdict()
        band = d["band"].upper()
        if band not in bands:
            continue
        key = (d["date"], d["time"], band)
        groups[key].append(f)
        expected_nseg[key] = int(d["nseg"])

    complete = {}
    for key, paths in groups.items():
        segs = []
        for p in paths:
            m = PATTERN.search(p.name)
            segs.append(int(m.group("seg")))
        need = expected_nseg[key]
        missing = [s for s in range(1, need + 1) if s not in set(segs)]
        if missing:
            print(f"[WARN] skip incomplete group {key}, missing={missing}")
            continue
        complete[key] = [str(p) for p in sorted(paths)]

    return complete


def utc_to_beijing_naive(date_token: str, time_token: str) -> pd.Timestamp:
    t = pd.to_datetime(date_token + time_token, format="%Y%m%d%H%M", utc=True)
    return t.tz_convert("Asia/Shanghai").tz_localize(None)


def lonlat_to_xy(area, lon: float, lat: float) -> tuple[int, int]:
    try:
        x, y = area.get_xy_from_lonlat(lon, lat)
    except Exception:
        x, y = area.get_array_coordinates_from_lonlat(lon, lat)
    x = int(round(float(np.asarray(x))))
    y = int(round(float(np.asarray(y))))
    return x, y


def patch_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = values.astype(float)
    valid = np.isfinite(arr)
    out: dict[str, float] = {}
    out[prefix + "valid_ratio"] = float(valid.mean()) if valid.size else np.nan
    if not valid.any():
        for name in ["mean", "std", "min", "max", "p10", "p90", "center", "center_minus_mean", "grad_mean", "cold_ratio_273", "cold_ratio_263"]:
            out[prefix + name] = np.nan
        return out

    center = arr[arr.shape[0] // 2, arr.shape[1] // 2]
    mean = np.nanmean(arr)
    out[prefix + "mean"] = float(mean)
    out[prefix + "std"] = float(np.nanstd(arr))
    out[prefix + "min"] = float(np.nanmin(arr))
    out[prefix + "max"] = float(np.nanmax(arr))
    out[prefix + "p10"] = float(np.nanpercentile(arr, 10))
    out[prefix + "p90"] = float(np.nanpercentile(arr, 90))
    out[prefix + "center"] = float(center) if np.isfinite(center) else np.nan
    out[prefix + "center_minus_mean"] = float(center - mean) if np.isfinite(center) else np.nan

    try:
        gy, gx = np.gradient(arr)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        out[prefix + "grad_mean"] = float(np.nanmean(grad))
    except Exception:
        out[prefix + "grad_mean"] = np.nan

    # For infrared brightness temperature bands, colder pixels often indicate higher/thicker clouds.
    # If values are not in Kelvin, these ratios will be less interpretable but still harmless as numerical features.
    out[prefix + "cold_ratio_273"] = float(np.nanmean(arr < 273.15))
    out[prefix + "cold_ratio_263"] = float(np.nanmean(arr < 263.15))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/himawari/raw_selective")
    p.add_argument("--info_csv", default="data/raw/A榜-训练集_分布式光伏发电预测_基本信息.csv")
    p.add_argument("--bands", nargs="+", default=["B13"])
    p.add_argument("--patch_radius", type=int, default=16, help="Patch half size in pixels. 16 means 33x33 patch.")
    p.add_argument("--out_csv", default="data/processed/himawari_features_sample.csv")
    p.add_argument("--max_slots", type=int, default=None, help="Optional debug limit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    info_csv = Path(args.info_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        from satpy import Scene
    except Exception as e:
        raise RuntimeError(
            "Satpy is required. Install it with:\n"
            "python -m pip install satpy pyresample dask[array] -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from e

    sites = load_sites(info_csv)
    print("[INFO] sites:")
    print(sites.to_string(index=False))

    groups = find_groups(raw_dir, args.bands)
    keys = sorted(groups.keys())
    if args.max_slots:
        keys = keys[: args.max_slots]

    print(f"[INFO] complete time-band groups: {len(keys)}")
    if not keys:
        raise FileNotFoundError("No complete Himawari groups found.")

    rows = []
    for date_token, time_token, band in tqdm(keys, desc="Himawari slots"):
        filenames = groups[(date_token, time_token, band)]
        dt_bj = utc_to_beijing_naive(date_token, time_token)

        try:
            scn = Scene(filenames=filenames, reader="ahi_hsd")
            scn.load([band])
            da = scn[band]
            area = da.attrs.get("area")
            if area is None:
                print(f"[WARN] skip {date_token} {time_token} {band}: no area attr")
                continue
        except Exception as e:
            print(f"[WARN] failed to read {date_token} {time_token} {band}: {e}")
            continue

        ydim, xdim = da.dims[-2], da.dims[-1]
        ny = da.sizes[ydim]
        nx = da.sizes[xdim]
        r = args.patch_radius

        for _, s in sites.iterrows():
            try:
                x, y = lonlat_to_xy(area, float(s["longitude"]), float(s["latitude"]))
                y0, y1 = max(0, y - r), min(ny, y + r + 1)
                x0, x1 = max(0, x - r), min(nx, x + r + 1)
                patch = da.isel({ydim: slice(y0, y1), xdim: slice(x0, x1)}).compute().values
                prefix = f"sat_{band.lower()}_"
                feat = patch_features(patch, prefix=prefix)
                feat.update({
                    "user": s["user"],
                    "datetime": dt_bj,
                    "sat_utc_date": date_token,
                    "sat_utc_time": time_token,
                    "sat_band": band,
                    "sat_x": x,
                    "sat_y": y,
                    "sat_patch_radius": r,
                    "sat_units": da.attrs.get("units", ""),
                    "sat_calibration": str(da.attrs.get("calibration", "")),
                })
                rows.append(feat)
            except Exception as e:
                print(f"[WARN] failed station {s['user']} at {date_token} {time_token} {band}: {e}")

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        raise RuntimeError("No features extracted.")

    # If multiple bands are extracted, pivot them to one row per user + datetime.
    id_cols = ["user", "datetime"]
    meta_cols = ["sat_utc_date", "sat_utc_time", "sat_x", "sat_y", "sat_patch_radius"]
    feature_cols = [c for c in feat_df.columns if c.startswith("sat_b")]
    base = feat_df[id_cols + meta_cols].drop_duplicates(id_cols).copy()
    wide = base
    for band in sorted(feat_df["sat_band"].unique()):
        sub = feat_df[feat_df["sat_band"] == band][id_cols + [c for c in feature_cols if c.startswith(f"sat_{band.lower()}_")]].copy()
        wide = wide.merge(sub, on=id_cols, how="left")

    wide = wide.sort_values(["user", "datetime"]).reset_index(drop=True)

    # Temporal difference features for mean and center.
    for col in list(wide.columns):
        if col.endswith("_mean") or col.endswith("_center"):
            if col.startswith("sat_b"):
                wide[col + "_diff"] = wide.groupby("user")[col].diff()

    wide.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[OK] saved:", out_csv)
    print("[OK] rows:", len(wide), "cols:", len(wide.columns))
    print("[OK] columns:", list(wide.columns))


if __name__ == "__main__":
    main()
