# -*- coding: utf-8 -*-
"""
Test reading one Himawari AHI HSD time slot with Satpy.

Usage:
    python src/test_read_himawari_one_slot.py --raw_dir data/himawari/raw_selective --date 20230415 --time 0000 --band B13

If this fails on .DAT.bz2 files, decompress one time slot and retry on .DAT files.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def find_slot_files(raw_dir: Path, date: str, time: str, band: str) -> list[str]:
    band = band.upper()
    files = sorted(raw_dir.rglob(f"HS_H*_{date}_{time}_{band}_FLDK_*.DAT*"))
    return [str(f) for f in files]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/himawari/raw_selective")
    p.add_argument("--date", required=True, help="UTC date, e.g. 20230415")
    p.add_argument("--time", required=True, help="UTC time, e.g. 0000")
    p.add_argument("--band", default="B13")
    p.add_argument("--lon", type=float, default=119.156033, help="test longitude, default f6")
    p.add_argument("--lat", type=float, default=25.449233, help="test latitude, default f6")
    p.add_argument("--patch_radius", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    files = find_slot_files(raw_dir, args.date, args.time, args.band)

    print(f"[INFO] matched files: {len(files)}")
    for f in files[:20]:
        print("   ", f)
    if not files:
        raise FileNotFoundError("No Himawari files found for the selected slot.")

    try:
        from satpy import Scene
    except Exception as e:
        raise RuntimeError(
            "Satpy is not installed or cannot be imported. Install it with:\n"
            "python -m pip install satpy pyresample dask[array] -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from e

    scn = Scene(filenames=files, reader="ahi_hsd")
    print("[INFO] available dataset names:")
    try:
        print(list(scn.available_dataset_names())[:50])
    except Exception as e:
        print("[WARN] could not list datasets:", e)

    band = args.band.upper()
    print(f"[INFO] loading {band}")
    scn.load([band])
    da = scn[band]

    print("[INFO] DataArray:")
    print("  dims:", da.dims)
    print("  shape:", da.shape)
    print("  attrs units:", da.attrs.get("units"))
    print("  attrs calibration:", da.attrs.get("calibration"))
    print("  attrs start_time:", da.attrs.get("start_time"))
    print("  attrs end_time:", da.attrs.get("end_time"))

    area = da.attrs.get("area")
    if area is None:
        raise RuntimeError("No area definition found in Satpy DataArray attrs.")

    try:
        x, y = area.get_xy_from_lonlat(args.lon, args.lat)
    except Exception:
        try:
            x, y = area.get_array_coordinates_from_lonlat(args.lon, args.lat)
        except Exception as e:
            raise RuntimeError("Could not convert lon/lat to image x/y coordinates.") from e

    x = int(round(float(np.asarray(x))))
    y = int(round(float(np.asarray(y))))
    print(f"[INFO] lon/lat=({args.lon}, {args.lat}) -> x/y=({x}, {y})")

    ydim, xdim = da.dims[-2], da.dims[-1]
    ny = da.sizes[ydim]
    nx = da.sizes[xdim]
    r = args.patch_radius
    y0, y1 = max(0, y - r), min(ny, y + r + 1)
    x0, x1 = max(0, x - r), min(nx, x + r + 1)

    patch = da.isel({ydim: slice(y0, y1), xdim: slice(x0, x1)}).compute().values.astype(float)
    valid = np.isfinite(patch)
    print("[INFO] patch shape:", patch.shape)
    print("[INFO] valid ratio:", valid.mean() if valid.size else np.nan)
    print("[INFO] patch mean/std/min/max:", np.nanmean(patch), np.nanstd(patch), np.nanmin(patch), np.nanmax(patch))

    print("\n[OK] Satpy can read this time slot and locate the PV station.")


if __name__ == "__main__":
    main()
