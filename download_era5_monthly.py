"""Download ERA5 single-level hourly data month by month for the PV thesis project.

Before running:
1. Register/login at https://cds.climate.copernicus.eu/
2. Put your personal token in ~/.cdsapirc
3. Open the ERA5 single-levels dataset page once and accept its Terms of Use.

Example:
python src/download_era5_monthly.py --start 2022-01 --end 2023-05 --out_dir data/era5
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import cdsapi


DEFAULT_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "total_column_water_vapour",
    "surface_solar_radiation_downwards",
    "total_precipitation",
]


def iter_months(start: str, end: str):
    """Yield (year, month) from YYYY-MM to YYYY-MM, inclusive."""
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01", help="Start month, e.g. 2022-01")
    parser.add_argument("--end", default="2023-05", help="End month, inclusive, e.g. 2023-05")
    parser.add_argument("--out_dir", type=Path, default=Path("data/era5"))
    parser.add_argument("--north", type=float, default=27.5)
    parser.add_argument("--west", type=float, default=116.5)
    parser.add_argument("--south", type=float, default=23.5)
    parser.add_argument("--east", type=float, default=120.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year, month in iter_months(args.start, args.end):
        ndays = calendar.monthrange(year, month)[1]
        target = args.out_dir / f"era5_single_levels_fujian_{year}{month:02d}.nc"
        if target.exists() and not args.overwrite:
            print(f"[SKIP] {target} already exists")
            continue

        request = {
            "product_type": ["reanalysis"],
            "variable": DEFAULT_VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, ndays + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            # CDS order is [North, West, South, East]
            "area": [args.north, args.west, args.south, args.east],
        }
        print(f"[DOWNLOAD] {year}-{month:02d} -> {target}")
        client.retrieve("reanalysis-era5-single-levels", request, str(target))


if __name__ == "__main__":
    main()
