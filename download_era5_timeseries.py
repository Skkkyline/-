
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import cdsapi
import pandas as pd


DEFAULT_START = "2022-01-02"
DEFAULT_END = "2023-05-01"

# ERA5 hourly time-series variables.
# 注意：ERA5 time-series 不一定支持 total_cloud_cover 等所有 ERA5 网格变量。
# 这里保留最稳、最常用的一组变量。
ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_precipitation",
]


def find_project_root() -> Path:
    """
    Try to find project root.

    If this file is in src/, project root is parent of src/.
    """
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "src":
        return here.parent.parent
    return Path.cwd()


def auto_find_info_csv(project_root: Path) -> Path:
    """
    Automatically find DataFountain basic info CSV.
    """
    candidates = []

    search_dirs = [
        project_root / "data" / "raw",
        project_root,
        project_root.parent,
    ]

    patterns = [
        "*基本信息*.csv",
        "*basic*.csv",
        "*info*.csv",
    ]

    for d in search_dirs:
        if d.exists():
            for pattern in patterns:
                candidates.extend(d.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            "没有自动找到基本信息 CSV。\n"
            "请确认文件放在 data/raw/ 目录下，或运行时指定：\n"
            "python src/download_era5_timeseries.py --info_csv \"你的基本信息.csv路径\""
        )

    # Prefer files containing Chinese dataset name.
    candidates = sorted(candidates, key=lambda p: (("基本信息" not in p.name), len(str(p))))
    return candidates[0]


def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Read Chinese CSV with common encodings.
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"读取 CSV 失败：{path}\n最后一次错误：{last_error}")


def guess_column(columns: list[str], keywords: list[str]) -> str:
    """
    Guess column name by keywords.
    """
    for kw in keywords:
        for col in columns:
            if kw.lower() in str(col).lower():
                return col

    raise KeyError(
        f"无法根据关键词 {keywords} 自动识别列名。\n"
        f"当前 CSV 列名为：{columns}"
    )


def load_sites(info_csv: Path) -> pd.DataFrame:
    """
    Load site id, longitude, latitude from DataFountain basic info CSV.

    This version is robust for columns like:
        光伏用户名称, 光伏用户编号, 装机容量(kW), 经度, 纬度
    """
    df = read_csv_smart(info_csv)
    cols = list(df.columns)

    print("[DEBUG] 基本信息 CSV 列名：")
    print(cols)
    print("[DEBUG] 基本信息 CSV 前几行：")
    print(df.head().to_string(index=False))

    # Longitude / latitude columns
    lon_col = guess_column(cols, ["经度", "longitude", "lon"])
    lat_col = guess_column(cols, ["纬度", "latitude", "lat"])

    # Prefer the real station id column.
    site_col = None

    preferred_site_cols = [
        "光伏用户编号",
        "用户编号",
        "user_id",
        "id",
        "编号",
        "光伏用户名称",
        "用户名称",
    ]

    for key in preferred_site_cols:
        for col in cols:
            if key.lower() in str(col).lower():
                site_col = col
                break
        if site_col is not None:
            break

    if site_col is None:
        raise KeyError(
            "无法识别站点编号列。\n"
            f"当前 CSV 列名为：{cols}"
        )

    sites = df[[site_col, lon_col, lat_col]].copy()
    sites.columns = ["site_id_raw", "longitude", "latitude"]

    # Extract f1, f2, ..., f9 from either "f1" or "f1光伏发电用户".
    sites["site_id"] = (
        sites["site_id_raw"]
        .astype(str)
        .str.strip()
        .str.extract(r"(f\d+)", expand=False)
    )

    sites["longitude"] = pd.to_numeric(sites["longitude"], errors="coerce")
    sites["latitude"] = pd.to_numeric(sites["latitude"], errors="coerce")

    sites = sites.dropna(subset=["site_id", "longitude", "latitude"]).copy()

    if sites.empty:
        raise ValueError(
            "基本信息 CSV 中没有识别到 f1-f9 站点。\n"
            "请检查站点编号列、经度列、纬度列是否正确。"
        )

    # Natural sort: f1, f2, ..., f9, f10
    sites["site_num"] = sites["site_id"].str.extract(r"f(\d+)").astype(int)
    sites = sites.sort_values("site_num").reset_index(drop=True)

    sites = sites[["site_id", "longitude", "latitude"]]

    print("[INFO] 成功识别站点经纬度：")
    print(sites.to_string(index=False))

    return sites


def extract_downloaded_csv(download_path: Path, out_csv: Path) -> None:
    """
    CDS time-series download may be a zip file or direct csv.
    This function handles both cases.
    """
    if download_path.suffix.lower() == ".csv":
        download_path.replace(out_csv)
        return

    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path, "r") as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise RuntimeError(f"压缩包中没有 CSV 文件：{download_path}")

            extracted = zf.extract(csv_names[0], download_path.parent)
            extracted_path = Path(extracted)

            if out_csv.exists():
                out_csv.unlink()

            extracted_path.replace(out_csv)
        return

    raise RuntimeError(
        f"无法识别下载文件格式：{download_path}\n"
        "它既不是 zip，也不是 csv。"
    )


def download_one_site(
    client: cdsapi.Client,
    site_id: str,
    longitude: float,
    latitude: float,
    start: str,
    end: str,
    out_dir: Path,
    overwrite: bool = False,
    sleep_seconds: float = 2.0,
) -> None:
    """
    Download ERA5 time-series CSV for one station.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"era5_timeseries_{site_id}.csv"
    tmp_download = out_dir / f"era5_timeseries_{site_id}.download"

    if out_csv.exists() and not overwrite:
        print(f"[SKIP] {site_id}: 已存在 {out_csv}")
        return

    if tmp_download.exists():
        tmp_download.unlink()

    request = {
        "variable": ERA5_VARIABLES,
        "location": {
            "longitude": float(longitude),
            "latitude": float(latitude),
        },
        "date": [f"{start}/{end}"],
        "data_format": "csv",
    }

    print("=" * 80)
    print(f"[INFO] 下载站点 {site_id}")
    print(f"       longitude = {longitude}")
    print(f"       latitude  = {latitude}")
    print(f"       date      = {start}/{end}")
    print(f"       output    = {out_csv}")
    print("=" * 80)

    try:
        result = client.retrieve(
            "reanalysis-era5-single-levels-timeseries",
            request,
        )

        result.download(str(tmp_download))
        extract_downloaded_csv(tmp_download, out_csv)

        if tmp_download.exists():
            tmp_download.unlink()

        # Quick check.
        check_df = read_csv_smart(out_csv)
        print(f"[OK] {site_id}: 保存成功，行数 = {len(check_df)}, 列数 = {len(check_df.columns)}")
        print(f"     columns = {list(check_df.columns)}")

    except Exception as e:
        print(f"[ERROR] {site_id}: 下载失败")
        print(e)
        raise

    time.sleep(sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--info_csv",
        type=str,
        default=None,
        help="DataFountain 基本信息 CSV 路径。如果不填，程序会自动从 data/raw/ 查找。",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="ERA5 输出目录。默认 data/era5_timeseries_new。",
    )

    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START,
        help="开始日期，格式 YYYY-MM-DD。默认 2022-01-02。",
    )

    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_END,
        help="结束日期，格式 YYYY-MM-DD。默认 2023-05-01。",
    )

    parser.add_argument(
        "--site",
        type=str,
        default="all",
        help="下载哪个站点，例如 f6；默认 all 下载全部站点。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果目标 CSV 已存在，是否重新下载覆盖。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()

    if args.info_csv is None:
        info_csv = auto_find_info_csv(project_root)
        print(f"[INFO] 自动找到基本信息文件：{info_csv}")
    else:
        info_csv = Path(args.info_csv).expanduser().resolve()

    if args.out_dir is None:
        out_dir = project_root / "data" / "era5_timeseries_new"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()

    if not info_csv.exists():
        raise FileNotFoundError(f"基本信息文件不存在：{info_csv}")

    sites = load_sites(info_csv)

    print("[INFO] 识别到站点：")
    print(sites.to_string(index=False))

    site_arg = args.site.strip().lower()

    if site_arg != "all":
        sites = sites[sites["site_id"].str.lower() == site_arg].copy()
        if sites.empty:
            raise ValueError(f"没有找到指定站点：{args.site}")

    client = cdsapi.Client()

    for _, row in sites.iterrows():
        download_one_site(
            client=client,
            site_id=row["site_id"],
            longitude=row["longitude"],
            latitude=row["latitude"],
            start=args.start,
            end=args.end,
            out_dir=out_dir,
            overwrite=args.overwrite,
        )

    print("\n[DONE] ERA5 time-series 下载完成。")
    print(f"[DONE] 输出目录：{out_dir}")


if __name__ == "__main__":
    main()