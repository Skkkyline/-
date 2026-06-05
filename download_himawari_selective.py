# -*- coding: utf-8 -*-
"""
Selective downloader for Himawari-9 AHI-L1b-FLDK files from AWS Open Data.

Purpose:
    Avoid downloading a full UTC day and all time slots.
    Download only selected local-time slots and selected bands.

Example:
    python src/download_himawari_selective.py --date 2023-04-15 --bands B13 --start 08:00 --end 16:00 --step 60 --dry_run

    python src/download_himawari_selective.py --date 2023-04-15 --bands B13 --start 08:00 --end 16:00 --step 60

    python src/download_himawari_selective.py --date 2023-04-15 --bands B03 B13 --start 06:00 --end 18:00 --step 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm


BUCKET = "noaa-himawari9"
BASE_PREFIX = "AHI-L1b-FLDK"


def parse_hhmm(text: str) -> tuple[int, int]:
    h, m = text.split(":")
    return int(h), int(m)


def human_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def build_target_utc_tokens(local_date: str, start: str, end: str, step: int) -> set[str]:
    """
    Build UTC time tokens matching Himawari filename pattern:
        YYYYMMDD_HHMM

    Input local time is Beijing time = UTC + 8.
    """
    date0 = datetime.strptime(local_date, "%Y-%m-%d")

    sh, sm = parse_hhmm(start)
    eh, em = parse_hhmm(end)

    local_start = date0.replace(hour=sh, minute=sm, second=0)
    local_end = date0.replace(hour=eh, minute=em, second=0)

    if local_end < local_start:
        local_end += timedelta(days=1)

    tokens = set()
    cur = local_start
    while cur <= local_end:
        utc = cur - timedelta(hours=8)
        tokens.add(utc.strftime("%Y%m%d_%H%M"))
        cur += timedelta(minutes=step)

    return tokens


def needed_utc_dates_from_tokens(tokens: set[str]) -> list[tuple[str, str, str]]:
    dates = sorted({t.split("_")[0] for t in tokens})
    out = []
    for d in dates:
        out.append((d[:4], d[4:6], d[6:8]))
    return out


def list_matching_objects(s3, tokens: set[str], bands: list[str]) -> list[dict]:
    """
    List S3 objects whose key contains:
        - one target UTC time token, e.g. 20230415_0000
        - one selected band, e.g. _B13_
    """
    matched = []

    utc_dates = needed_utc_dates_from_tokens(tokens)

    for year, month, day in utc_dates:
        prefix = f"{BASE_PREFIX}/{year}/{month}/{day}/"
        print(f"[INFO] Listing s3://{BUCKET}/{prefix}")

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]

                has_time = any(token in key for token in tokens)
                if not has_time:
                    continue

                has_band = any(f"_{band}_" in key or band in Path(key).name for band in bands)
                if not has_band:
                    continue

                matched.append(obj)

    matched = sorted(matched, key=lambda x: x["Key"])
    return matched


def download_objects(s3, objects: list[dict], out_dir: Path, overwrite: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for obj in tqdm(objects, desc="Downloading"):
        key = obj["Key"]

        # Keep a clean local structure:
        # data/himawari/raw_selective/2023/04/15/filename
        rel = key.replace(f"{BASE_PREFIX}/", "")
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            continue

        s3.download_file(BUCKET, key, str(dst))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--date", required=True, help="Local Beijing date, e.g. 2023-04-15")
    parser.add_argument("--bands", nargs="+", default=["B13"], help="Bands, e.g. B13 or B03 B13")
    parser.add_argument("--start", default="08:00", help="Local Beijing start time, default 08:00")
    parser.add_argument("--end", default="16:00", help="Local Beijing end time, default 16:00")
    parser.add_argument("--step", type=int, default=60, help="Time step in minutes, default 60")
    parser.add_argument("--out_dir", default="data/himawari/raw_selective", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Only list matched files and size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bands = [b.upper() for b in args.bands]
    tokens = build_target_utc_tokens(
        local_date=args.date,
        start=args.start,
        end=args.end,
        step=args.step,
    )

    print("[INFO] Local Beijing date:", args.date)
    print("[INFO] Local time range:", args.start, "-", args.end)
    print("[INFO] Step:", args.step, "min")
    print("[INFO] Bands:", bands)
    print("[INFO] UTC tokens:")
    for t in sorted(tokens):
        print("   ", t)

    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
    )

    objects = list_matching_objects(s3, tokens=tokens, bands=bands)

    total_size = sum(int(obj["Size"]) for obj in objects)

    print("\n[INFO] Matched file count:", len(objects))
    print("[INFO] Total size:", human_size(total_size))

    print("\n[INFO] First 20 matched files:")
    for obj in objects[:20]:
        print("   ", obj["Key"], human_size(int(obj["Size"])))

    if args.dry_run:
        print("\n[DRY RUN] No files downloaded.")
        return

    if not objects:
        print("[WARN] No matched objects. Check path, date, time, or band names.")
        return

    out_dir = Path(args.out_dir)
    download_objects(s3, objects=objects, out_dir=out_dir, overwrite=args.overwrite)

    print("\n[DONE] Download completed.")
    print("[DONE] Output directory:", out_dir.resolve())


if __name__ == "__main__":
    main()