# -*- coding: utf-8 -*-
"""
Check Himawari AHI HSD segment completeness.

Expected filename example:
    HS_H09_20230407_0000_B13_FLDK_R20_S0110.DAT.bz2

Usage:
    python src/check_himawari_segments.py --raw_dir data/himawari/raw_selective
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


PATTERN = re.compile(
    r"HS_H(?P<sat>\d{2})_(?P<date>\d{8})_(?P<time>\d{4})_(?P<band>B\d{2})_FLDK_(?P<res>R\d{2})_S(?P<seg>\d{2})(?P<nseg>\d{2})\.DAT(?:\.bz2)?$",
    re.IGNORECASE,
)


def human_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/himawari/raw_selective")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    files = list(raw_dir.rglob("*.DAT")) + list(raw_dir.rglob("*.DAT.bz2"))
    groups = defaultdict(list)
    bad = []

    for f in files:
        m = PATTERN.search(f.name)
        if not m:
            bad.append(f)
            continue
        d = m.groupdict()
        key = (d["date"], d["time"], d["band"].upper(), d["res"].upper())
        groups[key].append((int(d["seg"]), int(d["nseg"]), f, f.stat().st_size))

    print(f"[INFO] raw_dir: {raw_dir}")
    print(f"[INFO] total DAT/DAT.bz2 files: {len(files)}")
    print(f"[INFO] parsed groups: {len(groups)}")
    if bad:
        print(f"[WARN] unrecognized files: {len(bad)}")
        for f in bad[:20]:
            print("   ", f)

    problem_count = 0
    total_size = 0
    rows = []
    for key, items in sorted(groups.items()):
        date, time, band, res = key
        nseg_expected = max(nseg for _, nseg, _, _ in items)
        segs = sorted({seg for seg, _, _, _ in items})
        missing = [s for s in range(1, nseg_expected + 1) if s not in segs]
        size = sum(sz for _, _, _, sz in items)
        total_size += size
        ok = len(missing) == 0
        if not ok:
            problem_count += 1
        rows.append((date, time, band, res, len(items), nseg_expected, missing, size, ok))

    print("\n[INFO] first 30 groups:")
    for r in rows[:30]:
        date, time, band, res, n_files, nseg_expected, missing, size, ok = r
        print(
            f"  {date} {time} {band} {res}: files={n_files}/{nseg_expected}, "
            f"size={human_size(size)}, ok={ok}, missing={missing}"
        )

    print("\n[SUMMARY]")
    print(f"  groups: {len(rows)}")
    print(f"  incomplete groups: {problem_count}")
    print(f"  total size: {human_size(total_size)}")

    if problem_count > 0:
        print("\n[WARN] Some time-band groups are incomplete. Satpy may fail for those groups.")
    else:
        print("\n[OK] All parsed groups look complete.")


if __name__ == "__main__":
    main()
