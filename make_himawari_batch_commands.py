# -*- coding: utf-8 -*-
"""
Generate PowerShell commands to download Himawari selected dates using download_himawari_selective.py.

Example:
    python src/make_himawari_batch_commands.py --start 2023-03-01 --end 2023-04-30 --bands B13 --step 60
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-03-01")
    p.add_argument("--end", default="2023-04-30")
    p.add_argument("--bands", nargs="+", default=["B13"])
    p.add_argument("--local_start", default="08:00")
    p.add_argument("--local_end", default="16:00")
    p.add_argument("--step", type=int, default=60)
    p.add_argument("--out_ps1", default="scripts/download_himawari_batch.ps1")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dates = pd.date_range(args.start, args.end, freq="D")
    out = Path(args.out_ps1)
    out.parent.mkdir(parents=True, exist_ok=True)
    bands = " ".join(args.bands)
    dry = " --dry_run" if args.dry_run else ""

    lines = []
    lines.append("# Auto-generated Himawari batch download commands")
    lines.append("# Run from project root")
    lines.append("$ErrorActionPreference = 'Stop'")
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        lines.append(
            f".\\.venv\\Scripts\\python.exe src\\download_himawari_selective.py --date {date_str} --bands {bands} --start {args.local_start} --end {args.local_end} --step {args.step}{dry}"
        )
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    print("[OK] saved:", out)
    print("[OK] date count:", len(dates))
    print("[INFO] First commands:")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
