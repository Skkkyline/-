import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RENAME = {
    'u10': 'era5_u10_ms',
    'v10': 'era5_v10_ms',
    'd2m': 'era5_d2m_k',
    't2m': 'era5_t2m_k',
    'sp': 'era5_sp_pa',
    'ssrd': 'era5_ssrd_jm2',
    'tp': 'era5_tp_m',
    'latitude': 'era5_grid_lat',
    'longitude': 'era5_grid_lon',
}

def read_era5_csv(path: Path, user_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'valid_time' not in df.columns:
        raise ValueError(f'{path} does not contain valid_time column. Columns: {df.columns.tolist()}')
    # CDS valid_time is UTC; convert to China Standard Time (UTC+8) and make timezone-naive.
    df['valid_time_utc'] = pd.to_datetime(df['valid_time'], utc=True)
    df['dt'] = df['valid_time_utc'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    keep = ['dt'] + [c for c in df.columns if c.startswith('era5_')]
    df = df[keep].copy()
    df['user_id'] = user_id
    # Derived variables.
    if {'era5_u10_ms', 'era5_v10_ms'}.issubset(df.columns):
        df['era5_wind10_ms'] = np.sqrt(df['era5_u10_ms'] ** 2 + df['era5_v10_ms'] ** 2)
        # meteorological direction: direction wind comes from
        df['era5_wind10_dir_deg'] = (np.degrees(np.arctan2(-df['era5_u10_ms'], -df['era5_v10_ms'])) + 360) % 360
    if 'era5_t2m_k' in df.columns:
        df['era5_t2m_c'] = df['era5_t2m_k'] - 273.15
    if 'era5_d2m_k' in df.columns:
        df['era5_d2m_c'] = df['era5_d2m_k'] - 273.15
    if 'era5_ssrd_jm2' in df.columns:
        # ERA5 ssrd is accumulated energy over the preceding hour in J/m2.
        # Divide by 3600 to get approximate W/m2 hourly mean.
        df['era5_ssrd_wm2_1h'] = df['era5_ssrd_jm2'] / 3600.0
    return df.sort_values('dt')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed_csv', type=Path, required=True)
    ap.add_argument('--era5_dir', type=Path, required=True)
    ap.add_argument('--out_csv', type=Path, required=True)
    ap.add_argument('--tolerance_minutes', type=int, default=75)
    args = ap.parse_args()

    pv = pd.read_csv(args.processed_csv, parse_dates=['dt'])
    pv = pv.sort_values(['user_id','dt'])

    parts = []
    grids = []
    for path in sorted(args.era5_dir.glob('era5_timeseries_f*.csv')):
        user_id = path.stem.replace('era5_timeseries_', '')
        era = read_era5_csv(path, user_id)
        grids.append(era[['user_id','era5_grid_lat','era5_grid_lon']].drop_duplicates().head(1))
        sub = pv[pv['user_id'] == user_id].sort_values('dt').copy()
        if sub.empty:
            print(f'[WARN] no PV rows for {user_id}, skip {path.name}')
            continue
        merged = pd.merge_asof(
            sub,
            era.sort_values('dt'),
            on='dt',
            by='user_id',
            direction='backward',
            tolerance=pd.Timedelta(minutes=args.tolerance_minutes),
        )
        parts.append(merged)
        era_cols = [c for c in merged.columns if c.startswith('era5_')]
        missing_rate = merged[era_cols].isna().mean().mean() if era_cols else np.nan
        print(f'[OK] {user_id}: rows={len(merged)}, avg_era5_missing={missing_rate:.4%}')

    if not parts:
        raise RuntimeError('No merged station data created. Check --era5_dir and file names.')

    out = pd.concat(parts, ignore_index=True).sort_values(['user_id','dt'])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding='utf-8-sig')

    grid_df = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()
    if not grid_df.empty:
        print('\nERA5 grid points in uploaded files:')
        print(grid_df.to_string(index=False))
        n_unique = grid_df[['era5_grid_lat','era5_grid_lon']].drop_duplicates().shape[0]
        if n_unique == 1 and grid_df['user_id'].nunique() > 1:
            print('[WARN] All station CSVs use the same ERA5 grid point. This is fine for f6-only experiments, but redownload is recommended for multi-station experiments.')

    era_cols = [c for c in out.columns if c.startswith('era5_')]
    print(f'\nsaved: {args.out_csv}')
    print(f'rows={len(out)}, era5_cols={len(era_cols)}')
    print(out[era_cols].isna().mean().sort_values(ascending=False).head(20).to_string())

if __name__ == '__main__':
    main()
