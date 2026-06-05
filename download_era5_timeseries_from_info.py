import argparse
from pathlib import Path
import zipfile
import pandas as pd
import cdsapi

VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    'surface_pressure',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_solar_radiation_downwards',
    'total_precipitation',
]

def read_info(path: Path) -> pd.DataFrame:
    for enc in ['utf-8-sig', 'gbk', 'utf-8']:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

def download_one(client, site_id, lon, lat, start, end, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'era5_timeseries_{site_id}.csv'
    out_zip = out_dir / f'era5_timeseries_{site_id}.zip'
    if out_csv.exists():
        print(f'[SKIP] {out_csv}')
        return
    req = {
        'variable': VARIABLES,
        'location': {'longitude': float(lon), 'latitude': float(lat)},
        'date': [f'{start}/{end}'],
        'data_format': 'csv',
    }
    print(f'[INFO] download {site_id}: lon={lon}, lat={lat}, date={start}/{end}')
    result = client.retrieve('reanalysis-era5-single-levels-timeseries', req)
    result.download(str(out_zip))
    with zipfile.ZipFile(out_zip, 'r') as zf:
        csvs = [x for x in zf.namelist() if x.lower().endswith('.csv')]
        if not csvs:
            raise RuntimeError(f'No csv found in {out_zip}')
        extracted = Path(zf.extract(csvs[0], out_dir))
        extracted.replace(out_csv)
    print(f'[OK] saved {out_csv}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--info_csv', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, default=Path('data/era5_timeseries'))
    ap.add_argument('--start', default='2022-01-02')
    ap.add_argument('--end', default='2023-05-01')
    ap.add_argument('--site', default='all', help='all or one site id, e.g. f6')
    args = ap.parse_args()
    info = read_info(args.info_csv)
    col_id, col_lon, col_lat = '光伏用户编号', '经度', '纬度'
    if args.site != 'all':
        info = info[info[col_id] == args.site]
        if info.empty:
            raise ValueError(f'No site {args.site} in {args.info_csv}')
    client = cdsapi.Client()
    for _, r in info.iterrows():
        download_one(client, r[col_id], r[col_lon], r[col_lat], args.start, args.end, args.out_dir)

if __name__ == '__main__':
    main()
