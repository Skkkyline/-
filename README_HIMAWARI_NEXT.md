# Himawari next-step scripts

Copy these files into your project `src/` directory:

- `check_himawari_segments.py`
- `test_read_himawari_one_slot.py`
- `extract_himawari_features_satpy.py`
- `merge_himawari_features.py`
- `plot_himawari_feature_day.py`

Recommended order:

```powershell
cd "E:\Hartley\BUCT\毕设2026.3.13\毕设\pv_thesis_rebuild_with_era5_scripts\pv_thesis_rebuild"

.\.venv\Scripts\python.exe -m pip install satpy pyresample "dask[array]" tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

.\.venv\Scripts\python.exe src\check_himawari_segments.py --raw_dir data\himawari\raw_selective

.\.venv\Scripts\python.exe src\test_read_himawari_one_slot.py --raw_dir data\himawari\raw_selective --date 20230415 --time 0000 --band B13

.\.venv\Scripts\python.exe src\extract_himawari_features_satpy.py --raw_dir data\himawari\raw_selective --info_csv data\raw\A榜-训练集_分布式光伏发电预测_基本信息.csv --bands B13 --out_csv data\processed\himawari_features_sample.csv

.\.venv\Scripts\python.exe src\merge_himawari_features.py --processed_csv data\processed\processed_all_stations_era5_timeseries_new.csv --himawari_csv data\processed\himawari_features_sample.csv --out_csv data\processed\processed_all_stations_era5_himawari_sample.csv

.\.venv\Scripts\python.exe src\plot_himawari_feature_day.py --merged_csv data\processed\processed_all_stations_era5_himawari_sample.csv --site f6 --date 2023-04-15 --feature sat_b13_mean
```
