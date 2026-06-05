import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from train_baseline import WEATHER_COLS, TIME_COLS, add_supervised_features, chrono_split


def calc_metrics(y, pred):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
        'MAE': float(mean_absolute_error(y, pred)),
        'R2': float(r2_score(y, pred)),
        'n': int(len(y)),
    }


def run_one(df: pd.DataFrame, user: str, filter_mode: str):
    rows = []
    df_user = df[df['user_id'] == user].copy()
    if df_user.empty:
        raise ValueError(f'No rows for user={user}')
    cap = float(df_user['capacity_kw'].dropna().iloc[0])
    era5_cols = [c for c in df.columns if c.startswith('era5_') and c not in {'era5_grid_lat', 'era5_grid_lon'}]

    for h in [1, 2, 4]:
        featdf = add_supervised_features(df_user, h)
        power_cols = [c for c in featdf.columns if c.startswith(('lag_', 'roll_', 'delta_'))]
        base_required = ['target'] + power_cols
        sample = featdf.dropna(subset=base_required).copy()

        if filter_mode == 'day':
            sample = sample[(sample['irradiance'] > 0) | (sample['target'] > 0)]

        train, val, test = chrono_split(sample)
        pred_col = f'lag_{h}'
        rows.append({
            'user': user, 'filter': filter_mode, 'horizon_min': h * 15,
            'feature_set': 'persistence', 'model': 'Persistence',
            **calc_metrics(test['target'], test[pred_col])
        })

        feature_sets = {
            'power_only': power_cols + TIME_COLS,
            'power_weather': power_cols + TIME_COLS + WEATHER_COLS,
        }
        if era5_cols:
            feature_sets['power_weather_era5'] = power_cols + TIME_COLS + WEATHER_COLS + era5_cols

        for feature_set, cols in feature_sets.items():
            missing = [c for c in cols if c not in sample.columns]
            if missing:
                print(f'[WARN] skip {feature_set}, missing columns: {missing[:5]}')
                continue
            train_all = pd.concat([train, val], axis=0)
            # If ERA5 columns contain missing values, drop only rows needed for that feature set.
            train_all2 = train_all.dropna(subset=cols + ['target'])
            test2 = test.dropna(subset=cols + ['target'])
            model = XGBRegressor(
                n_estimators=450,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=4,
                reg_lambda=2.0,
                min_child_weight=2,
                eval_metric='rmse',
            )
            model.fit(train_all2[cols], train_all2['target'], verbose=False)
            pred = np.clip(model.predict(test2[cols]), 0, cap)
            rows.append({
                'user': user, 'filter': filter_mode, 'horizon_min': h * 15,
                'feature_set': feature_set, 'model': 'XGBoost',
                **calc_metrics(test2['target'], pred)
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_csv', type=Path, required=True)
    parser.add_argument('--user', default='f6')
    parser.add_argument('--filter', choices=['all', 'day'], default='day')
    parser.add_argument('--out_dir', type=Path, default=Path('results'))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.processed_csv, parse_dates=['dt'])
    res = run_one(df, args.user, args.filter)
    out = args.out_dir / f'baseline_era5_{args.user}_{args.filter}.csv'
    res.to_csv(out, index=False, encoding='utf-8-sig')
    print(res.to_string(index=False))
    print(f'saved: {out}')

if __name__ == '__main__':
    main()
