"""
Compute inventory lunar-calendar YoY growth rate (库存农历同比增速).

Formula:
    YOY(y,t) = (stock_lunar_t - mean(stock_lunar at same lunar date, past n years))
               / mean(stock_lunar at same lunar date, past n years)

Lunar calendar alignment captures Chinese holiday seasonal patterns
(Spring Festival restocking, post-holiday drawdowns) better than Gregorian.

Reference:
    吴先兴, 2019. 基本面逻辑下的因子改进与策略组合 (模拟盘篇). 天风证券.

Usage:
    python scripts/compute_inv_lunar_yoy.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from lunardate import LunarDate
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

DUCKDB_PATH = os.getenv("DUCKDB_PATH",
                         os.path.expanduser("~/quant/tick_data/kline_1min.duckdb"))


def _to_lunar_key(d):
    """Convert a date to lunar (month, day) tuple for YoY matching."""
    try:
        ld = LunarDate.fromSolarDate(d.year, d.month, d.day)
        return (ld.month, ld.day)
    except Exception:
        return None


def _lunar_year(d):
    """Get lunar year for a date."""
    try:
        ld = LunarDate.fromSolarDate(d.year, d.month, d.day)
        return ld.year
    except Exception:
        return None


def compute_inv_lunar_yoy(n_years=2):
    """Compute lunar YoY for inventory data.

    Args:
        n_years: number of prior years to average (default 2, paper recommends <=3)
    """
    data_dir = os.path.dirname(DUCKDB_PATH)
    inv_path = os.path.join(data_dir, "inventory_cache.parquet")

    if not os.path.exists(inv_path):
        print(f"Error: {inv_path} not found. Run scripts/fetch_edb_inventory.py first.")
        return

    inv_df = pd.read_parquet(inv_path)
    inv_df['time'] = pd.to_datetime(inv_df['time'])
    inv_df = inv_df.sort_values(['address', 'time']).reset_index(drop=True)
    print(f"Loaded inventory: {len(inv_df)} rows, {inv_df['address'].nunique()} products")
    print(f"  Date range: {inv_df['time'].min().date()} → {inv_df['time'].max().date()}")

    # Pre-compute lunar keys for all dates
    all_dates = sorted(inv_df['time'].dt.date.unique())
    date_to_lunar_key = {}
    date_to_lunar_year = {}
    for d in all_dates:
        date_to_lunar_key[d] = _to_lunar_key(d)
        date_to_lunar_year[d] = _lunar_year(d)

    results = []

    for address, group in inv_df.groupby('address'):
        group = group.set_index('time').sort_index()

        # Build lunar-key → {lunar_year: inventory_value} lookup
        lunar_lookup = {}  # (month, day) → {year: value}
        for dt, row in group.iterrows():
            d = dt.date()
            lk = date_to_lunar_key.get(d)
            ly = date_to_lunar_year.get(d)
            if lk is None or ly is None:
                continue
            if lk not in lunar_lookup:
                lunar_lookup[lk] = {}
            lunar_lookup[lk][ly] = row['inventory']

        # Compute YoY for each date
        for dt, row in group.iterrows():
            d = dt.date()
            lk = date_to_lunar_key.get(d)
            ly = date_to_lunar_year.get(d)
            if lk is None or ly is None:
                continue

            current_val = row['inventory']
            if current_val <= 0 or np.isnan(current_val):
                continue

            # Find same lunar date in past n years
            prior_vals = []
            year_data = lunar_lookup.get(lk, {})
            for yr_offset in range(1, n_years + 1):
                target_year = ly - yr_offset
                if target_year in year_data:
                    pv = year_data[target_year]
                    if pv > 0 and not np.isnan(pv):
                        prior_vals.append(pv)

            if not prior_vals:
                # Fallback: try nearest lunar dates (±3 days) in prior years
                for day_offset in range(1, 4):
                    for sign in [1, -1]:
                        neighbor_key = (lk[0], lk[1] + sign * day_offset)
                        if neighbor_key[1] < 1 or neighbor_key[1] > 30:
                            continue
                        year_data_nb = lunar_lookup.get(neighbor_key, {})
                        for yr_offset in range(1, n_years + 1):
                            target_year = ly - yr_offset
                            if target_year in year_data_nb:
                                pv = year_data_nb[target_year]
                                if pv > 0 and not np.isnan(pv):
                                    prior_vals.append(pv)
                    if prior_vals:
                        break

            if not prior_vals:
                continue

            prior_mean = np.mean(prior_vals)
            if prior_mean <= 0:
                continue

            yoy = (current_val - prior_mean) / prior_mean

            results.append({
                'time': pd.Timestamp(d),
                'address': address,
                'inv_lunar_yoy': float(yoy),
            })

    if not results:
        print("No results computed.")
        return

    out_df = pd.DataFrame(results)
    out_df = out_df.drop_duplicates(subset=['time', 'address'], keep='last')
    out_df = out_df.sort_values(['address', 'time']).reset_index(drop=True)

    cache_path = os.path.join(data_dir, "inv_lunar_yoy_cache.parquet")
    out_df.to_parquet(cache_path, index=False)

    print(f"Saved: {cache_path}")
    print(f"  {len(out_df)} rows, {out_df['address'].nunique()} products, "
          f"{out_df['time'].nunique()} dates")
    print(f"  Range: {out_df['time'].min().date()} → {out_df['time'].max().date()}")
    print(f"  Value stats: mean={out_df['inv_lunar_yoy'].mean():.4f}, "
          f"std={out_df['inv_lunar_yoy'].std():.4f}")
    return cache_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute lunar YoY inventory factor")
    parser.add_argument("--n-years", type=int, default=2,
                        help="Years of history for comparison (default: 2)")
    args = parser.parse_args()
    compute_inv_lunar_yoy(n_years=args.n_years)
