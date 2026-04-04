"""
Compute 偏度因子 (Return Skewness) from 5-min returns.

Algorithm (per product per day):
    1. Aggregate 1-min bars to 5-min bars
    2. Compute 5-min returns: ret_i = close_i / close_{i-1} - 1
    3. Over a rolling N-day window (default N=20):
       skew = E[((ret - μ) / σ)³]
       where μ, σ are the mean and std of all 5-min returns in the window

Reference:
    张革, 2022. 动量及高阶矩因子在商品期货截面上的运用. 中信期货.

Usage:
    python scripts/compute_ret_skew.py
    python scripts/compute_ret_skew.py --products rb al
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DUCKDB_PATH = os.getenv("DUCKDB_PATH",
                         os.path.expanduser("~/quant/tick_data/kline_1min.duckdb"))


def compute_ret_skew(products=None, lookback_days=20):
    import duckdb

    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    product_filter = ""
    if products:
        plist = ", ".join(f"'{p}'" for p in products)
        product_filter = f"AND product_id IN ({plist})"

    # Step 1: Get 5-min OHLCV bars for main contracts
    sql = f"""
        WITH daily_oi AS (
            SELECT product_id, exchange, symbol,
                   DATE_TRUNC('day', datetime) as day,
                   AVG(close_oi) as avg_oi
            FROM kline_1min
            WHERE 1=1 {product_filter}
            GROUP BY product_id, exchange, symbol, day
        ),
        main_contracts AS (
            SELECT day, product_id, exchange, symbol,
                   ROW_NUMBER() OVER (
                       PARTITION BY product_id, day ORDER BY avg_oi DESC
                   ) as rn
            FROM daily_oi
        ),
        bars_5min AS (
            SELECT mc.product_id || '.' || mc.exchange as address,
                   mc.day,
                   TIME_BUCKET(INTERVAL '5 minutes', k.datetime) as bar_time,
                   LAST(k.close ORDER BY k.datetime) as close
            FROM main_contracts mc
            JOIN kline_1min k ON k.symbol = mc.symbol
                AND DATE_TRUNC('day', k.datetime) = mc.day
            WHERE mc.rn = 1
            GROUP BY mc.product_id || '.' || mc.exchange, mc.day, bar_time
        )
        SELECT address, day, bar_time, close
        FROM bars_5min
        ORDER BY address, day, bar_time
    """

    print("Loading 5-min bars from 1-min data...")
    df = con.execute(sql).fetchdf()
    con.close()
    print(f"  5-min bars: {len(df)}")

    # Step 2: Compute 5-min returns per address per day
    df = df.sort_values(['address', 'day', 'bar_time'])
    df['prev_close'] = df.groupby(['address', 'day'])['close'].shift(1)
    df['ret_5min'] = df['close'] / df['prev_close'] - 1.0
    df = df.dropna(subset=['ret_5min'])

    # Step 3: Rolling N-day skewness of 5-min returns
    # Collect all 5-min returns per (address, day)
    daily_rets = df.groupby(['address', 'day'])['ret_5min'].apply(list).reset_index()
    daily_rets = daily_rets.sort_values(['address', 'day'])

    results = []
    print("Computing rolling skewness...")
    for addr, grp in daily_rets.groupby('address'):
        grp = grp.sort_values('day').reset_index(drop=True)
        days = grp['day'].values
        ret_lists = grp['ret_5min'].values

        for i in range(len(grp)):
            start_idx = max(0, i - lookback_days + 1)
            # Pool all 5-min returns from the lookback window
            pooled = []
            for j in range(start_idx, i + 1):
                pooled.extend(ret_lists[j])

            if len(pooled) < 10:
                continue

            arr = np.array(pooled, dtype=np.float64)
            mu = arr.mean()
            sigma = arr.std()
            if sigma < 1e-12:
                continue
            skew = np.mean(((arr - mu) / sigma) ** 3)

            results.append({
                'address': addr,
                'time': days[i],
                'ret_skew': float(skew),
            })

    out = pd.DataFrame(results)
    print(f"  Valid rows: {len(out)}")
    print(f"  Products: {out['address'].nunique()}")
    print(f"  Date range: {out['time'].min()} → {out['time'].max()}")
    print(f"  Skew range: [{out['ret_skew'].min():.4f}, {out['ret_skew'].max():.4f}]")

    out_path = os.path.join(os.path.dirname(DUCKDB_PATH), "ret_skew_cache.parquet")
    out.to_parquet(out_path, index=False)
    print(f"  Saved to: {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", nargs="*", default=None)
    parser.add_argument("--lookback", type=int, default=20)
    args = parser.parse_args()
    compute_ret_skew(args.products, args.lookback)
