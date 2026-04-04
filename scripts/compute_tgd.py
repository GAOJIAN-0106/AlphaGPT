"""
Compute TGD (跌幅时间重心偏离 / Time Gravity Deviation) from 1-minute data.

Algorithm (per day):
    1. For each product, compute 1-min returns: r_t = close_t / close_{t-1} - 1
    2. Separate into up-moves (r > 0) and down-moves (r < 0)
    3. Compute return-weighted time gravity centers:
       G_u = Σ(t_i * |r_i|) / Σ|r_i|  for up bars
       G_d = Σ(t_j * |r_j|) / Σ|r_j|  for down bars
    4. Cross-sectional regression: G_d = α + β·G_u + ε
    5. TGD = residual ε (deviation of down-move timing from up-move prediction)

Reference:
    魏建榕, 苗杰, 徐少楠, 2022. 日内分钟收益率的时间衍生变量. 开源证券.

Usage:
    python scripts/compute_tgd.py
    python scripts/compute_tgd.py --products rb al
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


def compute_tgd(products=None):
    import duckdb

    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    product_filter = ""
    if products:
        plist = ", ".join(f"'{p}'" for p in products)
        product_filter = f"WHERE product_id IN ({plist})"

    sql = f"""
        WITH daily_oi AS (
            SELECT product_id, exchange, symbol,
                   DATE_TRUNC('day', datetime) as day,
                   AVG(close_oi) as avg_oi
            FROM kline_1min
            {product_filter}
            GROUP BY product_id, exchange, symbol, day
        ),
        main_contracts AS (
            SELECT day, product_id, exchange, symbol,
                   ROW_NUMBER() OVER (
                       PARTITION BY product_id, day ORDER BY avg_oi DESC
                   ) as rn
            FROM daily_oi
        )
        SELECT
            mc.product_id || '.' || mc.exchange as address,
            mc.day,
            k.datetime,
            k.close
        FROM main_contracts mc
        JOIN kline_1min k ON k.symbol = mc.symbol
            AND DATE_TRUNC('day', k.datetime) = mc.day
        WHERE mc.rn = 1
        ORDER BY address, mc.day, k.datetime
    """

    print("Querying 1-min data from DuckDB...")
    df = con.execute(sql).fetchdf()
    con.close()
    print(f"Loaded {len(df)} rows, {df['address'].nunique()} products, "
          f"{df['day'].nunique()} days")

    # Step 1-3: compute G_u and G_d per product per day
    daily_records = []

    for (address, day), group in df.groupby(['address', 'day']):
        group = group.sort_values('datetime').reset_index(drop=True)
        if len(group) < 10:
            continue

        close = group['close'].values.astype(np.float64)
        ret = close[1:] / close[:-1] - 1.0
        # Time indices: 1-based minute within the day
        t_idx = np.arange(1, len(ret) + 1, dtype=np.float64)

        # Separate up and down
        up_mask = ret > 0
        dn_mask = ret < 0

        if up_mask.sum() < 3 or dn_mask.sum() < 3:
            continue

        abs_ret_up = np.abs(ret[up_mask])
        abs_ret_dn = np.abs(ret[dn_mask])
        t_up = t_idx[up_mask]
        t_dn = t_idx[dn_mask]

        sum_abs_up = abs_ret_up.sum()
        sum_abs_dn = abs_ret_dn.sum()
        if sum_abs_up < 1e-15 or sum_abs_dn < 1e-15:
            continue

        g_u = (t_up * abs_ret_up).sum() / sum_abs_up
        g_d = (t_dn * abs_ret_dn).sum() / sum_abs_dn

        daily_records.append({
            'time': pd.Timestamp(day),
            'address': address,
            'g_u': float(g_u),
            'g_d': float(g_d),
        })

    if not daily_records:
        print("No results computed.")
        return

    result_df = pd.DataFrame(daily_records)
    print(f"Computed G_u/G_d for {len(result_df)} product-day observations")

    # Step 4: cross-sectional regression per day: G_d = α + β·G_u + ε
    residuals = []

    for day, day_data in result_df.groupby('time'):
        if len(day_data) < 5:
            # Too few products for meaningful regression
            for _, row in day_data.iterrows():
                residuals.append({
                    'time': row['time'],
                    'address': row['address'],
                    'tgd': row['g_d'] - row['g_u'],  # fallback: simple difference
                })
            continue

        g_u = day_data['g_u'].values
        g_d = day_data['g_d'].values

        # OLS: G_d = α + β·G_u + ε
        X = np.column_stack([np.ones(len(g_u)), g_u])
        try:
            beta = np.linalg.lstsq(X, g_d, rcond=None)[0]
            eps = g_d - X @ beta
        except np.linalg.LinAlgError:
            eps = g_d - g_u  # fallback

        for i, (_, row) in enumerate(day_data.iterrows()):
            residuals.append({
                'time': row['time'],
                'address': row['address'],
                'tgd': float(eps[i]),
            })

    out_df = pd.DataFrame(residuals)
    out_df = out_df.sort_values(['address', 'time']).reset_index(drop=True)

    # Save cache
    data_dir = os.path.dirname(DUCKDB_PATH)
    cache_path = os.path.join(data_dir, "tgd_cache.parquet")
    out_df.to_parquet(cache_path, index=False)

    print(f"Saved: {cache_path}")
    print(f"  {len(out_df)} rows, {out_df['address'].nunique()} products, "
          f"{out_df['time'].nunique()} days")
    print(f"  Range: {out_df['time'].min().date()} → {out_df['time'].max().date()}")
    print(f"  Value stats: mean={out_df['tgd'].mean():.4f}, "
          f"std={out_df['tgd'].std():.4f}")
    return cache_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute TGD factor")
    parser.add_argument("--products", nargs='+', default=None,
                        help="Product IDs to compute (default: all)")
    args = parser.parse_args()
    compute_tgd(products=args.products)
