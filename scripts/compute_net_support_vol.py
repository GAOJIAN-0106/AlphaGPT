"""
Compute 净支撑成交量因子 (Net Support Volume) from 1-minute kline data.

Algorithm (per product per day):
    1. Compute mean of all minute-bar close prices for the day
    2. Support volume  = sum(volume) where minute close < daily mean close
    3. Resistance volume = sum(volume) where minute close > daily mean close
    4. net_support_ratio = (support_vol - resistance_vol) / total_vol

Reference:
    沈芷琳, 刘富兵, 2024. 基于趋势资金日内交易行为的选股因子. 国盛证券.

Usage:
    python scripts/compute_net_support_vol.py
    python scripts/compute_net_support_vol.py --products rb al
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DUCKDB_PATH = os.getenv("DUCKDB_PATH",
                         os.path.expanduser("~/quant/tick_data/kline_1min.duckdb"))


def compute_net_support_vol(products=None):
    import duckdb

    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    product_filter = ""
    if products:
        plist = ", ".join(f"'{p}'" for p in products)
        product_filter = f"AND product_id IN ({plist})"

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
        daily_mean AS (
            -- Mean of minute-bar close prices for each product-day
            SELECT mc.product_id || '.' || mc.exchange as address,
                   mc.day,
                   AVG(k.close)::DOUBLE as mean_close
            FROM main_contracts mc
            JOIN kline_1min k ON k.symbol = mc.symbol
                AND DATE_TRUNC('day', k.datetime) = mc.day
            WHERE mc.rn = 1
            GROUP BY mc.product_id || '.' || mc.exchange, mc.day
        ),
        support_resist AS (
            -- Support/resistance volume split by mean close
            SELECT mc.product_id || '.' || mc.exchange as address,
                   mc.day as time,
                   SUM(CASE WHEN k.close < dm.mean_close THEN k.volume ELSE 0 END)::DOUBLE as support_vol,
                   SUM(CASE WHEN k.close > dm.mean_close THEN k.volume ELSE 0 END)::DOUBLE as resist_vol,
                   SUM(k.volume)::DOUBLE as total_vol
            FROM main_contracts mc
            JOIN kline_1min k ON k.symbol = mc.symbol
                AND DATE_TRUNC('day', k.datetime) = mc.day
            JOIN daily_mean dm ON dm.address = mc.product_id || '.' || mc.exchange
                AND dm.day = mc.day
            WHERE mc.rn = 1
            GROUP BY mc.product_id || '.' || mc.exchange, mc.day
        )
        SELECT address, time,
               (support_vol - resist_vol) / NULLIF(total_vol, 0) as net_support_vol
        FROM support_resist
        ORDER BY address, time
    """

    print("Computing net support volume from 1-min data...")
    df = con.execute(sql).fetchdf()
    con.close()

    print(f"  Raw rows: {len(df)}")
    df = df.dropna(subset=['net_support_vol'])
    print(f"  Valid rows: {len(df)}")
    print(f"  Products: {df['address'].nunique()}")
    print(f"  Date range: {df['time'].min()} → {df['time'].max()}")

    # Save cache
    out_path = os.path.join(os.path.dirname(DUCKDB_PATH), "net_support_vol_cache.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  Saved to: {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", nargs="*", default=None)
    args = parser.parse_args()
    compute_net_support_vol(args.products)
