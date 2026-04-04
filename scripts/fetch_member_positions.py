"""
拉取期货会员持仓排名数据，计算 LS_raw 并缓存到 parquet。

LS_raw = Σ(top20 long_oi) - Σ(top20 short_oi)

数据来源: 天勤 api.query_symbol_ranking()
支持增量更新：已有缓存时只拉取新日期。

Usage:
    python scripts/fetch_member_positions.py
    python scripts/fetch_member_positions.py --days 500
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
from datetime import date, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_data_dir():
    db_path = os.getenv("DUCKDB_PATH",
                        os.path.expanduser("~/quant/tick_data/kline_1min.duckdb"))
    return os.path.dirname(db_path)


def _parse_product_id(symbol: str) -> tuple:
    """Extract (exchange, product_id) from 'SHFE.rb2605' → ('SHFE', 'rb')."""
    parts = symbol.split(".")
    if len(parts) != 2:
        return None, None
    exchange = parts[0]
    m = re.match(r'^([a-zA-Z]+)', parts[1])
    product_id = m.group(1) if m else parts[1]
    return exchange, product_id


def _discover_products(api):
    """Discover all products and their main contract symbols."""
    exchanges = os.getenv("TIANQIN_FUTURES_EXCHANGES",
                          "SHFE,DCE,CZCE,INE,GFEX").split(",")
    products = []
    for exch in [e.strip() for e in exchanges]:
        try:
            conts = api.query_cont_quotes(exchange_id=exch)
            for sym in conts:
                exchange, product_id = _parse_product_id(sym)
                if exchange and product_id:
                    products.append({
                        'exchange': exchange,
                        'product_id': product_id,
                        'main_contract': sym,
                    })
            print(f"  {exch}: {len([p for p in products if p['exchange'] == exch])} products")
        except Exception as e:
            print(f"  {exch}: error - {e}")
    return products


def _fetch_product_ls(api, contract, start_dt, fetch_days):
    """Fetch LS_raw and long concentration std for one product.

    Returns list of dicts with keys: time, ls_raw, long_conc_std.
    """
    rows = []
    chunk_size = 50
    cur_start = start_dt

    while cur_start <= date.today():
        remaining = (date.today() - cur_start).days + 1
        cur_days = min(chunk_size, remaining)
        if cur_days <= 0:
            break

        try:
            df_long = api.query_symbol_ranking(
                contract, ranking_type='LONG',
                days=cur_days, start_dt=cur_start,
            )
            df_short = api.query_symbol_ranking(
                contract, ranking_type='SHORT',
                days=cur_days, start_dt=cur_start,
            )

            if len(df_long) > 0 and len(df_short) > 0:
                long_daily = df_long.groupby('datetime')['long_oi'].sum()
                short_daily = df_short.groupby('datetime')['short_oi'].sum()
                merged = pd.DataFrame({
                    'sum_long': long_daily,
                    'sum_short': short_daily,
                }).dropna()

                # Compute long concentration std per day:
                # std(oi_long_i / total_oi_long) across top-20 members
                long_conc = {}
                for dt_str, day_data in df_long.groupby('datetime'):
                    oi_vals = day_data['long_oi'].dropna()
                    total = oi_vals.sum()
                    if total > 0 and len(oi_vals) > 1:
                        shares = oi_vals / total
                        long_conc[dt_str] = float(shares.std())
                    else:
                        long_conc[dt_str] = 0.0

                # Compute short weighted HHI per day:
                # WeightedS = Σ(short_i² / total_short)
                short_weighted = {}
                for dt_str, day_data in df_short.groupby('datetime'):
                    oi_vals = day_data['short_oi'].dropna()
                    total = oi_vals.sum()
                    if total > 0 and len(oi_vals) > 1:
                        short_weighted[dt_str] = float((oi_vals ** 2).sum() / total)
                    else:
                        short_weighted[dt_str] = 0.0

                for dt_str, row in merged.iterrows():
                    rows.append({
                        'time': pd.to_datetime(str(dt_str)),
                        'ls_raw': float(row['sum_long'] - row['sum_short']),
                        'long_conc_std': long_conc.get(dt_str, 0.0),
                        'short_weighted': short_weighted.get(dt_str, 0.0),
                    })
        except Exception:
            pass

        cur_start += timedelta(days=cur_days)

    return rows


def _connect_api(max_retries=3, wait_secs=30):
    """Create TqApi with retry logic for non-trading hours."""
    from tqsdk import TqApi, TqAuth
    user = os.getenv("TIANQIN_USER")
    password = os.getenv("TIANQIN_PASSWORD")
    if not user or not password:
        raise ValueError("TIANQIN_USER and TIANQIN_PASSWORD required in .env")

    import time
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Connecting to TianQin (attempt {attempt}/{max_retries})...")
            api = TqApi(auth=TqAuth(user, password))
            print("Connected.")
            return api
        except Exception as e:
            print(f"Connection failed: {e}")
            if attempt < max_retries:
                print(f"Retrying in {wait_secs}s...")
                time.sleep(wait_secs)
    raise ConnectionError(
        "Cannot connect to TianQin. This often happens during the "
        "maintenance window (~02:30-08:45 CST). Try again during "
        "trading hours (09:00-15:00 or 21:00-02:30)."
    )


def fetch_member_positions(days=400):
    data_dir = get_data_dir()
    cache_path = os.path.join(data_dir, "member_pos_cache.parquet")

    # Load existing cache
    existing_df = None
    latest_cached = None
    if os.path.exists(cache_path):
        existing_df = pd.read_parquet(cache_path)
        latest_cached = pd.to_datetime(existing_df['time']).max().date()
        print(f"Existing cache: {len(existing_df)} rows, latest={latest_cached}")

    api = _connect_api()
    try:
        # Discover products
        products = _discover_products(api)
        print(f"Total products: {len(products)}")

        # Date range
        if latest_cached:
            start_dt = latest_cached + timedelta(days=1)
            if start_dt > date.today():
                print("Cache is up to date.")
                return cache_path
            fetch_days = (date.today() - start_dt).days + 1
            print(f"Incremental: {start_dt} → today ({fetch_days} days)")
        else:
            fetch_days = days
            start_dt = date.today() - timedelta(days=fetch_days)
            print(f"Full fetch: {start_dt} → today ({fetch_days} days)")

        # Fetch each product
        results = []
        total = len(products)

        for idx, prod in enumerate(products):
            address = f"{prod['product_id']}.{prod['exchange']}"
            contract = prod['main_contract']

            rows = _fetch_product_ls(api, contract, start_dt, fetch_days)

            if rows:
                for r in rows:
                    r['address'] = address
                results.extend(rows)
                print(f"  [{idx+1}/{total}] {address}: {len(rows)} rows")
            else:
                print(f"  [{idx+1}/{total}] {address}: no data")

        if not results:
            print("No new data fetched.")
            return cache_path

        new_df = pd.DataFrame(results)
        # Deduplicate within new data (multiple contracts may cover same dates)
        new_df = new_df.drop_duplicates(subset=['time', 'address'], keep='last')
        print(f"\nNew: {len(new_df)} rows, "
              f"{new_df['address'].nunique()} products, "
              f"{new_df['time'].nunique()} dates")

        # Merge
        if existing_df is not None:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['time', 'address'], keep='last'
            ).sort_values(['address', 'time']).reset_index(drop=True)
        else:
            combined = new_df.sort_values(['address', 'time']).reset_index(drop=True)

        # Save ls_raw cache (original)
        ls_cols = ['time', 'address', 'ls_raw']
        combined_ls = combined[ls_cols].copy()
        combined_ls.to_parquet(cache_path, index=False)
        print(f"\nSaved: {cache_path} ({len(combined_ls)} rows)")

        # Save long_conc_std cache (new)
        if 'long_conc_std' in combined.columns:
            conc_path = os.path.join(data_dir, "long_conc_cache.parquet")
            conc_cols = ['time', 'address', 'long_conc_std']
            combined_conc = combined[conc_cols].copy()
            combined_conc.to_parquet(conc_path, index=False)
            print(f"Saved: {conc_path} ({len(combined_conc)} rows)")

        # Save short_weighted cache (空头主力加权持仓)
        if 'short_weighted' in combined.columns:
            sw_path = os.path.join(data_dir, "short_weighted_cache.parquet")
            sw_cols = ['time', 'address', 'short_weighted']
            combined_sw = combined[sw_cols].copy()
            combined_sw.to_parquet(sw_path, index=False)
            print(f"Saved: {sw_path} ({len(combined_sw)} rows)")

        return cache_path

    finally:
        api.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch member position rankings")
    parser.add_argument("--days", type=int, default=400,
                        help="Days of history to fetch (default: 400)")
    args = parser.parse_args()
    fetch_member_positions(days=args.days)
