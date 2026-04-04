"""
拉取 EDB 仓单数据并缓存到 parquet。

数据来源: 天勤 api.query_edb_data()
映射关系: model_core/edb_config.py → WAREHOUSE_EDB_MAP

Usage:
    python scripts/fetch_edb_warehouse.py
    python scripts/fetch_edb_warehouse.py --days 1000
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_data_dir():
    db_path = os.getenv("DUCKDB_PATH",
                        os.path.expanduser("~/quant/tick_data/kline_1min.duckdb"))
    return os.path.dirname(db_path)


def fetch_edb_warehouse(days=1000):
    from tqsdk import TqApi, TqAuth
    from model_core.edb_config import WAREHOUSE_EDB_MAP

    user = os.getenv("TIANQIN_USER")
    password = os.getenv("TIANQIN_PASSWORD")
    if not user or not password:
        raise ValueError("TIANQIN_USER and TIANQIN_PASSWORD required in .env")

    data_dir = get_data_dir()
    cache_path = os.path.join(data_dir, "warehouse_cache.parquet")

    # Collect all unique EDB IDs
    id_to_address = {}
    all_ids = []
    for address, ids in WAREHOUSE_EDB_MAP.items():
        edb_id = ids[0]  # use primary ID
        id_to_address[edb_id] = address
        all_ids.append(edb_id)

    print(f"Products mapped: {len(all_ids)}")

    api = TqApi(auth=TqAuth(user, password))
    try:
        # query_edb_data supports max 100 IDs per call
        # Fetch in batches
        all_rows = []
        batch_size = 80
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i+batch_size]
            print(f"  Fetching EDB batch {i//batch_size+1}: "
                  f"{len(batch_ids)} indicators, {days} days...")

            df = api.query_edb_data(
                ids=batch_ids, n=days,
                align='day', fill='ffill',
            )

            if len(df) == 0:
                print(f"    No data returned")
                continue

            # df: index=date (str), columns=EDB IDs, values=warehouse quantities
            for edb_id in batch_ids:
                if edb_id not in df.columns:
                    continue
                col = df[edb_id].dropna()
                if col.empty:
                    continue

                address = id_to_address[edb_id]
                for date_str, val in col.items():
                    all_rows.append({
                        'time': pd.to_datetime(date_str),
                        'address': address,
                        'warehouse': float(val),
                    })

            print(f"    Got data for {sum(1 for eid in batch_ids if eid in df.columns and df[eid].notna().any())} products")

        if not all_rows:
            print("No data fetched.")
            return cache_path

        result = pd.DataFrame(all_rows)
        # Drop weekends/holidays where ffill creates duplicates
        result = result.drop_duplicates(subset=['time', 'address'], keep='last')
        result = result.sort_values(['address', 'time']).reset_index(drop=True)

        result.to_parquet(cache_path, index=False)
        print(f"\nSaved: {cache_path}")
        print(f"  {len(result)} rows, {result['address'].nunique()} products, "
              f"{result['time'].nunique()} dates")
        print(f"  Range: {result['time'].min().date()} → {result['time'].max().date()}")
        return cache_path

    finally:
        api.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch EDB warehouse receipt data")
    parser.add_argument("--days", type=int, default=1000,
                        help="Days of history to fetch (default: 1000)")
    args = parser.parse_args()
    fetch_edb_warehouse(days=args.days)
