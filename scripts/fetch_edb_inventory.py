"""
拉取 EDB 期货库存（社会库存）数据并缓存到 parquet。

数据来源: 天勤 api.query_edb_data() — 周频，ffill 到日频
映射关系: model_core/edb_config.py → INVENTORY_EDB_MAP

Usage:
    python scripts/fetch_edb_inventory.py
    python scripts/fetch_edb_inventory.py --days 1000
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


def fetch_edb_inventory(days=1000):
    from tqsdk import TqApi, TqAuth
    from model_core.edb_config import INVENTORY_EDB_MAP

    user = os.getenv("TIANQIN_USER")
    password = os.getenv("TIANQIN_PASSWORD")
    if not user or not password:
        raise ValueError("TIANQIN_USER and TIANQIN_PASSWORD required in .env")

    data_dir = get_data_dir()
    cache_path = os.path.join(data_dir, "inventory_cache.parquet")

    id_to_address = {}
    all_ids = []
    for address, ids in INVENTORY_EDB_MAP.items():
        edb_id = ids[0]
        id_to_address[edb_id] = address
        all_ids.append(edb_id)

    print(f"Products mapped: {len(all_ids)}")

    api = TqApi(auth=TqAuth(user, password))
    try:
        print(f"Fetching {len(all_ids)} indicators, {days} days, align=day, fill=ffill...")
        df = api.query_edb_data(
            ids=all_ids, n=days,
            align='day', fill='ffill',
        )

        if len(df) == 0:
            print("No data returned.")
            return cache_path

        all_rows = []
        for edb_id in all_ids:
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
                    'inventory': float(val),
                })

        if not all_rows:
            print("No data fetched.")
            return cache_path

        result = pd.DataFrame(all_rows)
        result = result.drop_duplicates(subset=['time', 'address'], keep='last')
        result = result.sort_values(['address', 'time']).reset_index(drop=True)

        result.to_parquet(cache_path, index=False)
        print(f"Saved: {cache_path}")
        print(f"  {len(result)} rows, {result['address'].nunique()} products, "
              f"{result['time'].nunique()} dates")
        print(f"  Range: {result['time'].min().date()} → {result['time'].max().date()}")
        return cache_path

    finally:
        api.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch EDB inventory data")
    parser.add_argument("--days", type=int, default=1000,
                        help="Days of history (default: 1000)")
    args = parser.parse_args()
    fetch_edb_inventory(days=args.days)
