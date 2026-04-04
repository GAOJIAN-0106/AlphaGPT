"""
Incremental DuckDB update using get_kline_serial (works with free TqSDK account).

Fetches 1-min kline data for all active products from the last known date
in DuckDB to today, then appends to the existing database.

Usage:
    python scripts/update_duckdb.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DUCKDB_PATH = os.getenv("DUCKDB_PATH",
                         os.path.expanduser("~/quant/tick_data/kline_1min.duckdb"))


def update_duckdb():
    import duckdb
    from tqsdk import TqApi, TqAuth

    user = os.getenv("TIANQIN_USER")
    password = os.getenv("TIANQIN_PASSWORD")

    # Step 1: Find the latest date in DuckDB
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    last_dt = con.execute("SELECT MAX(datetime) FROM kline_1min").fetchone()[0]
    con.close()
    print(f"DuckDB last datetime: {last_dt}")

    # How many 1-min bars to fetch per contract
    # From last_dt to now, ~240 bars/day × trading days
    from datetime import timedelta
    days_gap = (datetime.now() - last_dt).days + 1
    bars_needed = min(days_gap * 280, 10000)  # 280 bars/day (night + day session)
    print(f"Gap: {days_gap} days, requesting {bars_needed} bars per contract")

    # Step 2: Connect to TianQin and discover all products
    api = TqApi(auth=TqAuth(user, password))
    try:
        exchanges = os.getenv("TIANQIN_FUTURES_EXCHANGES",
                              "SHFE,DCE,CZCE,INE,GFEX").split(",")
        products = []
        for exch in [e.strip() for e in exchanges]:
            try:
                conts = api.query_cont_quotes(exchange_id=exch)
                for sym in conts:
                    products.append(sym)  # e.g. "SHFE.rb2605"
                print(f"  {exch}: {len([s for s in products if s.startswith(exch)])} products")
            except Exception as e:
                print(f"  {exch}: error - {e}")

        # Also get secondary contracts for term structure
        all_symbols = set()
        for sym in products:
            parts = sym.split(".")
            if len(parts) == 2:
                exch = parts[0]
                m = re.match(r'^([a-zA-Z]+)', parts[1])
                if m:
                    pid = m.group(1)
                    # Get all non-expired contracts for this product
                    try:
                        active = api.query_quotes(
                            ins_class="FUTURE", exchange_id=exch,
                            product_id=pid, expired=False,
                        )
                        for s in active:
                            all_symbols.add(s)
                    except:
                        all_symbols.add(sym)

        print(f"\nTotal symbols to fetch: {len(all_symbols)}")

        # Step 3: Fetch kline data for each symbol
        all_rows = []
        total = len(all_symbols)
        for idx, symbol in enumerate(sorted(all_symbols)):
            try:
                klines = api.get_kline_serial(symbol, 60, data_length=bars_needed)
                if klines is None or len(klines) == 0:
                    continue

                df = klines.copy()
                # Filter to only new data (after last_dt)
                df['dt'] = pd.to_datetime(df['datetime'])
                df = df[df['dt'] > last_dt]

                if len(df) == 0:
                    continue

                # Parse exchange and product_id from symbol
                parts = symbol.split(".")
                exchange = parts[0]
                contract = parts[1]
                product_id = re.match(r'^([a-zA-Z]+)', contract).group(1)

                for _, row in df.iterrows():
                    all_rows.append({
                        'exchange': exchange,
                        'product_id': product_id,
                        'symbol': symbol,
                        'datetime': row['dt'],
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume']),
                        'open_oi': int(row['open_oi']),
                        'close_oi': int(row['close_oi']),
                    })

                if (idx + 1) % 20 == 0 or idx == total - 1:
                    print(f"  [{idx+1}/{total}] {len(all_rows):,} new rows so far")

                # Flush to DuckDB every 100 symbols to avoid OOM
                if len(all_rows) >= 500000:
                    _flush_to_duckdb(all_rows)
                    all_rows.clear()

            except Exception as e:
                if "不支持" not in str(e):
                    print(f"  [{idx+1}/{total}] {symbol}: {e}")

    finally:
        api.close()

    # Flush remaining rows
    if all_rows:
        _flush_to_duckdb(all_rows)

    # Final check
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    new_count = con.execute("SELECT COUNT(*) FROM kline_1min").fetchone()[0]
    new_max = con.execute("SELECT MAX(datetime) FROM kline_1min").fetchone()[0]
    con.close()
    print(f"\nDuckDB updated: {new_count:,} total rows, latest={new_max}")


def _flush_to_duckdb(rows):
    """Insert a batch of rows into DuckDB."""
    import duckdb
    batch_df = pd.DataFrame(rows)
    print(f"    Flushing {len(batch_df):,} rows to DuckDB...")
    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.execute("""
            INSERT INTO kline_1min
            SELECT exchange, product_id, symbol, datetime,
                   open, high, low, close, volume, open_oi, close_oi
            FROM batch_df
        """)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    print(f"    Flushed OK.")


if __name__ == "__main__":
    update_duckdb()
