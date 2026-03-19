import asyncio
from datetime import datetime, timedelta
from loguru import logger
from ..config import Config
from .base import DataProvider

try:
    import duckdb
except ImportError:
    duckdb = None
    logger.warning("duckdb not installed. Run: pip install duckdb")


# timeframe 字符串 → 分钟数
TIMEFRAME_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}


class DuckDBProvider(DataProvider):
    """本地 DuckDB 商品期货数据提供者，支持主力合约 + 多周期聚合"""

    def __init__(self):
        if duckdb is None:
            raise ImportError("duckdb is required for DuckDBProvider. Run: pip install duckdb")

        self.db_path = Config.DUCKDB_PATH
        self.products_filter = Config.DUCKDB_PRODUCTS
        self.exchanges_filter = Config.DUCKDB_EXCHANGES
        self.con = duckdb.connect(self.db_path, read_only=True)
        logger.info(f"DuckDBProvider initialized (path={self.db_path})")

    async def get_trending_tokens(self, limit: int = 500):
        """返回品种列表，按数据量排序"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_products, limit)

    def _get_products(self, limit: int):
        where_clauses = []
        params = []

        if self.products_filter:
            placeholders = ", ".join(["?"] * len(self.products_filter))
            where_clauses.append(f"product_id IN ({placeholders})")
            params.extend(self.products_filter)

        if self.exchanges_filter:
            placeholders = ", ".join(["?"] * len(self.exchanges_filter))
            where_clauses.append(f"exchange IN ({placeholders})")
            params.extend(self.exchanges_filter)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT product_id, exchange, COUNT(*) as cnt
            FROM kline_1min
            {where_sql}
            GROUP BY product_id, exchange
            ORDER BY cnt DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self.con.execute(sql, params).fetchall()

        results = []
        for product_id, exchange, cnt in rows:
            results.append({
                'address': f"{exchange}.{product_id}",
                'symbol': product_id,
                'name': f"{product_id} ({exchange})",
                'decimals': 2,
                'liquidity': 0,
                'fdv': 0,
            })
        return results

    async def get_token_history(self, session, address: str, days: int = 0, timeframe: str = '1d'):
        """
        获取主力合约连续K线数据
        address 格式: "EXCHANGE.PRODUCT" (如 "SHFE.al")
        返回: [(time, address, o, h, l, c, vol, close_oi, 0.0, 'duckdb'), ...]
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_main_contract_data, address, timeframe)

    def _fetch_main_contract_data(self, address: str, timeframe: str):
        parts = address.split(".", 1)
        if len(parts) != 2:
            logger.error(f"Invalid address format: {address}, expected 'EXCHANGE.PRODUCT'")
            return []

        exchange, product_id = parts
        minutes = TIMEFRAME_MINUTES.get(timeframe, 1440)

        try:
            # Step 1: 找每天的主力合约（按日均 close_oi 最大选取）
            main_contracts = self.con.execute("""
                WITH daily_oi AS (
                    SELECT symbol,
                           DATE_TRUNC('day', datetime) as day,
                           AVG(close_oi) as avg_oi
                    FROM kline_1min
                    WHERE product_id = ? AND exchange = ?
                    GROUP BY symbol, day
                ),
                ranked AS (
                    SELECT day, symbol,
                           ROW_NUMBER() OVER (PARTITION BY day ORDER BY avg_oi DESC) as rn
                    FROM daily_oi
                )
                SELECT day, symbol FROM ranked WHERE rn = 1
                ORDER BY day
            """, [product_id, exchange]).fetchall()

            if not main_contracts:
                logger.warning(f"No data for {address}")
                return []

            # Step 2: 取每天主力合约的 1min 数据，聚合为目标周期
            all_records = []
            for day, symbol in main_contracts:
                day_start = day
                day_end = day + timedelta(days=1)

                if minutes >= 1440:
                    # 日线：直接按天聚合
                    rows = self.con.execute("""
                        SELECT
                            DATE_TRUNC('day', datetime) as time,
                            FIRST(open ORDER BY datetime) as open,
                            MAX(high) as high,
                            MIN(low) as low,
                            LAST(close ORDER BY datetime) as close,
                            SUM(volume) as volume,
                            LAST(close_oi ORDER BY datetime) as close_oi
                        FROM kline_1min
                        WHERE symbol = ? AND datetime >= ? AND datetime < ?
                        GROUP BY DATE_TRUNC('day', datetime)
                        ORDER BY time
                    """, [symbol, day_start, day_end]).fetchall()
                else:
                    # 分钟级聚合
                    rows = self.con.execute(f"""
                        SELECT
                            TIME_BUCKET(INTERVAL '{minutes} minutes', datetime) as time,
                            FIRST(open ORDER BY datetime) as open,
                            MAX(high) as high,
                            MIN(low) as low,
                            LAST(close ORDER BY datetime) as close,
                            SUM(volume) as volume,
                            LAST(close_oi ORDER BY datetime) as close_oi
                        FROM kline_1min
                        WHERE symbol = ? AND datetime >= ? AND datetime < ?
                        GROUP BY TIME_BUCKET(INTERVAL '{minutes} minutes', datetime)
                        ORDER BY time
                    """, [symbol, day_start, day_end]).fetchall()

                for row in rows:
                    time_val, o, h, l, c, vol, oi = row
                    if isinstance(time_val, str):
                        time_val = datetime.fromisoformat(time_val)
                    all_records.append((
                        time_val,       # time
                        address,        # address
                        float(o),       # open
                        float(h),       # high
                        float(l),       # low
                        float(c),       # close
                        float(vol),     # volume
                        float(vol) * float(c),  # liquidity = 成交金额
                        0.0,            # fdv (期货无此概念)
                        'duckdb',       # source
                    ))

            logger.info(f"{address} [{timeframe}]: {len(all_records)} candles (main contracts: {len(main_contracts)} days)")
            return all_records

        except Exception as e:
            logger.error(f"DuckDB fetch error for {address} (tf={timeframe}): {e}")
            return []

    async def close(self):
        """关闭 DuckDB 连接"""
        try:
            self.con.close()
            logger.info("DuckDB connection closed.")
        except Exception as e:
            logger.warning(f"DuckDB close error: {e}")
