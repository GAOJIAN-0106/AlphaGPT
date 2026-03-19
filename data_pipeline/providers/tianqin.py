import asyncio
from datetime import datetime
from loguru import logger
from ..config import Config, TIMEFRAME_DURATION_MAP
from .base import DataProvider

try:
    from tqsdk import TqApi, TqAuth
except ImportError:
    TqApi = None
    TqAuth = None
    logger.warning("tqsdk not installed. Run: pip install tqsdk")


class TianQinProvider(DataProvider):
    """天勤量化数据提供者，支持期货和A股多周期数据"""

    def __init__(self):
        if TqApi is None:
            raise ImportError("tqsdk is required for TianQinProvider")

        user = Config.TIANQIN_USER
        password = Config.TIANQIN_PASSWORD
        if not user or not password:
            raise ValueError("TIANQIN_USER and TIANQIN_PASSWORD are required in .env")

        self.api = TqApi(auth=TqAuth(user, password))
        self.asset_type = Config.TIANQIN_ASSET_TYPE
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY)
        logger.info(f"TianQinProvider initialized (asset_type={self.asset_type})")

    async def get_trending_tokens(self, limit: int = 500):
        """获取合约/股票列表"""
        loop = asyncio.get_event_loop()
        try:
            if self.asset_type == "futures":
                result = await loop.run_in_executor(None, self._get_futures_instruments, limit)
            else:
                result = await loop.run_in_executor(None, self._get_stock_instruments, limit)
            logger.info(f"TianQin fetched {len(result)} instruments (type={self.asset_type})")
            return result[:limit]
        except Exception as e:
            logger.error(f"TianQin get_trending_tokens error: {e}")
            return []

    def _get_futures_instruments(self, limit: int):
        """获取期货主力连续合约列表"""
        results = []
        for exchange in Config.TIANQIN_FUTURES_EXCHANGES:
            exchange = exchange.strip()
            try:
                # 查询该交易所所有主力连续合约
                quotes = self.api.query_cont_quotes(exchange=exchange)
                for symbol in quotes:
                    if len(results) >= limit:
                        break
                    # symbol like "KQ.m@SHFE.rb"
                    results.append({
                        'address': symbol,
                        'symbol': symbol.split('.')[-1] if '.' in symbol else symbol,
                        'name': symbol,
                        'decimals': 2,
                        'liquidity': 0,
                        'fdv': 0,
                    })
            except Exception as e:
                logger.warning(f"TianQin query_cont_quotes error for {exchange}: {e}")
        return results

    def _get_stock_instruments(self, limit: int):
        """获取A股股票列表"""
        results = []
        for market in Config.TIANQIN_STOCK_POOL.split(","):
            market = market.strip()
            try:
                stock_list = self.api.query_quotes(ins_class="STOCK", exchange_id=market)
                for symbol in stock_list:
                    if len(results) >= limit:
                        break
                    results.append({
                        'address': symbol,
                        'symbol': symbol.split('.')[-1] if '.' in symbol else symbol,
                        'name': symbol,
                        'decimals': 2,
                        'liquidity': 0,
                        'fdv': 0,
                    })
            except Exception as e:
                logger.warning(f"TianQin query_quotes error for {market}: {e}")
        return results

    async def get_token_history(self, session, address: str, days: int = 0, timeframe: str = '1d'):
        """
        获取K线历史数据
        返回: [(time, address, o, h, l, c, vol, liquidity, fdv, source), ...]
        """
        loop = asyncio.get_event_loop()
        duration = TIMEFRAME_DURATION_MAP.get(timeframe, 86400)
        data_length = Config.TIANQIN_DATA_LENGTH

        async with self.semaphore:
            try:
                df = await loop.run_in_executor(
                    None,
                    self._fetch_kline_data,
                    address,
                    duration,
                    data_length,
                )

                if df is None or df.empty:
                    return []

                # 过滤起始日期
                start_dt = datetime.strptime(Config.TIANQIN_START_DATE, "%Y-%m-%d")

                formatted = []
                for _, row in df.iterrows():
                    dt = row['datetime'].to_pydatetime()
                    if dt < start_dt:
                        continue
                    formatted.append((
                        dt,                         # time
                        address,                    # address
                        float(row['open']),          # open
                        float(row['high']),          # high
                        float(row['low']),           # low
                        float(row['close']),         # close
                        float(row['volume']),        # volume
                        0.0,                         # liquidity
                        0.0,                         # fdv
                        'tianqin',                   # source
                    ))

                return formatted

            except Exception as e:
                logger.error(f"TianQin fetch error for {address} (tf={timeframe}): {e}")
                return []

    def _fetch_kline_data(self, symbol: str, duration_seconds: int, data_length: int):
        """同步获取K线数据"""
        try:
            klines = self.api.get_kline_serial(symbol, duration_seconds, data_length=data_length)
            return klines.copy()
        except Exception as e:
            logger.error(f"TianQin get_kline_serial error for {symbol}: {e}")
            return None

    async def close(self):
        """关闭天勤API连接"""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self.api.close)
            logger.info("TianQin API connection closed.")
        except Exception as e:
            logger.warning(f"TianQin close error: {e}")
