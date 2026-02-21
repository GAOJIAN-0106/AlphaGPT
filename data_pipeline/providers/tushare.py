import asyncio
from datetime import datetime, timedelta
from loguru import logger
from ..config import Config
from .base import DataProvider

try:
    import tushare as ts
except ImportError:
    ts = None
    logger.warning("tushare not installed. Run: pip install tushare")


class TushareProvider(DataProvider):
    """A股数据提供者，基于 Tushare Pro API"""

    # 股票池代码映射
    POOL_CODES = {
        "hs300": "000300.SH",   # 沪深300
        "zz500": "000905.SH",   # 中证500
        "zz1000": "000852.SH",  # 中证1000
        "sz50": "000016.SH",    # 上证50
    }

    def __init__(self):
        if ts is None:
            raise ImportError("tushare is required for TushareProvider")

        self.token = Config.TUSHARE_TOKEN
        if not self.token:
            raise ValueError("TUSHARE_TOKEN is required in .env")

        # 使用代理 API 连接方式
        self.pro = ts.pro_api()
        self.pro._DataApi__token = self.token
        self.pro._DataApi__http_url = Config.TUSHARE_API_URL

        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY)
        self.pool_type = Config.ASTOCK_POOL_TYPE
        logger.info(f"TushareProvider initialized with API: {Config.TUSHARE_API_URL}")

    async def get_trending_tokens(self, limit: int = 500):
        """
        获取股票池列表
        返回: [{address, symbol, name, decimals, liquidity, fdv}, ...]
        """
        loop = asyncio.get_event_loop()

        try:
            if self.pool_type == "all":
                # 全市场按市值排序
                stocks = await loop.run_in_executor(None, self._get_all_stocks_by_market_cap, limit)
            else:
                # 指数成分股
                stocks = await loop.run_in_executor(None, self._get_index_constituents)

            logger.info(f"Tushare fetched {len(stocks)} stocks from pool '{self.pool_type}'")
            return stocks[:limit]

        except Exception as e:
            logger.error(f"Tushare get_trending_tokens error: {e}")
            return []

    def _get_index_constituents(self):
        """获取指数成分股"""
        index_code = self.POOL_CODES.get(self.pool_type)
        if not index_code:
            logger.error(f"Unknown pool type: {self.pool_type}")
            return []

        # 获取指数成分股
        df = self.pro.index_weight(index_code=index_code)
        if df is None or df.empty:
            logger.warning(f"No constituents found for index {index_code}")
            return []

        # 获取最新交易日的成分股
        latest_date = df['trade_date'].max()
        df = df[df['trade_date'] == latest_date]

        # 获取股票基本信息和市值
        codes = df['con_code'].tolist()
        return self._enrich_stock_info(codes)

    def _get_all_stocks_by_market_cap(self, limit: int):
        """获取全市场股票，按市值排序"""
        # 获取所有上市股票
        df_basic = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name'
        )

        if df_basic is None or df_basic.empty:
            return []

        # 获取最新交易日
        today = datetime.now().strftime('%Y%m%d')
        df_daily = self.pro.daily_basic(
            trade_date=today,
            fields='ts_code,total_mv,turnover_rate_f'
        )

        # 如果今天无数据，尝试获取最近交易日
        if df_daily is None or df_daily.empty:
            # 获取最近5个交易日的数据
            for i in range(1, 6):
                prev_date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
                df_daily = self.pro.daily_basic(
                    trade_date=prev_date,
                    fields='ts_code,total_mv,turnover_rate_f'
                )
                if df_daily is not None and not df_daily.empty:
                    break

        if df_daily is None or df_daily.empty:
            logger.warning("Failed to get daily basic data")
            return []

        # 合并数据
        df = df_basic.merge(df_daily, on='ts_code', how='inner')

        # 过滤市值
        min_mv = Config.ASTOCK_MIN_MARKET_CAP / 10000  # 转换为万元
        df = df[df['total_mv'] >= min_mv]

        # 按市值排序
        df = df.sort_values('total_mv', ascending=False).head(limit)

        results = []
        for _, row in df.iterrows():
            results.append({
                'address': row['ts_code'],
                'symbol': row['symbol'],
                'name': row['name'],
                'decimals': 2,  # A股价格精度
                'liquidity': 0,  # 后续在 get_token_history 中填充
                'fdv': float(row['total_mv']) * 10000 if row['total_mv'] else 0  # 万元转元
            })

        return results

    def _enrich_stock_info(self, codes: list):
        """补充股票基本信息和市值数据"""
        if not codes:
            return []

        # 获取股票基本信息
        df_basic = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name'
        )

        if df_basic is None or df_basic.empty:
            return []

        df_basic = df_basic[df_basic['ts_code'].isin(codes)]

        # 获取最新市值数据
        today = datetime.now().strftime('%Y%m%d')
        df_daily = self.pro.daily_basic(
            trade_date=today,
            fields='ts_code,total_mv'
        )

        if df_daily is None or df_daily.empty:
            for i in range(1, 6):
                prev_date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
                df_daily = self.pro.daily_basic(
                    trade_date=prev_date,
                    fields='ts_code,total_mv'
                )
                if df_daily is not None and not df_daily.empty:
                    break

        if df_daily is not None and not df_daily.empty:
            df = df_basic.merge(df_daily, on='ts_code', how='left')
        else:
            df = df_basic
            df['total_mv'] = 0

        results = []
        for _, row in df.iterrows():
            fdv = float(row['total_mv']) * 10000 if row.get('total_mv') else 0
            results.append({
                'address': row['ts_code'],
                'symbol': row['symbol'],
                'name': row['name'],
                'decimals': 2,
                'liquidity': 0,
                'fdv': fdv
            })

        return results

    async def get_token_history(self, session, address: str, days: int = Config.HISTORY_DAYS):
        """
        获取股票 OHLCV 历史数据
        返回: [(time, address, o, h, l, c, vol, liquidity, fdv, source), ...]
        """
        loop = asyncio.get_event_loop()

        async with self.semaphore:
            try:
                # 计算日期范围
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = Config.ASTOCK_START_DATE

                # 使用线程池执行同步 API 调用
                df = await loop.run_in_executor(
                    None,
                    self._fetch_daily_data,
                    address,
                    start_date,
                    end_date
                )

                if df is None or df.empty:
                    return []

                # 添加延迟避免触发限流
                await asyncio.sleep(0.1)

                formatted = []
                for _, row in df.iterrows():
                    trade_time = datetime.strptime(str(row['trade_date']), '%Y%m%d')
                    formatted.append((
                        trade_time,                                    # time
                        address,                                       # address (ts_code)
                        float(row['open']),                            # open
                        float(row['high']),                            # high
                        float(row['low']),                             # low
                        float(row['close']),                           # close
                        float(row['vol']) * 100 if row['vol'] else 0,  # volume (手转股)
                        float(row['amount']) * 1000 if row['amount'] else 0,  # liquidity (千元转元)
                        0.0,                                           # fdv (日线数据无此字段)
                        'tushare'                                      # source
                    ))

                return formatted

            except Exception as e:
                logger.error(f"Tushare fetch error for {address}: {e}")
                return []

    def _fetch_daily_data(self, ts_code: str, start_date: str, end_date: str):
        """同步获取日线数据"""
        try:
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='trade_date,open,high,low,close,vol,amount'
            )
            return df
        except Exception as e:
            logger.error(f"Tushare daily API error for {ts_code}: {e}")
            return None
