import asyncio
import aiohttp
from loguru import logger
from .config import Config, DataSourceMode
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider
from .providers.tushare import TushareProvider
from .providers.tianqin import TianQinProvider
from .providers.duckdb_provider import DuckDBProvider


class DataManager:
    def __init__(self, mode: DataSourceMode = None):
        self.db = DBManager()
        self.mode = mode or Config.DATA_SOURCE_MODE
        self.provider = self._create_provider()
        if self.mode == DataSourceMode.TIANQIN:
            self.chain = Config.TIANQIN_ASSET_TYPE
        elif self.mode == DataSourceMode.ASTOCK:
            self.chain = "astock"
        elif self.mode == DataSourceMode.DUCKDB:
            self.chain = "futures"
        else:
            self.chain = Config.CHAIN

    def _create_provider(self):
        """工厂方法：根据模式创建数据提供者"""
        if self.mode == DataSourceMode.DUCKDB:
            return DuckDBProvider()
        elif self.mode == DataSourceMode.TIANQIN:
            return TianQinProvider()
        elif self.mode == DataSourceMode.ASTOCK:
            return TushareProvider()
        else:
            return BirdeyeProvider()
        
    async def initialize(self):
        await self.db.connect()
        await self.db.init_schema()

    async def close(self):
        await self.provider.close()
        await self.db.close()

    async def pipeline_sync_daily(self):
        logger.info(f"Step 1: Discovering tokens (mode={self.mode.value})...")
        limit = 500 if self.mode in (DataSourceMode.ASTOCK, DataSourceMode.TIANQIN) or Config.BIRDEYE_IS_PAID else 100
        candidates = await self.provider.get_trending_tokens(limit=limit)

        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        for t in candidates:
            fdv = t.get('fdv', 0)

            if self.mode == DataSourceMode.ASTOCK:
                if fdv < Config.ASTOCK_MIN_MARKET_CAP:
                    continue
            elif self.mode in (DataSourceMode.TIANQIN, DataSourceMode.DUCKDB):
                # 期货模式：不按 FDV 过滤
                pass
            else:
                liq = t.get('liquidity', 0)
                if liq < Config.MIN_LIQUIDITY_USD:
                    continue
                if fdv < Config.MIN_FDV:
                    continue
                if fdv > Config.MAX_FDV:
                    continue

            selected_tokens.append(t)

        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")

        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return

        db_tokens = [(t['address'], t['symbol'], t['name'], t['decimals'], self.chain) for t in selected_tokens]
        await self.db.upsert_tokens(db_tokens)

        # 确定时间周期列表
        if self.mode == DataSourceMode.DUCKDB:
            timeframes = Config.DUCKDB_TIMEFRAMES
        elif self.mode == DataSourceMode.TIANQIN:
            timeframes = Config.TIANQIN_TIMEFRAMES
        else:
            timeframes = ["1d"]

        logger.info(f"Step 2: Fetching OHLCV for {len(selected_tokens)} tokens, timeframes={timeframes}...")

        # 根据数据源选择不同的 session 处理方式
        if self.mode in (DataSourceMode.ASTOCK, DataSourceMode.TIANQIN, DataSourceMode.DUCKDB):
            session = None
        else:
            session = aiohttp.ClientSession(headers=self.provider.headers)

        try:
            total_candles = 0

            for tf in timeframes:
                logger.info(f"Fetching timeframe={tf}...")
                tasks = []
                for t in selected_tokens:
                    tasks.append(self.provider.get_token_history(session, t['address'], timeframe=tf))

                batch_size = 20

                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i+batch_size]
                    results = await asyncio.gather(*batch)

                    # 每条10-tuple追加 timeframe 构成11-tuple
                    records = []
                    for sublist in results:
                        if not sublist:
                            continue
                        for item in sublist:
                            records.append(item + (tf,))

                    await self.db.batch_insert_ohlcv(records)
                    total_candles += len(records)
                    logger.info(f"[{tf}] Processed batch {i}/{len(tasks)}. Inserted {len(records)} candles.")

            logger.success(f"Pipeline complete. Total candles stored: {total_candles}")
        finally:
            if session is not None:
                await session.close()