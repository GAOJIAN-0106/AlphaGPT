import asyncio
import aiohttp
from loguru import logger
from .config import Config, DataSourceMode
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider
from .providers.tushare import TushareProvider


class DataManager:
    def __init__(self, mode: DataSourceMode = None):
        self.db = DBManager()
        self.mode = mode or Config.DATA_SOURCE_MODE
        self.provider = self._create_provider()
        self.chain = "astock" if self.mode == DataSourceMode.ASTOCK else Config.CHAIN

    def _create_provider(self):
        """工厂方法：根据模式创建数据提供者"""
        if self.mode == DataSourceMode.ASTOCK:
            return TushareProvider()
        else:
            return BirdeyeProvider()
        
    async def initialize(self):
        await self.db.connect()
        await self.db.init_schema()

    async def close(self):
        await self.db.close()

    async def pipeline_sync_daily(self):
        logger.info(f"Step 1: Discovering tokens (mode={self.mode.value})...")
        limit = 500 if self.mode == DataSourceMode.ASTOCK or Config.BIRDEYE_IS_PAID else 100
        candidates = await self.provider.get_trending_tokens(limit=limit)
        
        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        for t in candidates:
            fdv = t.get('fdv', 0)

            if self.mode == DataSourceMode.ASTOCK:
                # A股：按市值过滤，已在 provider 中完成初步筛选
                if fdv < Config.ASTOCK_MIN_MARKET_CAP:
                    continue
            else:
                # Solana：按流动性和 FDV 过滤
                liq = t.get('liquidity', 0)
                if liq < Config.MIN_LIQUIDITY_USD:
                    continue
                if fdv < Config.MIN_FDV:
                    continue
                if fdv > Config.MAX_FDV:
                    continue  # 剔除像 WIF/BONK 这种巨无霸，专注于早期高成长

            selected_tokens.append(t)
            
        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")
        
        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return

        db_tokens = [(t['address'], t['symbol'], t['name'], t['decimals'], self.chain) for t in selected_tokens]
        await self.db.upsert_tokens(db_tokens)

        logger.info(f"Step 4: Fetching OHLCV for {len(selected_tokens)} tokens...")

        # 根据数据源选择不同的 session 处理方式
        if self.mode == DataSourceMode.ASTOCK:
            # Tushare 不需要 HTTP session
            session = None
        else:
            session = aiohttp.ClientSession(headers=self.provider.headers)

        try:
            tasks = []
            for t in selected_tokens:
                tasks.append(self.provider.get_token_history(session, t['address']))
            
            batch_size = 20
            total_candles = 0
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                results = await asyncio.gather(*batch)
                
                records = [item for sublist in results if sublist for item in sublist]
                
                # 批量写入
                await self.db.batch_insert_ohlcv(records)
                total_candles += len(records)
                logger.info(f"Processed batch {i}/{len(tasks)}. Inserted {len(records)} candles.")

            logger.success(f"Pipeline complete. Total candles stored: {total_candles}")
        finally:
            if session is not None:
                await session.close()