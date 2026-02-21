import argparse
import asyncio
from loguru import logger
from .data_manager import DataManager
from .config import Config, DataSourceMode


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaGPT Data Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["solana", "astock"],
        default=None,
        help="Data source mode: solana or astock (default: from .env DATA_SOURCE_MODE)"
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=["hs300", "zz500", "zz1000", "sz50", "all"],
        default=None,
        help="A股 stock pool type (default: from .env ASTOCK_POOL_TYPE)"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # 确定数据源模式
    if args.mode:
        mode = DataSourceMode(args.mode)
    else:
        mode = Config.DATA_SOURCE_MODE

    # 如果指定了股票池，更新配置
    if args.pool:
        Config.ASTOCK_POOL_TYPE = args.pool

    # 验证 API Key
    if mode == DataSourceMode.ASTOCK:
        if not Config.TUSHARE_TOKEN:
            logger.error("TUSHARE_TOKEN is missing in .env")
            return
        logger.info(f"Using A股 mode with pool: {Config.ASTOCK_POOL_TYPE}")
    else:
        if not Config.BIRDEYE_API_KEY:
            logger.error("BIRDEYE_API_KEY is missing in .env")
            return
        logger.info("Using Solana mode with Birdeye API")

    manager = DataManager(mode=mode)

    try:
        await manager.initialize()
        await manager.pipeline_sync_daily()
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
