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
        choices=["solana", "astock", "tianqin", "duckdb"],
        default=None,
        help="Data source mode: solana, astock, tianqin, or duckdb (default: from .env DATA_SOURCE_MODE)"
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=["hs300", "zz500", "zz1000", "sz50", "all"],
        default=None,
        help="A股 stock pool type (default: from .env ASTOCK_POOL_TYPE)"
    )
    parser.add_argument(
        "--asset-type",
        type=str,
        choices=["stock", "futures"],
        default=None,
        help="TianQin asset type: stock or futures (default: from .env TIANQIN_ASSET_TYPE)"
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default=None,
        help="Comma-separated timeframes, e.g. '1d,5m,15m' (default: from .env)"
    )
    parser.add_argument(
        "--products",
        type=str,
        default=None,
        help="DuckDB mode: comma-separated product filter, e.g. 'al,cu,ag'"
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

    # 天勤 CLI 参数覆盖
    if args.asset_type:
        Config.TIANQIN_ASSET_TYPE = args.asset_type
    if args.timeframes:
        tfs = [tf.strip() for tf in args.timeframes.split(",")]
        Config.TIANQIN_TIMEFRAMES = tfs
        Config.DUCKDB_TIMEFRAMES = tfs

    # DuckDB CLI 参数覆盖
    if args.products:
        Config.DUCKDB_PRODUCTS = [p.strip() for p in args.products.split(",")]

    # 验证凭证
    if mode == DataSourceMode.DUCKDB:
        import os
        if not os.path.exists(Config.DUCKDB_PATH):
            logger.error(f"DuckDB file not found: {Config.DUCKDB_PATH}")
            return
        logger.info(f"Using DuckDB mode (path={Config.DUCKDB_PATH}, products={Config.DUCKDB_PRODUCTS or 'all'}, timeframes={Config.DUCKDB_TIMEFRAMES})")
    elif mode == DataSourceMode.TIANQIN:
        if not Config.TIANQIN_USER:
            logger.error("TIANQIN_USER is missing in .env")
            return
        logger.info(f"Using TianQin mode (asset={Config.TIANQIN_ASSET_TYPE}, timeframes={Config.TIANQIN_TIMEFRAMES})")
    elif mode == DataSourceMode.ASTOCK:
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
