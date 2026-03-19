import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class DataSourceMode(Enum):
    SOLANA = "solana"
    ASTOCK = "astock"
    TIANQIN = "tianqin"
    DUCKDB = "duckdb"


class Config:
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "crypto_quant")
    DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    CHAIN = "solana"
    TIMEFRAME = "1m" # 也支持 15min
    MIN_LIQUIDITY_USD = 500000.0  
    MIN_FDV = 10000000.0            
    MAX_FDV = float('inf') 
    BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
    BIRDEYE_IS_PAID = True
    USE_DEXSCREENER = False
    CONCURRENCY = 20
    HISTORY_DAYS = 7

    # Data source mode
    DATA_SOURCE_MODE = DataSourceMode(os.getenv("DATA_SOURCE_MODE", "solana"))

    # A股 Tushare configurations
    TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
    TUSHARE_API_URL = os.getenv("TUSHARE_API_URL", "http://1w2b.xiximiao.com/dataapi")  # 代理 API
    ASTOCK_POOL_TYPE = os.getenv("ASTOCK_POOL_TYPE", "hs300")  # hs300/zz500/zz1000/sz50/all
    ASTOCK_START_DATE = os.getenv("ASTOCK_START_DATE", "20200101")
    ASTOCK_MIN_MARKET_CAP = float(os.getenv("ASTOCK_MIN_MARKET_CAP", "1000000000"))  # 最小市值10亿

    # 天勤量化 TianQin (tqsdk) configurations
    TIANQIN_USER = os.getenv("TIANQIN_USER", "")
    TIANQIN_PASSWORD = os.getenv("TIANQIN_PASSWORD", "")
    TIANQIN_ASSET_TYPE = os.getenv("TIANQIN_ASSET_TYPE", "futures")  # futures / stock
    TIANQIN_TIMEFRAMES = os.getenv("TIANQIN_TIMEFRAMES", "1d").split(",")  # e.g. "1d,5m,15m"
    TIANQIN_FUTURES_EXCHANGES = os.getenv("TIANQIN_FUTURES_EXCHANGES", "SHFE,DCE,CZCE,INE,CFFEX,GFEX").split(",")
    TIANQIN_STOCK_POOL = os.getenv("TIANQIN_STOCK_POOL", "SSE,SZSE")  # 上交所,深交所
    TIANQIN_DATA_LENGTH = int(os.getenv("TIANQIN_DATA_LENGTH", "8000"))  # K线条数
    TIANQIN_START_DATE = os.getenv("TIANQIN_START_DATE", "2020-01-01")

    # DuckDB 本地商品期货数据源
    DUCKDB_PATH = os.getenv("DUCKDB_PATH", os.path.expanduser("~/quant/tick_data/kline_1min.duckdb"))
    DUCKDB_PRODUCTS = [p.strip() for p in os.getenv("DUCKDB_PRODUCTS", "").split(",") if p.strip()]
    DUCKDB_EXCHANGES = [e.strip() for e in os.getenv("DUCKDB_EXCHANGES", "").split(",") if e.strip()]
    DUCKDB_TIMEFRAMES = os.getenv("DUCKDB_TIMEFRAMES", "1d").split(",")



# 时间周期到秒数的映射 (用于 tqsdk duration_seconds)
TIMEFRAME_DURATION_MAP = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "1d": 86400,
}