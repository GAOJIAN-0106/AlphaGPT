import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class DataSourceMode(Enum):
    SOLANA = "solana"
    ASTOCK = "astock"


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