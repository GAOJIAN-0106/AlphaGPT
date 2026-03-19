import torch
import os
import math
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Asset class profiles: trading cost / liquidity defaults
# ---------------------------------------------------------------------------
ASSET_PROFILES = {
    'crypto':  {'base_fee': 0.005,  'min_liq': 5000.0,       'trade_size': 1000.0},
    'astock':  {'base_fee': 0.0015, 'min_liq': 10_000_000.0, 'trade_size': 100_000.0},
    'futures': {'base_fee': 0.0003, 'min_liq': 5_000_000.0,   'trade_size': 100_000.0},
}

# Timeframe → bars per trading day (Chinese futures/A-stock: ~4h session = 240 1min bars)
TIMEFRAME_BARS_PER_DAY = {
    '1m': 240, '5m': 48, '15m': 16, '30m': 8, '1h': 4, '4h': 1, '1d': 1,
}


class TimeframeProfile:
    """Derives all timeframe- and asset-dependent parameters from (timeframe, asset_class)."""

    def __init__(self, timeframe='1d', asset_class='astock'):
        self.timeframe = timeframe
        self.asset_class = asset_class
        self.bars_per_day = TIMEFRAME_BARS_PER_DAY.get(timeframe, 1)

        # Annualization: sqrt(trading_days * bars_per_day)
        self.annualization = math.sqrt(252 * self.bars_per_day)

        # Return clamp per bar: scale from daily ±20% by sqrt(bar_fraction)
        if self.bars_per_day > 1:
            self.ret_clamp = round(0.2 * math.sqrt(1.0 / self.bars_per_day), 4)
        else:
            self.ret_clamp = 0.2

        # Asset class — scale liquidity and trade size by bars_per_day
        # (e.g. 1h bar has ~1/4 of daily liquidity for 4h trading sessions)
        ap = ASSET_PROFILES.get(asset_class, ASSET_PROFILES['astock'])
        self.base_fee = ap['base_fee']
        self.min_liq = ap['min_liq'] / self.bars_per_day
        self.trade_size = ap['trade_size'] / self.bars_per_day

        # Turnover targets: distribute daily budget across bars
        self.target_turnover = 0.05 / self.bars_per_day
        self.lazy_threshold = 0.01 / self.bars_per_day


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 128
    TRAIN_STEPS = 3000
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    INPUT_DIM = 12
    TIMEFRAME = os.getenv("MODEL_TIMEFRAME", "1d")

    # Auto-detect asset class from DATA_SOURCE_MODE, overridable via MODEL_ASSET_CLASS
    _dsm = os.getenv("DATA_SOURCE_MODE", "solana")
    _asset_map = {'solana': 'crypto', 'astock': 'astock', 'tianqin': 'futures', 'duckdb': 'futures'}
    ASSET_CLASS = os.getenv("MODEL_ASSET_CLASS", _asset_map.get(_dsm, 'crypto'))

    @staticmethod
    def get_feature_dim():
        # Intraday microstructure features (18-dim) only available for daily futures
        # Term-structure factors extend to 21-dim when dual-contract data is available
        if ModelConfig.ASSET_CLASS == 'futures' and ModelConfig.TIMEFRAME == '1d':
            if os.getenv('ENABLE_TERM_STRUCTURE', '1') == '1':
                return 28  # IC-screened feature subset
            return 18  # DuckDBFeatureEngineer only
        return 12      # FeatureEngineer

    @staticmethod
    def get_ops_config():
        from .ops import OPS_CONFIG, OPS_CONFIG_EXTENDED
        # Extended ops (with cross-sectional ops) only for daily futures
        if ModelConfig.ASSET_CLASS == 'futures' and ModelConfig.TIMEFRAME == '1d':
            return OPS_CONFIG_EXTENDED
        return OPS_CONFIG

    @staticmethod
    def get_max_formula_len():
        # Keep formula length at 12 for all asset classes.
        # Longer formulas (16) encourage degenerate padding with GATE cascades.
        return 12