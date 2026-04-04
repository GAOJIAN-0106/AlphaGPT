"""
DuckDBFeatureEngineer — 18-dimensional feature space for commodity futures.

Features 0-11: identical to FeatureEngineer (daily OHLCV)
Features 12-17: intraday microstructure (aggregated from 1-min data)
    12  VWAP_DEV    — (VWAP - close) / close, institutional flow proxy
    13  OI_CHANGE   — daily open interest change, position buildup
    14  VOL_SKEW    — AM/PM volume ratio, session timing pattern
    15  VOL_CONC    — max_30min_vol / total_vol, volume burst detection
    16  SMART_MONEY — last_hour_vol / first_hour_vol, smart money activity
    17  TWAP_DEV    — (TWAP - close) / close, execution quality proxy
"""

import torch
from .factors import FeatureEngineer

FEATURES_V1_LIST = [
    'RET', 'LIQ', 'PRESSURE', 'FOMO', 'DEV', 'LOG_VOL',
    'VOL_CLUSTER', 'MOM_REV', 'REL_STR', 'HL_RANGE', 'CLOSE_POS', 'VOL_TREND',
]

FEATURES_V2_LIST = FEATURES_V1_LIST + [
    'VWAP_DEV', 'OI_CHANGE', 'VOL_SKEW', 'VOL_CONC', 'SMART_MONEY', 'TWAP_DEV',
]

# V3 full: all registry factors (before IC screening)
FEATURES_V3_FULL = FEATURES_V2_LIST + [
    'TERM_SPREAD', 'PINDYCK_VOL', 'CARRY', 'BASIS', 'XS_MOM', 'AMIHUD_ILLIQ', 'MPB', 'VP_SKEW', 'BUY_WILL', 'TS_REGRESS', 'HEDGE_PRESS', 'NEAR_SUB_SPREAD', 'OVERNIGHT_GAP', 'VOL_ENTROPY_STD', 'LARGE_ORDER', 'ROBUST_MOM', 'OI_CHANGE_RATE', 'INTRADAY_RSI', 'INFLECTION', 'VOL_TIDE', 'JUMP_INTENSITY', 'TSMOM', 'SIGN_MOM', 'IVOL', 'RS_VOL', 'SHADOW_VOL', 'VOL_RATIO', 'DAZZLE_RET', 'INTRADAY_REV', 'INTRADAY_CVAR', 'AMB_VOL_CORR', 'RET_VOL_COV', 'OI_LEVEL', 'AB_NIGHT_REV', 'SALIENCE_RET', 'FOLLOW_CROWD', 'MA_ALIGN', 'ATTN_SPILL', 'BASIS_MOM', 'LONE_GOOSE', 'HURST', 'PRICE_OI_CORR', 'MORNING_FOG', 'PROSPECT_TK', 'MEMBER_LS', 'WAREHOUSE_CHG', 'INVENTORY_MOM', 'LS_STRENGTH', 'LONG_CONC_STD', 'FLOW_IN_RATIO', 'SHADING_TREE', 'OVERNIGHT_TREND', 'LRSR', 'TGD', 'BASIS_MOM_LOG', 'INV_LUNAR_YOY', 'IDEAL_AMP', 'NET_SUPPORT_VOL', 'RET_SKEW', 'SHORT_WEIGHTED',
]

# V3: IC-screened subset (hierarchical clustering |corr|>0.7 merged + ICIR > 0.05)
# Re-screened 2026-04-01 with 77 candidate factors, top-20 by ICIR after dedup
# Fixed at 20 factors for stable token mapping (quarterly incremental updates ±3 max)
FEATURES_V3_LIST = [
    # Intraday microstructure (5)
    'TWAP_DEV', 'CLOSE_POS', 'SMART_MONEY', 'PRESSURE', 'OI_CHANGE',
    # Volatility (4)
    'INTRADAY_CVAR', 'VOL_TREND', 'VOL_SKEW', 'IVOL',
    # Term structure (2)
    'TERM_SPREAD', 'BASIS_MOM_LOG',
    # Behavioral/sentiment (2)
    'PROSPECT_TK', 'SALIENCE_RET',
    # Price pattern (2)
    'HL_RANGE', 'REL_STR',
    # Liquidity (1)
    'AMIHUD_ILLIQ',
    # Order flow (2)
    'LARGE_ORDER', 'NET_SUPPORT_VOL',
    # Fundamental (1)
    'TGD',
    # Overnight (1)
    'AB_NIGHT_REV',
]


class DuckDBFeatureEngineer:
    """Feature engineer for futures with 18 features (12 daily + 6 intraday)."""

    INPUT_DIM = 18

    @staticmethod
    def compute_features(raw_dict):
        """Compute 18-dim feature tensor.

        raw_dict must contain the standard 7 keys (open/high/low/close/volume/
        liquidity/fdv) plus the 6 intraday keys (vwap_dev/oi_change/vol_skew/
        vol_conc/smart_money/twap_dev).
        """
        # First 12 features — delegate to base FeatureEngineer
        base_features = FeatureEngineer.compute_features(raw_dict)  # [N, 12, T]

        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        # Intraday microstructure features (already computed by DuckDBDataLoader)
        vwap_dev = robust_norm(raw_dict['vwap_dev'])
        oi_change = robust_norm(raw_dict['oi_change'])
        vol_skew = robust_norm(raw_dict['vol_skew'])
        vol_conc = robust_norm(raw_dict['vol_conc'])
        smart_money = robust_norm(raw_dict['smart_money'])
        twap_dev = robust_norm(raw_dict['twap_dev'])

        intraday = torch.stack([
            vwap_dev,    # 12
            oi_change,   # 13
            vol_skew,    # 14
            vol_conc,    # 15
            smart_money, # 16
            twap_dev,    # 17
        ], dim=1)  # [N, 6, T]

        return torch.cat([base_features, intraday], dim=1)  # [N, 18, T]
