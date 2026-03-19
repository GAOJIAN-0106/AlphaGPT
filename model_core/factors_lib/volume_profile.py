"""
Volume distribution / profile factors for commodity futures.

References:
    郑北跂, 2022. 成交量分布中的Alpha. 兴业证券.
"""

import torch
from model_core.factor_registry import FactorBase


class VolumeProfileSkew(FactorBase):
    """B-type volume distribution factor (b型成交量分布).

    Uses the Volume Support Price (VSP) — the price level with the highest
    intraday volume — computed from 1-minute data in the DuckDB SQL query.

    vsp_position = (VSP - Low) / (High - Low)
      → 0: volume peak near the low (b-type) → bearish
      → 1: volume peak near the high (p-type) → bullish

    The factor uses 20-day rolling mean for stability and to match
    the monthly aggregation from the paper.

    Falls back to OHLC approximation if vsp_position is not available.
    """
    name = 'VP_SKEW'
    frequency = '1d'
    data_keys = ['close', 'high', 'low']
    category = 'volume_profile'
    description = 'Volume profile skew from 1min VSP (b-type distribution), 20d rolling'

    def compute(self, raw_dict, K=20):
        if 'vsp_position' in raw_dict:
            # Use real VSP position computed from 1-minute data
            vsp_pos = raw_dict['vsp_position']  # [N, T] in [0, 1]
        else:
            # Fallback: approximate with close position in HL range
            close = raw_dict['close']
            high = raw_dict['high']
            low = raw_dict['low']
            vsp_pos = (close - low) / (high - low + 1e-9)

        # Center around 0.5: positive = volume skewed high, negative = low
        vsp_centered = vsp_pos - 0.5

        # Rolling K-day mean for stability
        result = self.rolling_mean(vsp_centered, window=K)

        return self.robust_norm(result)


class VolEntropyStd(FactorBase):
    """Volume distribution instability (成交量分桶熵的标准差).

    Original: compute daily volume entropy from 1-min bucketed distribution,
    then take 20-day rolling std of the entropy series.

    Approximation using existing daily features:
        VOL_CONC = max_30min_vol / total_vol is an inverse-entropy proxy
        (high concentration ≈ low entropy). Its 20-day rolling std captures
        instability in volume distribution patterns.

    High instability → emotional/speculative trading → bearish signal.

    Reference: 郑北跂, 2022, 成交量分布中的Alpha, 兴业证券.
    """
    name = 'VOL_ENTROPY_STD'
    frequency = '1d'
    data_keys = ['vol_conc']
    category = 'volume_profile'
    description = 'Volume concentration instability (20d std of VOL_CONC)'

    def compute(self, raw_dict, K=20):
        vol_conc = raw_dict['vol_conc']  # [N, T]

        # 20-day rolling std of volume concentration
        result = self.rolling_std(vol_conc, window=K)

        return self.robust_norm(result)
