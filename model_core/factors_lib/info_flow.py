"""
Information flow / price-volume lead-lag factors.

References:
    曹有晗, 2023. 推动个股价格变化的因素分解与"在隐林间"因子. 方正证券.
"""

import torch
from model_core.factor_registry import FactorBase


class MorningFog(FactorBase):
    """Morning fog factor (朝没晨雾因子).

    From 1-min data:
    1. ret_t = close_t / close_{t-1} - 1
    2. voldiff_t = volume_t - volume_{t-1}
    3. Regress: ret_t = α0 + α1*voldiff_{t-1} + ... + α5*voldiff_{t-5} + ε
    4. Factor = std(t-values of α1..α5)

    Measures stability of volume-price lead-lag relationship.
    Low std → stable info flow → "cold" stock → potential upside.

    Computed from 1-min data, cached in parquet.
    Uses 20-day rolling mean.
    """
    name = 'MORNING_FOG'
    frequency = '1d'
    data_keys = ['morning_fog']
    category = 'microstructure'
    description = 'Morning fog (volume-price lead-lag stability, 20d mean)'

    def compute(self, raw_dict, K=20):
        mf = raw_dict['morning_fog']
        result = self.rolling_mean(mf, window=K)
        return self.robust_norm(result)
