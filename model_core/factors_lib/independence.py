"""
Market independence / herding factors for commodity futures.

References:
    曹有晗, 2023. 个股成交额的市场趋附性与"水中行舟"因子. 方正证券.
"""

import torch
from model_core.factor_registry import FactorBase


class LoneGoose(FactorBase):
    """Lone goose factor (孤雁出群因子).

    Full implementation from 1-minute cross-sectional data:
    1. Per minute: market divergence = cross-sectional std of returns
    2. Find non-divergent moments (divergence < daily mean)
    3. In those moments: compute |Pearson corr| of each product's
       turnover vs all other products' turnover
    4. Mean |corr| = daily factor
    5. 20-day rolling mean + std → monthly factor

    Low correlation during calm markets → independent behavior →
    potentially discovering new trend → bullish signal.

    Computed from 1-min data, cached in parquet.
    """
    name = 'LONE_GOOSE'
    frequency = '1d'
    data_keys = ['lone_goose']
    category = 'microstructure'
    description = 'Lone goose (turnover independence during calm markets, 20d)'

    def compute(self, raw_dict, K=20):
        lg = raw_dict['lone_goose']  # [N, T]
        result = self.rolling_mean(lg, window=K)
        return self.robust_norm(result)
