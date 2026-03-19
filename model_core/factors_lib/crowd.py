"""
Crowd behavior / herding factors for commodity futures.

References:
    曹有晗, 2021. 个股成交额在价格区间的分布特征与"随波逐流"因子. 方正证券.
"""

import torch
from model_core.factor_registry import FactorBase


class FollowTheCrowd(FactorBase):
    """Follow-the-crowd factor (随波逐流因子).

    Full implementation from 1-minute data:
    1. Per minute: relative-to-open return = close_m / open_day - 1
    2. High-side turnover = Σ turnover where ret > "fair return" (≈0)
    3. Low-side turnover = Σ turnover where ret < "fair return"
    4. hl_diff = (high - low) / total turnover
    5. Rolling 20-day cross-sectional Spearman correlation of hl_diff
       vs all other products → mean |ρ| = herding intensity

    High follow_crowd → prices move in lockstep with market →
    more co-movement → potential reversal when crowd disperses.

    Computed from 1-min data, cached in parquet.
    """
    name = 'FOLLOW_CROWD'
    frequency = '1d'
    data_keys = ['follow_crowd']
    category = 'microstructure'
    description = 'Follow-the-crowd (cross-sectional herding intensity from 1min)'

    def compute(self, raw_dict):
        fc = raw_dict['follow_crowd']  # [N, T]
        return self.robust_norm(fc)
