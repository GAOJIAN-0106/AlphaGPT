"""
Inventory lunar-calendar YoY growth factor (库存农历同比增速).

Formula:
    YOY = (stock_t - mean(stock at same lunar date, past n years))
          / mean(stock at same lunar date, past n years)

Lunar calendar alignment captures Chinese holiday seasonal patterns
better than Gregorian (Spring Festival restocking, post-holiday drawdown).

Precomputed in scripts/compute_inv_lunar_yoy.py, cached as parquet.

Reference:
    吴先兴, 2019. 基本面逻辑下的因子改进与策略组合. 天风证券.
"""

import torch
from model_core.factor_registry import FactorBase


class InvLunarYoY(FactorBase):
    """Inventory lunar YoY growth (库存农历同比增速).

    Uses precomputed daily values from inv_lunar_yoy_cache.parquet.
    Applies 20-day rolling mean for smoothing (weekly source, ffilled to daily).

    Inventory buildup (positive YoY) → supply pressure → bearish → negate.
    """
    name = 'INV_LUNAR_YOY'
    frequency = '1d'
    data_keys = ['inv_lunar_yoy']
    category = 'fundamental'
    description = 'Inventory lunar YoY growth (seasonal alignment, 20d rolling)'

    def compute(self, raw_dict, K=20):
        yoy = raw_dict['inv_lunar_yoy']  # [N, T]
        smoothed = self.rolling_mean(yoy, window=K)
        # Negate: inventory buildup → bearish
        return self.robust_norm(-smoothed)
