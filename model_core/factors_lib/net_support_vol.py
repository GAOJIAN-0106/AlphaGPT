"""
Net Support Volume factor (净支撑成交量因子).

Pre-computed from 1-min data (net_support_vol_cache.parquet):
    net_support_ratio = (support_vol - resistance_vol) / total_vol
    where support_vol = volume at minutes with close < daily mean close
          resistance_vol = volume at minutes with close > daily mean close

This factor takes the rolling 20-day mean of the daily ratio.

Positive → support volume dominates → bullish (price floor is strong).
Negative → resistance volume dominates → bearish.

Reference:
    沈芷琳, 刘富兵, 2024. 基于趋势资金日内交易行为的选股因子. 国盛证券.
"""

import torch
from model_core.factor_registry import FactorBase


class NetSupportVol(FactorBase):
    """Net support volume = rolling_mean(support - resistance ratio, 20)."""

    name = 'NET_SUPPORT_VOL'
    frequency = '1d'
    data_keys = ['net_support_vol']
    category = 'volume'
    description = 'Net support volume ratio: support vs resistance volume split by mean price (20d avg)'

    def compute(self, raw_dict, K=20):
        raw = raw_dict['net_support_vol']  # [N, T] daily ratio from cache
        # Rolling 20-day mean (matches paper: monthly lookback)
        smoothed = self.rolling_mean(raw, window=K)
        return self.robust_norm(smoothed)
