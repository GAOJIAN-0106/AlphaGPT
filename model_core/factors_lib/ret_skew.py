"""
Return Skewness factor (偏度因子).

Pre-computed from 5-min returns (ret_skew_cache.parquet):
    skew_t = E[((ret_i - μ) / σ)³]

where ret_i are 5-min returns over a rolling 20-day window,
μ and σ are the pooled mean/std of all 5-min returns in the window.

Positive skew → strong buying pressure, past large up-moves → mean-revert short.
Negative skew → strong selling pressure, past large down-moves → mean-revert long.

Reference:
    张革, 2022. 动量及高阶矩因子在商品期货截面上的运用. 中信期货.
"""

import torch
from model_core.factor_registry import FactorBase


class RetSkew(FactorBase):
    """5-min return skewness (偏度), pre-computed from 1-min kline."""

    name = 'RET_SKEW'
    frequency = '1d'
    data_keys = ['ret_skew']
    category = 'volatility'
    description = 'Return skewness from 5-min returns (20d rolling window)'

    def compute(self, raw_dict):
        skew = raw_dict['ret_skew']  # [N, T] already rolling-computed
        return self.robust_norm(skew)
