"""
Short Weighted Position factor (空头主力加权持仓因子).

Formula:
    WeightedS_t = Σ(oi_short_i,t × oi_short_i,t / total_oi_short_t)
               = Σ(oi_short_i² / total_short)  (HHI-style concentration)

where oi_short_i,t is member i's short position on day t,
total_oi_short_t is the sum of all top-20 members' short positions.

High WeightedS → short positions concentrated in few members → one-sided
short bet by dominant player → short-squeeze risk → bearish for shorts (bullish).
Low WeightedS → shorts evenly distributed → consensus short → more persistent.

Uses 20-day rolling mean for stability.

Reference:
    吴先兴, 何青青, 2019. 持仓龙虎榜中蕴藏的投资机会. 天风证券.
"""

import torch
from model_core.factor_registry import FactorBase


class ShortWeightedPosition(FactorBase):
    """Short-side HHI weighted position of top-20 exchange members."""

    name = 'SHORT_WEIGHTED'
    frequency = '1d'
    data_keys = ['short_weighted']
    category = 'positioning'
    description = 'Short position HHI-weighted concentration of top-20 members (20d avg)'

    def compute(self, raw_dict, K=20):
        sw = raw_dict['short_weighted']  # [N, T]
        # Rolling K-day mean for stability
        smoothed = self.rolling_mean(sw, window=K)
        # Negate: high concentration → short-squeeze risk → bullish
        return self.robust_norm(-smoothed)
