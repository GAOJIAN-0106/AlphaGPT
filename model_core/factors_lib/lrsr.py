"""
Long-short relative strength of top members (持仓主力多空相对强弱).

Formula:
    LRSR = Σ(top20 oi_long_i) / total_oi_long - Σ(top20 oi_short_i) / total_oi_short

In futures, total_oi_long = total_oi_short = total_oi, so:
    LRSR = ls_raw / total_oi

where ls_raw = Σ(top20 long) - Σ(top20 short).

Top members (from exchange position rankings) are key informed traders.
LRSR > 0 → informed traders hold larger share of longs → bullish.

Reference:
    吴先兴, 何青青, 2019. 持仓龙虎榜中蕴藏的投资机会. 天风证券.
"""

import torch
from model_core.factor_registry import FactorBase


class LRSR(FactorBase):
    """Long-short relative strength of top-20 members.

    LRSR = ls_raw / total_oi (exact OI from close_oi).
    Uses 20-day rolling mean for stability.

    Positive → top members hold disproportionately more longs → bullish.
    Negative → top members skewed short → bearish.
    """
    name = 'LRSR'
    frequency = '1d'
    data_keys = ['ls_raw', 'total_oi']
    category = 'positioning'
    description = 'Top-20 member long-short relative strength (ls_raw / total_oi, 20d)'

    def compute(self, raw_dict, K=20):
        ls = raw_dict['ls_raw']       # [N, T]
        oi = raw_dict['total_oi']     # [N, T]

        # LRSR = ls_raw / total_oi
        lrsr = ls / (oi.abs() + 1e-9)

        # Rolling K-day mean for stability
        result = self.rolling_mean(lrsr, window=K)

        return self.robust_norm(result)
