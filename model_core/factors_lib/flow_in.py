"""
Money flow in ratio factor for commodity futures.

flowInRatio = Σ_j (Volume_j * Close_j * sign(Close_j - Close_{j-1})) / Σ_j (Volume_j * Close_j)

Computed from 1-minute data in SQL (DuckDB loader), this factor captures
the proportion of turnover flowing in (up-tick) vs out (down-tick).

Short lookback: short-term reversal (factor direction negative).
Long lookback (>5d): momentum characteristic (factor direction positive).

Reference:
    高佳睿, 班石, 2018. 高频量价因子在股票与期货中的表现. 海通证券.
"""

import torch
from model_core.factor_registry import FactorBase


class FlowInRatio(FactorBase):
    """Money flow in ratio (资金流入占比).

    Uses precomputed daily flow_in_ratio from 1-min data (SQL CTE).
    Applies 20-day rolling mean for stability, matching the paper's
    recommended holding period for momentum-direction usage.
    """
    name = 'FLOW_IN_RATIO'
    frequency = '1d'
    data_keys = ['flow_in_ratio']
    category = 'microstructure'
    description = 'Money flow in ratio from 1min sign(delta_close) weighting, 20d rolling'

    def compute(self, raw_dict, K=20):
        fir = raw_dict['flow_in_ratio']  # [N, T] in ~[-1, 1]

        # Rolling K-day mean for stability
        result = self.rolling_mean(fir, window=K)

        return self.robust_norm(result)
