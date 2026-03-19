"""
Ambiguity / uncertainty factors for commodity futures.
"""

import torch
from model_core.factor_registry import FactorBase


class AmbiguityCorrelation(FactorBase):
    """Ambiguity-volume correlation (模糊关联度).

    Daily correlation between intraday "ambiguity" (vol-of-vol from 1-min)
    and intraday volume. Computed from 1-min data, cached in parquet.

    High positive correlation → investors trade more when uncertainty
    spikes → ambiguity aversion → bearish (overreaction leads to reversal).

    Uses 20-day rolling mean for monthly aggregation.
    """
    name = 'AMB_VOL_CORR'
    frequency = '1d'
    data_keys = ['amb_vol_corr']
    category = 'microstructure'
    description = 'Ambiguity-volume correlation (vol-of-vol vs turnover, 20d mean)'

    def compute(self, raw_dict, K=20):
        corr = raw_dict['amb_vol_corr']  # [N, T]
        result = self.rolling_mean(corr, window=K)
        return self.robust_norm(result)
