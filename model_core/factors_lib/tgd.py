"""
Time Gravity Deviation factor (跌幅时间重心偏离).

Measures the timing asymmetry between up and down price moves within
the trading day, after removing the common intraday pattern via
cross-sectional regression.

G_u = return-weighted average time of up-moves (1-min)
G_d = return-weighted average time of down-moves (1-min)
Cross-sectional regression: G_d = α + β·G_u + ε
TGD = 20-day average of ε

Negative TGD → down-moves cluster earlier than predicted → selling
pressure front-loaded → positive future return (reversal).

Precomputed in scripts/compute_tgd.py, cached as parquet.

Reference:
    魏建榕, 苗杰, 徐少楠, 2022. 日内分钟收益率的时间衍生变量. 开源证券.
"""

import torch
from model_core.factor_registry import FactorBase


class TimeGravityDeviation(FactorBase):
    """Time gravity deviation of down-moves (跌幅时间重心偏离).

    Uses precomputed daily TGD from tgd_cache.parquet.
    Applies 20-day rolling mean matching the paper's recommendation.
    """
    name = 'TGD'
    frequency = '1d'
    data_keys = ['tgd']
    category = 'microstructure'
    description = 'Down-move time gravity deviation (cross-sectional residual, 20d)'

    def compute(self, raw_dict, K=20):
        tgd = raw_dict['tgd']  # [N, T]
        result = self.rolling_mean(tgd, window=K)
        return self.robust_norm(result)
