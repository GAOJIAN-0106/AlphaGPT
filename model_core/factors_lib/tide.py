"""
Volume tide factor for commodity futures.

The "tide" captures the full cycle of volume buildup (tide-in) → peak →
volume decline (tide-out), and measures the price change rate during
this entire cycle.

References:
    曹春明, 2022. 个股成交量的潮汐变化及"潮汐"因子构建. 方正证券.
"""

import torch
from model_core.factor_registry import FactorBase


class VolumeTide(FactorBase):
    """Volume tide price velocity (成交量潮汐价格变动速率).

    For each trading day:
    1. Compute 9-minute domain volume (±4 bars rolling sum) for each 1-min bar
    2. Find volume peak time t (max domain volume)
    3. Find tide-in time m (min domain volume before peak, bars 5..t-1)
    4. Find tide-out time n (min domain volume after peak, bars t+1..t+233)
    5. Tide speed = (C_n - C_m) / C_m / (n - m)

    Computed via separate SQL query in DuckDBDataLoader._compute_tide_factor().
    The factor is the 20-day rolling mean of daily tide speed.

    Positive tide → price rises during volume cycle → bullish.
    Negative tide → price falls during volume cycle → bearish.
    """
    name = 'VOL_TIDE'
    frequency = '1d'
    data_keys = ['tide_speed']
    category = 'volume_profile'
    description = 'Volume tide price velocity (full cycle speed, 20d mean)'

    def compute(self, raw_dict, K=20):
        tide = raw_dict['tide_speed']  # [N, T]

        # Rolling K-day mean
        result = self.rolling_mean(tide, window=K)

        return self.robust_norm(result)
