"""
Hurst exponent / R-S analysis factor for commodity futures.

References:
    张嘉颖, 洛升银, 2024. 技术指标因子高频化. 中信建投.
"""

import torch
from model_core.factor_registry import FactorBase


class HurstIndex(FactorBase):
    """Hurst index from intraday R/S analysis (Hurst 指标).

    Computed from 1-min close prices:
        R(n) = max(cumsum(close - mean)) - min(cumsum(close - mean))
        Hurst = R / std(close)

    Hurst > avg → trending (momentum) behavior
    Hurst < avg → mean-reverting behavior

    Uses 20-day time-decay weighted rolling mean.
    Computed from 1-min data, cached in parquet.
    """
    name = 'HURST'
    frequency = '1d'
    data_keys = ['hurst']
    category = 'trend'
    description = 'Hurst index from intraday R/S analysis (trend persistence)'

    def compute(self, raw_dict, K=20):
        h = raw_dict['hurst']  # [N, T]
        N, T = h.shape
        device = h.device

        # Time-decay weighted rolling mean
        weights = torch.arange(1, K + 1, dtype=torch.float32, device=device) / K
        weight_sum = weights.sum()
        pad = torch.zeros(N, K - 1, device=device)
        padded = torch.cat([pad, h], dim=1)
        windows = padded.unfold(1, K, 1)
        result = (windows * weights).sum(dim=-1) / weight_sum

        return self.robust_norm(result)
"""
"""
