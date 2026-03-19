"""
Jump / discontinuity factors for commodity futures.

References:
    Tauchen, G. & Zhou, H., 2011. Realized jumps on financial markets
    and predicting credit spreads. Journal of Econometrics, 160(1), 102-118.
    闻飞翔, 罗宇宁, 2022. 构造价格跳跃因子. 广发证券.
"""

import torch
from model_core.factor_registry import FactorBase


class JumpIntensity(FactorBase):
    """Jump arrival intensity (跳跃强度).

    Simplified Tauchen-Zhou approach:
        jump_ratio_t = 1 - BV_t / RV_t

    where:
        RV = Σ r_i² (realized variance from 1-min returns)
        BV = (π/2) × Σ |r_i| × |r_{i-1}| (bipower variation)

    jump_ratio ≈ 0 → no jumps (continuous diffusion)
    jump_ratio ≈ 1 → large jump dominated the day

    Factor = 20-day rolling mean of jump_ratio (jump arrival rate).
    Computed from 1-min data, cached in parquet.

    High jump intensity → volatile/uncertain → potential risk premium.
    """
    name = 'JUMP_INTENSITY'
    frequency = '1d'
    data_keys = ['jump_ratio']
    category = 'volatility'
    description = 'Jump arrival intensity (BV/RV ratio from 1min, 20d mean)'

    def compute(self, raw_dict, K=20):
        jr = raw_dict['jump_ratio']  # [N, T]

        # Clamp to [0, 1] — negative values from numerical issues
        jr = torch.clamp(jr, 0.0, 1.0)

        # Rolling K-day mean = jump arrival rate
        result = self.rolling_mean(jr, window=K)

        return self.robust_norm(result)
