"""
Liquidity factors for commodity futures.

References:
    Amihud, Y., 2002. Illiquidity and stock returns: cross-section and
    time-series effects. Journal of Financial Markets, 5(1), 31-56.
    高伊素, 张石右, 2017. 商品期货因子投研. 海通期货.
"""

import torch
from model_core.factor_registry import FactorBase


class AmihudILLIQ(FactorBase):
    """Amihud illiquidity ratio — price impact per unit of turnover.

    Formula:
        ILLIQ_t = 1/R × Σ_{i=t-R+1}^{t} |r_i| / Amount_i

    Where r_i = daily log return, Amount_i = daily turnover (volume × close).
    Higher ILLIQ = less liquid = larger price impact per unit traded.

    Long illiquid assets earns the illiquidity premium.
    R ≤ 40 trading days works best per the reference.
    """
    name = 'AMIHUD_ILLIQ'
    frequency = '1d'
    data_keys = ['close', 'liquidity']
    category = 'liquidity'
    description = 'Amihud illiquidity ratio (40d rolling), price impact measure'

    def compute(self, raw_dict, R=40):
        close = raw_dict['close']      # [N, T]
        amount = raw_dict['liquidity']  # [N, T] — volume × close

        # Daily absolute log return
        prev_close = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        abs_ret = torch.abs(torch.log(close / (prev_close + 1e-9)))

        # |r_i| / Amount_i  (clamp amount to avoid division by zero)
        ratio = abs_ret / (amount + 1e-9)

        # Rolling R-day mean
        illiq = self.rolling_mean(ratio, window=R)

        return self.robust_norm(illiq)
