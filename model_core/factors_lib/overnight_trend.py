"""
Overnight trend factor for commodity futures.

Formula:
    (1/K) × Σ_{t=T-K+1}^{T} (open_t - close_{t-1}) / close_{t-1}

Average signed overnight return over K days. Captures the momentum
component embedded in overnight price gaps.

Reference:
    王冬黎, 常海晴, 2022. 商品多因子模型框架再探究. 东证期货.
"""

import torch
from model_core.factor_registry import FactorBase


class OvernightTrend(FactorBase):
    """Overnight trend (隔夜趋势因子).

    K-day average of signed overnight returns.
    Paper recommends K in 10-40; we use K=20.

    Positive → persistent overnight buying → bullish momentum.
    Negative → persistent overnight selling → bearish momentum.
    """
    name = 'OVERNIGHT_TREND'
    frequency = '1d'
    data_keys = ['open', 'close']
    category = 'momentum'
    description = 'Average signed overnight return (20d), momentum component'

    def compute(self, raw_dict, K=20):
        open_p = raw_dict['open']    # [N, T]
        close_p = raw_dict['close']  # [N, T]

        # Overnight return: (open_t - close_{t-1}) / close_{t-1}
        prev_close = torch.cat([close_p[:, :1], close_p[:, :-1]], dim=1)
        overnight_ret = (open_p - prev_close) / (prev_close + 1e-9)

        # Rolling K-day mean
        result = self.rolling_mean(overnight_ret, window=K)

        return self.robust_norm(result)
