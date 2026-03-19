"""
Risk / tail-risk factors for commodity futures.

References:
    严佳炜, 2020. 分钟线的尾部特征. 方正证券.
"""

import torch
from model_core.factor_registry import FactorBase


class IntradayCVaR(FactorBase):
    """Intraday conditional VaR (日内条件在险价值).

    VWAR = volume-weighted average return from 1-min data.
    CVaR = 5th percentile of daily VWAR over rolling 20-day window.

    Computed from 1-min data, cached in parquet:
        1. VWAR_t = Σ(ret_i × vol_i) / Σ(vol_i) per day
        2. CVaR = quantile(VWAR, 0.05) over 20-day window

    More negative CVaR → higher tail risk → potential risk premium.

    Differs from VOL_CLUSTER (symmetric volatility) by capturing
    only the left tail of volume-weighted returns.
    """
    name = 'INTRADAY_CVAR'
    frequency = '1d'
    data_keys = ['cvar_5pct']
    category = 'risk'
    description = 'Intraday CVaR (5th pct of VWAR, 20d rolling)'

    def compute(self, raw_dict):
        cvar = raw_dict['cvar_5pct']  # [N, T] — negative values
        return self.robust_norm(cvar)


class RetVolCovariance(FactorBase):
    """Return-volatility covariance (日内收益波动比 / 灾后重建因子).

    Full implementation from 1-minute OHLC data:
    1. Per minute: "optimal vol" = std(5-bar OHLC, 20 prices) / mean
    2. ret_vol_ratio = minute_return / optimal_vol
    3. Daily cov(ret_vol_ratio, optimal_vol)
    4. 20-day rolling mean for monthly factor

    Computed from 1-min data, cached in parquet.

    Negative covariance → when vol rises, ret/vol falls → investors
    underreact to volatility spikes → future recovery (bullish).

    Reference: Moreira & Muir, 2017, JF; 鲁希存, 2022, 方正证券.
    """
    name = 'RET_VOL_COV'
    frequency = '1d'
    data_keys = ['ret_vol_cov']
    category = 'risk'
    description = 'Return-vol covariance from 1min OHLC (disaster recovery, 20d)'

    def compute(self, raw_dict, K=20):
        cov = raw_dict['ret_vol_cov']  # [N, T]
        result = self.rolling_mean(cov, window=K)
        return self.robust_norm(result)
