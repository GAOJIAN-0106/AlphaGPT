"""
Member position factor (会员持仓因子).

Formula:
    Factor_T = (LS_raw,T - (1/K) × Σ LS_raw,t) / |1/K × Σ LS_raw,t|

Where LS_raw = Σ(top20 long_oi) - Σ(top20 short_oi)

The top-20 member position data approximates informed trader positioning.
A rising LS_raw relative to its K-day average indicates net long buildup
by large participants — bullish signal.

Reference:
    王冬黎, 常海锋, 2022. 商品多因子模型框架再探究. 东证期货.

Data source:
    Cached from exchange member position rankings via
    scripts/fetch_member_positions.py → member_pos_cache.parquet
"""

import torch
from model_core.factor_registry import FactorBase


class MemberPositionFactor(FactorBase):
    """Net long/short deviation of top-20 exchange members.

    Factor = (LS_raw_T - MA(LS_raw, K)) / |MA(LS_raw, K)|

    Positive → members adding longs faster than recent average → bullish.
    Negative → members reducing longs or adding shorts → bearish.
    """
    name = 'MEMBER_LS'
    frequency = '1d'
    data_keys = ['ls_raw']
    category = 'positioning'
    description = 'Member position net L/S deviation (top20, K=20)'

    def compute(self, raw_dict, K=20):
        ls = raw_dict['ls_raw']  # [N, T]
        mean_ls = self.rolling_mean(ls, window=K)
        factor = (ls - mean_ls) / (mean_ls.abs() + 1e-9)
        return self.robust_norm(factor)


class LongConcStd(FactorBase):
    """Long position concentration std (多头主力持仓占比标准差).

    Formula:
        StdLR_t = std(oi_long_i,t / total_oi_long_t)  for i in top-20

    Standard deviation of each member's long position share.
    High StdLR → one or few members dominate longs → concentrated bet.
    Low StdLR → evenly distributed → diffuse consensus.

    Higher concentration of longs historically precedes price declines
    (crowded trade risk), so we negate the signal.

    Reference: 吴先兴, 何书青, 2019, 天风证券.
    """
    name = 'LONG_CONC_STD'
    frequency = '1d'
    data_keys = ['long_conc_std']
    category = 'positioning'
    description = 'Top-20 long position concentration std (crowding risk)'

    def compute(self, raw_dict, K=20):
        conc = raw_dict['long_conc_std']  # [N, T]
        # K-day rolling mean for smoothing
        smoothed = self.rolling_mean(conc, window=K)
        # Negate: high concentration → bearish (crowded trade risk)
        return self.robust_norm(-smoothed)


class LSStrength(FactorBase):
    """Long-short strength factor (多空强弱因子).

    DEPRECATED: Superseded by LRSR which uses exact total_oi from close_oi
    instead of the rough oi_change.cumsum() approximation.

    Kept as alias for backward compatibility — name maps to LRSR computation.

    Reference: 王冬黎, 常海锋, 2022, 东证期货.
    """
    name = 'LS_STRENGTH'
    frequency = '1d'
    data_keys = ['ls_raw', 'total_oi']
    category = 'positioning'
    description = 'Member net L/S as fraction of total OI (K=20 avg, uses exact OI)'

    def compute(self, raw_dict, K=20):
        ls = raw_dict['ls_raw']       # [N, T]
        oi = raw_dict['total_oi']     # [N, T]

        ratio = ls / (oi.abs() + 1e-9)
        factor = self.rolling_mean(ratio, window=K)

        return self.robust_norm(factor)
