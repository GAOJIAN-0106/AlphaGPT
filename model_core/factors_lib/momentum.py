"""
Momentum factors for commodity futures.

References:
    任菁, 2022. 动量及高阶矩因子在商品期货截面上的应用. 中信期货.
"""

import torch
from model_core.factor_registry import FactorBase


class CrossSectionalMomentum(FactorBase):
    """Cross-sectional momentum with lag.

    Formula:
        XSMom_t = (P_{t-L} - P_{t-L-J}) / P_{t-L-J}

    Where J=20 (lookback), L=3 (lag to avoid short-term reversal).
    The lag L skips the most recent days to capture the momentum
    effect while avoiding the mean-reversion at very short horizons.

    This is a pure cross-sectional signal: long the top gainers,
    short the bottom losers across all commodity products.
    """
    name = 'XS_MOM'
    frequency = '1d'
    data_keys = ['close']
    category = 'momentum'
    description = 'Cross-sectional momentum (J=20 lookback, L=3 lag)'

    def compute(self, raw_dict, J=20, L=3):
        close = raw_dict['close']  # [N, T]
        N, T = close.shape

        # P_{t-L}: price L days ago
        # P_{t-L-J}: price L+J days ago
        shift_L = L
        shift_LJ = L + J

        # Build shifted tensors with zero padding
        # P_{t-L}
        P_recent = torch.cat([
            torch.zeros(N, shift_L, device=close.device),
            close[:, :T - shift_L]
        ], dim=1)

        # P_{t-L-J}
        P_past = torch.cat([
            torch.zeros(N, shift_LJ, device=close.device),
            close[:, :T - shift_LJ]
        ], dim=1)

        # XSMom = (P_{t-L} - P_{t-L-J}) / P_{t-L-J}
        mom = (P_recent - P_past) / (P_past + 1e-9)

        # Zero out the first shift_LJ bars (no valid data)
        mom[:, :shift_LJ] = 0.0

        return self.robust_norm(mom)


class OvernightGap(FactorBase):
    """Cumulative absolute overnight gap (隔夜跳空).

    Formula:
        absRet_night_t = Σ_{i=t-K+1}^{t} |ln(Open_i / Close_{i-1})|

    Larger cumulative gaps → prices tend to mean-revert (negative
    correlation with future returns). Overnight gaps reflect
    information shocks during non-trading hours; excessive gaps
    indicate instability that typically reverses.

    Reference: 朱定豪, 2020, 华安证券.
    """
    name = 'OVERNIGHT_GAP'
    frequency = '1d'
    data_keys = ['open', 'close']
    category = 'mean_reversion'
    description = 'Cumulative absolute overnight gap (20d), reversal signal'

    def compute(self, raw_dict, K=20):
        open_p = raw_dict['open']    # [N, T]
        close_p = raw_dict['close']  # [N, T]

        # Overnight return: ln(Open_t / Close_{t-1})
        prev_close = torch.cat([close_p[:, :1], close_p[:, :-1]], dim=1)
        overnight_ret = torch.log(open_p / (prev_close + 1e-9))
        abs_overnight = overnight_ret.abs()

        # Rolling K-day sum
        pad = torch.zeros(abs_overnight.shape[0], K - 1, device=abs_overnight.device)
        padded = torch.cat([pad, abs_overnight], dim=1)
        cum_gap = padded.unfold(1, K, 1).sum(dim=-1)  # [N, T]

        return self.robust_norm(cum_gap)


class SignMomentum(FactorBase):
    """Sign momentum (符号动量) — win rate over K days.

    Factor = Σ 1(r_t >= 0) / K = fraction of positive-return days.

    Captures the consistency/frequency of positive returns, regardless
    of magnitude. Robust to outliers since it only counts direction.

    Reference: 王冬黎, 范海韵, 2022, 东证期货.
    """
    name = 'SIGN_MOM'
    frequency = '1d'
    data_keys = ['close']
    category = 'momentum'
    description = 'Sign momentum (positive return frequency over K=20 days)'

    def compute(self, raw_dict, K=20):
        close = raw_dict['close']
        prev = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        ret = torch.log(close / (prev + 1e-9))

        # 1 if return >= 0, 0 otherwise
        positive = (ret >= 0).float()

        # Rolling K-day mean = win rate
        result = self.rolling_mean(positive, window=K)

        return self.robust_norm(result)


class TimeSeriesMomentum(FactorBase):
    """Time-series momentum (时间序列动量, TSMOM).

    Signal = sign(r_{t-h,t}) / σ_t

    Positive past return → long, negative → short, scaled by inverse vol.
    Purely time-series: each asset's signal depends only on its own history.

    Reference: Moskowitz, Ooi & Pedersen, 2012, JFE.
    """
    name = 'TSMOM'
    frequency = '1d'
    data_keys = ['close']
    category = 'momentum'
    description = 'Time-series momentum signal (sign × inverse vol, h=20)'

    def compute(self, raw_dict, h=20, vol_window=60):
        close = raw_dict['close']  # [N, T]
        N, T = close.shape
        device = close.device

        # Daily log returns
        prev = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        ret = torch.log(close / (prev + 1e-9))

        # Cumulative return over past h days: ln(close_t / close_{t-h})
        close_shifted = torch.cat([
            close[:, :1].expand(N, h), close[:, :-h]
        ], dim=1)
        cum_ret = torch.log(close / (close_shifted + 1e-9))

        # Direction signal
        direction = torch.sign(cum_ret)

        # Rolling volatility (annualized)
        vol = self.rolling_std(ret, window=vol_window) * (252 ** 0.5) + 1e-9

        # TSMOM signal = direction / vol
        signal = direction / vol
        signal[:, :h] = 0.0

        return self.robust_norm(signal)


class RobustMomentum(FactorBase):
    """Robust cross-sectional momentum (稳健动量因子).

    Instead of raw returns, uses cross-sectional rank normalization
    to reduce the impact of extreme price moves:

        rank_t = (y(r_t) - (N+1)/2) / √((N+1)(N-1)/12)

    where y(r_t) is the ascending rank of asset's return among all N assets.
    Then takes K-day rolling mean of the standardized rank.

    More robust than XS_MOM which uses raw returns and can be
    dominated by outlier price movements.

    Reference: 王冬黎, 范海韵, 2022, 东证期货.
    """
    name = 'ROBUST_MOM'
    frequency = '1d'
    data_keys = ['close']
    category = 'momentum'
    description = 'Cross-sectional rank-normalized momentum (K=20 rolling)'

    def compute(self, raw_dict, K=20):
        close = raw_dict['close']  # [N, T]
        N, T = close.shape

        # Daily log returns
        prev = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        ret = torch.log(close / (prev + 1e-9))

        # Cross-sectional rank at each time step
        # argsort of argsort gives ranks (0-indexed)
        ranks = torch.zeros_like(ret)
        for t in range(T):
            r = ret[:, t]
            order = r.argsort()
            rank_vals = torch.zeros_like(r)
            rank_vals[order] = torch.arange(1, N + 1, dtype=torch.float32,
                                            device=r.device)
            # Standardize: (rank - (N+1)/2) / sqrt((N+1)(N-1)/12)
            mean_rank = (N + 1.0) / 2.0
            std_rank = ((N + 1.0) * (N - 1.0) / 12.0) ** 0.5
            ranks[:, t] = (rank_vals - mean_rank) / (std_rank + 1e-9)

        # Rolling K-day mean
        result = self.rolling_mean(ranks, window=K)

        return self.robust_norm(result)


class IntradayRSI(FactorBase):
    """High-frequency RSI with time-decay (相对强弱 高频版).

    Computed from 1-minute data in SQL:
        RSI_daily = avg(gain_1min) / avg(|change_1min|) * 100

    Then 20-day time-decay weighted rolling mean:
        Factor = Σ(j/20 × RSI_j) / Σ(j/20)

    Differs from daily REL_STR (RSI-14) by using intraday granularity
    and longer effective window.

    Reference: 张嘉颖, 洛升银, 2024, 中信建投.
    """
    name = 'INTRADAY_RSI'
    frequency = '1d'
    data_keys = ['intraday_rsi']
    category = 'momentum'
    description = 'Intraday RSI from 1min data, 20d time-decay averaged'

    def compute(self, raw_dict, K=20):
        rsi = raw_dict['intraday_rsi']  # [N, T] in ~[30, 70]
        N, T = rsi.shape
        device = rsi.device

        # Center around 50 (neutral)
        centered = rsi - 50.0

        # Time-decay weighted rolling mean
        weights = torch.arange(1, K + 1, dtype=torch.float32, device=device) / K
        weight_sum = weights.sum()

        pad = torch.zeros(N, K - 1, device=device)
        padded = torch.cat([pad, centered], dim=1)
        windows = padded.unfold(1, K, 1)
        result = (windows * weights).sum(dim=-1) / weight_sum

        return self.robust_norm(result)
