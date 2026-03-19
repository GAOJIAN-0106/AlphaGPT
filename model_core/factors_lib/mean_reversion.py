"""
Mean-reversion / technical factors for commodity futures.

References:
    张嘉颖, 洛升银, 2024. 技术指标因子高频化. 中信建投.
"""

import torch
from model_core.factor_registry import FactorBase


class BiasDecay(FactorBase):
    """BIAS with time-decay weighting (乖离率).

    Original formula (1-min frequency):
        BIAS_t = (Close_t - MA240_t) / MA240_t × 100

    Since DuckDB loader already computes twap_dev = (AVG(1min close) - close) / close
    which is equivalent to -BIAS/100, we reuse that and add the time-decay
    aggregation from the paper:

        Factor_t = Σ_{j=1}^{K} (j/K) × BIAS_{t-K+j} / Σ_{j=1}^{K} (j/K)

    This gives more weight to recent days (linear decay), converting the
    noisy daily BIAS into a smoother monthly signal.

    Captures mean-reversion: extreme deviation from intraday average
    tends to revert, especially when persistent over multiple days.
    """
    # Disabled: too correlated with MPB (r=0.72) and TWAP_DEV.
    name = ''  # empty name prevents auto-registration


class IntradayReversal(FactorBase):
    """Intraday momentum reversal (日内动量反转).

    OHLC approximation of the extrema-based factor:
        Original: (extrema_first - extrema_behind) / extrema_first

    Proxy using OHLC:
        reversal = (open - close) / (high - low)

    This captures the degree to which the day's move reversed from open.
    - Positive = open > close (bearish reversal from initial move)
    - Negative = close > open (bullish continuation)

    Takes 20-day rolling mean for stability.

    Reference: 赵明 et al., 2024, 国金证券.
    """
    name = 'INTRADAY_REV'
    frequency = '1d'
    data_keys = ['open', 'close', 'high', 'low']
    category = 'mean_reversion'
    description = 'Intraday reversal (open-close vs range, 20d mean)'

    def compute(self, raw_dict, K=20):
        o = raw_dict['open']
        c = raw_dict['close']
        h = raw_dict['high']
        l = raw_dict['low']

        # (open - close) normalized by day range
        hl_range = h - l + 1e-9
        reversal_daily = (o - c) / hl_range

        # Rolling K-day mean
        result = self.rolling_mean(reversal_daily, window=K)

        return self.robust_norm(result)
    frequency = '1d'
    data_keys = ['twap_dev']
    category = 'mean_reversion'
    description = 'Time-decay weighted BIAS (20d), intraday mean-reversion signal'

    def compute(self, raw_dict, K=20):
        # twap_dev = (TWAP - close) / close ≈ -BIAS/100
        # Flip sign so positive = close above average (overextended)
        bias_daily = -raw_dict['twap_dev']  # [N, T]
        N, T = bias_daily.shape
        device = bias_daily.device

        # Time-decay weighted rolling sum:
        #   weight_j = j/K for j=1..K (most recent gets weight 1.0)
        weights = torch.arange(1, K + 1, dtype=torch.float32, device=device) / K
        weight_sum = weights.sum()

        # Pad and unfold
        pad = torch.zeros(N, K - 1, device=device)
        padded = torch.cat([pad, bias_daily], dim=1)
        windows = padded.unfold(1, K, 1)  # [N, T, K]

        # Weighted average
        result = (windows * weights).sum(dim=-1) / weight_sum  # [N, T]

        return self.robust_norm(result)


class AttentionSpillover(FactorBase):
    """Attention spillover (注意力溢出).

    ATTN_i = rolling variance of abnormal returns (r_i - r_market)²
    SPILL = mean(ATTN_all) - ATTN_i

    Positive SPILL → peers getting more attention → attention
    will spill over → future outperformance.

    Reference: Chen et al., 2023, JF; 陈升锐, 2024, 中信建投.
    """
    name = 'ATTN_SPILL'
    frequency = '1d'
    data_keys = ['close']
    category = 'mean_reversion'
    description = 'Attention spillover (market attention vs self, 20d)'

    def compute(self, raw_dict, K=20):
        close = raw_dict['close']
        prev = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        ret = torch.log(close / (prev + 1e-9))
        r_bar = ret.mean(dim=0, keepdim=True).expand_as(ret)
        abnormal_sq = (ret - r_bar) ** 2
        attn = self.rolling_mean(abnormal_sq, window=K)
        attn_mean = attn.mean(dim=0, keepdim=True).expand_as(attn)
        spill = attn_mean - attn
        return self.robust_norm(spill)


class AbnormalNighttimeReversal(FactorBase):
    """Abnormal nighttime reversal frequency (反向日内逆转异常频率).

    NR = fraction of days with overnight > 0 AND intraday < 0.
    AB_NR = NR_short / NR_long — abnormality vs 1-year baseline.

    High AB_NR → intensifying tug-of-war between overnight and
    intraday traders → price correction overshooting.

    Reference: Cheema et al., 2022, Pacific-Basin Finance Journal.
    """
    name = 'AB_NIGHT_REV'
    frequency = '1d'
    data_keys = ['open', 'close']
    category = 'mean_reversion'
    description = 'Abnormal overnight-reversal frequency (vs 1Y baseline)'

    def compute(self, raw_dict, K_short=20, K_long=240):
        o = raw_dict['open']
        c = raw_dict['close']

        prev_close = torch.cat([c[:, :1], c[:, :-1]], dim=1)
        overnight_ret = o / (prev_close + 1e-9) - 1.0
        intraday_ret = c / (o + 1e-9) - 1.0

        reversal = ((overnight_ret > 0) & (intraday_ret < 0)).float()

        nr_short = self.rolling_mean(reversal, window=K_short)
        K_long = min(K_long, c.shape[1] - 1)
        nr_long = self.rolling_mean(reversal, window=max(K_long, 60)) + 1e-6

        ab_nr = nr_short / nr_long
        return self.robust_norm(ab_nr)


class ProspectTK(FactorBase):
    """Prospect theory value TK (前景价值 TK).

    Uses past N days' returns as empirical distribution, applies
    Kahneman-Tversky value and probability weighting functions:

        v(x) = x^α if x≥0, -λ(-x)^α if x<0
        w+(P) = P^γ / (P^γ + (1-P)^γ)^(1/γ)
        w-(P) = P^δ / (P^δ + (1-P)^δ)^(1/δ)

    TK = Σ v(r_i) × π_i  (decision weights)

    Parameters: α=0.88, λ=2.25, γ=0.61, δ=0.69

    High TK → overvalued by prospect theory → future underperformance.

    Reference: Barberis, Mukherjee & Wang, 2016, RFS.
    """
    name = 'PROSPECT_TK'
    frequency = '1d'
    data_keys = ['close']
    category = 'mean_reversion'
    description = 'Prospect theory TK value (behavioral overvaluation, 60d)'

    def compute(self, raw_dict, N=60, alpha=0.88, lam=2.25, gamma=0.61, delta=0.69):
        close = raw_dict['close']  # [N_assets, T]
        N_assets, T = close.shape
        device = close.device

        # Daily returns
        prev = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        ret = torch.log(close / (prev + 1e-9))

        # Pad and unfold
        pad = torch.zeros(N_assets, N - 1, device=device)
        padded = torch.cat([pad, ret], dim=1)
        windows = padded.unfold(1, N, 1)  # [N_assets, T, N]

        # Sort returns within each window
        sorted_ret, _ = windows.sort(dim=-1)  # ascending

        # Value function: v(x) = x^α if x≥0, -λ(-x)^α if x<0
        pos_mask = sorted_ret >= 0
        v = torch.where(
            pos_mask,
            sorted_ret.abs().pow(alpha),
            -lam * sorted_ret.abs().pow(alpha)
        )

        # Probability weighting
        # Each return has probability 1/N
        # Cumulative probabilities
        probs = torch.arange(1, N + 1, device=device, dtype=torch.float32) / N
        probs_prev = torch.arange(0, N, device=device, dtype=torch.float32) / N

        # w+(P) and w-(P)
        def w_plus(p):
            pg = p.pow(gamma)
            return pg / (pg + (1 - p).pow(gamma) + 1e-12).pow(1.0 / gamma)

        def w_minus(p):
            pd = p.pow(delta)
            return pd / (pd + (1 - p).pow(delta) + 1e-12).pow(1.0 / delta)

        # Decision weights π_i
        # For gains (positive sorted returns from right):
        # π_i = w+(P_i) - w+(P_{i-1}) where P counts from the right
        # For losses (negative sorted returns from left):
        # π_i = w-(P_i) - w-(P_{i-1}) where P counts from the left
        # Simplified: use uniform decision weights with probability distortion
        wp = w_plus(probs) - w_plus(probs_prev)
        wm = w_minus(probs) - w_minus(probs_prev)

        # Apply: gains use w+, losses use w-
        # For each return in sorted order, check if gain or loss
        pi = torch.where(pos_mask, wp, wm)  # [N_assets, T, N]

        # TK = Σ v(r_i) × π_i
        tk = (v * pi).sum(dim=-1)  # [N_assets, T]

        return self.robust_norm(tk)


class SalienceReturn(FactorBase):
    """Salience theory return (凸显性收益 STR).

    1. Salience coefficient: σ = |r - r̄| / (|r| + |r̄| + θ)
       where r̄ = cross-sectional median return, θ = 0.1
    2. Salience weight: ω = δ^rank (δ=0.7, rank by salience descending)
    3. STR = cov(ω, r) over rolling K-day window

    Captures investor overreaction to "salient" (extreme vs market)
    returns. High STR → over-attention → likely to reverse.

    Reference: Bordalo et al., 2012, QJE; 仲腊, 2022, 招商证券.
    """
    name = 'SALIENCE_RET'
    frequency = '1d'
    data_keys = ['close']
    category = 'mean_reversion'
    description = 'Salience theory return (attention-weighted cov, 20d)'

    def compute(self, raw_dict, K=20, theta=0.1, delta=0.7):
        close = raw_dict['close']  # [N, T]
        N, T = close.shape
        device = close.device

        # Daily returns
        prev = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        ret = torch.log(close / (prev + 1e-9))  # [N, T]

        # Cross-sectional median return per day
        r_bar = ret.median(dim=0, keepdim=True)[0].expand(N, T)  # [N, T]

        # Salience coefficient
        sigma = torch.abs(ret - r_bar) / (torch.abs(ret) + torch.abs(r_bar) + theta)

        # Per-day salience rank (descending: most salient = rank 1)
        # and weight = δ^rank
        weights = torch.zeros_like(sigma)
        for t in range(T):
            ranks = sigma[:, t].argsort(descending=True).argsort().float() + 1  # [N]
            weights[:, t] = delta ** ranks

        # Rolling covariance between weights and returns
        pad = torch.zeros(N, K - 1, device=device)
        w_pad = torch.cat([pad, weights], dim=1)
        r_pad = torch.cat([pad, ret], dim=1)

        w_win = w_pad.unfold(1, K, 1)  # [N, T, K]
        r_win = r_pad.unfold(1, K, 1)

        w_dm = w_win - w_win.mean(dim=-1, keepdim=True)
        r_dm = r_win - r_win.mean(dim=-1, keepdim=True)
        str_val = (w_dm * r_dm).mean(dim=-1)  # [N, T]

        return self.robust_norm(str_val)
