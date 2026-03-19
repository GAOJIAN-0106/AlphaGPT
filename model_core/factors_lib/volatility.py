"""
Volatility factors for commodity futures.

References:
    王冬黎, 范海韵, 2022. 商品多因子模型框架再探究. 东证期货.
"""

import torch
from model_core.factor_registry import FactorBase


class SimpleVol(FactorBase):
    """Annualized realized volatility (简单波动率).

    Formula:
        σ_t = √(252 × 1/K × Σ_{i=t-K+1}^{t} (r_i - r̄)²)

    Where r_i is the daily log return and r̄ is the mean return
    over the K-day window.

    Differs from VOL_CLUSTER (which uses √(mean(r²)) without demeaning)
    by removing the mean return component, giving a purer measure of
    dispersion. This matters when assets have strong trends — VOL_CLUSTER
    conflates trend strength with volatility, while SimpleVol isolates
    the uncertainty component.

    Long high-vol / short low-vol earns volatility risk premium.
    """
    # Disabled: r=0.71 with VOL_CLUSTER after robust_norm. Kept for reference.
    name = ''  # empty name prevents auto-registration


class IdiosyncraticVol(FactorBase):
    """Idiosyncratic volatility (特质波动率).

    For each asset, regress daily returns on the equal-weighted market
    return (cross-sectional mean). The rolling std of the residuals
    = idiosyncratic volatility.

    r_{i,t} = α + β × r_market,t + ε_{i,t}
    IVol_i = rolling_std(ε, K)

    Low IVol → more predictable / less noisy → tends to outperform.

    Reference: Fuertes, Miffre & Fernández-Pérez, 2015, JFM.
    """
    name = 'IVOL'
    frequency = '1d'
    data_keys = ['close']
    category = 'volatility'
    description = 'Idiosyncratic volatility (residual vol after market beta, 60d)'

    def compute(self, raw_dict, K=60):
        close = raw_dict['close']  # [N, T]
        N, T = close.shape
        device = close.device

        # Daily returns
        prev = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        ret = torch.log(close / (prev + 1e-9))  # [N, T]

        # Market return = equal-weighted mean across assets
        mkt_ret = ret.mean(dim=0, keepdim=True).expand(N, T)  # [N, T]

        # Rolling regression: for each asset, regress ret on mkt_ret
        # β = cov(r, mkt) / var(mkt), residual = r - β × mkt
        pad = torch.zeros(N, K - 1, device=device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mkt_pad = torch.cat([pad.mean(dim=0, keepdim=True).expand(N, K - 1), mkt_ret], dim=1)

        ret_win = ret_pad.unfold(1, K, 1)   # [N, T, K]
        mkt_win = mkt_pad.unfold(1, K, 1)   # [N, T, K]

        # β = cov / var
        ret_dm = ret_win - ret_win.mean(dim=-1, keepdim=True)
        mkt_dm = mkt_win - mkt_win.mean(dim=-1, keepdim=True)
        cov = (ret_dm * mkt_dm).mean(dim=-1)
        var_mkt = (mkt_dm ** 2).mean(dim=-1) + 1e-12
        beta = cov / var_mkt  # [N, T]

        # Residuals within each window
        resid = ret_win - beta.unsqueeze(-1) * mkt_win  # [N, T, K]
        ivol = resid.std(dim=-1)  # [N, T]

        return self.robust_norm(ivol)


class RogersSatchellVol(FactorBase):
    """Rogers-Satchell volatility estimator.

    σ_RS = √(252/K × Σ [h(h-c) - l(l-c)])
    where h = ln(H/O), c = ln(C/O), l = ln(L/O)

    Uses full OHLC to estimate volatility, more efficient than
    close-to-close estimators. Robust to drift (trending markets).

    Reference: 王冬黎, 范海韵, 2022, 东证期货.
    """
    name = 'RS_VOL'
    frequency = '1d'
    data_keys = ['open', 'high', 'low', 'close']
    category = 'volatility'
    description = 'Rogers-Satchell OHLC volatility estimator (K=20)'

    def compute(self, raw_dict, K=20):
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        c = raw_dict['close']

        # Log ratios relative to open
        h_log = torch.log(h / (o + 1e-9))
        c_log = torch.log(c / (o + 1e-9))
        l_log = torch.log(l / (o + 1e-9))

        # RS variance per bar: h(h-c) - l(l-c)
        rs_var = h_log * (h_log - c_log) - l_log * (l_log - c_log)
        rs_var = torch.clamp(rs_var, min=0.0)  # ensure non-negative

        # Rolling K-day mean, annualized
        rs_mean = self.rolling_mean(rs_var, window=K)
        rs_vol = torch.sqrt(252.0 * rs_mean + 1e-12)

        return self.robust_norm(rs_vol)


class ShadowVolatility(FactorBase):
    """Shadow (wick) volatility — std of daily shadow/close ratio.

    shadow_ratio_t = (upper_shadow + lower_shadow) / close
    factor = rolling_std(shadow_ratio, K=20)

    Upper shadow = high - max(open, close)
    Lower shadow = min(open, close) - low

    Shadows reflect intraday rejection / reversal attempts.
    High std of shadow ratio → unstable market microstructure,
    aggressive gaming between bulls and bears → bearish signal.

    Reference: 王路, 2023, JASON's alpha, 东北证券.
    """
    name = 'SHADOW_VOL'
    frequency = '1d'
    data_keys = ['open', 'high', 'low', 'close']
    category = 'volatility'
    description = 'Shadow line volatility (20d std of wick/close ratio)'

    def compute(self, raw_dict, K=20):
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        c = raw_dict['close']

        upper_shadow = h - torch.maximum(o, c)
        lower_shadow = torch.minimum(o, c) - l
        shadow_ratio = (upper_shadow + lower_shadow) / (c + 1e-9)

        # Rolling K-day std
        result = self.rolling_std(shadow_ratio, window=K)

        return self.robust_norm(result)
    frequency = '1d'
    data_keys = ['close']
    category = 'volatility'
    description = 'Annualized realized volatility (K=20, demeaned)'

    def compute(self, raw_dict, K=20):
        close = raw_dict['close']  # [N, T]

        # Daily log returns
        prev = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        ret = torch.log(close / (prev + 1e-9))

        # Rolling K-day variance (demeaned)
        pad = torch.zeros(ret.shape[0], K - 1, device=ret.device)
        padded = torch.cat([pad, ret], dim=1)
        windows = padded.unfold(1, K, 1)  # [N, T, K]

        # Annualized volatility
        vol = torch.sqrt(252.0 * windows.var(dim=-1) + 1e-12)

        return self.robust_norm(vol)
