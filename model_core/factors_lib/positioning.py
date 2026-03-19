"""
Positioning / hedging pressure factors for commodity futures.

References:
    张革, 2022. 商品期货截面风格因子初探. 中信期货.
    Fan, J.H. and Zhang, T., 2019. The untold story of commodity futures
    in China. Journal of Futures Markets, 40(4), 671-706.
"""

import torch
from model_core.factor_registry import FactorBase


class HedgingPressure(FactorBase):
    """Hedging pressure ratio (套期保值比例).

    Formula:
        HP_t = |OI_t - OI_{t-J}| / Σ(j=0..J-1) Vol_{t-j}

    Since we don't have absolute OI in raw_data_cache, we reconstruct
    relative OI changes from oi_change (daily delta) and use cumulative
    sums. The absolute difference over J days = |Σ oi_change over J days|.

    J=243 (≈1 trading year) per the paper, but we also support shorter
    lookbacks since data may be limited.

    Higher HP → more hedger activity → speculators earn risk premium.
    """
    name = 'HEDGE_PRESS'
    frequency = '1d'
    data_keys = ['oi_change', 'volume']
    category = 'positioning'
    description = 'Hedging pressure ratio (OI change / cumulative volume, ~1Y)'

    def compute(self, raw_dict, J=243):
        oi_change = raw_dict['oi_change']  # [N, T] daily OI change
        volume = raw_dict['volume']        # [N, T] daily volume
        N, T = oi_change.shape
        device = oi_change.device

        # Clamp J to available data
        J = min(J, T - 1)
        if J < 20:
            return torch.zeros(N, T, device=device)

        # |OI_t - OI_{t-J}| = |cumsum of oi_change over [t-J+1, t]|
        # Use rolling sum of oi_change over J days
        pad_oi = torch.zeros(N, J - 1, device=device)
        oi_padded = torch.cat([pad_oi, oi_change], dim=1)
        oi_windows = oi_padded.unfold(1, J, 1)  # [N, T, J]
        abs_oi_delta = oi_windows.sum(dim=-1).abs()  # [N, T]

        # Σ Vol over J days
        pad_vol = torch.zeros(N, J - 1, device=device)
        vol_padded = torch.cat([pad_vol, volume], dim=1)
        vol_windows = vol_padded.unfold(1, J, 1)
        cum_vol = vol_windows.sum(dim=-1)  # [N, T]

        # HP = |ΔOI_J| / Σ Vol_J
        hp = abs_oi_delta / (cum_vol + 1e-9)

        return self.robust_norm(hp)


class OIChangeRate(FactorBase):
    """Open interest log change rate (持仓量变动因子).

    Formula:
        ΔOI_t = ln(OI_t) - ln(OI_{t-J})

    Since absolute OI is not in raw_data_cache, we reconstruct it
    from cumulative oi_change (daily delta). The J-day change of
    cumulative OI approximates the log ratio when changes are moderate.

    Long assets with growing OI (more participation) → bullish.
    OI growth reflects macro activity and price expectations.

    Reference: Hong & Yogo, 2012, JFE.
    """
    name = 'OI_CHANGE_RATE'
    frequency = '1d'
    data_keys = ['oi_change']
    category = 'positioning'
    description = 'OI log change rate (J=20), participation growth signal'

    def compute(self, raw_dict, J=20):
        oi_change = raw_dict['oi_change']  # [N, T] daily OI delta
        N, T = oi_change.shape

        # Reconstruct relative OI via cumulative sum
        oi_cum = oi_change.cumsum(dim=1)

        # J-day change: OI_cum_t - OI_cum_{t-J}
        oi_prev = torch.cat([
            torch.zeros(N, J, device=oi_change.device),
            oi_cum[:, :-J]
        ], dim=1)
        delta_oi = oi_cum - oi_prev

        # Zero out first J bars
        delta_oi[:, :J] = 0.0

        return self.robust_norm(delta_oi)


class OILevel(FactorBase):
    """Open interest level factor (持仓总量因子).

    OI_T / MA(OI, K) - 1

    Current OI relative to its K-day moving average.
    Measures capital flow intensity — OI above average indicates
    growing participation / attention on this product.

    Reference: 王冬黎, 范海韵, 2022, 东证期货.
    """
    name = 'OI_LEVEL'
    frequency = '1d'
    data_keys = ['oi_change']
    category = 'positioning'
    description = 'OI level relative to K-day average (capital flow intensity)'

    def compute(self, raw_dict, K=20):
        oi_change = raw_dict['oi_change']  # [N, T] daily delta

        # Reconstruct relative OI via cumsum (add large offset for positivity)
        oi_cum = oi_change.cumsum(dim=1) + 1e6

        # K-day moving average of OI
        ma_oi = self.rolling_mean(oi_cum, window=K)

        # OI / MA(OI) - 1
        oi_ratio = oi_cum / (ma_oi + 1e-9) - 1.0

        return self.robust_norm(oi_ratio)


class PriceOICorrelation(FactorBase):
    """Price-OI correlation (仓价相关性).

    Daily Pearson correlation between 1-min close price and close_oi.
    Computed from 1-min data, cached in parquet.

    ρ > 0 → price up + OI up = bullish (longs adding)
    ρ < 0 → price up + OI down = bearish (shorts covering)

    Uses 20-day rolling mean.

    Reference: 冯佳睿, 2018, 海通证券.
    """
    name = 'PRICE_OI_CORR'
    frequency = '1d'
    data_keys = ['price_oi_corr']
    category = 'positioning'
    description = 'Price-OI intraday correlation (1min, 20d mean)'

    def compute(self, raw_dict, K=20):
        corr = raw_dict['price_oi_corr']
        result = self.rolling_mean(corr, window=K)
        return self.robust_norm(result)
