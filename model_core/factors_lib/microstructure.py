"""
Market microstructure factors for commodity futures.

References:
    丁鲁明, 陈升锐, 2020. 高频量价选股因子初探. 中信建投证券.
    Corwin, S. & Schultz, P., 2012. A Simple Way to Estimate Bid-Ask Spreads
    from Daily High and Low Prices. Journal of Finance, 67(2), 719-760.
    赵明, 高智威, 第子瑞, 2024. 基于日内高频博弈信息的商品CTA策略. 国金证券.
"""

import torch
from model_core.factor_registry import FactorBase


class MarketPriceBias(FactorBase):
    """Market price bias (市价偏离度) — proxy using OHLC data.

    Original formula (requires tick data):
        MPB_t = VWAP_t - MidPrice_t
    where MidPrice = (Bid + Ask) / 2.

    OHLC approximation:
        MPB_t ≈ VWAP_t - (High_t + Low_t) / 2

    We use the DuckDB loader's precomputed TWAP (= AVG of 1-min close prices)
    as the VWAP proxy, and (High + Low) / 2 as the mid-price proxy.

    Interpretation:
        MPB > 0 → trades execute above mid → sell pressure, bearish
        MPB < 0 → trades execute below mid → buy pressure, bullish

    The factor is averaged over K days for stability, matching the
    time-decay methodology from the original paper.
    """
    name = 'MPB'
    frequency = '1d'
    data_keys = ['close', 'high', 'low']
    category = 'microstructure'
    description = 'Market price bias (VWAP vs HL midpoint), order flow proxy'

    def compute(self, raw_dict, K=20):
        close = raw_dict['close']  # [N, T]
        high = raw_dict['high']
        low = raw_dict['low']

        # Mid-price proxy: (High + Low) / 2
        mid = (high + low) / 2.0

        # MPB = close position relative to mid, scaled by range
        # This captures where the close sits vs the HL midpoint
        hl_range = high - low + 1e-9
        mpb_daily = (close - mid) / hl_range

        # Rolling K-day mean for stability
        mpb = self.rolling_mean(mpb_daily, window=K)

        return self.robust_norm(mpb)


class BuyWillingness(FactorBase):
    """Buy willingness factor (买入意愿因子).

    Original formula (requires L1 order book snapshots):
        buy_sell = count(bid > ask) / count(all snapshots)

    1-minute K-line approximation:
        buy_sell ≈ count(close > open) / count(all 1min bars)

    A 1-min bar closing above its open indicates buy-side dominance
    during that minute. The daily ratio of such bars proxies the
    fraction of time buyers are in control.

    Higher ratio → stronger buy willingness → bullish signal.
    Uses 20-day rolling mean for monthly stability.
    """
    name = 'BUY_WILL'
    frequency = '1d'
    data_keys = ['buy_ratio']
    category = 'microstructure'
    description = 'Buy willingness (1min close>open ratio), order flow proxy'

    def compute(self, raw_dict, K=20):
        buy_ratio = raw_dict['buy_ratio']  # [N, T] in ~[0.2, 0.5]

        # Center around 0.5 (neutral)
        centered = buy_ratio - 0.5

        # Rolling K-day mean
        result = self.rolling_mean(centered, window=K)

        return self.robust_norm(result)


class DazzleReturn(FactorBase):
    """Dazzle return (耀眼收益率).

    Return during volume surge moments (vol_diff > mean + std).
    Computed from 1-min data, cached in parquet.

    Step 4: subtract cross-sectional mean, take absolute value.
    Uses 20-day rolling mean for monthly factor.

    Negative correlation with future returns when large.

    Reference: 曹有晗, 2022, 方正证券.
    """
    name = 'DAZZLE_RET'
    frequency = '1d'
    data_keys = ['dazzle_ret']
    category = 'microstructure'
    description = 'Dazzle return (avg return during volume surges, 20d)'

    def compute(self, raw_dict, K=20):
        dr = raw_dict['dazzle_ret']  # [N, T]
        # Step 4: subtract cross-sectional mean, take abs
        cs_mean = dr.mean(dim=0, keepdim=True).expand_as(dr)
        moderate = (dr - cs_mean).abs()
        # Rolling K-day mean
        result = self.rolling_mean(moderate, window=K)
        return self.robust_norm(result)


class VolumeRatio(FactorBase):
    """Volume ratio (成交量比率) — up-volume / down-volume.

    VR = Σ Vol(close > open) / Σ Vol(close ≤ open)
    from 1-minute data, computed in SQL.

    VR > 1 → more volume on up bars → buy pressure dominates.
    Uses 20-day time-decay weighted rolling mean.

    Reference: 张嘉颖, 洛升银, 2024, 中信建投.
    """
    name = 'VOL_RATIO'
    frequency = '1d'
    data_keys = ['vol_ratio']
    category = 'microstructure'
    description = 'Volume ratio (up-vol / down-vol from 1min, 20d decay)'

    def compute(self, raw_dict, K=20):
        vr = raw_dict['vol_ratio']  # [N, T]
        N, T = vr.shape
        device = vr.device

        # Log transform for symmetry (VR=2 and VR=0.5 are equidistant from neutral)
        log_vr = torch.log(vr + 1e-9)

        # Time-decay weighted rolling mean
        weights = torch.arange(1, K + 1, dtype=torch.float32, device=device) / K
        weight_sum = weights.sum()
        pad = torch.zeros(N, K - 1, device=device)
        padded = torch.cat([pad, log_vr], dim=1)
        windows = padded.unfold(1, K, 1)
        result = (windows * weights).sum(dim=-1) / weight_sum

        return self.robust_norm(result)


class InflectionPoint(FactorBase):
    """Price inflection point factor (价格拐点).

    inflection_point_t = inflection_t - mean(inflection_{t-5..t-1})

    inflection_ratio = fraction of 1-min bars where price direction
    reverses (bar_move * prev_bar_move < 0). Computed in SQL.

    The factor is the deviation of today's inflection ratio from
    its recent 5-day average. High deviation → sudden increase in
    market disagreement → bearish.

    Reference: 赵明 et al., 2024, 国金证券.
    """
    name = 'INFLECTION'
    frequency = '1d'
    data_keys = ['inflection_ratio']
    category = 'microstructure'
    description = 'Price inflection point deviation (1min direction changes)'

    def compute(self, raw_dict):
        ratio = raw_dict['inflection_ratio']  # [N, T]
        avg_5d = self.rolling_mean(ratio, window=5)
        deviation = ratio - avg_5d
        return self.robust_norm(deviation)


class LargeOrderImpact(FactorBase):
    """Large order impact (大单影响力).

    Average return during high-volume (>90th percentile) 1-minute bars.
    Computed in SQL from 1-min data:
        r_large = mean(return_i) for bars where volume >= P90

    Positive → large orders push price up → informed buying → bullish.
    Negative → large orders push price down → informed selling → bearish.

    Uses 20-day rolling mean for stability.

    Reference: 赵明 et al., 2024, 国金证券.
    """
    name = 'LARGE_ORDER'
    frequency = '1d'
    data_keys = ['large_order_ret']
    category = 'microstructure'
    description = 'Large order price impact (avg return during P90 volume bars)'

    def compute(self, raw_dict, K=20):
        lor = raw_dict['large_order_ret']  # [N, T]

        # Rolling K-day mean for stability
        result = self.rolling_mean(lor, window=K)

        return self.robust_norm(result)
