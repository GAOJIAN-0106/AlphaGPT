"""
Fundamental factors for commodity futures (基本面因子).

Data sources: TianQin EDB (economic database) via cached parquet files.
These factors use non-price data like warehouse receipts, spot prices,
and production costs.

Reference:
    王冬黎, 常海锋, 2022. 商品多因子模型框架再探究. 东证期货.
"""

import torch
from model_core.factor_registry import FactorBase


class WarehouseChange(FactorBase):
    """Warehouse receipt change factor (仓单因子).

    Formula:
        S(L,K) = Σ(t=T-L+1..T) Warehouse_t / Σ(t=T-L-K+1..T-L) Warehouse_t - 1

    This is the ratio of recent L-day sum vs prior K-day sum, minus 1.
    Equivalent to the period-over-period change rate of warehouse receipts.

    Warehouse receipt decline → inventory drawdown → supply tightening → bullish.
    So we negate the raw ratio: negative change (drawdown) → positive signal.

    Default: L=20, K=20 (compare recent 20d vs prior 20d).
    """
    name = 'WAREHOUSE_CHG'
    frequency = '1d'
    data_keys = ['warehouse']
    category = 'fundamental'
    description = 'Warehouse receipt period-over-period change (L=20, K=20)'

    def compute(self, raw_dict, L=20, K=20):
        wh = raw_dict['warehouse']  # [N, T]
        N, T = wh.shape
        device = wh.device
        window = L + K

        if T < window + 1:
            return torch.zeros(N, T, device=device)

        # Rolling sum of recent L days
        pad = torch.zeros(N, L - 1, device=device)
        wh_pad = torch.cat([pad, wh], dim=1)
        recent_sum = wh_pad.unfold(1, L, 1).sum(dim=-1)  # [N, T]

        # Rolling sum of prior K days (shifted by L)
        pad2 = torch.zeros(N, window - 1, device=device)
        wh_pad2 = torch.cat([pad2, wh], dim=1)
        # prior K days = [T-L-K+1 .. T-L], shift the window
        prior_sum = wh_pad2.unfold(1, K, 1).sum(dim=-1)  # [N, T+L]
        # Align: prior_sum at position t corresponds to sum ending at t,
        # we need sum ending at t-L, so shift by L
        if prior_sum.shape[1] > T:
            prior_sum = prior_sum[:, :T]
        # Actually: we need prior_sum to be the K-day sum ending L days before
        # recent_sum position. Let's recompute properly.
        # At time T: recent = sum(T-L+1..T), prior = sum(T-L-K+1..T-L)
        # prior_sum uses same unfold but we take it L steps earlier
        full_pad = torch.zeros(N, window - 1, device=device)
        wh_full = torch.cat([full_pad, wh], dim=1)
        # For each position t (0-indexed in original), we need:
        #   recent = wh[t-L+1:t+1].sum()  (L values ending at t)
        #   prior  = wh[t-L-K+1:t-L+1].sum()  (K values ending at t-L)
        # Using unfold on wh_full (padded by window-1):
        #   recent at pos t in wh = unfold position (t + window-1 - L + 1) with size L
        #   Simpler: compute two rolling sums and shift
        rolling_L = wh_full.unfold(1, L, 1).sum(dim=-1)   # [N, T+K]
        rolling_K = wh_full.unfold(1, K, 1).sum(dim=-1)   # [N, T+L]

        # rolling_L[t] = sum of L values ending at position t in wh_full
        # For original wh position t, rolling_L corresponds to index t+K (after K padding)
        # rolling_K[t] = sum of K values ending at position t in wh_full
        # For prior sum at original t, we need K-sum ending at t-L,
        # which is rolling_K at index t+K-L+L = wait, let me simplify.

        # Direct approach: just use two unfold on the original tensor with proper padding
        # recent_L: rolling sum of last L bars
        pad_L = torch.zeros(N, L - 1, device=device)
        recent = torch.cat([pad_L, wh], dim=1).unfold(1, L, 1).sum(dim=-1)  # [N, T]

        # prior_K: rolling sum of K bars ending L bars ago
        # = shift recent by L, but with K-width window
        pad_LK = torch.zeros(N, L + K - 1, device=device)
        all_k_sums = torch.cat([pad_LK, wh], dim=1).unfold(1, K, 1).sum(dim=-1)  # [N, T+L]
        prior = all_k_sums[:, :T]  # K-sum ending L bars before position t

        # S = recent / prior - 1
        ratio = recent / (prior + 1e-9) - 1.0

        # Negate: warehouse decline → positive signal (bullish)
        factor = -ratio

        # Zero out warm-up period
        factor[:, :window] = 0.0

        return self.robust_norm(factor)


class InventoryMomentum(FactorBase):
    """Social inventory momentum factor (库存因子).

    Formula:
        S(K) = 1/K × Σ(t=T-K+1..T) (stock_t / stock_{t-1} - 1)

    Rolling average of daily inventory change rate over K days.
    Measures the speed of inventory drawdown or buildup.

    Inventory decrease (drawdown) → supply tightening → bullish.
    So we negate: negative inventory momentum → positive signal.

    Data: EDB 期货库存 (weekly, ffill to daily).
    """
    name = 'INVENTORY_MOM'
    frequency = '1d'
    data_keys = ['inventory']
    category = 'fundamental'
    description = 'Social inventory momentum (K=20 avg daily change rate)'

    def compute(self, raw_dict, K=20):
        inv = raw_dict['inventory']  # [N, T]
        N, T = inv.shape
        device = inv.device

        if T < K + 2:
            return torch.zeros(N, T, device=device)

        # Daily change rate: stock_t / stock_{t-1} - 1
        inv_prev = torch.cat([inv[:, :1], inv[:, :-1]], dim=1)
        daily_chg = inv / (inv_prev + 1e-9) - 1.0

        # Clamp extreme daily changes (data glitches)
        daily_chg = torch.clamp(daily_chg, -0.5, 0.5)
        daily_chg[:, 0] = 0.0  # first bar has no valid change

        # K-day rolling mean of daily change rate
        avg_chg = self.rolling_mean(daily_chg, window=K)

        # Negate: inventory decline → positive signal
        factor = -avg_chg

        return self.robust_norm(factor)
