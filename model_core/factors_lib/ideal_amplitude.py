"""
Ideal Amplitude factor (理想振幅因子).

For each asset, over a rolling N-day window:
1. Compute daily amplitude: amp_t = high_t / low_t - 1
2. Rank days by close price within the window
3. V_high(λ) = mean amplitude of days with close in top λ quantile
4. V_low(λ)  = mean amplitude of days with close in bottom λ quantile
5. V = V_high - V_low

Rationale: splits amplitude by price regime. High-price days tend to have
larger amplitude (negative selection ability), so V > 0 is bearish.

Reference: 魏建榕, 2020, 振幅因子的隐藏结构, 开源证券.
"""

import torch
from model_core.factor_registry import FactorBase


class IdealAmplitude(FactorBase):
    """Ideal amplitude = mean_amp(high-close days) - mean_amp(low-close days)."""

    name = 'IDEAL_AMP'
    frequency = '1d'
    data_keys = ['high', 'low', 'close']
    category = 'volatility'
    description = 'Ideal amplitude factor: amp split by close rank (N=20, λ=0.25)'

    def compute(self, raw_dict, N=20, lam=0.25):
        high = raw_dict['high']   # [N_assets, T]
        low = raw_dict['low']
        close = raw_dict['close']

        N_assets, T = close.shape
        device = close.device

        # Daily amplitude
        amp = high / (low + 1e-9) - 1.0  # [N_assets, T]

        # Pad for rolling window
        pad_amp = torch.zeros(N_assets, N - 1, device=device)
        pad_close = torch.zeros(N_assets, N - 1, device=device)
        amp_padded = torch.cat([pad_amp, amp], dim=1)
        close_padded = torch.cat([pad_close, close], dim=1)

        # Unfold into rolling windows
        amp_win = amp_padded.unfold(1, N, 1)     # [N_assets, T, N]
        close_win = close_padded.unfold(1, N, 1)  # [N_assets, T, N]

        # Rank close within each window (0-based rank / (N-1) → [0, 1])
        # argsort twice gives ranks
        ranks = close_win.argsort(dim=-1).argsort(dim=-1).float()  # [N_assets, T, N]
        quantile_pos = ranks / (N - 1)  # normalize to [0, 1]

        # Top λ (high close days) and bottom λ (low close days)
        top_k = max(int(N * lam), 1)
        high_mask = quantile_pos >= (1.0 - lam)  # top 25%
        low_mask = quantile_pos <= lam            # bottom 25%

        # Masked mean amplitude
        # Replace False positions with NaN for nanmean-like behavior
        amp_high = amp_win.clone()
        amp_high[~high_mask] = 0.0
        v_high = amp_high.sum(dim=-1) / (high_mask.float().sum(dim=-1) + 1e-9)

        amp_low = amp_win.clone()
        amp_low[~low_mask] = 0.0
        v_low = amp_low.sum(dim=-1) / (low_mask.float().sum(dim=-1) + 1e-9)

        # Ideal amplitude
        ideal_amp = v_high - v_low  # [N_assets, T]

        return self.robust_norm(ideal_amp)
