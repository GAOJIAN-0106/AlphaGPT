"""
Trend / time-series regression factors for commodity futures.

References:
    周遥, 2022. 商品期货Alpha因子构造. 中信期货.
"""

import torch
from model_core.factor_registry import FactorBase


class TSRegression(FactorBase):
    """Time-series regression trend factor (时序回归因子).

    Fits a quadratic regression to the past J days of close prices:
        close_t = a + b*t + c*t²

    Factor = b × R²

    Where b is the linear trend slope and R² is the coefficient of
    determination. The product ensures that only well-fitted trends
    (high R²) generate strong signals.

    Vectorized implementation using the normal equations for OLS.
    """
    name = 'TS_REGRESS'
    frequency = '1d'
    data_keys = ['close']
    category = 'trend'
    description = 'Quadratic trend regression slope × R² (J=20 lookback)'

    def compute(self, raw_dict, J=20):
        close = raw_dict['close']  # [N, T]
        N, T = close.shape
        device = close.device

        # Build time indices [1, 2, ..., J]
        t = torch.arange(1, J + 1, dtype=torch.float32, device=device)
        t_sq = t ** 2

        # Design matrix columns (centered for numerical stability)
        t_mean = t.mean()
        t_sq_mean = t_sq.mean()
        t_c = t - t_mean
        t_sq_c = t_sq - t_sq_mean

        # Precompute design matrix dot products (constant for all windows)
        # X = [1, t_c, t_sq_c], we need (X'X)^-1 X'y for each window
        # For efficiency, use the fact that centered [1, t, t²] are nearly orthogonal
        sum_tc2 = (t_c ** 2).sum()
        sum_tsqc2 = (t_sq_c ** 2).sum()
        sum_tc_tsqc = (t_c * t_sq_c).sum()

        # Pad and unfold
        pad = torch.zeros(N, J - 1, device=device)
        padded = torch.cat([pad, close], dim=1)
        windows = padded.unfold(1, J, 1)  # [N, T, J]

        # Center y within each window
        y_mean = windows.mean(dim=-1, keepdim=True)  # [N, T, 1]
        y_c = windows - y_mean  # [N, T, J]

        # Compute regression coefficients via closed-form
        # b (linear slope) using partial regression
        xy_t = (y_c * t_c).sum(dim=-1)      # [N, T]
        xy_tsq = (y_c * t_sq_c).sum(dim=-1)  # [N, T]

        # 2x2 system: [sum_tc2, sum_tc_tsqc; sum_tc_tsqc, sum_tsqc2] [b; c] = [xy_t; xy_tsq]
        det = sum_tc2 * sum_tsqc2 - sum_tc_tsqc ** 2 + 1e-12
        b = (sum_tsqc2 * xy_t - sum_tc_tsqc * xy_tsq) / det

        # R² = 1 - SS_res / SS_tot
        c = (sum_tc2 * xy_tsq - sum_tc_tsqc * xy_t) / det
        y_hat_c = b.unsqueeze(-1) * t_c + c.unsqueeze(-1) * t_sq_c  # [N, T, J]
        ss_res = ((y_c - y_hat_c) ** 2).sum(dim=-1)  # [N, T]
        ss_tot = (y_c ** 2).sum(dim=-1) + 1e-12       # [N, T]
        r_sq = torch.clamp(1.0 - ss_res / ss_tot, 0.0, 1.0)

        # Normalize b by price level to make cross-product comparable
        b_norm = b / (y_mean.squeeze(-1) + 1e-9)

        # Factor = b_normalized × R²
        factor = b_norm * r_sq

        # Zero out first J-1 bars (insufficient data)
        factor[:, :J - 1] = 0.0

        return self.robust_norm(factor)


class MAAlignment(FactorBase):
    """Moving average alignment (均线排列).

    sig = Σ sign(MA_i - MA_{i+1})

    For MA windows [5, 10, 20, 40, 60], counts how many adjacent
    pairs have the shorter MA above the longer MA.

    sig = +4 → perfect bull alignment (MA5 > MA10 > MA20 > MA40 > MA60)
    sig = -4 → perfect bear alignment
    sig ≈ 0  → mixed / ranging market

    Reference: 吴先兴, 2017, 天风证券.
    """
    name = 'MA_ALIGN'
    frequency = '1d'
    data_keys = ['close']
    category = 'trend'
    description = 'Moving average alignment (bull/bear arrangement of 5 MAs)'

    def compute(self, raw_dict):
        close = raw_dict['close']  # [N, T]
        windows = [5, 10, 20, 40, 60]

        mas = []
        for w in windows:
            mas.append(self.rolling_mean(close, window=w))

        # Count sign(MA_i - MA_{i+1}) for adjacent pairs
        sig = torch.zeros_like(close)
        for i in range(len(mas) - 1):
            sig = sig + torch.sign(mas[i] - mas[i + 1])

        return self.robust_norm(sig)
