"""
Term-structure factors for commodity futures.

Requires dual-contract data (near-month F1 + next-month F2) from DuckDBDataLoader.

References:
    Pindyck, R., 2004. Volatility and Commodity Price Dynamics.
    Journal of Futures Markets, 29, 1029-1047.
    王冬黎, 范海韵, 2022. 商品多因子模型框架再探究. 东证期货.
"""

import torch
from model_core.factor_registry import FactorBase


class TermSpread(FactorBase):
    """Log ratio of near-month to next-month contract price.

    Positive = backwardation (F1 > F2), negative = contango (F1 < F2).
    Captures convenience yield and storage cost dynamics.
    """
    name = 'TERM_SPREAD'
    frequency = '1d'
    data_keys = ['F1_close', 'F2_close']
    category = 'term_structure'
    description = 'ln(F1/F2), term structure slope'

    def compute(self, raw_dict):
        F1 = raw_dict['F1_close']  # [N, T]
        F2 = raw_dict['F2_close']  # [N, T]
        spread = torch.log(F1 / (F2 + 1e-9))
        return self.robust_norm(spread)


class PindyckVol(FactorBase):
    """Pindyck (2004) realized volatility from implied spot prices.

    1. Back out spot price:  P_t = F1_t * (F1_t / F2_t) ^ (n0_t / n1)
    2. Compute standardized log return:  r = (logP_t - logP_{t-1}) / (std_ratio)
    3. Rolling 25-day realized volatility:  σ = std(r, window=25)

    Key insight: using term structure to remove roll noise gives cleaner
    volatility that reflects true supply/demand shocks.
    """
    name = 'PINDYCK_VOL'
    frequency = '1d'
    data_keys = ['F1_close', 'F2_close', 'days_to_expiry', 'contract_gap_days']
    category = 'volatility'
    description = 'Pindyck term-structure implied realized volatility (25d)'

    def compute(self, raw_dict):
        F1 = raw_dict['F1_close']
        F2 = raw_dict['F2_close']
        n0 = raw_dict['days_to_expiry']       # [N, T] days to near-month expiry
        n1 = raw_dict['contract_gap_days']     # [N, T] gap between F1 and F2 expiry

        # Step 1: implied spot price
        ratio = F1 / (F2 + 1e-9)
        exponent = n0 / (n1 + 1e-9)
        # Clamp exponent to avoid numerical explosion
        exponent = torch.clamp(exponent, -3.0, 3.0)
        P_spot = F1 * torch.pow(torch.clamp(ratio, 0.5, 2.0), exponent)

        # Step 2: log returns
        log_P = torch.log(P_spot + 1e-9)
        log_ret = log_P[:, 1:] - log_P[:, :-1]
        log_ret = torch.cat([torch.zeros_like(log_ret[:, :1]), log_ret], dim=1)

        # Step 3: standardize by rolling 1-day std (ŝ₁)
        ret_std_1d = self.rolling_std(log_ret, window=5) + 1e-9
        r_standardized = log_ret / ret_std_1d

        # Step 4: 25-day rolling volatility
        vol = self.rolling_std(r_standardized, window=25)

        return self.robust_norm(vol)


class Carry(FactorBase):
    """Annualized carry (roll yield) from term structure.

    carry = (F1 - F2) / F2 * (365 / days_between_contracts)

    Positive carry = backwardation → historically profitable to be long.
    """
    name = 'CARRY'
    frequency = '1d'
    data_keys = ['F1_close', 'F2_close', 'contract_gap_days']
    category = 'term_structure'
    description = 'Annualized carry/roll yield from F1-F2 spread'

    def compute(self, raw_dict):
        F1 = raw_dict['F1_close']
        F2 = raw_dict['F2_close']
        gap_days = raw_dict['contract_gap_days']

        raw_carry = (F1 - F2) / (F2 + 1e-9)
        # Annualize
        ann_factor = 365.0 / (gap_days + 1e-9)
        # Clamp annualization to avoid blowup near expiry
        ann_factor = torch.clamp(ann_factor, 0.0, 30.0)
        carry = raw_carry * ann_factor

        return self.robust_norm(carry)


class Basis(FactorBase):
    """Annualized basis factor — rolling K-day mean of (main - spot) / spot.

    Formula (王冬黎, 2022):
        BASIS_t = -1/K × Σ_{i=t-K+1}^{t} [(P_main,i - P_spot,i) / P_spot,i × 365/M_main,i]

    We approximate P_spot using the Pindyck term-structure extrapolation:
        P_spot = F1 × (F1/F2)^(n0/n1)

    Negative sign convention: backwardation (spot > futures) → positive signal
    → go long backwardated, short contango. Captures roll yield via basis
    convergence.

    Differs from CARRY (which uses F1-F2 raw spread) by:
      1. Using implied spot instead of F2
      2. Averaging over K days (smoother, more stable)
    """
    name = 'BASIS'
    frequency = '1d'
    data_keys = ['F1_close', 'F2_close', 'days_to_expiry', 'contract_gap_days']
    category = 'term_structure'
    description = 'Rolling K-day annualized basis (main vs implied spot)'

    def compute(self, raw_dict, K=20):
        F1 = raw_dict['F1_close']       # P_main: main contract close [N, T]
        F2 = raw_dict['F2_close']
        n0 = raw_dict['days_to_expiry']       # M_main: days to expiry
        n1 = raw_dict['contract_gap_days']

        # Implied spot price via Pindyck extrapolation
        ratio = F1 / (F2 + 1e-9)
        exponent = n0 / (n1 + 1e-9)
        exponent = torch.clamp(exponent, -3.0, 3.0)
        P_spot = F1 * torch.pow(torch.clamp(ratio, 0.5, 2.0), exponent)

        # Annualized basis at each day: (P_main - P_spot) / P_spot × 365/M
        ann_factor = 365.0 / (n0 + 1e-9)
        ann_factor = torch.clamp(ann_factor, 0.0, 50.0)
        daily_basis = (F1 - P_spot) / (P_spot + 1e-9) * ann_factor

        # Rolling K-day mean (negative sign: backwardation = positive)
        basis = -self.rolling_mean(daily_basis, window=K)

        return self.robust_norm(basis)


class NearSubSpread(FactorBase):
    """Near-sub contract spread (主次价差因子).

    Formula:
        Factor_t = -1/K × Σ_{i=t-K+1}^{t}
            [(P_sub,i - P_main,i) / P_main,i × 365 / (M_sub,i - M_main,i)]

    Where P_main = F1 (near-month), P_sub = F2 (next-month),
    M_main/M_sub = days to expiry.

    Similar to CARRY but:
      1. Uses K-day rolling mean (smoother signal)
      2. Normalizes by F1 instead of F2
      3. Uses exact contract gap days instead of inter-expiry gap

    Long contango (sub > main) historically reverts → negative sign.
    """
    name = 'NEAR_SUB_SPREAD'
    frequency = '1d'
    data_keys = ['F1_close', 'F2_close', 'contract_gap_days']
    category = 'term_structure'
    description = 'Near-sub annualized spread (K=20 rolling mean)'

    def compute(self, raw_dict, K=20):
        F1 = raw_dict['F1_close']  # P_main
        F2 = raw_dict['F2_close']  # P_sub
        gap = raw_dict['contract_gap_days']  # M_sub - M_main

        # Daily annualized spread
        spread = -(F2 - F1) / (F1 + 1e-9) * 365.0 / (gap + 1e-9)
        spread = torch.clamp(spread, -50.0, 50.0)

        # Rolling K-day mean
        result = self.rolling_mean(spread, window=K)

        return self.robust_norm(result)


class BasisMomentum(FactorBase):
    """Basis momentum (基差动量).

    BasisMom = cumret(F1, R) - cumret(F2, R)
             = Π(1 + r_F1,i) - Π(1 + r_F2,i)  over R days

    Measures the change in term structure slope over R days.
    Positive → near-month outperforming far-month → steepening.

    Reference: Boons & Prado, 2019, JF; 冯佳睿, 2017, 海通证券.
    """
    name = 'BASIS_MOM'
    frequency = '1d'
    data_keys = ['F1_close', 'F2_close']
    category = 'term_structure'
    description = 'Basis momentum (F1 vs F2 cumulative return diff, R=20)'

    def compute(self, raw_dict, R=20):
        F1 = raw_dict['F1_close']  # [N, T]
        F2 = raw_dict['F2_close']

        # R-day cumulative return for each contract
        F1_prev = torch.cat([F1[:, :1].expand(-1, R), F1[:, :-R]], dim=1)
        F2_prev = torch.cat([F2[:, :1].expand(-1, R), F2[:, :-R]], dim=1)

        cum_ret_F1 = F1 / (F1_prev + 1e-9) - 1.0
        cum_ret_F2 = F2 / (F2_prev + 1e-9) - 1.0

        basis_mom = cum_ret_F1 - cum_ret_F2
        basis_mom[:, :R] = 0.0

        return self.robust_norm(basis_mom)
