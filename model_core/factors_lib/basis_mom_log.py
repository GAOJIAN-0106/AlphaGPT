"""
Log basis momentum factor for commodity futures.

Formula:
    BN_t = log(F1_t / F1_{t-J}) - log(F2_t / F2_{t-J})

where F1 = near-month contract, F2 = far-month contract, J = 243 days.

Captures long-term changes in term structure slope. Strong basis momentum
means near-month has outperformed far-month over ~1 year, which tends
to revert as arbitrageurs participate and the near-far spread mean-reverts.

References:
    Boons M, Prado M P, 2019. Basis Momentum. Journal of Finance, 74(1), 239-279.
    周涵, 2022. 商品期货 alpha 因子精选. 中信期货.
"""

import torch
from model_core.factor_registry import FactorBase


class BasisMomentumLog(FactorBase):
    """Log basis momentum (基差动量对数版).

    BN_t = log(F1_t/F1_{t-J}) - log(F2_t/F2_{t-J}), J=243.

    Positive → near-month outperformed far-month over 1Y → steepening.
    Paper finds J=243 optimal; shorter J captured by existing BASIS_MOM (R=20).
    """
    name = 'BASIS_MOM_LOG'
    frequency = '1d'
    data_keys = ['F1_close', 'F2_close']
    category = 'term_structure'
    description = 'Log basis momentum (F1 vs F2 log-return diff, J=243)'

    def compute(self, raw_dict, J=243):
        F1 = raw_dict['F1_close']  # [N, T]
        F2 = raw_dict['F2_close']

        # J-day lagged values
        F1_lag = torch.cat([F1[:, :1].expand(-1, J), F1[:, :-J]], dim=1)
        F2_lag = torch.cat([F2[:, :1].expand(-1, J), F2[:, :-J]], dim=1)

        # Log returns
        log_ret_F1 = torch.log(F1 / (F1_lag + 1e-9) + 1e-9)
        log_ret_F2 = torch.log(F2 / (F2_lag + 1e-9) + 1e-9)

        bn = log_ret_F1 - log_ret_F2
        bn[:, :J] = 0.0  # no valid lookback in first J days

        return self.robust_norm(bn)
