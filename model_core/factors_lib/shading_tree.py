"""
"午蔽古木" (Shading Tree) factor for commodity futures.

Measures the degree to which fundamental/long-term information (vs noise)
drives price changes. Small factor value → price driven by noise (F-all small)
and sudden information shocks don't move price → future alpha opportunity.

Computed from 1-min data via OLS regression of minute returns on lagged
volume diffs. Precomputed in scripts/compute_shading_tree.py, cached as parquet.

Reference:
    曹有梅, 2023. 驱动个股价格变化的因素分解与"花隐仙间"因子. 方正证券.
"""

import torch
from model_core.factor_registry import FactorBase


class ShadingTree(FactorBase):
    """Shading tree factor (午蔽古木).

    Uses precomputed daily values from shading_tree_cache.parquet.
    Applies 20-day rolling mean for stability.

    Factor interpretation:
        Low value (negative) → noise-driven, low information efficiency → bullish
        High value (positive) → information-driven → less alpha
    """
    name = 'SHADING_TREE'
    frequency = '1d'
    data_keys = ['shading_tree']
    category = 'microstructure'
    description = 'Shading tree (OLS intercept t-stat adjusted by F-all, 20d rolling)'

    def compute(self, raw_dict, K=20):
        st = raw_dict['shading_tree']  # [N, T]

        # Rolling K-day mean for stability
        result = self.rolling_mean(st, window=K)

        return self.robust_norm(result)
