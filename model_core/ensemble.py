"""
Formula Ensemble Module — Multi-strategy aggregation for AlphaGPT.

Inspired by the Kaggle Jane Street competition approach:
multiple seeds → average predictions → reduced variance.
"""

import torch
import numpy as np
from .vm import StackVM


class FormulaEnsemble:
    """
    Holds N factor formulas and aggregates their signals.

    Supports two aggregation modes:
      - 'mean': simple weighted average of sigmoid signals
      - 'rank_mean': average of per-formula rank-normalized signals
        (more robust to scale differences between formulas)
    """

    def __init__(self, formulas, weights=None, mode='mean'):
        """
        Args:
            formulas: list of formula token lists, e.g. [[0,1,6], [2,10], ...]
            weights: optional list/array of floats (auto-normalized to sum=1)
            mode: 'mean' or 'rank_mean'
        """
        if not formulas:
            raise ValueError("formulas list must not be empty")

        self.formulas = formulas
        self.n = len(formulas)
        self.mode = mode
        self.vm = StackVM()

        if weights is None:
            self.weights = torch.ones(self.n) / self.n
        else:
            w = torch.tensor(weights, dtype=torch.float32)
            self.weights = w / w.sum()

    def predict(self, feat_tensor):
        """
        Execute all formulas and return aggregated signal.

        Args:
            feat_tensor: [num_tokens, num_features, time_steps]

        Returns:
            Aggregated signal tensor [num_tokens, time_steps], or None if
            all formulas fail.
        """
        signals = []
        valid_weights = []

        for i, formula in enumerate(self.formulas):
            res = self.vm.execute(formula, feat_tensor)
            if res is None:
                continue
            if res.std() < 1e-8:
                continue
            # Normalize shape: some ops produce extra dims
            target_shape = feat_tensor.shape[0], feat_tensor.shape[2]  # [N, T]
            if res.shape != target_shape:
                while res.dim() > 2:
                    res = res.squeeze(-1)
                if res.dim() == 1:
                    res = res.unsqueeze(-1)
                res = res[:target_shape[0], :target_shape[1]]
            signals.append(res)
            valid_weights.append(self.weights[i])

        if not signals:
            return None

        if self.mode == 'rank_mean':
            return self._rank_mean(signals, valid_weights)
        else:
            return self._weighted_mean(signals, valid_weights)

    def predict_individual(self, feat_tensor):
        """
        Execute all formulas and return each signal separately.

        Returns:
            list of (formula_index, signal_tensor) tuples for valid formulas.
        """
        results = []
        for i, formula in enumerate(self.formulas):
            res = self.vm.execute(formula, feat_tensor)
            if res is not None and res.std() >= 1e-8:
                results.append((i, res))
        return results

    def _weighted_mean(self, signals, valid_weights):
        """Simple weighted average of sigmoid-transformed signals."""
        stacked = torch.stack(signals, dim=0)  # [N, num_tokens, time_steps]
        w = torch.tensor(valid_weights, dtype=torch.float32, device=stacked.device)
        w = w / w.sum()

        weighted = (stacked * w.view(-1, 1, 1)).sum(dim=0)
        return weighted

    def _rank_mean(self, signals, valid_weights):
        """
        Rank-normalize each signal across the time dimension, then average.
        More robust when formulas produce signals on different scales.
        """
        ranked = []
        for sig in signals:
            # Rank along time dimension for each token
            ranks = sig.argsort(dim=1).argsort(dim=1).float()
            n_steps = sig.shape[1]
            ranks = ranks / max(n_steps - 1, 1)  # normalize to [0, 1]
            ranked.append(ranks)

        stacked = torch.stack(ranked, dim=0)  # [N, num_tokens, time_steps]
        w = torch.tensor(valid_weights, dtype=torch.float32, device=stacked.device)
        w = w / w.sum()
        weighted = (stacked * w.view(-1, 1, 1)).sum(dim=0)
        return weighted

    @property
    def num_formulas(self):
        return self.n

    @property
    def num_valid(self):
        """Number of formulas that would produce valid output (estimated)."""
        return self.n

    def to_dict(self):
        """Serialize ensemble to a dict for JSON saving."""
        return {
            'formulas': self.formulas,
            'weights': self.weights.tolist(),
            'mode': self.mode,
            'num_formulas': self.n,
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct ensemble from a saved dict."""
        return cls(
            formulas=d['formulas'],
            weights=d.get('weights'),
            mode=d.get('mode', 'mean'),
        )
