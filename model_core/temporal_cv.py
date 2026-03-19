"""
Time Series Cross-Validation for AlphaGPT.

Provides expanding-window and rolling-window CV splitters designed for
temporal financial data, where future data must never leak into training.
"""

import numpy as np
from .vm import StackVM
from .backtest import MemeBacktest


class TimeSeriesCV:
    """
    Time series cross-validation splitter.

    Two modes:
      - 'expanding': train window grows each fold (anchored at t=0)
      - 'rolling':   train window slides forward with fixed size

    Each fold yields (train_start, train_end, test_start, test_end).
    """

    def __init__(self, n_splits=5, min_train_pct=0.3, gap=0, mode='expanding'):
        """
        Args:
            n_splits: number of CV folds (must be >= 1)
            min_train_pct: minimum training set as fraction of total (0 < x < 1)
            gap: number of steps to skip between train and test (purge gap, >= 0)
            mode: 'expanding' or 'rolling'
        """
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not (0 < min_train_pct < 1):
            raise ValueError("min_train_pct must be in (0, 1)")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        if mode not in ('expanding', 'rolling'):
            raise ValueError(f"mode must be 'expanding' or 'rolling', got '{mode}'")

        self.n_splits = n_splits
        self.min_train_pct = min_train_pct
        self.gap = gap
        self.mode = mode

    def split(self, total_steps):
        """
        Generate train/test index ranges for each fold.

        Args:
            total_steps: total number of time steps in dataset

        Yields:
            (train_start, train_end, test_start, test_end) tuples

        Raises:
            ValueError: if total_steps is too few for the requested configuration
        """
        min_train = int(total_steps * self.min_train_pct)
        remaining = total_steps - min_train
        # Each fold needs at least 1 test step + gap
        min_test_per_fold = 5  # each fold needs at least 5 test steps
        min_needed = self.n_splits * (min_test_per_fold + self.gap)
        if remaining < min_needed:
            raise ValueError(
                f"too few steps ({total_steps}) for {self.n_splits} folds "
                f"with min_train_pct={self.min_train_pct} and gap={self.gap}. "
                f"Need at least {min_train + min_needed} steps."
            )

        # Divide the remaining space into n_splits test folds
        # Account for gaps: usable test space = remaining - n_splits * gap
        usable_test = remaining - self.n_splits * self.gap
        test_size = usable_test // self.n_splits

        folds = []
        for i in range(self.n_splits):
            # Test window boundaries
            test_start_raw = min_train + i * (test_size + self.gap) + self.gap
            if i < self.n_splits - 1:
                test_end = test_start_raw + test_size
            else:
                # Last fold takes all remaining steps
                test_end = total_steps

            test_start = test_start_raw

            # Train window
            train_end = test_start - self.gap
            if self.mode == 'expanding':
                train_start = 0
            else:
                # Rolling: fixed train size = min_train
                train_start = max(0, train_end - min_train)

            folds.append((train_start, train_end, test_start, test_end))

        return folds


def evaluate_formula_cv(formula, feat_tensor, target_ret, raw_data, cv):
    """
    Evaluate a single formula across all CV folds.

    Args:
        formula: list of token ids (RPN formula)
        feat_tensor: [num_tokens, num_features, time_steps]
        target_ret: [num_tokens, time_steps]
        raw_data: dict with 'liquidity' etc.
        cv: TimeSeriesCV instance

    Returns:
        dict with fold_results, mean_sharpe, std_sharpe, etc.
    """
    vm = StackVM()
    bt = MemeBacktest()

    total_steps = feat_tensor.shape[2]
    folds = cv.split(total_steps)

    fold_results = []
    valid_folds = 0

    for fold_idx, (ts, te, vs, ve) in enumerate(folds):
        # Slice data for this fold's test window
        test_feat = feat_tensor[:, :, vs:ve]
        test_ret = target_ret[:, vs:ve]
        test_raw = {k: (v[:, vs:ve] if v.dim() > 1 else v[vs:ve])
                    for k, v in raw_data.items()}

        # Execute formula on test fold
        res = vm.execute(formula, test_feat)
        if res is None or res.std() < 1e-8:
            fold_results.append({
                'fold': fold_idx,
                'sharpe': float('nan'),
                'max_drawdown': float('nan'),
                'avg_turnover': float('nan'),
                'valid': False,
            })
            continue

        # Evaluate
        base_factors = test_feat[:, 0, :]
        formula_len = len([t for t in formula if t != 0])
        _, _, diag = bt.evaluate(
            res, test_raw, test_ret,
            formula_length=formula_len,
            base_factors=base_factors,
            return_diagnostics=True,
        )

        fold_results.append({
            'fold': fold_idx,
            'sharpe': diag['sharpe'],
            'max_drawdown': diag['max_drawdown'],
            'avg_turnover': diag['avg_turnover'],
            'train_steps': te - ts,
            'test_steps': ve - vs,
            'valid': True,
        })
        valid_folds += 1

    # Aggregate
    valid_sharpes = [f['sharpe'] for f in fold_results if f['valid']]
    valid_dds = [f['max_drawdown'] for f in fold_results if f['valid']]

    return {
        'fold_results': fold_results,
        'num_valid_folds': valid_folds,
        'mean_sharpe': float(np.mean(valid_sharpes)) if valid_sharpes else float('nan'),
        'std_sharpe': float(np.std(valid_sharpes)) if valid_sharpes else float('nan'),
        'mean_max_drawdown': float(np.mean(valid_dds)) if valid_dds else float('nan'),
    }
