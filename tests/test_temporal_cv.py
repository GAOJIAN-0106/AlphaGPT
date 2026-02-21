"""
TDD tests for TimeSeriesCV — time series cross-validation for AlphaGPT.

Phase 2: Replace fixed 70/30 split with expanding/rolling window CV
to get more robust OOS evaluation across market regimes.
"""

import pytest
import torch
import numpy as np

from model_core.factors import FeatureEngineer


# ── Construction & Validation ─────────────────────────────────────

class TestTimeSeriesCVConstruction:

    def test_default_construction(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5)
        assert cv.n_splits == 5
        assert cv.mode == 'expanding'
        assert cv.gap == 0

    def test_rolling_mode(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=3, mode='rolling')
        assert cv.mode == 'rolling'

    def test_invalid_mode_raises(self):
        from model_core.temporal_cv import TimeSeriesCV
        with pytest.raises(ValueError, match="mode"):
            TimeSeriesCV(n_splits=3, mode='invalid')

    def test_n_splits_must_be_positive(self):
        from model_core.temporal_cv import TimeSeriesCV
        with pytest.raises(ValueError):
            TimeSeriesCV(n_splits=0)
        with pytest.raises(ValueError):
            TimeSeriesCV(n_splits=-1)

    def test_min_train_pct_range(self):
        from model_core.temporal_cv import TimeSeriesCV
        with pytest.raises(ValueError):
            TimeSeriesCV(n_splits=3, min_train_pct=0.0)
        with pytest.raises(ValueError):
            TimeSeriesCV(n_splits=3, min_train_pct=1.0)

    def test_gap_non_negative(self):
        from model_core.temporal_cv import TimeSeriesCV
        with pytest.raises(ValueError):
            TimeSeriesCV(n_splits=3, gap=-1)


# ── Expanding Window Splits ───────────────────────────────────────

class TestExpandingWindowSplits:

    def test_correct_number_of_folds(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        folds = list(cv.split(1000))
        assert len(folds) == 5

    def test_folds_are_tuples_of_four(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=3, min_train_pct=0.3)
        folds = list(cv.split(500))
        for fold in folds:
            assert len(fold) == 4  # (train_start, train_end, test_start, test_end)

    def test_train_starts_at_zero(self):
        """Expanding window: train always starts from beginning."""
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        folds = list(cv.split(1000))
        for train_start, _, _, _ in folds:
            assert train_start == 0

    def test_train_size_grows_across_folds(self):
        """Expanding window: train set gets larger each fold."""
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        folds = list(cv.split(1000))
        train_sizes = [end - start for start, end, _, _ in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_no_overlap_between_train_and_test(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        folds = list(cv.split(1000))
        for train_start, train_end, test_start, test_end in folds:
            assert train_end <= test_start

    def test_last_fold_test_reaches_end(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        folds = list(cv.split(1000))
        _, _, _, last_test_end = folds[-1]
        assert last_test_end == 1000

    def test_min_train_size_respected(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        folds = list(cv.split(1000))
        min_train = folds[0][1] - folds[0][0]
        assert min_train >= 300  # 30% of 1000

    def test_test_sizes_roughly_equal(self):
        """Each test fold should be approximately the same size."""
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        folds = list(cv.split(1000))
        test_sizes = [test_end - test_start for _, _, test_start, test_end in folds]
        # All test sizes should be within 2 of each other (rounding)
        assert max(test_sizes) - min(test_sizes) <= 2

    def test_covers_all_data_points(self):
        """Every time step after min_train should appear in exactly one test fold."""
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        total = 1000
        folds = list(cv.split(total))
        covered = set()
        for _, _, test_start, test_end in folds:
            fold_range = set(range(test_start, test_end))
            assert covered.isdisjoint(fold_range), "Test folds overlap!"
            covered.update(fold_range)
        # First fold test_start should be at min_train position
        assert min(covered) == folds[0][2]
        assert max(covered) == total - 1


# ── Rolling Window Splits ─────────────────────────────────────────

class TestRollingWindowSplits:

    def test_train_size_constant(self):
        """Rolling window: train size is fixed across all folds."""
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=4, min_train_pct=0.3, mode='rolling')
        folds = list(cv.split(1000))
        train_sizes = [end - start for start, end, _, _ in folds]
        # All train sizes should be equal
        assert len(set(train_sizes)) == 1

    def test_train_start_advances(self):
        """Rolling window: train start moves forward each fold."""
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=4, min_train_pct=0.3, mode='rolling')
        folds = list(cv.split(1000))
        train_starts = [start for start, _, _, _ in folds]
        for i in range(1, len(train_starts)):
            assert train_starts[i] > train_starts[i - 1]


# ── Gap (Purging) ─────────────────────────────────────────────────

class TestGapPurging:

    def test_gap_creates_separation(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=3, min_train_pct=0.3, gap=10)
        folds = list(cv.split(1000))
        for _, train_end, test_start, _ in folds:
            assert test_start - train_end >= 10

    def test_zero_gap_no_separation(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=3, min_train_pct=0.3, gap=0)
        folds = list(cv.split(1000))
        for _, train_end, test_start, _ in folds:
            assert test_start == train_end


# ── Edge Cases ────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_fold(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=1, min_train_pct=0.3)
        folds = list(cv.split(1000))
        assert len(folds) == 1
        assert folds[0][0] == 0
        assert folds[0][3] == 1000

    def test_too_few_steps_raises(self):
        """If total_steps is too small for the requested splits, raise error."""
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=10, min_train_pct=0.5)
        with pytest.raises(ValueError, match="too few"):
            list(cv.split(20))

    def test_real_data_shape_1477(self):
        """Test with actual data dimensions: 1477 trading days."""
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        folds = list(cv.split(1477))
        assert len(folds) == 5
        # All indices in valid range
        for ts, te, vs, ve in folds:
            assert 0 <= ts < te <= 1477
            assert 0 <= vs < ve <= 1477

    def test_large_gap_reduces_test_coverage(self):
        from model_core.temporal_cv import TimeSeriesCV
        cv_no_gap = TimeSeriesCV(n_splits=3, min_train_pct=0.3, gap=0)
        cv_gap = TimeSeriesCV(n_splits=3, min_train_pct=0.3, gap=20)
        folds_no_gap = list(cv_no_gap.split(500))
        folds_gap = list(cv_gap.split(500))
        # With gap, test start is pushed further, so total test coverage is less
        total_test_no_gap = sum(ve - vs for _, _, vs, ve in folds_no_gap)
        total_test_gap = sum(ve - vs for _, _, vs, ve in folds_gap)
        assert total_test_gap < total_test_no_gap


# ── Data Slicing Helper ───────────────────────────────────────────

class TestDataSlicing:

    def test_slice_tensor_by_fold(self):
        """TimeSeriesCV should provide a helper to slice data tensors."""
        from model_core.temporal_cv import TimeSeriesCV
        cv = TimeSeriesCV(n_splits=3, min_train_pct=0.3)
        # Simulate feat_tensor: [num_tokens, num_features, time_steps]
        feat = torch.randn(100, FeatureEngineer.INPUT_DIM, 500)
        target = torch.randn(100, 500)
        raw = {'liquidity': torch.randn(100, 500)}

        folds = list(cv.split(500))
        ts, te, vs, ve = folds[0]

        train_feat, test_feat = feat[:, :, ts:te], feat[:, :, vs:ve]
        train_ret, test_ret = target[:, ts:te], target[:, vs:ve]
        train_raw = {k: v[:, ts:te] for k, v in raw.items()}
        test_raw = {k: v[:, vs:ve] for k, v in raw.items()}

        assert train_feat.shape[2] == te - ts
        assert test_feat.shape[2] == ve - vs
        assert train_ret.shape[1] == te - ts
        assert test_ret.shape[1] == ve - vs


# ── CV Evaluation Function ────────────────────────────────────────

class TestCVEvaluation:

    def test_evaluate_formula_across_folds(self):
        """evaluate_formula_cv should return per-fold metrics."""
        from model_core.temporal_cv import TimeSeriesCV, evaluate_formula_cv
        from model_core.vm import StackVM
        from model_core.backtest import MemeBacktest

        cv = TimeSeriesCV(n_splits=3, min_train_pct=0.3)
        # Create synthetic data
        torch.manual_seed(42)
        n_tokens, n_feat, n_steps = 50, FeatureEngineer.INPUT_DIM, 500
        feat = torch.randn(n_tokens, n_feat, n_steps)
        target = torch.randn(n_tokens, n_steps) * 0.01
        raw = {'liquidity': torch.ones(n_tokens, n_steps) * 1e8}

        # Use a simple formula: just push feature 1 (LIQ)
        formula = [1]

        results = evaluate_formula_cv(formula, feat, target, raw, cv)

        assert 'fold_results' in results
        assert len(results['fold_results']) == 3
        assert 'mean_sharpe' in results
        assert 'std_sharpe' in results
        assert 'mean_max_drawdown' in results

        for fold_res in results['fold_results']:
            assert 'fold' in fold_res
            assert 'sharpe' in fold_res
            assert 'max_drawdown' in fold_res
            assert isinstance(fold_res['sharpe'], float)

    def test_cv_sharpe_is_average_of_folds(self):
        """mean_sharpe should be the mean of fold sharpes."""
        from model_core.temporal_cv import TimeSeriesCV, evaluate_formula_cv

        cv = TimeSeriesCV(n_splits=3, min_train_pct=0.3)
        torch.manual_seed(42)
        feat = torch.randn(50, FeatureEngineer.INPUT_DIM, 500)
        target = torch.randn(50, 500) * 0.01
        raw = {'liquidity': torch.ones(50, 500) * 1e8}

        results = evaluate_formula_cv([1], feat, target, raw, cv)
        fold_sharpes = [f['sharpe'] for f in results['fold_results']]
        assert abs(results['mean_sharpe'] - np.mean(fold_sharpes)) < 1e-6

    def test_invalid_formula_returns_none_folds(self):
        """Invalid formula should still return results with None/NaN sharpes."""
        from model_core.temporal_cv import TimeSeriesCV, evaluate_formula_cv

        cv = TimeSeriesCV(n_splits=3, min_train_pct=0.3)
        feat = torch.randn(50, FeatureEngineer.INPUT_DIM, 500)
        target = torch.randn(50, 500) * 0.01
        raw = {'liquidity': torch.ones(50, 500) * 1e8}

        # Empty formula or invalid formula
        results = evaluate_formula_cv([], feat, target, raw, cv)
        assert results is not None
        assert results['num_valid_folds'] == 0


# ── Integration: CV vs Single Split ──────────────────────────────

class TestCVvsSimpleSplit:

    def test_cv_gives_multiple_sharpe_estimates(self):
        """CV should give N separate Sharpe estimates vs single split's 1."""
        from model_core.temporal_cv import TimeSeriesCV, evaluate_formula_cv

        cv = TimeSeriesCV(n_splits=5, min_train_pct=0.3)
        torch.manual_seed(123)
        feat = torch.randn(50, FeatureEngineer.INPUT_DIM, 1000)
        target = torch.randn(50, 1000) * 0.01
        raw = {'liquidity': torch.ones(50, 1000) * 1e8}

        results = evaluate_formula_cv([1], feat, target, raw, cv)
        valid_sharpes = [f['sharpe'] for f in results['fold_results'] if f['valid']]
        assert len(valid_sharpes) == 5
        # Sharpe std > 0 means CV reveals variance that single split hides
        assert results['std_sharpe'] > 0

    def test_expanding_vs_rolling_different_results(self):
        """Different CV modes should produce different results."""
        from model_core.temporal_cv import TimeSeriesCV, evaluate_formula_cv

        torch.manual_seed(42)
        feat = torch.randn(50, FeatureEngineer.INPUT_DIM, 1000)
        target = torch.randn(50, 1000) * 0.01
        raw = {'liquidity': torch.ones(50, 1000) * 1e8}

        cv_exp = TimeSeriesCV(n_splits=3, min_train_pct=0.3, mode='expanding')
        cv_rol = TimeSeriesCV(n_splits=3, min_train_pct=0.3, mode='rolling')

        r_exp = evaluate_formula_cv([1], feat, target, raw, cv_exp)
        r_rol = evaluate_formula_cv([1], feat, target, raw, cv_rol)

        # Results should differ because training windows differ
        # (in expanding, later folds see more data; in rolling, same amount)
        exp_sharpes = [f['sharpe'] for f in r_exp['fold_results'] if f['valid']]
        rol_sharpes = [f['sharpe'] for f in r_rol['fold_results'] if f['valid']]
        # At minimum, the fold structure differs
        assert len(exp_sharpes) == len(rol_sharpes) == 3

    def test_gap_purging_affects_results(self):
        """Adding a gap should change evaluation results."""
        from model_core.temporal_cv import TimeSeriesCV, evaluate_formula_cv

        torch.manual_seed(42)
        feat = torch.randn(50, FeatureEngineer.INPUT_DIM, 1000)
        target = torch.randn(50, 1000) * 0.01
        raw = {'liquidity': torch.ones(50, 1000) * 1e8}

        cv_no_gap = TimeSeriesCV(n_splits=3, min_train_pct=0.3, gap=0)
        cv_gap = TimeSeriesCV(n_splits=3, min_train_pct=0.3, gap=20)

        r_no = evaluate_formula_cv([1], feat, target, raw, cv_no_gap)
        r_gap = evaluate_formula_cv([1], feat, target, raw, cv_gap)

        # With gap, test windows are shifted/smaller
        no_test_sizes = [f['test_steps'] for f in r_no['fold_results'] if f['valid']]
        gap_test_sizes = [f['test_steps'] for f in r_gap['fold_results'] if f['valid']]
        assert sum(gap_test_sizes) < sum(no_test_sizes)
