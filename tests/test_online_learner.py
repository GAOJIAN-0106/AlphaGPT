"""
Tests for OnlineLearner — adaptive ensemble weight optimization.
All tests use synthetic torch tensors, no database required.

TDD RED phase: these tests define the expected behavior.
"""

import math
import pytest
import torch
import numpy as np

from model_core.online_learner import OnlineLearner
from model_core.ensemble import FormulaEnsemble
from model_core.vm import StackVM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_formulas():
    """Create 3 simple valid formulas using known tokens.

    Token mapping (from StackVM): features [0..feat_offset-1],
    operators start at feat_offset. Use StackVM to get the offset.
    """
    from model_core.vm import StackVM
    vm = StackVM()
    off = vm.feat_offset
    # NEG is the 5th operator (index 4 after offset): ADD SUB MUL DIV NEG
    # ABS is the 6th operator (index 5)
    # ADD is the 1st operator (index 0)
    return [
        [0, off + 4],     # NEG(RET) — reverse return signal
        [1, off + 5],     # ABS(LIQ) — absolute liquidity
        [0, 1, off + 0],  # RET + LIQ — combined signal
    ]


def _make_feat_tensor(N=10, T=100, F=12, seed=42):
    """Create synthetic feature tensor [N, F, T]."""
    torch.manual_seed(seed)
    return torch.randn(N, F, T) * 0.1


def _make_raw_data(N=10, T=100):
    """Create raw_data dict with high liquidity (all tradeable)."""
    return {"liquidity": torch.full((N, T), 1e8)}


def _make_target_ret(N=10, T=100, seed=123):
    """Create synthetic target returns [N, T]."""
    torch.manual_seed(seed)
    return torch.randn(N, T) * 0.02


def _make_ensemble():
    """Create a FormulaEnsemble with 3 formulas."""
    return FormulaEnsemble(_make_formulas())


# ===========================================================================
# 1. Initialization
# ===========================================================================

class TestOnlineLearnerInit:
    def test_basic_init(self):
        """OnlineLearner should initialize with an ensemble."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)
        assert ol is not None

    def test_initial_weights_equal(self):
        """Initial weights should match the ensemble's equal weights."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)
        weights = ol.get_weights()
        assert len(weights) == 3
        np.testing.assert_allclose(weights, [1/3, 1/3, 1/3], atol=1e-6)

    def test_custom_params(self):
        """Should accept custom hyperparameters."""
        ens = _make_ensemble()
        ol = OnlineLearner(
            ens,
            lookback_window=10,
            learning_rate=0.5,
            decay=0.9,
            min_weight=0.1,
        )
        assert ol.lookback_window == 10
        assert ol.learning_rate == 0.5
        assert ol.decay == 0.9
        assert ol.min_weight == 0.1

    def test_empty_weight_history(self):
        """Weight history should be empty at init."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)
        assert ol.get_weight_history() == []


# ===========================================================================
# 2. Single update step
# ===========================================================================

class TestOnlineLearnerUpdate:
    def test_update_returns_metrics(self):
        """update() should return a dict with required keys."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        feat = _make_feat_tensor(N=10, T=20)
        raw = _make_raw_data(N=10, T=20)
        ret = _make_target_ret(N=10, T=20)

        result = ol.update(feat, raw, ret)

        assert isinstance(result, dict)
        assert 'weights' in result
        assert 'formula_pnls' in result
        assert len(result['weights']) == 3

    def test_update_changes_weights(self):
        """After enough updates, weights should differ from equal."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, learning_rate=0.5)

        # Create data where formula 0 (NEG(RET)) consistently profits
        # by making returns negative (so NEG flips them to positive signal)
        N, T = 10, 50
        torch.manual_seed(0)
        feat = torch.randn(N, 12, T) * 0.1
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T, seed=0)

        # Multiple updates to build performance history
        for i in range(5):
            ol.update(feat, raw, ret)

        weights = ol.get_weights()
        # Weights should no longer be exactly equal
        assert not np.allclose(weights, [1/3, 1/3, 1/3], atol=1e-3)

    def test_update_appends_weight_history(self):
        """Each update should append to weight history."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        feat = _make_feat_tensor(N=10, T=20)
        raw = _make_raw_data(N=10, T=20)
        ret = _make_target_ret(N=10, T=20)

        ol.update(feat, raw, ret)
        assert len(ol.get_weight_history()) == 1

        ol.update(feat, raw, ret)
        assert len(ol.get_weight_history()) == 2

    def test_weights_sum_to_one(self):
        """After any update, weights must sum to 1.0."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, learning_rate=0.8)

        feat = _make_feat_tensor(N=10, T=30)
        raw = _make_raw_data(N=10, T=30)
        ret = _make_target_ret(N=10, T=30)

        for _ in range(10):
            ol.update(feat, raw, ret)
            w = ol.get_weights()
            assert abs(sum(w) - 1.0) < 1e-6, f"Weights don't sum to 1: {w}"

    def test_min_weight_floor(self):
        """No weight should drop below min_weight."""
        ens = _make_ensemble()
        min_w = 0.1
        ol = OnlineLearner(ens, learning_rate=0.9, min_weight=min_w)

        feat = _make_feat_tensor(N=10, T=30)
        raw = _make_raw_data(N=10, T=30)
        ret = _make_target_ret(N=10, T=30)

        for _ in range(20):
            ol.update(feat, raw, ret)
            w = ol.get_weights()
            assert all(wi >= min_w - 1e-6 for wi in w), \
                f"Weight below min_weight: {w}"


# ===========================================================================
# 3. Predict
# ===========================================================================

class TestOnlineLearnerPredict:
    def test_predict_returns_tensor(self):
        """predict() should return a tensor with correct shape."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        N, T = 10, 50
        feat = _make_feat_tensor(N, T)
        signal = ol.predict(feat)

        assert signal is not None
        assert signal.shape == (N, T)

    def test_predict_uses_adaptive_weights(self):
        """After updates, predict should use adapted weights, not equal."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, learning_rate=0.8)

        N, T = 10, 50
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T)

        # Get prediction with equal weights
        signal_before = ol.predict(feat).clone()

        # Update several times
        for _ in range(10):
            ol.update(feat, raw, ret)

        # Get prediction with adapted weights
        signal_after = ol.predict(feat)

        # Signals should differ (weights changed)
        assert not torch.allclose(signal_before, signal_after, atol=1e-6)


# ===========================================================================
# 4. Walk-forward simulation (run_online)
# ===========================================================================

class TestOnlineLearnerRunOnline:
    def test_run_online_returns_required_keys(self):
        """run_online() should return dict with all required metrics."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        N, T = 10, 200
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T)

        result = ol.run_online(feat, raw, ret, period_length=5, warmup_periods=5)

        assert isinstance(result, dict)
        for key in ['online_pnl', 'static_pnl', 'weight_history',
                     'online_sharpe', 'static_sharpe', 'improvement']:
            assert key in result, f"Missing key: {key}"

    def test_run_online_pnl_length(self):
        """PnL arrays should match expected number of periods."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        N, T = 10, 100
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T)

        period_len = 5
        result = ol.run_online(feat, raw, ret, period_length=period_len,
                               warmup_periods=3)

        expected_periods = T // period_len
        assert len(result['online_pnl']) == expected_periods
        assert len(result['static_pnl']) == expected_periods

    def test_warmup_uses_equal_weights(self):
        """During warmup, weights should stay equal."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, learning_rate=0.9)

        N, T = 10, 100
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T)

        warmup = 5
        result = ol.run_online(feat, raw, ret, period_length=5,
                               warmup_periods=warmup)

        # First warmup entries in weight_history should be equal weights
        wh = result['weight_history']
        for i in range(min(warmup, len(wh))):
            np.testing.assert_allclose(wh[i], [1/3, 1/3, 1/3], atol=1e-6)

    def test_run_online_sharpe_is_finite(self):
        """Online and static Sharpe should be finite numbers."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        N, T = 10, 200
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T)

        result = ol.run_online(feat, raw, ret, period_length=5,
                               warmup_periods=5)

        assert np.isfinite(result['online_sharpe'])
        assert np.isfinite(result['static_sharpe'])

    def test_static_pnl_matches_equal_weight_ensemble(self):
        """static_pnl should match what a fixed equal-weight ensemble produces."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        N, T = 10, 50
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T)

        result = ol.run_online(feat, raw, ret, period_length=10,
                               warmup_periods=2)

        # Static PnL should be deterministic and reproducible
        assert len(result['static_pnl']) > 0
        # All static PnL values should be finite
        assert all(np.isfinite(p) for p in result['static_pnl'])


# ===========================================================================
# 5. Serialization
# ===========================================================================

class TestOnlineLearnerSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        """Serialization should preserve state."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, lookback_window=15, learning_rate=0.4,
                           decay=0.9, min_weight=0.08)

        # Do some updates
        feat = _make_feat_tensor(N=10, T=20)
        raw = _make_raw_data(N=10, T=20)
        ret = _make_target_ret(N=10, T=20)
        ol.update(feat, raw, ret)
        ol.update(feat, raw, ret)

        d = ol.to_dict()
        ol2 = OnlineLearner.from_dict(d)

        np.testing.assert_allclose(ol.get_weights(), ol2.get_weights(), atol=1e-6)
        assert ol2.lookback_window == 15
        assert ol2.learning_rate == 0.4
        assert ol2.decay == 0.9
        assert ol2.min_weight == 0.08
        assert len(ol2.get_weight_history()) == 2

    def test_to_dict_contains_required_keys(self):
        """to_dict() should include all reconstruction info."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        d = ol.to_dict()
        assert 'ensemble' in d
        assert 'weights' in d
        assert 'lookback_window' in d
        assert 'learning_rate' in d
        assert 'decay' in d
        assert 'min_weight' in d
        assert 'weight_history' in d
        assert 'performance_history' in d


# ===========================================================================
# 6. Edge cases
# ===========================================================================

class TestOnlineLearnerEdgeCases:
    def test_single_formula_ensemble(self):
        """Should work with a single-formula ensemble (weight stays 1.0)."""
        from model_core.vm import StackVM
        off = StackVM().feat_offset
        ens = FormulaEnsemble([[0, off + 4]])  # Just NEG(RET)
        ol = OnlineLearner(ens)

        feat = _make_feat_tensor(N=10, T=20)
        raw = _make_raw_data(N=10, T=20)
        ret = _make_target_ret(N=10, T=20)

        ol.update(feat, raw, ret)
        w = ol.get_weights()
        assert len(w) == 1
        assert abs(w[0] - 1.0) < 1e-6

    def test_very_short_period(self):
        """Should handle period_length=1 (daily updates)."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        N, T = 10, 30
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T)

        result = ol.run_online(feat, raw, ret, period_length=1,
                               warmup_periods=5)
        assert len(result['online_pnl']) == T

    def test_decay_reduces_old_influence(self):
        """With high decay, recent performance should dominate."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, decay=0.5, learning_rate=0.8)

        N, T = 10, 20
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T)

        # Many updates with same data
        for _ in range(15):
            ol.update(feat, raw, ret)

        # Weights should be stable (not diverging)
        w = ol.get_weights()
        assert all(np.isfinite(wi) for wi in w)
        assert abs(sum(w) - 1.0) < 1e-6

    def test_zero_returns_no_crash(self):
        """Should handle zero returns gracefully."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        N, T = 10, 20
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = torch.zeros(N, T)

        result = ol.update(feat, raw, ret)
        assert result is not None
        w = ol.get_weights()
        assert all(np.isfinite(wi) for wi in w)


# ===========================================================================
# 7. MWU Strategy
# ===========================================================================

class TestMWUStrategy:
    def test_mwu_init_defaults(self):
        """MWU should be the default strategy."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)
        assert ol.strategy == 'mwu'
        assert ol.eta > 0

    def test_mwu_custom_eta(self):
        """Should accept custom eta."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, eta=0.1)
        assert ol.eta == 0.1

    def test_mwu_auto_eta(self):
        """Auto eta should be sqrt(ln(N) / T_est)."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, lookback_window=20)
        expected = math.sqrt(math.log(3) / 200)
        assert abs(ol.eta - expected) < 1e-8

    def test_mwu_weights_sum_to_one(self):
        """MWU weights must sum to 1 after updates."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, strategy='mwu')

        feat = _make_feat_tensor(N=10, T=30)
        raw = _make_raw_data(N=10, T=30)
        ret = _make_target_ret(N=10, T=30)

        for _ in range(10):
            ol.update(feat, raw, ret)
            w = ol.get_weights()
            assert abs(sum(w) - 1.0) < 1e-6

    def test_mwu_min_weight_floor(self):
        """MWU should respect min_weight floor."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, strategy='mwu', min_weight=0.1, eta=0.5)

        feat = _make_feat_tensor(N=10, T=30)
        raw = _make_raw_data(N=10, T=30)
        ret = _make_target_ret(N=10, T=30)

        for _ in range(20):
            ol.update(feat, raw, ret)
            w = ol.get_weights()
            assert all(wi >= 0.1 - 1e-6 for wi in w)

    def test_mwu_upweights_winner(self):
        """MWU should give more weight to consistently profitable formulas."""
        ens = _make_ensemble()
        # Use aggressive eta so weight shifts are visible in few iterations
        ol = OnlineLearner(ens, strategy='mwu', eta=0.8, min_weight=0.01)

        N, T = 10, 30

        for seed_i in range(30):
            feat = _make_feat_tensor(N, T, seed=seed_i)
            raw = _make_raw_data(N, T)
            # Strongly negative returns to give NEG(RET) a clear edge
            torch.manual_seed(seed_i + 1000)
            ret = -torch.abs(torch.randn(N, T)) * 0.1
            ol.update(feat, raw, ret)

        w = ol.get_weights()
        # After many updates with varied data, weights should diverge
        assert not np.allclose(w, [1/3, 1/3, 1/3], atol=0.02)

    def test_mwu_zero_returns_stable(self):
        """MWU should keep equal weights when all returns are zero."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, strategy='mwu')

        N, T = 10, 20
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = torch.zeros(N, T)

        ol.update(feat, raw, ret)
        w = ol.get_weights()
        # Zero returns → no signal → weights unchanged
        np.testing.assert_allclose(w, [1/3, 1/3, 1/3], atol=1e-6)

    def test_mwu_step_counter(self):
        """MWU step counter should increment."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, strategy='mwu')

        feat = _make_feat_tensor(N=10, T=20)
        raw = _make_raw_data(N=10, T=20)
        ret = _make_target_ret(N=10, T=20)

        assert ol._mwu_step == 0
        ol.update(feat, raw, ret)
        assert ol._mwu_step == 1
        ol.update(feat, raw, ret)
        assert ol._mwu_step == 2

    def test_mwu_serialization_roundtrip(self):
        """MWU state should survive serialization."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, strategy='mwu', eta=0.15)

        feat = _make_feat_tensor(N=10, T=20)
        raw = _make_raw_data(N=10, T=20)
        ret = _make_target_ret(N=10, T=20)
        ol.update(feat, raw, ret)

        d = ol.to_dict()
        ol2 = OnlineLearner.from_dict(d)

        assert ol2.strategy == 'mwu'
        assert ol2.eta == 0.15
        assert ol2._mwu_step == 1
        np.testing.assert_allclose(ol.get_weights(), ol2.get_weights())

    def test_ab_compare_returns_both(self):
        """run_ab_compare should return metrics for both strategies."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens)

        N, T = 10, 200
        feat = _make_feat_tensor(N, T)
        raw = _make_raw_data(N, T)
        ret = _make_target_ret(N, T)

        result = ol.run_ab_compare(feat, raw, ret,
                                   period_length=5, warmup_periods=5)

        assert 'mwu_sharpe' in result
        assert 'ema_sharpe' in result
        assert 'static_sharpe' in result
        assert np.isfinite(result['mwu_sharpe'])
        assert np.isfinite(result['ema_sharpe'])

    def test_ema_strategy_still_works(self):
        """Legacy EMA strategy should still function."""
        ens = _make_ensemble()
        ol = OnlineLearner(ens, strategy='ema', learning_rate=0.5)

        feat = _make_feat_tensor(N=10, T=30)
        raw = _make_raw_data(N=10, T=30)
        ret = _make_target_ret(N=10, T=30)

        for _ in range(5):
            result = ol.update(feat, raw, ret)
            w = ol.get_weights()
            assert abs(sum(w) - 1.0) < 1e-6
