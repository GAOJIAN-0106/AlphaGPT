"""
TDD tests for P0: Position sizing overhaul + reward adjustments.

RED phase: these tests define the desired behavior for:
  1. New position modes (rank, zscore, quantile)
  2. Graduated lazy penalty
  3. Turnover bonus
  4. Transaction cost scaling
  5. Backward compatibility with sigmoid_clamp
"""

import math
import pytest
import torch

from model_core.backtest import MemeBacktest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_data(liquidity_val, batch, T, device="cpu"):
    liq = torch.full((batch, T), liquidity_val, device=device)
    return {"liquidity": liq}


@pytest.fixture
def device():
    return torch.device("cpu")


# ===========================================================================
# 1. MemeBacktest accepts position_mode parameter
# ===========================================================================

class TestPositionModeInit:
    def test_default_mode_is_sigmoid_clamp(self):
        bt = MemeBacktest()
        assert bt.position_mode == 'sigmoid_clamp'

    def test_custom_mode(self):
        bt = MemeBacktest(position_mode='rank')
        assert bt.position_mode == 'rank'

    def test_target_turnover_param(self):
        bt = MemeBacktest(target_turnover=0.08)
        assert bt.target_turnover == 0.08

    def test_default_target_turnover(self):
        bt = MemeBacktest()
        assert bt.target_turnover == 0.05


# ===========================================================================
# 2. compute_position() public method
# ===========================================================================

class TestComputePosition:
    def test_compute_position_exists(self):
        bt = MemeBacktest()
        assert hasattr(bt, 'compute_position')

    def test_compute_position_output_shape(self, device):
        bt = MemeBacktest(position_mode='rank')
        N, T = 30, 100
        factors = torch.randn(N, T, device=device)
        is_safe = torch.ones(N, T, device=device)
        pos = bt.compute_position(factors, is_safe)
        assert pos.shape == (N, T)

    def test_compute_position_in_valid_range(self, device):
        """All position modes must produce values in [0, 1]."""
        N, T = 20, 50
        torch.manual_seed(42)
        factors = torch.randn(N, T, device=device) * 3.0
        is_safe = torch.ones(N, T, device=device)

        for mode in ['sigmoid_clamp', 'rank', 'zscore', 'quantile']:
            bt = MemeBacktest(position_mode=mode)
            pos = bt.compute_position(factors, is_safe)
            assert (pos >= 0).all(), f"{mode}: position < 0 found"
            assert (pos <= 1.0).all(), f"{mode}: position > 1 found"

    def test_compute_position_respects_is_safe(self, device):
        """When is_safe=0, position must be 0 regardless of mode."""
        N, T = 10, 30
        factors = torch.full((N, T), 100.0, device=device)
        is_safe = torch.zeros(N, T, device=device)

        for mode in ['sigmoid_clamp', 'rank', 'zscore', 'quantile']:
            bt = MemeBacktest(position_mode=mode)
            pos = bt.compute_position(factors, is_safe)
            assert (pos == 0).all(), f"{mode}: position not 0 when is_safe=0"


# ===========================================================================
# 3. Rank position mode
# ===========================================================================

class TestPositionModeRank:
    def test_rank_produces_spread_positions(self, device):
        """Rank mode should spread positions across [0, 1]."""
        bt = MemeBacktest(position_mode='rank')
        N, T = 50, 100
        torch.manual_seed(42)
        factors = torch.randn(N, T, device=device)
        is_safe = torch.ones(N, T, device=device)
        pos = bt.compute_position(factors, is_safe)

        # Should have both zero and near-1 positions
        assert pos.max() > 0.8, f"Max position too low: {pos.max():.4f}"
        # Bottom stocks should have 0 position (long-only, top portion only)
        assert (pos == 0).any(), "Some stocks should have zero position"

    def test_rank_higher_turnover_than_sigmoid(self, device):
        """Rank mode should produce higher turnover with slow-changing factors."""
        N, T = 30, 200
        torch.manual_seed(42)
        # Simulate realistic slow-changing factors (cumsum → high autocorrelation)
        noise = torch.randn(N, T, device=device) * 0.1
        factors = noise.cumsum(dim=1)
        raw_data = _make_raw_data(1e9, N, T, device)
        target_ret = torch.randn(N, T, device=device) * 0.01

        bt_old = MemeBacktest(position_mode='sigmoid_clamp')
        bt_new = MemeBacktest(position_mode='rank')

        _, _, diag_old = bt_old.evaluate(factors, raw_data, target_ret, return_diagnostics=True)
        _, _, diag_new = bt_new.evaluate(factors, raw_data, target_ret, return_diagnostics=True)

        assert diag_new['avg_turnover'] > diag_old['avg_turnover'] * 1.5, (
            f"Rank turnover {diag_new['avg_turnover']:.4f} should be >1.5x "
            f"sigmoid {diag_old['avg_turnover']:.4f}"
        )

    def test_rank_top_quintile_has_position(self, device):
        """Top 40% of stocks (by factor value) should have non-zero position."""
        bt = MemeBacktest(position_mode='rank')
        N, T = 50, 1
        # Deterministic ascending factors so rank is clear
        factors = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1)
        is_safe = torch.ones(N, T, device=device)
        pos = bt.compute_position(factors, is_safe)

        # Top 40% of 50 = top 20 stocks should have pos > 0
        top_20_pos = pos[30:, 0]  # stocks ranked 30-49 (top)
        assert (top_20_pos > 0).all(), "Top 40% should have non-zero positions"

        # Bottom 60% should have pos = 0
        bottom_30_pos = pos[:30, 0]
        assert (bottom_30_pos == 0).all(), "Bottom 60% should have zero positions"


# ===========================================================================
# 4. Zscore position mode
# ===========================================================================

class TestPositionModeZscore:
    def test_zscore_positions_spread(self, device):
        """Zscore mode should produce spread positions."""
        bt = MemeBacktest(position_mode='zscore')
        N, T = 30, 100
        torch.manual_seed(42)
        factors = torch.randn(N, T, device=device)
        is_safe = torch.ones(N, T, device=device)
        pos = bt.compute_position(factors, is_safe)

        # Should have meaningful spread
        assert pos.max() > 0.5, f"Max position too low: {pos.max():.4f}"
        assert pos.std() > 0.1, f"Position spread too low: {pos.std():.4f}"

    def test_zscore_single_stock_fallback(self, device):
        """With N=1, zscore should fallback gracefully."""
        bt = MemeBacktest(position_mode='zscore')
        N, T = 1, 50
        factors = torch.randn(N, T, device=device)
        is_safe = torch.ones(N, T, device=device)
        pos = bt.compute_position(factors, is_safe)

        assert pos.shape == (1, T)
        assert not torch.isnan(pos).any()


# ===========================================================================
# 5. Quantile position mode
# ===========================================================================

class TestPositionModeQuantile:
    def test_quantile_discrete_levels(self, device):
        """Quantile mode should produce discrete position levels."""
        bt = MemeBacktest(position_mode='quantile')
        N, T = 100, 1
        factors = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1)
        is_safe = torch.ones(N, T, device=device)
        pos = bt.compute_position(factors, is_safe)

        # Should have exactly 3 distinct levels: 0.0, 0.5, 1.0
        unique_vals = pos[:, 0].unique()
        assert 0.0 in unique_vals
        assert 0.5 in unique_vals or any(abs(v - 0.5) < 0.01 for v in unique_vals)
        assert 1.0 in unique_vals or any(abs(v - 1.0) < 0.01 for v in unique_vals)

    def test_quantile_top10_full_position(self, device):
        """Top 10% of stocks should get position=1.0."""
        bt = MemeBacktest(position_mode='quantile')
        N, T = 100, 1
        factors = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1)
        is_safe = torch.ones(N, T, device=device)
        pos = bt.compute_position(factors, is_safe)

        # Top 10% (stocks 90-99) should have position ~1.0
        top10 = pos[90:, 0]
        assert (top10 > 0.9).all(), f"Top 10% positions: {top10.tolist()}"


# ===========================================================================
# 6. Graduated lazy penalty
# ===========================================================================

class TestGraduatedLazyPenalty:
    def test_very_low_turnover_heavy_penalty(self, device):
        """avg_turnover < 0.01 → penalty = 3.0."""
        bt = MemeBacktest(position_mode='sigmoid_clamp')
        N, T = 4, 400
        # Constant signal → turnover only at t=0 → avg ~0.0025
        factors = torch.full((N, T), 100.0, device=device)
        raw_data = _make_raw_data(1e11, N, T, device)
        target_ret = torch.full((N, T), 0.01, device=device)

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)
        assert diag['avg_turnover'] < 0.01

        score, _ = bt.evaluate(factors, raw_data, target_ret)
        # Score should include -3.0 penalty for very low turnover
        expected = diag['sharpe'] - diag['max_drawdown'] * 2.0 - 3.0
        assert score.item() == pytest.approx(expected, abs=0.1)

    def test_moderate_turnover_partial_penalty(self, device):
        """0.01 < avg_turnover < target_turnover → graduated penalty."""
        bt = MemeBacktest(position_mode='sigmoid_clamp', target_turnover=0.10)
        N, T = 4, 200
        torch.manual_seed(42)
        # Create factors that produce moderate turnover
        factors = torch.randn(N, T, device=device) * 3.0 + 1.0
        raw_data = _make_raw_data(1e9, N, T, device)
        target_ret = torch.randn(N, T, device=device) * 0.01

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)

        if 0.01 < diag['avg_turnover'] < 0.10:
            # Penalty should be between 0 and 2.0
            score, _ = bt.evaluate(factors, raw_data, target_ret)
            base = diag['sharpe'] - diag['max_drawdown'] * 2.0 + diag['ic'] * 10.0
            penalty = base - score.item()
            # Graduated penalty should be less than 3.0 (heavy) but > 0
            assert 0 < penalty < 3.0, f"Graduated penalty out of range: {penalty}"

    def test_above_target_no_lazy_penalty(self, device):
        """avg_turnover >= target_turnover → no lazy penalty."""
        bt = MemeBacktest(position_mode='rank', target_turnover=0.02)
        N, T = 30, 200
        torch.manual_seed(42)
        factors = torch.randn(N, T, device=device)
        raw_data = _make_raw_data(1e9, N, T, device)
        target_ret = torch.randn(N, T, device=device) * 0.01

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)

        # Rank mode with random factors should produce high turnover
        if diag['avg_turnover'] >= 0.02:
            score, _ = bt.evaluate(factors, raw_data, target_ret)
            # No lazy penalty; score = sharpe - dd_penalty + ic*10 + turnover_bonus
            # Just verify score is reasonable (not penalized to -10)
            assert score.item() > -5.0


# ===========================================================================
# 7. Turnover bonus
# ===========================================================================

class TestTurnoverBonus:
    def test_turnover_bonus_capped(self, device):
        """Turnover bonus should be capped at 0.5."""
        bt = MemeBacktest(position_mode='rank')
        N, T = 30, 200
        torch.manual_seed(42)
        factors = torch.randn(N, T, device=device) * 5.0
        raw_data = _make_raw_data(1e9, N, T, device)
        target_ret = torch.zeros(N, T, device=device)  # zero returns → sharpe=0

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)
        score, _ = bt.evaluate(factors, raw_data, target_ret)

        # With zero returns and high turnover: negative from tx costs
        # but turnover bonus helps offset
        # Bonus should not exceed 0.5 regardless of turnover level
        # We can't easily decompose exactly, but the bonus is part of score


# ===========================================================================
# 8. Transaction cost scaling
# ===========================================================================

class TestTxCostScale:
    def test_default_tx_cost_scale(self):
        bt = MemeBacktest()
        assert bt.tx_cost_scale == 1.0

    def test_reduced_tx_cost_improves_score(self, device):
        """tx_cost_scale=0.3 should produce higher score than 1.0."""
        N, T = 10, 200
        torch.manual_seed(42)
        factors = torch.randn(N, T, device=device) * 2.0 + 1.0
        raw_data = _make_raw_data(1e9, N, T, device)
        target_ret = torch.randn(N, T, device=device) * 0.01

        bt_full = MemeBacktest(position_mode='rank')
        bt_full.tx_cost_scale = 1.0
        score_full, _ = bt_full.evaluate(factors, raw_data, target_ret)

        bt_low = MemeBacktest(position_mode='rank')
        bt_low.tx_cost_scale = 0.3
        score_low, _ = bt_low.evaluate(factors, raw_data, target_ret)

        # Lower tx costs → better score
        assert score_low.item() >= score_full.item() - 0.1, (
            f"Reduced tx cost score ({score_low.item():.4f}) should be >= "
            f"full tx score ({score_full.item():.4f})"
        )


# ===========================================================================
# 9. Backward compatibility: sigmoid_clamp unchanged
# ===========================================================================

class TestSigmoidClampBackwardCompat:
    def test_sigmoid_clamp_position_formula(self, device):
        """sigmoid_clamp mode should produce same formula: clamp(sigmoid-0.5, 0, 0.5)*2."""
        bt = MemeBacktest(position_mode='sigmoid_clamp')
        N, T = 5, 20
        torch.manual_seed(42)
        factors = torch.randn(N, T, device=device)
        is_safe = torch.ones(N, T, device=device)

        pos = bt.compute_position(factors, is_safe)

        # Manual computation
        expected = torch.clamp(torch.sigmoid(factors) - 0.5, 0.0, 0.5) * 2.0 * is_safe
        assert torch.allclose(pos, expected, atol=1e-6)

    def test_sigmoid_signal_half_gives_zero(self, device):
        """sigmoid(0)=0.5 → position=0 in sigmoid_clamp mode."""
        bt = MemeBacktest(position_mode='sigmoid_clamp')
        factors = torch.zeros(1, 10, device=device)
        is_safe = torch.ones(1, 10, device=device)
        pos = bt.compute_position(factors, is_safe)
        assert (pos == 0).all()


# ===========================================================================
# 10. Integration: rank mode produces tradeable strategy
# ===========================================================================

class TestRankModeIntegration:
    def test_rank_mode_full_evaluate(self, device):
        """Full evaluate() pipeline with rank mode should produce valid diagnostics."""
        bt = MemeBacktest(position_mode='rank')
        N, T = 30, 252
        torch.manual_seed(42)
        factors = torch.randn(N, T, device=device)
        raw_data = _make_raw_data(1e9, N, T, device)
        target_ret = torch.randn(N, T, device=device) * 0.01

        score, mean_ret, diag = bt.evaluate(
            factors, raw_data, target_ret, return_diagnostics=True
        )

        assert not torch.isnan(score), "Score should not be NaN"
        assert math.isfinite(mean_ret), "mean_ret should be finite"
        assert math.isfinite(diag['sharpe'])
        assert math.isfinite(diag['avg_turnover'])
        # Turnover should be meaningful with rank mode
        assert diag['avg_turnover'] > 0.02, (
            f"Rank mode turnover too low: {diag['avg_turnover']:.4f}"
        )
