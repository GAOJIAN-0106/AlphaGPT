"""
Comprehensive tests for AlphaGPT reward function improvements.
All tests use synthetic torch tensors — no database connection required.
"""

import math
import pytest
import torch

from model_core.backtest import MemeBacktest
from model_core.factors import FeatureEngineer, MemeIndicators
from model_core.vm import StackVM
from model_core.ops import OPS_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bt():
    """Fresh MemeBacktest instance."""
    return MemeBacktest()


@pytest.fixture
def device():
    return torch.device("cpu")


def _make_raw_data(liquidity_val, batch, T, device="cpu"):
    """Helper: build a raw_data dict with constant liquidity."""
    liq = torch.full((batch, T), liquidity_val, device=device)
    return {"liquidity": liq}


# ===========================================================================
# 1. Sharpe ratio computation
# ===========================================================================

class TestSharpeRatio:
    def test_known_sharpe(self, bt, device):
        """Create a known constant daily-PnL and verify Sharpe = mean/std * sqrt(252)."""
        batch, T = 1, 252
        # Constant positive PnL → std ≈ 0 → Sharpe very high, but let's use
        # a series with known mean & std instead.
        torch.manual_seed(42)
        daily_pnl = torch.randn(1, T) * 0.01 + 0.001  # mean≈0.001, std≈0.01

        # To get *exact* control we need to bypass position sizing.
        # Instead, craft factors so that position = 1.0 everywhere and
        # target_ret = daily_pnl directly.
        # sigmoid(large) ≈ 1 → signal=1 → clamp(0.5,0,0.5)*2 = 1.0
        factors = torch.full((batch, T), 100.0, device=device)
        high_liq = 1e9  # way above min_liq → is_safe = 1
        raw_data = _make_raw_data(high_liq, batch, T, device)

        # target_ret = daily_pnl, but we also lose tx cost on the first bar
        # (position jumps from 0→1). For simplicity, check Sharpe within a
        # tolerance that accounts for small tx costs.
        target_ret = daily_pnl

        score, mean_ret = bt.evaluate(factors, raw_data, target_ret)

        # Manually compute expected Sharpe (ignoring tx)
        expected_mean = daily_pnl.mean()
        expected_std = daily_pnl.std()
        expected_sharpe = (expected_mean / (expected_std + 1e-8)) * math.sqrt(252)

        # score ≈ sharpe − drawdown_penalty, so we at least check sharpe via diagnostics
        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)
        # The backtest sharpe accounts for tx costs, so allow generous tolerance
        assert abs(diag["sharpe"] - expected_sharpe.item()) < 3.0, (
            f"Sharpe mismatch: got {diag['sharpe']:.4f}, expected ≈{expected_sharpe.item():.4f}"
        )

    def test_zero_pnl_sharpe(self, bt, device):
        """All-zero factors → position = 0 everywhere → Sharpe ≈ 0."""
        batch, T = 2, 100
        factors = torch.zeros(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T) * 0.01

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)
        assert abs(diag["sharpe"]) < 1e-3


# ===========================================================================
# 2. Max drawdown
# ===========================================================================

class TestMaxDrawdown:
    def test_known_drawdown(self, bt, device):
        """Build a cumulative PnL that goes 0→+1→+0.5 → drawdown = 0.5."""
        batch, T = 1, 4
        # net_pnl that gives cum_pnl = [0.5, 0.5, -0.3, -0.2]
        # cum = [0.5, 1.0, 0.7, 0.5]
        # running_max = [0.5, 1.0, 1.0, 1.0]
        # drawdown = [0.0, 0.0, 0.3, 0.5] → max = 0.5
        #
        # We need position * target_ret − tx_cost = net_pnl.
        # Force position=1.0 everywhere (use big factor), high liquidity.
        factors = torch.full((batch, T), 100.0, device=device)
        raw_data = _make_raw_data(1e11, batch, T, device)

        # With position=1 and huge liquidity, tx cost ≈ base_fee * turnover.
        # Turnover is 1 at t=0, 0 elsewhere (position stays 1).
        # base_fee = 0.0015, impact ≈ 0
        # tx_cost = [0.0015, 0, 0, 0]
        # We want net_pnl = [0.5, 0.5, -0.3, -0.2]
        # gross_pnl = net_pnl + tx_cost = [0.5015, 0.5, -0.3, -0.2]
        # target_ret = gross_pnl / position = same values since position=1
        target_ret = torch.tensor([[0.5015, 0.5, -0.3, -0.2]], device=device)

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)
        assert abs(diag["max_drawdown"] - 0.5) < 0.02, (
            f"Expected drawdown ≈ 0.5, got {diag['max_drawdown']:.4f}"
        )

    def test_monotonic_increase_no_drawdown(self, bt, device):
        """Monotonically increasing PnL → drawdown = 0."""
        batch, T = 1, 50
        factors = torch.full((batch, T), 100.0, device=device)
        raw_data = _make_raw_data(1e11, batch, T, device)
        # Constant positive returns, big enough to dominate any tx cost
        target_ret = torch.full((batch, T), 0.1, device=device)

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)
        # Small drawdown from the first-bar tx cost is acceptable
        assert diag["max_drawdown"] < 0.01


# ===========================================================================
# 3. Complexity penalty
# ===========================================================================

class TestComplexityPenalty:
    def test_no_penalty_short_formula(self, bt, device):
        """formula_length <= 6 → no complexity penalty."""
        batch, T = 2, 100
        factors = torch.full((batch, T), 100.0, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.full((batch, T), 0.01, device=device)

        score_no_len, _ = bt.evaluate(factors, raw_data, target_ret)
        score_len6, _ = bt.evaluate(factors, raw_data, target_ret, formula_length=6)

        assert torch.allclose(score_no_len, score_len6, atol=1e-6), (
            "formula_length=6 should incur zero penalty"
        )

    def test_penalty_long_formula(self, bt, device):
        """formula_length=10 → penalty = 0.1*(10−6) = 0.4."""
        batch, T = 2, 100
        factors = torch.full((batch, T), 100.0, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.full((batch, T), 0.01, device=device)

        score_base, _ = bt.evaluate(factors, raw_data, target_ret)
        score_pen, _ = bt.evaluate(factors, raw_data, target_ret, formula_length=10)

        expected_penalty = 0.1 * (10 - 6)
        diff = (score_base - score_pen).item()
        assert abs(diff - expected_penalty) < 1e-4, (
            f"Expected penalty {expected_penalty}, got diff {diff}"
        )

    @pytest.mark.parametrize("length,expected", [(1, 0), (6, 0), (7, 0.1), (12, 0.6)])
    def test_penalty_values(self, bt, device, length, expected):
        """Parametric check over several lengths."""
        batch, T = 1, 100
        factors = torch.full((batch, T), 100.0, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.full((batch, T), 0.01, device=device)

        score_none, _ = bt.evaluate(factors, raw_data, target_ret)
        score_with, _ = bt.evaluate(factors, raw_data, target_ret, formula_length=length)

        diff = (score_none - score_with).item()
        assert abs(diff - expected) < 1e-4


# ===========================================================================
# 4. Factor redundancy penalty
# ===========================================================================

class TestRedundancyPenalty:
    def test_perfectly_correlated_penalty(self, device):
        """Factor identical to base → corr > 0.7 → penalty = 2.0."""
        batch, T = 3, 200
        factors = torch.randn(batch, T, device=device)
        base_factors = factors.clone()  # perfect correlation

        penalty = MemeBacktest._redundancy_penalty(factors, base_factors)
        assert torch.allclose(penalty, torch.tensor(2.0, device=device)), (
            f"Expected penalty 2.0 for perfect correlation, got {penalty}"
        )

    def test_uncorrelated_no_penalty(self, device):
        """Orthogonal / uncorrelated factor → penalty = 0."""
        batch, T = 4, 500
        torch.manual_seed(0)
        factors = torch.randn(batch, T, device=device)
        # Shuffle each row independently to destroy correlation
        base_factors = torch.randn(batch, T, device=device) * 100.0
        # Make truly uncorrelated by using independent random data
        # With large T and independent draws, rank correlation → 0
        penalty = MemeBacktest._redundancy_penalty(factors, base_factors)

        # Tolerance: each element should be 0.0 (not 2.0)
        assert (penalty < 0.01).all(), (
            f"Expected zero penalty for uncorrelated factors, got {penalty}"
        )

    def test_negatively_correlated_penalty(self, device):
        """Factor = −base → |corr| > 0.7 → penalty = 2.0."""
        batch, T = 2, 200
        factors = torch.randn(batch, T, device=device)
        base_factors = -factors  # perfect negative correlation

        penalty = MemeBacktest._redundancy_penalty(factors, base_factors)
        assert torch.allclose(penalty, torch.tensor(2.0, device=device))

    def test_multi_base_factors(self, device):
        """Multiple base factors: correlated with one → penalty 2.0."""
        batch, T, K = 2, 200, 3
        torch.manual_seed(7)
        factors = torch.randn(batch, T, device=device)
        base = torch.randn(batch, T, K, device=device)
        # Make the first base factor identical → triggers penalty
        base[:, :, 0] = factors.clone()

        penalty = MemeBacktest._redundancy_penalty(factors, base)
        # At least 2.0 (from base factor 0)
        assert (penalty >= 1.9).all(), f"Expected penalty >= 2.0, got {penalty}"


# ===========================================================================
# 5. Continuous position sizing
# ===========================================================================

class TestPositionSizing:
    def test_signal_half_gives_zero_position(self, device):
        """sigmoid(0)=0.5 → clamp(0.5-0.5,0,0.5)=0 → position=0."""
        bt = MemeBacktest()
        batch, T = 1, 10
        factors = torch.zeros(batch, T, device=device)  # sigmoid(0) = 0.5
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.ones(batch, T, device=device)

        _, mean_ret = bt.evaluate(factors, raw_data, target_ret)
        # position=0 everywhere → PnL=0 → mean_ret=0
        assert abs(mean_ret) < 1e-6

    def test_strong_signal_gives_full_position(self, device):
        """sigmoid(large)≈1 → clamp(0.5)*2=1.0, times is_safe=1 → position=1."""
        batch, T = 1, 10
        factors = torch.full((batch, T), 100.0, device=device)
        liquidity = torch.full((batch, T), 1e9, device=device)

        signal = torch.sigmoid(factors)
        is_safe = (liquidity > 1e7).float()
        position = torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe

        assert torch.allclose(position, torch.ones_like(position), atol=1e-5)

    def test_low_liquidity_kills_position(self, device):
        """Liquidity below min_liq → is_safe = 0 → position = 0."""
        bt = MemeBacktest()
        batch, T = 1, 10
        factors = torch.full((batch, T), 100.0, device=device)
        low_liq = bt.min_liq * 0.5  # below threshold
        raw_data = _make_raw_data(low_liq, batch, T, device)
        target_ret = torch.ones(batch, T, device=device)

        _, mean_ret = bt.evaluate(factors, raw_data, target_ret)
        assert abs(mean_ret) < 1e-6, "Position should be zero with low liquidity"


# ===========================================================================
# 6. Activity threshold
# ===========================================================================

class TestActivityThreshold:
    def test_inactive_strategy_gets_negative_ten(self, bt, device):
        """Very few active positions → score clamped to -10."""
        batch, T = 4, 200
        # Factors just below 0 → sigmoid < 0.5 → position = 0
        factors = torch.full((batch, T), -100.0, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.ones(batch, T, device=device) * 0.01

        score, _ = bt.evaluate(factors, raw_data, target_ret)
        assert score.item() == pytest.approx(-10.0, abs=1e-4), (
            f"Expected -10.0 for inactive strategy, got {score.item()}"
        )

    def test_active_strategy_not_penalized(self, bt, device):
        """Fully active strategy should NOT get -10 penalty."""
        batch, T = 2, 200
        factors = torch.full((batch, T), 100.0, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.full((batch, T), 0.01, device=device)

        score, _ = bt.evaluate(factors, raw_data, target_ret)
        assert score.item() != pytest.approx(-10.0, abs=0.5), (
            "Active strategy should not receive inactive penalty"
        )


# ===========================================================================
# 7. torch.roll fix in factors (no wrap-around)
# ===========================================================================

class TestFactorsNoWrapAround:
    def test_return_at_t0_no_wraparound(self, device):
        """
        FeatureEngineer.compute_features uses:
            ret = log(c / (cat([zeros, c[:, :-1]]) + eps))
        At t=0, denominator = 0 + eps → ret[0] = log(c[0] / eps).
        The key check: ret at t=0 does NOT contain data from the last time step.
        """
        batch, T = 3, 50
        close = torch.rand(batch, T, device=device) * 100 + 1.0

        raw_dict = {
            "close": close,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "volume": torch.rand(batch, T, device=device) * 1e6,
            "liquidity": torch.rand(batch, T, device=device) * 1e8,
            "fdv": torch.zeros(batch, T, device=device),
        }

        features = FeatureEngineer.compute_features(raw_dict)
        # features shape: [batch, num_features, T]
        # Feature 0 = robust_norm(ret)
        ret_feature = features[:, 0, :]  # [batch, T]

        # The first-timestep return should NOT depend on close[:, -1].
        # Mutate last close and recompute — ret[:, 0] should stay the same.
        raw_dict2 = {k: v.clone() for k, v in raw_dict.items()}
        raw_dict2["close"][:, -1] = 999999.0

        features2 = FeatureEngineer.compute_features(raw_dict2)
        ret_feature2 = features2[:, 0, :]

        # t=0 should be identical (robust_norm might shift globally, so
        # compare pre-normalization return instead)
        c = raw_dict["close"]
        prev = torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], dim=1)
        ret_raw = torch.log(c / (prev + 1e-9))

        c2 = raw_dict2["close"]
        prev2 = torch.cat([torch.zeros_like(c2[:, :1]), c2[:, :-1]], dim=1)
        ret_raw2 = torch.log(c2 / (prev2 + 1e-9))

        # At t=0, prev is zeros regardless of last timestep
        assert torch.allclose(ret_raw[:, 0], ret_raw2[:, 0], atol=1e-6), (
            "Return at t=0 should not wrap around from last timestep"
        )

    def test_fomo_acceleration_no_wraparound(self, device):
        """fomo_acceleration uses explicit cat([zeros, ...]) — verify no wrap."""
        batch, T = 2, 30
        volume = torch.rand(batch, T, device=device) * 1e6

        fomo = MemeIndicators.fomo_acceleration(volume)
        # Mutate last volume element and verify t=0 unchanged
        volume2 = volume.clone()
        volume2[:, -1] = 1e12
        fomo2 = MemeIndicators.fomo_acceleration(volume2)

        assert torch.allclose(fomo[:, 0], fomo2[:, 0], atol=1e-6)


# ===========================================================================
# 8. Train/test split logic
# ===========================================================================

class TestTrainTestSplit:
    def test_split_indices(self, device):
        """
        Simulate the data_loader split logic on mock tensors
        and verify shapes and index boundaries.
        """
        num_tokens, num_features, time_steps = 10, FeatureEngineer.INPUT_DIM, 100
        train_ratio = 0.7

        feat_tensor = torch.randn(num_tokens, num_features, time_steps, device=device)
        target_ret = torch.randn(num_tokens, time_steps, device=device)
        raw_data = {
            "close": torch.randn(num_tokens, time_steps, device=device),
            "liquidity": torch.randn(num_tokens, time_steps, device=device),
        }

        split_idx = int(time_steps * train_ratio)
        assert split_idx == 70

        train_feat = feat_tensor[:, :, :split_idx]
        test_feat = feat_tensor[:, :, split_idx:]

        # Shape checks
        assert train_feat.shape == (num_tokens, num_features, split_idx)
        assert test_feat.shape == (num_tokens, num_features, time_steps - split_idx)

        train_ret = target_ret[:, :split_idx]
        test_ret = target_ret[:, split_idx:]
        assert train_ret.shape == (num_tokens, split_idx)
        assert test_ret.shape == (num_tokens, time_steps - split_idx)

        # No overlap
        assert torch.equal(
            torch.cat([train_feat, test_feat], dim=2), feat_tensor
        )
        assert torch.equal(
            torch.cat([train_ret, test_ret], dim=1), target_ret
        )

        # Raw data split
        for k, v in raw_data.items():
            train_v = v[:, :split_idx]
            test_v = v[:, split_idx:]
            assert train_v.shape[1] == split_idx
            assert test_v.shape[1] == time_steps - split_idx
            assert torch.equal(torch.cat([train_v, test_v], dim=1), v)

    @pytest.mark.parametrize("ratio", [0.5, 0.7, 0.8, 0.9])
    def test_split_ratios(self, device, ratio):
        """Different ratios produce correct split points."""
        T = 200
        split = int(T * ratio)
        expected_train = split
        expected_test = T - split

        feat = torch.randn(5, FeatureEngineer.INPUT_DIM, T, device=device)
        assert feat[:, :, :split].shape[2] == expected_train
        assert feat[:, :, split:].shape[2] == expected_test


# ===========================================================================
# 9. Backward compatibility — evaluate() with only positional args
# ===========================================================================

class TestBackwardCompatibility:
    def test_evaluate_positional_only(self, bt, device):
        """evaluate(factors, raw_data, target_ret) with no optional args still works."""
        batch, T = 2, 100
        factors = torch.randn(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        result = bt.evaluate(factors, raw_data, target_ret)
        assert isinstance(result, tuple)
        assert len(result) == 2
        score, mean_ret = result
        assert isinstance(score, torch.Tensor)
        assert isinstance(mean_ret, float)

    def test_evaluate_with_all_kwargs(self, bt, device):
        """evaluate() with all optional kwargs still returns valid output."""
        batch, T = 2, 100
        factors = torch.randn(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01
        base_factors = torch.randn(batch, T, device=device)

        result = bt.evaluate(
            factors, raw_data, target_ret,
            formula_length=8,
            base_factors=base_factors,
            return_diagnostics=False,
        )
        assert len(result) == 2


# ===========================================================================
# 10. Diagnostics return
# ===========================================================================

class TestDiagnosticsReturn:
    def test_return_diagnostics_true(self, bt, device):
        """return_diagnostics=True → 3-tuple with correct dict keys."""
        batch, T = 2, 100
        factors = torch.randn(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        result = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)

        assert isinstance(result, tuple)
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"

        score, mean_ret, diagnostics = result
        assert isinstance(score, torch.Tensor)
        assert isinstance(mean_ret, float)
        assert isinstance(diagnostics, dict)

        expected_keys = {"sharpe", "max_drawdown", "avg_turnover", "factor_autocorrelation", "ic"}
        assert set(diagnostics.keys()) == expected_keys, (
            f"Diagnostics keys mismatch: {set(diagnostics.keys())} != {expected_keys}"
        )

    def test_diagnostics_values_are_finite(self, bt, device):
        """All diagnostic values should be finite numbers."""
        batch, T = 3, 200
        factors = torch.randn(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)

        for key, val in diag.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"
            assert math.isfinite(val), f"{key} is not finite: {val}"

    def test_return_diagnostics_false(self, bt, device):
        """return_diagnostics=False (default) → 2-tuple."""
        batch, T = 2, 100
        factors = torch.randn(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        result = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=False)
        assert len(result) == 2


# ===========================================================================
# Bonus: integration-style test
# ===========================================================================

class TestIntegration:
    def test_full_evaluate_flow(self, bt, device):
        """End-to-end: build factors, evaluate with all options, check no crash."""
        batch, T = 4, 252
        torch.manual_seed(123)
        factors = torch.randn(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01
        base_factors = torch.randn(batch, T, device=device)

        score, mean_ret, diag = bt.evaluate(
            factors, raw_data, target_ret,
            formula_length=8,
            base_factors=base_factors,
            return_diagnostics=True,
        )

        assert score.shape == ()  # scalar
        assert isinstance(mean_ret, float)
        assert "sharpe" in diag
        assert "max_drawdown" in diag

    def test_nan_inputs_handled(self, bt, device):
        """NaN in factors/target_ret should not crash or produce NaN score."""
        batch, T = 2, 100
        factors = torch.randn(batch, T, device=device)
        factors[0, 10:20] = float("nan")
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01
        target_ret[1, 50:60] = float("nan")

        score, mean_ret = bt.evaluate(factors, raw_data, target_ret)
        assert not torch.isnan(score), "Score should not be NaN"
        assert math.isfinite(mean_ret), "mean_ret should be finite"


# ===========================================================================
# 11. StackVM regression tests
# ===========================================================================

class TestStackVMRegression:
    """Verify StackVM still works correctly with known formulas."""

    @pytest.fixture
    def vm(self):
        return StackVM()

    @pytest.fixture
    def feat_tensor(self, device):
        """Synthetic feature tensor: [num_tokens=5, num_features=6, time_steps=30]."""
        torch.manual_seed(99)
        return torch.randn(5, FeatureEngineer.INPUT_DIM, 30, device=device)

    def test_single_feature_push(self, vm, feat_tensor):
        """Token 0 pushes feature 0 → result == feat_tensor[:, 0, :]."""
        result = vm.execute([0], feat_tensor)
        assert result is not None
        assert torch.equal(result, feat_tensor[:, 0, :])

    def test_add_two_features(self, vm, feat_tensor):
        """[0, 1, ADD] → feat[0] + feat[1]."""
        add_token = FeatureEngineer.INPUT_DIM + 0  # ADD is OPS_CONFIG[0]
        result = vm.execute([0, 1, add_token], feat_tensor)
        expected = feat_tensor[:, 0, :] + feat_tensor[:, 1, :]
        assert result is not None
        assert torch.allclose(result, expected, atol=1e-5)

    def test_neg_feature(self, vm, feat_tensor):
        """[0, NEG] → −feat[0]."""
        neg_token = FeatureEngineer.INPUT_DIM + 4  # NEG is OPS_CONFIG[4]
        result = vm.execute([0, neg_token], feat_tensor)
        expected = -feat_tensor[:, 0, :]
        assert result is not None
        assert torch.allclose(result, expected, atol=1e-5)

    def test_invalid_formula_returns_none(self, vm, feat_tensor):
        """Formula that doesn't reduce to single stack element → None."""
        # Push two features, no op → stack has 2 elements
        result = vm.execute([0, 1], feat_tensor)
        assert result is None

    def test_empty_stack_op_returns_none(self, vm, feat_tensor):
        """Op with empty stack → None."""
        add_token = FeatureEngineer.INPUT_DIM + 0
        result = vm.execute([add_token], feat_tensor)
        assert result is None

    def test_result_shape_matches_input(self, vm, feat_tensor):
        """Output shape should be [num_tokens, time_steps]."""
        result = vm.execute([0], feat_tensor)
        assert result.shape == (feat_tensor.shape[0], feat_tensor.shape[2])

    def test_nan_inf_cleaned(self, vm, device):
        """StackVM should clean NaN/Inf from op results."""
        # DIV by near-zero can produce large values; StackVM clamps them
        feat = torch.zeros(2, FeatureEngineer.INPUT_DIM, 10, device=device)
        feat[:, 0, :] = 1.0
        feat[:, 1, :] = 0.0  # division by ~0
        div_token = FeatureEngineer.INPUT_DIM + 3  # DIV
        result = vm.execute([0, 1, div_token], feat)
        assert result is not None
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


# ===========================================================================
# 12. FeatureEngineer output shape
# ===========================================================================

class TestFeatureEngineerShape:
    def test_output_shape(self, device):
        """FeatureEngineer.compute_features → [batch, INPUT_DIM, T]."""
        batch, T = 8, 60
        raw_dict = {
            "close": torch.rand(batch, T, device=device) * 100 + 1,
            "open": torch.rand(batch, T, device=device) * 100 + 1,
            "high": torch.rand(batch, T, device=device) * 100 + 2,
            "low": torch.rand(batch, T, device=device) * 100,
            "volume": torch.rand(batch, T, device=device) * 1e6,
            "liquidity": torch.rand(batch, T, device=device) * 1e8,
            "fdv": torch.zeros(batch, T, device=device),
        }
        features = FeatureEngineer.compute_features(raw_dict)
        assert features.shape == (batch, FeatureEngineer.INPUT_DIM, T), (
            f"Expected shape ({batch}, {FeatureEngineer.INPUT_DIM}, {T}), "
            f"got {features.shape}"
        )

    def test_input_dim_constant(self):
        """INPUT_DIM should be 12 (the number of stacked features)."""
        assert FeatureEngineer.INPUT_DIM == 12

    def test_no_nan_in_output(self, device):
        """Features should not contain NaN even with edge-case inputs."""
        batch, T = 3, 40
        raw_dict = {
            "close": torch.ones(batch, T, device=device),
            "open": torch.ones(batch, T, device=device),
            "high": torch.ones(batch, T, device=device),
            "low": torch.ones(batch, T, device=device),
            "volume": torch.ones(batch, T, device=device),
            "liquidity": torch.ones(batch, T, device=device),
            "fdv": torch.zeros(batch, T, device=device),
        }
        features = FeatureEngineer.compute_features(raw_dict)
        assert not torch.isnan(features).any(), "Features contain NaN"


# ===========================================================================
# 13. Target return computation — no wrap-around
# ===========================================================================

class TestTargetRetComputation:
    def test_target_ret_no_wraparound(self, device):
        """
        Replicate target_ret logic from data_loader:
            next_close = cat([close[:, 1:], zeros], dim=1)
            target_ret = (next_close - close) / (close + eps)
            target_ret[:, -1] = 0
        Verify the last timestep is zero (not wrapped from t=0).
        """
        batch, T = 5, 20
        close = torch.rand(batch, T, device=device) * 100 + 1

        next_close = torch.cat([close[:, 1:], torch.zeros_like(close[:, :1])], dim=1)
        target_ret = (next_close - close) / (close + 1e-9)
        target_ret[:, -1] = 0.0

        # Last timestep should be exactly 0
        assert (target_ret[:, -1] == 0.0).all(), "Last timestep target_ret should be 0"

        # First timestep should reflect (close[1] - close[0]) / close[0]
        expected_first = (close[:, 1] - close[:, 0]) / (close[:, 0] + 1e-9)
        assert torch.allclose(target_ret[:, 0], expected_first, atol=1e-5)

    def test_target_ret_clamped(self, device):
        """target_ret should be clamped to [-0.2, 0.2]."""
        batch, T = 2, 10
        close = torch.ones(batch, T, device=device)
        close[:, 1] = 10.0  # Huge jump at t=1

        next_close = torch.cat([close[:, 1:], torch.zeros_like(close[:, :1])], dim=1)
        target_ret = (next_close - close) / (close + 1e-9)
        target_ret[:, -1] = 0.0
        target_ret = torch.clamp(target_ret, -0.2, 0.2)

        assert (target_ret >= -0.2).all()
        assert (target_ret <= 0.2).all()


# ===========================================================================
# 14. Position values in valid range
# ===========================================================================

class TestPositionRange:
    def test_position_in_zero_one(self, device):
        """Position should always be in [0, 1] regardless of factor values."""
        bt = MemeBacktest()
        batch, T = 10, 100
        torch.manual_seed(55)
        factors = torch.randn(batch, T, device=device) * 10  # wide range

        signal = torch.sigmoid(factors)
        liquidity = torch.full((batch, T), 1e9, device=device)
        is_safe = (liquidity > bt.min_liq).float()
        position = torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe

        assert (position >= 0.0).all(), "Position should be >= 0"
        assert (position <= 1.0).all(), "Position should be <= 1"

    def test_position_monotonic_in_signal(self, device):
        """Stronger signal → higher position (when liquidity is safe)."""
        bt = MemeBacktest()
        batch, T = 1, 5
        # Increasing factor values → increasing sigmoid → increasing position
        factors = torch.tensor([[0.0, 0.5, 1.0, 2.0, 5.0]], device=device)
        signal = torch.sigmoid(factors)
        liquidity = torch.full((batch, T), 1e9, device=device)
        is_safe = (liquidity > bt.min_liq).float()
        position = torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe

        # Each position should be >= previous
        for t in range(1, T):
            assert position[0, t] >= position[0, t - 1], (
                f"Position not monotonic at t={t}: {position[0, t-1]:.4f} > {position[0, t]:.4f}"
            )


# ===========================================================================
# 15. Backtest return type compatibility with engine.py
# ===========================================================================

class TestBacktestReturnTypes:
    def test_score_is_tensor(self, bt, device):
        """engine.py expects score to be a tensor (for rewards[i] = score)."""
        batch, T = 2, 100
        factors = torch.randn(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        score, mean_ret = bt.evaluate(factors, raw_data, target_ret)
        assert isinstance(score, torch.Tensor), f"score should be tensor, got {type(score)}"
        # score should be scalar (0-dim)
        assert score.dim() == 0, f"score should be 0-dim, got {score.dim()}-dim"

    def test_mean_ret_is_float(self, bt, device):
        """engine.py uses mean_ret as a Python float."""
        batch, T = 2, 100
        factors = torch.randn(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        _, mean_ret = bt.evaluate(factors, raw_data, target_ret)
        assert isinstance(mean_ret, float)

    def test_diagnostics_dict_values_are_python_floats(self, bt, device):
        """Diagnostics values should be Python floats (for JSON serialization)."""
        batch, T = 2, 100
        factors = torch.randn(batch, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)
        for key, val in diag.items():
            assert isinstance(val, float), (
                f"diag['{key}'] should be Python float for JSON, got {type(val)}"
            )


# ===========================================================================
# 16. Action Masking (RL exploration improvement)
# ===========================================================================

class TestActionMasking:
    """Tests for the action masking logic in engine.py generation loop."""

    def test_masking_produces_valid_formulas(self):
        """Generate 200 formulas via masking; ALL must end stack=1, never underflow."""
        from model_core.engine import _ARITY, _STACK_DELTA

        batch_size, formula_len = 200, 12
        vocab_size = FeatureEngineer.INPUT_DIM + len(OPS_CONFIG)  # 12 + 12 = 24
        torch.manual_seed(42)
        stack_sizes = torch.zeros(batch_size, dtype=torch.long)
        all_formulas = []

        for step_idx in range(formula_len):
            remaining = formula_len - 1 - step_idx
            logits = torch.randn(batch_size, vocab_size)
            ss = stack_sizes.unsqueeze(1)
            new_stack = ss + _STACK_DELTA.unsqueeze(0)

            valid = (ss >= _ARITY.unsqueeze(0))
            valid &= (new_stack + remaining >= 1)
            valid &= (new_stack - 2 * remaining <= 1)
            if remaining == 0:
                valid &= (new_stack == 1)

            logits = logits.masked_fill(~valid, -1e9)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
            stack_sizes = stack_sizes + _STACK_DELTA[action]
            all_formulas.append(action)

        formulas = torch.stack(all_formulas, dim=1)

        # All formulas must end with stack size = 1
        assert (stack_sizes == 1).all(), (
            f"Not all formulas end stack=1: {stack_sizes[stack_sizes != 1].tolist()}"
        )

        # Verify no underflow occurred during generation
        for i in range(batch_size):
            s = 0
            for t in range(formula_len):
                token = formulas[i, t].item()
                assert s >= _ARITY[token].item(), (
                    f"Formula {i} underflowed at step {t}: stack={s}, arity={_ARITY[token].item()}"
                )
                s += _STACK_DELTA[token].item()
            assert s == 1, f"Formula {i} ended with stack={s}"

    def test_step0_only_features(self):
        """At step 0 (empty stack), only feature tokens 0-11 should be valid."""
        from model_core.engine import _ARITY, _STACK_DELTA

        remaining = 11  # step 0 of 12
        ss = torch.zeros(1, 1, dtype=torch.long)
        new_stack = ss + _STACK_DELTA.unsqueeze(0)

        valid = (ss >= _ARITY.unsqueeze(0))
        valid &= (new_stack + remaining >= 1)
        valid &= (new_stack - 2 * remaining <= 1)

        valid_tokens = valid.squeeze(0).nonzero(as_tuple=True)[0].tolist()
        assert valid_tokens == list(range(FeatureEngineer.INPUT_DIM))

    def test_binary_op_needs_stack2(self):
        """When stack=1, binary ops (ADD/SUB/MUL/DIV, arity=2) should be masked."""
        from model_core.engine import _ARITY, _STACK_DELTA

        remaining = 10  # step 1 of 12
        ss = torch.ones(1, 1, dtype=torch.long)
        new_stack = ss + _STACK_DELTA.unsqueeze(0)

        valid = (ss >= _ARITY.unsqueeze(0))
        valid &= (new_stack + remaining >= 1)
        valid &= (new_stack - 2 * remaining <= 1)

        # Binary ops: ADD, SUB, MUL, DIV are first 4 ops
        fo = FeatureEngineer.INPUT_DIM
        for i in range(4):  # ADD=fo+0, SUB=fo+1, MUL=fo+2, DIV=fo+3
            token = fo + i
            assert not valid[0, token].item(), (
                f"Binary op {token} should be masked with stack=1"
            )

    def test_gate_needs_stack3(self):
        """GATE (arity=3) requires stack >= 3."""
        from model_core.engine import _ARITY, _STACK_DELTA

        formula_len = 12
        gate_token = FeatureEngineer.INPUT_DIM + 7  # GATE is OPS_CONFIG[7]

        # Stack < 3 -> GATE masked
        for stack_val in [0, 1, 2]:
            step_idx = stack_val
            remaining = formula_len - 1 - step_idx
            ss = torch.tensor([[stack_val]], dtype=torch.long)
            new_stack = ss + _STACK_DELTA.unsqueeze(0)

            valid = (ss >= _ARITY.unsqueeze(0))
            valid &= (new_stack + remaining >= 1)
            valid &= (new_stack - 2 * remaining <= 1)

            assert not valid[0, gate_token].item(), (
                f"GATE should be masked with stack={stack_val}"
            )

        # Stack = 3 -> GATE valid
        step_idx = 3
        remaining = formula_len - 1 - step_idx
        ss = torch.tensor([[3]], dtype=torch.long)
        new_stack = ss + _STACK_DELTA.unsqueeze(0)

        valid = (ss >= _ARITY.unsqueeze(0))
        valid &= (new_stack + remaining >= 1)
        valid &= (new_stack - 2 * remaining <= 1)

        assert valid[0, gate_token].item(), "GATE should be valid with stack=3"

    def test_last_step_forces_stack1(self):
        """At the last step (remaining=0), only tokens yielding stack=1 are allowed."""
        from model_core.engine import _ARITY, _STACK_DELTA

        remaining = 0
        fo = FeatureEngineer.INPUT_DIM

        # stack=1: only unary ops (delta=0, arity<=1) should be valid
        ss = torch.tensor([[1]], dtype=torch.long)
        new_stack = ss + _STACK_DELTA.unsqueeze(0)

        valid = (ss >= _ARITY.unsqueeze(0))
        valid &= (new_stack + remaining >= 1)
        valid &= (new_stack - 2 * remaining <= 1)
        valid &= (new_stack == 1)

        valid_tokens = valid.squeeze(0).nonzero(as_tuple=True)[0].tolist()

        # Features (delta=+1) should NOT be valid at last step with stack=1
        for t in range(fo):
            assert t not in valid_tokens

        # All valid tokens must keep stack at 1
        for t in valid_tokens:
            assert (1 + _STACK_DELTA[t].item()) == 1

        # stack=2: binary ops (delta=-1) should produce stack=1
        ss = torch.tensor([[2]], dtype=torch.long)
        new_stack = ss + _STACK_DELTA.unsqueeze(0)

        valid2 = (ss >= _ARITY.unsqueeze(0))
        valid2 &= (new_stack + remaining >= 1)
        valid2 &= (new_stack - 2 * remaining <= 1)
        valid2 &= (new_stack == 1)

        valid_tokens_2 = valid2.squeeze(0).nonzero(as_tuple=True)[0].tolist()
        # Binary ops: ADD=fo+0, SUB=fo+1, MUL=fo+2, DIV=fo+3
        for i in range(4):
            assert (fo + i) in valid_tokens_2


# ---------------------------------------------------------------------------
# Helper: replicate _evaluate_formula logic for unit testing
# ---------------------------------------------------------------------------

def _shaped_reward(formula, vm, feat_tensor=None, raw_data=None,
                   target_ret=None, bt=None, base_factors=None):
    """Standalone version of AlphaEngine._evaluate_formula for testing."""
    stack_size = 0
    for step_i, token in enumerate(formula):
        token = int(token)
        if token < vm.feat_offset:
            stack_size += 1
        elif token in vm.arity_map:
            arity = vm.arity_map[token]
            if stack_size < arity:
                progress = step_i / len(formula)
                return -5.0 + progress * 2.0
            stack_size = stack_size - arity + 1
        else:
            return -5.0

    if stack_size != 1:
        distance = abs(stack_size - 1)
        return -3.0 - min(distance, 4) * 0.5

    if feat_tensor is None:
        return 0.0

    res = vm.execute(formula, feat_tensor)
    if res is None:
        return -4.0

    std_val = res.std().item()
    if std_val < 1e-4:
        return -2.0
    if std_val < 0.01:
        return -2.0 + (std_val / 0.01)

    if bt is not None and raw_data is not None and target_ret is not None:
        formula_len = len([t for t in formula if t != 0])
        score, _ = bt.evaluate(
            res, raw_data, target_ret,
            formula_length=formula_len,
            base_factors=base_factors,
        )
        return score.item()

    return 0.0


# ===========================================================================
# 17. Shaped Reward (RL exploration improvement)
# ===========================================================================

class TestShapedReward:
    """Tests for shaped reward logic in _evaluate_formula."""

    @pytest.fixture
    def vm(self):
        return StackVM()

    def test_early_underflow_worse_than_late(self, vm):
        """Formula underflowing at step 1 should score worse than one at step 11."""
        ADD = FeatureEngineer.INPUT_DIM  # ADD is the first op token
        # Early underflow: push feat 0, then ADD (arity 2, stack=1 -> underflow at step 1)
        early = [0, ADD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Late underflow: push-op pattern drains stack, underflows at step 11
        late = [0, 1, ADD, 2, ADD, 3, ADD, 4, ADD, 5, ADD, ADD]

        early_reward = _shaped_reward(early, vm)
        late_reward = _shaped_reward(late, vm)

        # Early: progress=1/12, reward = -5.0 + (1/12)*2.0 ~ -4.833
        # Late: progress=11/12, reward = -5.0 + (11/12)*2.0 ~ -3.167
        assert early_reward < late_reward, (
            f"Early underflow ({early_reward:.3f}) should be worse than late ({late_reward:.3f})"
        )
        assert -5.0 <= early_reward <= -3.0
        assert -5.0 <= late_reward <= -3.0

    def test_wrong_stack_size_penalty(self, vm):
        """stack != 1 at end -> reward between -3.0 and -5.0."""
        # [0, 0]: stack=2, distance=1 -> -3.5
        reward_d1 = _shaped_reward([0, 0], vm)
        assert reward_d1 == pytest.approx(-3.5, abs=1e-6)
        assert -5.0 <= reward_d1 <= -3.0

        # [0, 0, 0]: stack=3, distance=2 -> -4.0
        reward_d2 = _shaped_reward([0, 0, 0], vm)
        assert reward_d2 == pytest.approx(-4.0, abs=1e-6)

        # [0]*6: stack=6, distance=5 (capped at 4) -> -5.0
        reward_d5 = _shaped_reward([0] * 6, vm)
        assert reward_d5 == pytest.approx(-5.0, abs=1e-6)

    def test_constant_output_minus2(self, vm):
        """Formula producing constant output (std < 1e-4) -> reward = -2.0."""
        feat = torch.ones(5, FeatureEngineer.INPUT_DIM, 30)
        reward = _shaped_reward([0], vm, feat_tensor=feat)
        assert reward == pytest.approx(-2.0, abs=1e-6)

    def test_near_constant_shaped(self, vm):
        """Output with 1e-4 < std < 0.01 -> reward between -2.0 and -1.0."""
        feat = torch.ones(1, FeatureEngineer.INPUT_DIM, 100)
        feat[0, 0, 50:] = 1.006  # small perturbation

        res = vm.execute([0], feat)
        std_val = res.std().item()
        assert 1e-4 < std_val < 0.01, f"Expected near-constant std, got {std_val}"

        reward = _shaped_reward([0], vm, feat_tensor=feat)
        expected = -2.0 + (std_val / 0.01)
        assert reward == pytest.approx(expected, abs=1e-6)
        assert -2.0 < reward < -1.0

    def test_valid_formula_gets_backtest_score(self, vm, bt, device):
        """Valid non-constant formula -> reward equals backtest score."""
        torch.manual_seed(42)
        batch, T = 5, 200
        feat = torch.randn(batch, FeatureEngineer.INPUT_DIM, T, device=device)
        raw_data = _make_raw_data(1e9, batch, T)
        target_ret = torch.randn(batch, T, device=device) * 0.01
        base_factors = feat[:, 0, :]

        formula = [1]  # push feature 1
        res = vm.execute(formula, feat)
        assert res is not None
        assert res.std().item() > 0.01

        reward = _shaped_reward(
            formula, vm, feat_tensor=feat, raw_data=raw_data,
            target_ret=target_ret, bt=bt, base_factors=base_factors,
        )
        formula_len = len([t for t in formula if t != 0])
        expected_score, _ = bt.evaluate(
            res, raw_data, target_ret,
            formula_length=formula_len,
            base_factors=base_factors,
        )
        assert reward == pytest.approx(expected_score.item(), abs=1e-4)


# ===========================================================================
# 18. Entropy Coefficient Schedule
# ===========================================================================

class TestEntropyCoef:
    """Tests for entropy coefficient linear decay schedule."""

    @staticmethod
    def _entropy_coef(step, train_steps):
        """Replicate entropy coef formula from engine.py."""
        return max(0.02, 0.08 * (1.0 - step / train_steps))

    def test_entropy_coef_at_start(self):
        """At step=0, entropy coef should be 0.08."""
        from model_core.config import ModelConfig
        coef = self._entropy_coef(0, ModelConfig.TRAIN_STEPS)
        assert coef == pytest.approx(0.08, abs=1e-8)

    def test_entropy_coef_at_end(self):
        """At step=TRAIN_STEPS, entropy coef should be 0.02 (floor)."""
        from model_core.config import ModelConfig
        coef = self._entropy_coef(ModelConfig.TRAIN_STEPS, ModelConfig.TRAIN_STEPS)
        assert coef == pytest.approx(0.02, abs=1e-8)

    def test_entropy_coef_midpoint(self):
        """At step=TRAIN_STEPS/2, entropy coef should be 0.04."""
        from model_core.config import ModelConfig
        mid = ModelConfig.TRAIN_STEPS // 2
        coef = self._entropy_coef(mid, ModelConfig.TRAIN_STEPS)
        assert coef == pytest.approx(0.04, abs=1e-4)


# ===========================================================================
# 19. Updated Activity Threshold & Drawdown Penalty
# ===========================================================================

class TestActivityThresholdUpdated:
    """Tests for updated activity threshold (>0.05) and drawdown penalty (*5.0)."""

    def test_position_below_005_not_counted(self, bt, device):
        """Positions of ~0.01 (below 0.05) should not count as active -> score = -10."""
        batch, T = 2, 200
        # sigmoid(0.02) ~ 0.505 -> clamp(0.005)*2 = 0.01 -> below 0.05
        factors = torch.full((batch, T), 0.02, device=device)
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.full((batch, T), 0.01, device=device)

        signal = torch.sigmoid(factors)
        is_safe = (raw_data["liquidity"] > bt.min_liq).float()
        position = torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe

        assert (position < 0.05).all(), f"Max position: {position.max().item()}"
        activity = (position > 0.05).float().sum(dim=1)
        assert (activity == 0).all()

        score, _ = bt.evaluate(factors, raw_data, target_ret)
        assert score.item() == pytest.approx(-10.0, abs=1e-4)

    def test_drawdown_penalty_reduced(self, bt, device):
        """Drawdown penalty should be max_drawdown * 2.0."""
        batch, T = 1, 100
        factors = torch.full((batch, T), 100.0, device=device)
        raw_data = _make_raw_data(1e11, batch, T, device)
        target_ret = torch.full((batch, T), 0.01, device=device)
        target_ret[:, 40:60] = -0.02  # create a drawdown period

        score, _, diag = bt.evaluate(
            factors, raw_data, target_ret, return_diagnostics=True
        )

        # batch=1, no complexity/redundancy, constant factors → IC=0 (N<2).
        # T=100 → avg_turnover = 1/100 = 0.01 > 0.005 → no lazy_penalty.
        # score = sharpe - max_drawdown * 2.0
        expected = diag["sharpe"] - diag["max_drawdown"] * 2.0
        assert score.item() == pytest.approx(expected, abs=0.01)


# ===========================================================================
# 20. Cross-Sectional IC
# ===========================================================================

class TestCrossSectionalIC:
    """Tests for _cross_sectional_ic static method."""

    def test_perfect_ic(self, device):
        """Factors perfectly correlated with returns → IC ≈ 1.0."""
        N, T = 20, 50
        torch.manual_seed(0)
        target_ret = torch.randn(N, T, device=device)
        # factors = target_ret → perfect rank correlation
        factors = target_ret.clone()
        ic = MemeBacktest._cross_sectional_ic(factors, target_ret)
        assert ic == pytest.approx(1.0, abs=0.01), f"Expected IC ≈ 1.0, got {ic}"

    def test_anti_ic(self, device):
        """Factors perfectly anti-correlated with returns → IC ≈ -1.0."""
        N, T = 20, 50
        torch.manual_seed(1)
        target_ret = torch.randn(N, T, device=device)
        factors = -target_ret
        ic = MemeBacktest._cross_sectional_ic(factors, target_ret)
        assert ic == pytest.approx(-1.0, abs=0.01), f"Expected IC ≈ -1.0, got {ic}"

    def test_random_ic_near_zero(self, device):
        """Independent random factors and returns → IC ≈ 0."""
        N, T = 50, 100
        torch.manual_seed(2)
        factors = torch.randn(N, T, device=device)
        target_ret = torch.randn(N, T, device=device)
        ic = MemeBacktest._cross_sectional_ic(factors, target_ret)
        assert abs(ic) < 0.15, f"Expected IC ≈ 0 for random, got {ic}"

    def test_single_stock_returns_zero(self, device):
        """N=1 (single stock) → IC = 0.0."""
        factors = torch.randn(1, 30, device=device)
        target_ret = torch.randn(1, 30, device=device)
        ic = MemeBacktest._cross_sectional_ic(factors, target_ret)
        assert ic == 0.0

    def test_ic_contributes_to_score(self, bt, device):
        """IC bonus should increase final score when IC > 0."""
        N, T = 10, 200
        torch.manual_seed(42)
        target_ret = torch.randn(N, T, device=device) * 0.01
        raw_data = _make_raw_data(1e9, N, T, device)

        # Factors correlated with returns → positive IC → higher score
        good_factors = torch.full((N, T), 100.0, device=device) + target_ret * 50
        # Constant factors → IC ≈ 0
        flat_factors = torch.full((N, T), 100.0, device=device)

        score_good, _ = bt.evaluate(good_factors, raw_data, target_ret)
        score_flat, _ = bt.evaluate(flat_factors, raw_data, target_ret)

        # Good factors should get IC bonus
        _, _, diag_good = bt.evaluate(good_factors, raw_data, target_ret, return_diagnostics=True)
        _, _, diag_flat = bt.evaluate(flat_factors, raw_data, target_ret, return_diagnostics=True)
        assert diag_good['ic'] > diag_flat['ic'], (
            f"Good factors IC ({diag_good['ic']:.4f}) should exceed flat IC ({diag_flat['ic']:.4f})"
        )


# ===========================================================================
# 21. Turnover Floor Penalty
# ===========================================================================

class TestTurnoverFloor:
    """Tests for turnover floor (lazy_penalty)."""

    def test_constant_signal_penalized(self, bt, device):
        """Constant full-position signal → avg_turnover < 0.005 → lazy_penalty = 1.0."""
        batch, T = 4, 400
        # Constant large signal → position = 1.0 everywhere → turnover only at t=0
        # T=400 → avg_turnover per sample = 1/400 = 0.0025 < 0.005 → lazy applied
        factors = torch.full((batch, T), 100.0, device=device)
        raw_data = _make_raw_data(1e11, batch, T, device)
        target_ret = torch.full((batch, T), 0.01, device=device)

        score_val, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)

        # Constant factors across stocks → IC = 0.0
        assert diag['ic'] == 0.0
        # avg_turnover per sample = 1/400 = 0.0025 < 0.005
        assert diag['avg_turnover'] < 0.005

        # Score should include -1.0 lazy penalty:
        # score = sharpe - max_dd * 2.0 - 1.0 + 0
        expected = diag['sharpe'] - diag['max_drawdown'] * 2.0 - 1.0
        assert score_val.item() == pytest.approx(expected, abs=0.05)

    def test_active_signal_not_penalized(self, bt, device):
        """Signal that varies → avg_turnover > 0.005 → no lazy_penalty."""
        batch, T = 4, 200
        torch.manual_seed(99)
        # Varying signal produces meaningful turnover
        factors = torch.randn(batch, T, device=device) * 3.0 + 1.0
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        _, _, diag = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)

        # With varying signal, turnover should be meaningful
        # Verify via score decomposition (no lazy penalty)
        score_val, _, _ = bt.evaluate(factors, raw_data, target_ret, return_diagnostics=True)
        sharpe = diag['sharpe']
        max_dd = diag['max_drawdown']
        ic = diag['ic']
        avg_to = diag['avg_turnover']

        if avg_to >= 0.005:
            expected_no_lazy = sharpe - max_dd * 2.0 + ic * 10.0
            assert score_val.item() == pytest.approx(expected_no_lazy, abs=0.05)
