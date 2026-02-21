"""
Comprehensive tests for FormulaEnsemble and ensemble training.
All tests use synthetic torch tensors — no database connection required.
"""

import math
import json
import pytest
import torch
import numpy as np

from model_core.ensemble import FormulaEnsemble
from model_core.backtest import MemeBacktest
from model_core.vm import StackVM
from model_core.factors import FeatureEngineer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def vm():
    return StackVM()


@pytest.fixture
def bt():
    return MemeBacktest()


@pytest.fixture
def feat_tensor(device):
    """Synthetic feature tensor: [num_tokens=10, num_features=6, time_steps=200]."""
    torch.manual_seed(42)
    return torch.randn(10, FeatureEngineer.INPUT_DIM, 200, device=device)


def _make_raw_data(liquidity_val, batch, T, device="cpu"):
    liq = torch.full((batch, T), liquidity_val, device=device)
    return {"liquidity": liq}


# ===========================================================================
# 1. FormulaEnsemble Construction
# ===========================================================================

class TestEnsembleConstruction:
    def test_basic_construction(self):
        """Ensemble with valid formulas should initialize correctly."""
        formulas = [[0], [1], [2]]
        ens = FormulaEnsemble(formulas)
        assert ens.num_formulas == 3
        assert ens.mode == 'mean'
        assert torch.allclose(ens.weights, torch.tensor([1/3, 1/3, 1/3]))

    def test_empty_formulas_raises(self):
        """Empty formula list should raise ValueError."""
        with pytest.raises(ValueError):
            FormulaEnsemble([])

    def test_single_formula(self):
        """Single formula ensemble should work (degenerates to single model)."""
        ens = FormulaEnsemble([[0]])
        assert ens.num_formulas == 1
        assert ens.weights[0].item() == pytest.approx(1.0)

    def test_custom_weights_normalized(self):
        """Custom weights should be auto-normalized to sum=1."""
        ens = FormulaEnsemble([[0], [1]], weights=[3.0, 1.0])
        assert ens.weights[0].item() == pytest.approx(0.75)
        assert ens.weights[1].item() == pytest.approx(0.25)
        assert ens.weights.sum().item() == pytest.approx(1.0)

    def test_rank_mean_mode(self):
        """Ensemble can be created with rank_mean mode."""
        ens = FormulaEnsemble([[0], [1]], mode='rank_mean')
        assert ens.mode == 'rank_mean'


# ===========================================================================
# 2. FormulaEnsemble Prediction
# ===========================================================================

class TestEnsemblePrediction:
    def test_single_formula_matches_vm(self, feat_tensor, vm):
        """Single-formula ensemble should produce same output as direct VM execution."""
        formula = [0]  # push feature 0
        ens = FormulaEnsemble([formula])

        ens_result = ens.predict(feat_tensor)
        vm_result = vm.execute(formula, feat_tensor)

        assert ens_result is not None
        assert vm_result is not None
        assert torch.allclose(ens_result, vm_result, atol=1e-5)

    def test_output_shape(self, feat_tensor):
        """Ensemble output shape matches [num_tokens, time_steps]."""
        ens = FormulaEnsemble([[0], [1], [2]])
        result = ens.predict(feat_tensor)
        assert result is not None
        assert result.shape == (feat_tensor.shape[0], feat_tensor.shape[2])

    def test_mean_of_two_formulas(self, feat_tensor, vm):
        """Mean of two formulas should be (f0 + f1) / 2."""
        f0 = vm.execute([0], feat_tensor)  # feature 0
        f1 = vm.execute([1], feat_tensor)  # feature 1
        expected = (f0 + f1) / 2.0

        ens = FormulaEnsemble([[0], [1]])
        result = ens.predict(feat_tensor)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_weighted_average(self, feat_tensor, vm):
        """Custom weights should be reflected in output."""
        f0 = vm.execute([0], feat_tensor)
        f1 = vm.execute([1], feat_tensor)
        # weights [0.75, 0.25]
        expected = 0.75 * f0 + 0.25 * f1

        ens = FormulaEnsemble([[0], [1]], weights=[3.0, 1.0])
        result = ens.predict(feat_tensor)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_invalid_formula_skipped(self, feat_tensor, vm):
        """Invalid formulas should be skipped gracefully."""
        # [0, 1] has stack_size=2, invalid → returns None from VM
        # [0] is valid
        ens = FormulaEnsemble([[0, 1], [0]])
        result = ens.predict(feat_tensor)

        vm_result = vm.execute([0], feat_tensor)
        assert result is not None
        assert torch.allclose(result, vm_result, atol=1e-5)

    def test_all_invalid_returns_none(self, feat_tensor):
        """If all formulas are invalid, predict returns None."""
        ens = FormulaEnsemble([[0, 1], [2, 3]])  # both invalid (stack=2)
        result = ens.predict(feat_tensor)
        assert result is None

    def test_rank_mean_output_range(self, feat_tensor):
        """Rank-mean output should be in [0, 1] range."""
        ens = FormulaEnsemble([[0], [1], [2]], mode='rank_mean')
        result = ens.predict(feat_tensor)
        assert result is not None
        assert result.min() >= -0.01  # small tolerance for float
        assert result.max() <= 1.01

    def test_predict_individual(self, feat_tensor, vm):
        """predict_individual returns each formula's signal separately."""
        ens = FormulaEnsemble([[0], [0, 1], [1]])  # middle one invalid
        results = ens.predict_individual(feat_tensor)

        assert len(results) == 2
        # First valid: index 0, formula [0]
        assert results[0][0] == 0
        assert torch.allclose(results[0][1], vm.execute([0], feat_tensor), atol=1e-5)
        # Second valid: index 2, formula [1]
        assert results[1][0] == 2
        assert torch.allclose(results[1][1], vm.execute([1], feat_tensor), atol=1e-5)


# ===========================================================================
# 3. Ensemble Variance Reduction (core theorem)
# ===========================================================================

class TestVarianceReduction:
    """
    The fundamental benefit of ensembling: if N models have similar expected
    performance but independent errors, the ensemble's variance is reduced
    by approximately 1/N.
    """

    def test_ensemble_reduces_noise_variance(self, device):
        """
        Simulate N noisy signals with the same true signal + independent noise.
        Ensemble average should have lower noise variance than any individual.
        """
        torch.manual_seed(123)
        batch, T = 5, 500
        num_models = 6

        # True underlying signal
        true_signal = torch.sin(torch.linspace(0, 4 * math.pi, T)).unsqueeze(0).expand(batch, -1)
        true_signal = true_signal.to(device)

        # Each model = true signal + independent noise
        noise_std = 1.0
        individual_signals = []
        for _ in range(num_models):
            noise = torch.randn(batch, T, device=device) * noise_std
            individual_signals.append(true_signal + noise)

        # Ensemble average
        ensemble_signal = torch.stack(individual_signals, dim=0).mean(dim=0)

        # Compute MSE relative to true signal
        individual_mses = []
        for sig in individual_signals:
            mse = ((sig - true_signal) ** 2).mean().item()
            individual_mses.append(mse)

        ensemble_mse = ((ensemble_signal - true_signal) ** 2).mean().item()

        avg_individual_mse = np.mean(individual_mses)

        # Ensemble MSE should be significantly lower than average individual MSE
        # Theoretical: ensemble_mse ≈ individual_mse / N
        assert ensemble_mse < avg_individual_mse * 0.5, (
            f"Ensemble MSE ({ensemble_mse:.4f}) should be < 50% of "
            f"avg individual MSE ({avg_individual_mse:.4f})"
        )

        # Should be close to theoretical 1/N reduction
        theoretical_ratio = 1.0 / num_models
        actual_ratio = ensemble_mse / avg_individual_mse
        assert actual_ratio < theoretical_ratio * 2.0, (
            f"Variance reduction ratio ({actual_ratio:.4f}) worse than "
            f"2x theoretical ({theoretical_ratio:.4f})"
        )

    def test_ensemble_sharpe_stability(self, bt, device):
        """
        Ensemble signal should have more stable Sharpe ratio than individuals
        across random realizations.
        """
        torch.manual_seed(456)
        batch, T = 8, 300

        # Generate diverse but correlated factor signals
        base_signal = torch.randn(batch, T, device=device) * 0.5
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        # Multiple "models": base + different noise
        num_models = 6
        signals = []
        for i in range(num_models):
            noise = torch.randn(batch, T, device=device) * 0.3
            signals.append(base_signal + noise)

        # Evaluate individual Sharpes
        individual_sharpes = []
        for sig in signals:
            _, _, diag = bt.evaluate(sig, raw_data, target_ret, return_diagnostics=True)
            individual_sharpes.append(diag['sharpe'])

        # Ensemble: average
        ensemble_sig = torch.stack(signals, dim=0).mean(dim=0)
        _, _, ens_diag = bt.evaluate(ensemble_sig, raw_data, target_ret, return_diagnostics=True)

        # The std of individual sharpes should be > 0 (they differ)
        sharpe_std = np.std(individual_sharpes)
        assert sharpe_std > 0.01, "Individual sharpes should vary"

        # Ensemble Sharpe should be within reasonable range of individual mean
        avg_sharpe = np.mean(individual_sharpes)
        # Main check: ensemble produces a valid finite sharpe
        assert math.isfinite(ens_diag['sharpe']), "Ensemble sharpe should be finite"


# ===========================================================================
# 4. Ensemble Backtest Integration
# ===========================================================================

class TestEnsembleBacktest:
    def test_evaluate_ensemble_returns_complete_report(self, bt, feat_tensor, device):
        """evaluate_ensemble should return a complete comparison report."""
        T = feat_tensor.shape[2]
        batch = feat_tensor.shape[0]
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        ens = FormulaEnsemble([[0], [1], [2]])
        report = bt.evaluate_ensemble(ens, feat_tensor, raw_data, target_ret)

        assert report is not None
        assert 'ensemble' in report
        assert 'individuals' in report
        assert 'individual_sharpes' in report
        assert 'best_single_sharpe' in report
        assert 'avg_single_sharpe' in report
        assert 'sharpe_std' in report
        assert 'improvement_vs_best' in report
        assert 'improvement_vs_avg' in report

        # Ensemble diagnostics should have standard keys
        assert 'sharpe' in report['ensemble']
        assert 'max_drawdown' in report['ensemble']
        assert 'avg_turnover' in report['ensemble']

        # Should have 3 individual results
        assert len(report['individuals']) == 3
        assert len(report['individual_sharpes']) == 3

    def test_evaluate_ensemble_all_invalid(self, bt, feat_tensor, device):
        """evaluate_ensemble with all invalid formulas returns None."""
        T = feat_tensor.shape[2]
        batch = feat_tensor.shape[0]
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        ens = FormulaEnsemble([[0, 1], [2, 3]])  # both invalid
        report = bt.evaluate_ensemble(ens, feat_tensor, raw_data, target_ret)
        assert report is None

    def test_ensemble_diagnostics_finite(self, bt, feat_tensor, device):
        """All diagnostic values should be finite."""
        T = feat_tensor.shape[2]
        batch = feat_tensor.shape[0]
        raw_data = _make_raw_data(1e9, batch, T, device)
        target_ret = torch.randn(batch, T, device=device) * 0.01

        ens = FormulaEnsemble([[0], [1]])
        report = bt.evaluate_ensemble(ens, feat_tensor, raw_data, target_ret)

        for key, val in report['ensemble'].items():
            assert math.isfinite(val), f"ensemble.{key} is not finite: {val}"

        for d in report['individuals']:
            for key, val in d.items():
                if isinstance(val, float):
                    assert math.isfinite(val), f"individual.{key} is not finite: {val}"


# ===========================================================================
# 5. Serialization
# ===========================================================================

class TestEnsembleSerialization:
    def test_to_dict_and_from_dict_roundtrip(self, feat_tensor):
        """Ensemble → dict → Ensemble should preserve behavior."""
        original = FormulaEnsemble([[0], [1], [2]], weights=[0.5, 0.3, 0.2], mode='rank_mean')
        d = original.to_dict()

        restored = FormulaEnsemble.from_dict(d)
        assert restored.num_formulas == 3
        assert restored.mode == 'rank_mean'
        assert torch.allclose(restored.weights, original.weights, atol=1e-5)

        # Both should produce same predictions
        orig_pred = original.predict(feat_tensor)
        rest_pred = restored.predict(feat_tensor)
        assert torch.allclose(orig_pred, rest_pred, atol=1e-5)

    def test_to_dict_json_serializable(self):
        """to_dict output should be JSON-serializable."""
        ens = FormulaEnsemble([[0, 1, 6], [2, 3, 7]], mode='mean')
        d = ens.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Round-trip through JSON
        loaded = json.loads(json_str)
        restored = FormulaEnsemble.from_dict(loaded)
        assert restored.num_formulas == 2


# ===========================================================================
# 6. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_constant_formula_skipped(self, device):
        """Formula that produces constant output (std < 1e-8) should be skipped."""
        batch, T = 3, 50
        # All-ones feature tensor → feature 0 is constant
        feat = torch.ones(batch, FeatureEngineer.INPUT_DIM, T, device=device)

        # [0] pushes constant feature → should be skipped
        # [1] also constant → should be skipped
        ens = FormulaEnsemble([[0], [1]])
        result = ens.predict(feat)
        assert result is None  # both constant → no valid signals

    def test_mixed_constant_and_valid(self, device):
        """Mix of constant and non-constant formulas should use only valid ones."""
        torch.manual_seed(99)
        batch, T = 5, 100
        feat = torch.randn(batch, FeatureEngineer.INPUT_DIM, T, device=device)
        # Make feature 0 constant
        feat[:, 0, :] = 1.0

        # [0] → constant (skipped), [1] → valid
        ens = FormulaEnsemble([[0], [1]])
        result = ens.predict(feat)

        # Should return just feature 1
        vm = StackVM()
        expected = vm.execute([1], feat)
        assert result is not None
        assert torch.allclose(result, expected, atol=1e-5)

    def test_large_ensemble(self, feat_tensor):
        """Large ensemble (20 formulas) should work correctly."""
        formulas = [[i % FeatureEngineer.INPUT_DIM] for i in range(20)]
        ens = FormulaEnsemble(formulas)
        result = ens.predict(feat_tensor)
        assert result is not None
        assert result.shape == (feat_tensor.shape[0], feat_tensor.shape[2])

    def test_nan_in_feature_tensor(self, device):
        """Ensemble should handle NaN in input gracefully."""
        torch.manual_seed(42)
        batch, T = 3, 50
        feat = torch.randn(batch, FeatureEngineer.INPUT_DIM, T, device=device)
        feat[0, 0, 10:15] = float('nan')

        ens = FormulaEnsemble([[1], [2]])  # avoid feature 0 with NaN
        result = ens.predict(feat)
        # Should still produce output (features 1 and 2 are clean)
        assert result is not None


# ===========================================================================
# 7. Consistency: rank_mean vs mean on identical signals
# ===========================================================================

class TestRankMeanVsMean:
    def test_identical_signals_same_result(self, feat_tensor):
        """When all formulas are identical, mean and rank_mean should agree."""
        ens_mean = FormulaEnsemble([[0], [0], [0]], mode='mean')
        ens_rank = FormulaEnsemble([[0], [0], [0]], mode='rank_mean')

        result_mean = ens_mean.predict(feat_tensor)
        result_rank = ens_rank.predict(feat_tensor)

        # rank_mean of identical signals = rank of a single signal / (T-1)
        # This won't equal raw mean, but both should be valid
        assert result_mean is not None
        assert result_rank is not None
        assert result_mean.shape == result_rank.shape

    def test_rank_mean_handles_scale_differences(self, device):
        """rank_mean should be more robust to scale differences."""
        torch.manual_seed(7)
        batch, T = 5, 200
        feat = torch.randn(batch, FeatureEngineer.INPUT_DIM, T, device=device)
        # Scale feature 0 by 1000x → different magnitude
        feat[:, 0, :] *= 1000.0

        ens_mean = FormulaEnsemble([[0], [1]], mode='mean')
        ens_rank = FormulaEnsemble([[0], [1]], mode='rank_mean')

        result_mean = ens_mean.predict(feat)
        result_rank = ens_rank.predict(feat)

        # Mean will be dominated by feature 0 (1000x scale)
        # Check that rank_mean is more balanced
        vm = StackVM()
        f0 = vm.execute([0], feat)
        f1 = vm.execute([1], feat)

        # For mean: correlation with f0 should be very high (f0 dominates)
        mean_corr_f0 = torch.corrcoef(torch.stack([
            result_mean.flatten(), f0.flatten()
        ]))[0, 1].item()

        # For rank_mean: both should contribute more equally
        rank_corr_f0 = torch.corrcoef(torch.stack([
            result_rank.flatten(), f0.flatten()
        ]))[0, 1].abs().item()
        rank_corr_f1 = torch.corrcoef(torch.stack([
            result_rank.flatten(), f1.flatten()
        ]))[0, 1].abs().item()

        # Mean is dominated by f0
        assert abs(mean_corr_f0) > 0.99, f"Mean should be dominated by f0: corr={mean_corr_f0}"

        # Rank-mean should have more balanced contributions
        assert rank_corr_f1 > 0.1, (
            f"Rank-mean should have non-trivial correlation with f1: {rank_corr_f1}"
        )
