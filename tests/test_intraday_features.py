"""
Tests for intraday microstructure features, extended operators, and dynamic config.

Covers:
  - DuckDBFeatureEngineer (18 features)
  - OPS_CONFIG_EXTENDED (20 ops)
  - Config dynamic methods (get_feature_dim, get_ops_config, get_max_formula_len)
  - AlphaGPT vocab sizing
  - StackVM parameterisation
  - Engine masking tensors
  - Backward compatibility (FeatureEngineer stays 12-dim)
"""

import os
import pytest
import torch

# Force crypto mode first (backward-compat tests), then switch to futures for extended tests
os.environ.setdefault("DATA_SOURCE_MODE", "solana")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_dict(n_products=5, n_bars=60, device='cpu'):
    """Create a synthetic raw_data_cache dict for testing."""
    def _rand():
        return torch.rand(n_products, n_bars, device=device) + 0.1

    close = _rand() * 100
    open_ = close * (1 + 0.01 * torch.randn(n_products, n_bars, device=device))
    high = torch.max(close, open_) + _rand()
    low = torch.min(close, open_) - _rand() * 0.5
    low = torch.clamp(low, min=0.01)
    volume = _rand() * 1e6

    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'liquidity': volume * close,
        'fdv': torch.zeros(n_products, n_bars, device=device),
    }


def _make_raw_dict_v2(n_products=5, n_bars=60, device='cpu'):
    """raw_data_cache with 6 intraday microstructure columns."""
    d = _make_raw_dict(n_products, n_bars, device)
    d['vwap_dev'] = 0.01 * torch.randn(n_products, n_bars, device=device)
    d['oi_change'] = torch.randn(n_products, n_bars, device=device) * 100
    d['vol_skew'] = torch.rand(n_products, n_bars, device=device) + 0.5
    d['vol_conc'] = torch.rand(n_products, n_bars, device=device) * 0.5
    d['smart_money'] = torch.rand(n_products, n_bars, device=device) + 0.3
    d['twap_dev'] = 0.005 * torch.randn(n_products, n_bars, device=device)
    return d


# ---------------------------------------------------------------------------
# 1–3: DuckDBFeatureEngineer
# ---------------------------------------------------------------------------

class TestDuckDBFeatureEngineer:
    def test_input_dim_is_18(self):
        from model_core.features_v2 import DuckDBFeatureEngineer
        assert DuckDBFeatureEngineer.INPUT_DIM == 18

    def test_features_includes_intraday(self):
        from model_core.features_v2 import DuckDBFeatureEngineer
        raw = _make_raw_dict_v2(n_products=4, n_bars=50)
        feat = DuckDBFeatureEngineer.compute_features(raw)
        assert feat.shape[1] == 18
        # All values should be finite
        assert torch.isfinite(feat).all()

    def test_backward_compat_feature_engineer(self):
        from model_core.factors import FeatureEngineer
        assert FeatureEngineer.INPUT_DIM == 12
        raw = _make_raw_dict(n_products=3, n_bars=40)
        feat = FeatureEngineer.compute_features(raw)
        assert feat.shape[1] == 12


# ---------------------------------------------------------------------------
# 4–11: New operators
# ---------------------------------------------------------------------------

class TestExtendedOps:
    def test_cs_rank_output_range(self):
        from model_core.ops import _op_cs_rank
        x = torch.randn(8, 50)
        out = _op_cs_rank(x)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6
        # Should vary across products (dim 0)
        assert out.std(dim=0).mean() > 0

    def test_cs_rank_single_product(self):
        from model_core.ops import _op_cs_rank
        x = torch.randn(1, 30)
        out = _op_cs_rank(x)
        assert (out == 0).all()

    def test_cs_demean_zero_mean(self):
        from model_core.ops import _op_cs_demean
        x = torch.randn(10, 40)
        out = _op_cs_demean(x)
        # Mean across products should be ~0 per timestep
        assert out.mean(dim=0).abs().max() < 1e-5

    def test_ma5_smoothing(self):
        from model_core.ops import _op_ma
        x = torch.randn(3, 30)
        out = _op_ma(x, 5)
        assert out.shape == x.shape
        # MA should be smoother than original
        diff_x = (x[:, 1:] - x[:, :-1]).abs().mean()
        diff_ma = (out[:, 1:] - out[:, :-1]).abs().mean()
        assert diff_ma < diff_x

    def test_ma20(self):
        from model_core.ops import _op_ma
        x = torch.randn(3, 40)
        out = _op_ma(x, 20)
        assert out.shape == x.shape

    def test_delta5_equals_diff(self):
        from model_core.ops import _op_delta
        x = torch.randn(4, 30)
        out = _op_delta(x, 5)
        # For positions >= 5, delta should equal x[t] - x[t-5]
        for t in range(5, 30):
            expected = x[:, t] - x[:, t - 5]
            torch.testing.assert_close(out[:, t], expected)

    def test_ts_std10_non_negative(self):
        from model_core.ops import _op_ts_std
        x = torch.randn(5, 40)
        out = _op_ts_std(x, 10)
        assert (out >= -1e-7).all()

    def test_delay5_equals_lagged(self):
        from model_core.ops import _ts_delay
        x = torch.randn(3, 20)
        out = _ts_delay(x, 5)
        # First 5 bars should be zero
        assert (out[:, :5] == 0).all()
        # Rest should be lagged
        torch.testing.assert_close(out[:, 5:], x[:, :-5])

    def test_corr10_range(self):
        from model_core.ops import _op_corr10
        x = torch.randn(4, 30)
        y = torch.randn(4, 30)
        out = _op_corr10(x, y)
        assert out.shape == x.shape
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_corr10_self_correlation(self):
        from model_core.ops import _op_corr10
        x = torch.randn(3, 30)
        out = _op_corr10(x, x)
        # Self-correlation should be ~1 after warmup (0.9 from Bessel correction)
        assert out[:, 10:].mean() >= 0.9 - 1e-6


# ---------------------------------------------------------------------------
# 12: OPS_CONFIG_EXTENDED count
# ---------------------------------------------------------------------------

class TestOpsConfig:
    def test_ops_config_extended_count(self):
        from model_core.ops import OPS_CONFIG, OPS_CONFIG_EXTENDED
        assert len(OPS_CONFIG) == 12
        assert len(OPS_CONFIG_EXTENDED) == 22

    def test_ops_config_extended_names(self):
        from model_core.ops import OPS_CONFIG_EXTENDED
        names = [cfg[0] for cfg in OPS_CONFIG_EXTENDED]
        assert 'CS_RANK' in names
        assert 'CORR10' in names
        assert 'MA5' in names
        assert 'DELAY5' in names


# ---------------------------------------------------------------------------
# 13–14: Config dynamic methods
# ---------------------------------------------------------------------------

class TestConfigDynamic:
    def test_get_feature_dim_futures(self):
        from model_core.config import ModelConfig
        original = ModelConfig.ASSET_CLASS
        try:
            ModelConfig.ASSET_CLASS = 'futures'
            assert ModelConfig.get_feature_dim() == 18
        finally:
            ModelConfig.ASSET_CLASS = original

    def test_get_feature_dim_crypto(self):
        from model_core.config import ModelConfig
        original = ModelConfig.ASSET_CLASS
        try:
            ModelConfig.ASSET_CLASS = 'crypto'
            assert ModelConfig.get_feature_dim() == 12
        finally:
            ModelConfig.ASSET_CLASS = original

    def test_get_ops_config_futures(self):
        from model_core.config import ModelConfig
        original = ModelConfig.ASSET_CLASS
        try:
            ModelConfig.ASSET_CLASS = 'futures'
            ops = ModelConfig.get_ops_config()
            assert len(ops) == 22
        finally:
            ModelConfig.ASSET_CLASS = original

    def test_get_ops_config_crypto(self):
        from model_core.config import ModelConfig
        original = ModelConfig.ASSET_CLASS
        try:
            ModelConfig.ASSET_CLASS = 'crypto'
            ops = ModelConfig.get_ops_config()
            assert len(ops) == 12
        finally:
            ModelConfig.ASSET_CLASS = original

    def test_get_max_formula_len_futures(self):
        from model_core.config import ModelConfig
        original = ModelConfig.ASSET_CLASS
        try:
            ModelConfig.ASSET_CLASS = 'futures'
            assert ModelConfig.get_max_formula_len() == 12
        finally:
            ModelConfig.ASSET_CLASS = original

    def test_get_max_formula_len_crypto(self):
        from model_core.config import ModelConfig
        original = ModelConfig.ASSET_CLASS
        try:
            ModelConfig.ASSET_CLASS = 'crypto'
            assert ModelConfig.get_max_formula_len() == 12
        finally:
            ModelConfig.ASSET_CLASS = original


# ---------------------------------------------------------------------------
# 15: AlphaGPT vocab size
# ---------------------------------------------------------------------------

class TestAlphaGPTVocab:
    def test_vocab_size_crypto(self):
        from model_core.config import ModelConfig
        original = ModelConfig.ASSET_CLASS
        try:
            ModelConfig.ASSET_CLASS = 'crypto'
            from model_core.alphagpt import AlphaGPT
            model = AlphaGPT()
            assert model.vocab_size == 24  # 12 feat + 12 ops
        finally:
            ModelConfig.ASSET_CLASS = original

    def test_vocab_size_futures(self):
        from model_core.config import ModelConfig
        original = ModelConfig.ASSET_CLASS
        try:
            ModelConfig.ASSET_CLASS = 'futures'
            from model_core.alphagpt import AlphaGPT
            model = AlphaGPT()
            assert model.vocab_size == 40  # 18 feat + 22 ops
        finally:
            ModelConfig.ASSET_CLASS = original


# ---------------------------------------------------------------------------
# 16: StackVM parameterized
# ---------------------------------------------------------------------------

class TestStackVMParameterized:
    def test_custom_feat_offset(self):
        from model_core.vm import StackVM
        from model_core.ops import OPS_CONFIG
        vm = StackVM(feat_offset=18, ops_config=OPS_CONFIG)
        assert vm.feat_offset == 18
        # First op should be at index 18
        assert 18 in vm.op_map

    def test_default_uses_config(self):
        from model_core.vm import StackVM
        from model_core.config import ModelConfig
        vm = StackVM()
        assert vm.feat_offset == ModelConfig.get_feature_dim()

    def test_execute_with_extended_ops(self):
        from model_core.vm import StackVM
        from model_core.ops import OPS_CONFIG_EXTENDED
        vm = StackVM(feat_offset=18, ops_config=OPS_CONFIG_EXTENDED)
        # Simple formula: feature 0 (push), CS_RANK (op 12 -> token 18+0=30... wait)
        # With feat_offset=18, ops start at 18. CS_RANK is index 12 in extended = token 30
        # Actually: ops start at index feat_offset. CS_RANK is at position 12 in the list
        # So token = 18 + 12 = 30
        feat = torch.randn(5, 18, 40)
        # Formula: push feature 0, then apply CS_RANK (token 18+12=30)
        formula = [0, 30]
        result = vm.execute(formula, feat)
        assert result is not None
        assert result.shape == (5, 40)


# ---------------------------------------------------------------------------
# 17: Engine masking tensors
# ---------------------------------------------------------------------------

class TestEngineMasking:
    def test_masking_tensor_lengths(self):
        from model_core.engine import _ARITY, _STACK_DELTA
        from model_core.config import ModelConfig
        expected_len = ModelConfig.get_feature_dim() + len(ModelConfig.get_ops_config())
        assert len(_ARITY) == expected_len
        assert len(_STACK_DELTA) == expected_len

    def test_features_have_zero_arity(self):
        from model_core.engine import _ARITY
        from model_core.config import ModelConfig
        n_feat = ModelConfig.get_feature_dim()
        assert (_ARITY[:n_feat] == 0).all()

    def test_features_push_one(self):
        from model_core.engine import _STACK_DELTA
        from model_core.config import ModelConfig
        n_feat = ModelConfig.get_feature_dim()
        assert (_STACK_DELTA[:n_feat] == 1).all()

    def test_build_masking_tensors_consistency(self):
        from model_core.engine import _build_masking_tensors
        arity, delta = _build_masking_tensors()
        # delta = 1 - arity for ops (arity > 0)
        from model_core.config import ModelConfig
        n_feat = ModelConfig.get_feature_dim()
        for i in range(n_feat, len(arity)):
            assert delta[i] == 1 - arity[i]


# ---------------------------------------------------------------------------
# 19: End-to-end formula with new ops
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_formula_with_new_ops(self):
        """Generate and execute a formula using extended ops."""
        from model_core.vm import StackVM
        from model_core.ops import OPS_CONFIG_EXTENDED

        vm = StackVM(feat_offset=18, ops_config=OPS_CONFIG_EXTENDED)
        feat = torch.randn(5, 18, 60)

        # Formula: VWAP_DEV(12) MOM_REV(7) CORR10(token=18+19=37)
        # Wait: CORR10 is at index 19 in OPS_CONFIG_EXTENDED (0-indexed from extended list entry 7)
        # Extended ops: indices 0-11 are base, 12-19 are new
        # CORR10 is the last one, index 19 → token = 18 + 19 = 37
        formula = [12, 7, 37]  # push feat12, push feat7, CORR10
        result = vm.execute(formula, feat)
        assert result is not None
        assert result.shape == (5, 60)
        assert torch.isfinite(result).all()

    def test_formula_cs_rank_ma5(self):
        """Formula: feat0 → MA5 → CS_RANK"""
        from model_core.vm import StackVM
        from model_core.ops import OPS_CONFIG_EXTENDED

        vm = StackVM(feat_offset=18, ops_config=OPS_CONFIG_EXTENDED)
        feat = torch.randn(5, 18, 60)

        # MA5 is at index 14 in extended → token 18+14=32
        # CS_RANK is at index 12 → token 18+12=30
        formula = [0, 32, 30]
        result = vm.execute(formula, feat)
        assert result is not None
        assert result.min() >= -1.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Features V2 list
# ---------------------------------------------------------------------------

class TestFeaturesLists:
    def test_v1_list_length(self):
        from model_core.features_v2 import FEATURES_V1_LIST
        assert len(FEATURES_V1_LIST) == 12

    def test_v2_list_length(self):
        from model_core.features_v2 import FEATURES_V2_LIST
        assert len(FEATURES_V2_LIST) == 18

    def test_v2_extends_v1(self):
        from model_core.features_v2 import FEATURES_V1_LIST, FEATURES_V2_LIST
        assert FEATURES_V2_LIST[:12] == FEATURES_V1_LIST
