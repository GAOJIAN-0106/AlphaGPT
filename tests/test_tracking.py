"""
Tests for Weights & Biases experiment tracking integration.

Tests cover:
  - DummyTracker: all methods callable, no-op behavior
  - WandbTracker: initialization, log_step, log_summary, log_artifact, finish
  - Engine integration: tracker creation based on use_wandb flag
  - Interface consistency: both trackers implement the same methods
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. DummyTracker tests
# ---------------------------------------------------------------------------

class TestDummyTracker:
    """DummyTracker must be zero-dependency and all methods must be callable no-ops."""

    def test_import_without_wandb(self):
        """DummyTracker can be imported even if wandb is not installed."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        assert tracker is not None

    def test_log_step_callable(self):
        """log_step accepts step and metrics dict without error."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        tracker.log_step(0, {"avg_reward": 1.5, "best_score": 2.0})

    def test_log_step_returns_none(self):
        """log_step returns None (no-op)."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        result = tracker.log_step(10, {"metric": 42.0})
        assert result is None

    def test_log_summary_callable(self):
        """log_summary accepts a metrics dict without error."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        tracker.log_summary({"final_score": 3.14})

    def test_log_summary_returns_none(self):
        """log_summary returns None (no-op)."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        result = tracker.log_summary({"final_score": 3.14})
        assert result is None

    def test_log_artifact_callable(self):
        """log_artifact accepts name and filepath without error."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        tracker.log_artifact("model", "/tmp/model.pt")

    def test_log_artifact_returns_none(self):
        """log_artifact returns None (no-op)."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        result = tracker.log_artifact("model", "/tmp/model.pt")
        assert result is None

    def test_finish_callable(self):
        """finish can be called without error."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        tracker.finish()

    def test_finish_returns_none(self):
        """finish returns None (no-op)."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        result = tracker.finish()
        assert result is None

    def test_multiple_calls_no_error(self):
        """Calling methods multiple times does not raise."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        for i in range(100):
            tracker.log_step(i, {"x": float(i)})
        tracker.log_summary({"done": True})
        tracker.log_artifact("a", "b")
        tracker.finish()


# ---------------------------------------------------------------------------
# 2. WandbTracker tests (all wandb calls mocked)
# ---------------------------------------------------------------------------

class TestWandbTracker:
    """WandbTracker wraps wandb; all external calls are mocked."""

    def _make_mock_wandb(self):
        """Create a mock wandb module with the attributes WandbTracker needs."""
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.init.return_value.summary = {}
        mock_wandb.Artifact.return_value = MagicMock()
        return mock_wandb

    def test_init_calls_wandb_init(self):
        """WandbTracker.__init__ calls wandb.init with project, name, config."""
        mock_wandb = self._make_mock_wandb()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import WandbTracker
            tracker = WandbTracker(
                project="test-project",
                name="run-42",
                config={"batch_size": 128, "seed": 42},
            )

        mock_wandb.init.assert_called_once_with(
            project="test-project",
            name="run-42",
            config={"batch_size": 128, "seed": 42},
        )

    def test_init_with_group(self):
        """WandbTracker passes group parameter to wandb.init when provided."""
        mock_wandb = self._make_mock_wandb()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import WandbTracker
            tracker = WandbTracker(
                project="proj",
                name="run-1",
                config={},
                group="ensemble-run",
            )

        mock_wandb.init.assert_called_once_with(
            project="proj",
            name="run-1",
            config={},
            group="ensemble-run",
        )

    def test_log_step_calls_wandb_log(self):
        """log_step calls wandb.log with step and metrics."""
        mock_wandb = self._make_mock_wandb()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import WandbTracker
            tracker = WandbTracker(project="p", name="n", config={})
            tracker.log_step(5, {"avg_reward": 1.0, "best_score": 2.0})

        mock_wandb.log.assert_called_once_with(
            {"avg_reward": 1.0, "best_score": 2.0},
            step=5,
        )

    def test_log_step_multiple_calls(self):
        """Multiple log_step calls result in multiple wandb.log calls."""
        mock_wandb = self._make_mock_wandb()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import WandbTracker
            tracker = WandbTracker(project="p", name="n", config={})
            tracker.log_step(0, {"a": 1.0})
            tracker.log_step(1, {"a": 2.0})
            tracker.log_step(2, {"a": 3.0})

        assert mock_wandb.log.call_count == 3

    def test_log_summary_updates_run_summary(self):
        """log_summary calls wandb.run.summary.update with the metrics dict."""
        mock_wandb = self._make_mock_wandb()
        mock_run = MagicMock()
        mock_run.summary = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import WandbTracker
            tracker = WandbTracker(project="p", name="n", config={})
            tracker.log_summary({"best_score": 5.0, "test_sharpe": 1.2})

        mock_run.summary.update.assert_called_once_with(
            {"best_score": 5.0, "test_sharpe": 1.2}
        )

    def test_log_artifact_creates_and_logs_artifact(self):
        """log_artifact creates a wandb.Artifact, adds file, and logs it."""
        mock_wandb = self._make_mock_wandb()
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import WandbTracker
            tracker = WandbTracker(project="p", name="n", config={})
            tracker.log_artifact("best-model", "/path/to/model.pt")

        mock_wandb.Artifact.assert_called_once_with("best-model", type="model")
        mock_artifact.add_file.assert_called_once_with("/path/to/model.pt")
        mock_wandb.log_artifact.assert_called_once_with(mock_artifact)

    def test_finish_calls_wandb_finish(self):
        """finish calls wandb.finish()."""
        mock_wandb = self._make_mock_wandb()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import WandbTracker
            tracker = WandbTracker(project="p", name="n", config={})
            tracker.finish()

        mock_wandb.finish.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Interface consistency
# ---------------------------------------------------------------------------

class TestTrackerInterface:
    """Both DummyTracker and WandbTracker must implement the same methods."""

    REQUIRED_METHODS = ["log_step", "log_summary", "log_artifact", "finish"]

    def test_dummy_tracker_has_all_methods(self):
        """DummyTracker implements all required tracker methods."""
        from model_core.tracking import DummyTracker
        tracker = DummyTracker()
        for method_name in self.REQUIRED_METHODS:
            assert hasattr(tracker, method_name), f"DummyTracker missing {method_name}"
            assert callable(getattr(tracker, method_name)), f"DummyTracker.{method_name} not callable"

    def test_wandb_tracker_has_all_methods(self):
        """WandbTracker implements all required tracker methods."""
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.init.return_value.summary = {}

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import WandbTracker
            tracker = WandbTracker(project="p", name="n", config={})

        for method_name in self.REQUIRED_METHODS:
            assert hasattr(tracker, method_name), f"WandbTracker missing {method_name}"
            assert callable(getattr(tracker, method_name)), f"WandbTracker.{method_name} not callable"

    def test_method_signatures_compatible(self):
        """Both trackers' methods accept the same arguments."""
        import inspect
        from model_core.tracking import DummyTracker

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.init.return_value.summary = {}

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import WandbTracker
            wandb_tracker = WandbTracker(project="p", name="n", config={})

        dummy_tracker = DummyTracker()

        for method_name in self.REQUIRED_METHODS:
            dummy_sig = inspect.signature(getattr(dummy_tracker, method_name))
            wandb_sig = inspect.signature(getattr(wandb_tracker, method_name))
            dummy_params = list(dummy_sig.parameters.keys())
            wandb_params = list(wandb_sig.parameters.keys())
            assert dummy_params == wandb_params, (
                f"Signature mismatch for {method_name}: "
                f"DummyTracker{dummy_params} vs WandbTracker{wandb_params}"
            )


# ---------------------------------------------------------------------------
# 4. create_tracker factory function
# ---------------------------------------------------------------------------

class TestCreateTracker:
    """Test the factory function that creates the appropriate tracker."""

    def test_create_tracker_disabled_returns_dummy(self):
        """When use_wandb=False, create_tracker returns a DummyTracker."""
        from model_core.tracking import create_tracker, DummyTracker
        tracker = create_tracker(use_wandb=False)
        assert isinstance(tracker, DummyTracker)

    def test_create_tracker_enabled_returns_wandb_tracker(self):
        """When use_wandb=True and wandb available, returns WandbTracker."""
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.init.return_value.summary = {}

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from model_core.tracking import create_tracker, WandbTracker
            tracker = create_tracker(
                use_wandb=True,
                project="test",
                name="run",
                config={"seed": 42},
            )
            assert isinstance(tracker, WandbTracker)

    def test_create_tracker_enabled_but_wandb_missing_returns_dummy(self):
        """When use_wandb=True but wandb not installed, falls back to DummyTracker."""
        from model_core.tracking import create_tracker, DummyTracker

        with patch.dict("sys.modules", {"wandb": None}):
            tracker = create_tracker(
                use_wandb=True,
                project="test",
                name="run",
                config={},
            )
            assert isinstance(tracker, DummyTracker)


# ---------------------------------------------------------------------------
# 5. Engine integration tests
# ---------------------------------------------------------------------------

class TestEngineTrackerIntegration:
    """Test that AlphaEngine correctly uses the tracker based on use_wandb flag."""

    def test_engine_accepts_wandb_params(self):
        """AlphaEngine.__init__ accepts use_wandb and wandb_project parameters."""
        import inspect
        from model_core.engine import AlphaEngine
        sig = inspect.signature(AlphaEngine.__init__)
        param_names = list(sig.parameters.keys())
        assert "use_wandb" in param_names, "AlphaEngine missing use_wandb parameter"
        assert "wandb_project" in param_names, "AlphaEngine missing wandb_project parameter"

    def test_engine_creates_dummy_tracker_by_default(self):
        """When use_wandb is not specified (default False), engine uses DummyTracker."""
        from model_core.tracking import DummyTracker
        import torch

        with patch("model_core.engine.CryptoDataLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader

            with patch("model_core.engine.DuckDBDataLoader", MockLoader):
                with patch("model_core.engine.AlphaGPT") as MockModel:
                    mock_model = MagicMock()
                    mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
                    MockModel.return_value.to.return_value = mock_model

                    from model_core.engine import AlphaEngine
                    engine = AlphaEngine(use_lord_regularization=False)
                    assert isinstance(engine.tracker, DummyTracker)

    def test_engine_creates_wandb_tracker_when_enabled(self):
        """When use_wandb=True, engine creates a WandbTracker."""
        import torch
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.init.return_value.summary = {}

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch("model_core.engine.CryptoDataLoader") as MockLoader:
                mock_loader = MagicMock()
                MockLoader.return_value = mock_loader

                with patch("model_core.engine.DuckDBDataLoader", MockLoader):
                    with patch("model_core.engine.AlphaGPT") as MockModel:
                        mock_model = MagicMock()
                        mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
                        MockModel.return_value.to.return_value = mock_model

                        from model_core.engine import AlphaEngine
                        engine = AlphaEngine(
                            use_lord_regularization=False,
                            use_wandb=True,
                            wandb_project="test-alphagpt",
                        )

                        from model_core.tracking import WandbTracker
                        assert isinstance(engine.tracker, WandbTracker)

    def test_engine_passes_config_to_tracker(self):
        """Engine passes relevant config dict when creating the tracker."""
        import torch
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.init.return_value.summary = {}

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch("model_core.engine.CryptoDataLoader") as MockLoader:
                mock_loader = MagicMock()
                MockLoader.return_value = mock_loader

                with patch("model_core.engine.DuckDBDataLoader", MockLoader):
                    with patch("model_core.engine.AlphaGPT") as MockModel:
                        mock_model = MagicMock()
                        mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
                        mock_model.named_parameters.return_value = [("w", torch.nn.Parameter(torch.zeros(1)))]
                        MockModel.return_value.to.return_value = mock_model

                        with patch("model_core.engine.NewtonSchulzLowRankDecay"):
                            with patch("model_core.engine.StableRankMonitor"):
                                from model_core.engine import AlphaEngine
                                engine = AlphaEngine(
                                    use_lord_regularization=True,
                                    lord_decay_rate=0.002,
                                    seed=123,
                                    use_wandb=True,
                                    wandb_project="my-project",
                                )

        # Verify wandb.init was called with the right config keys
        init_call = mock_wandb.init.call_args
        config = init_call.kwargs.get("config", init_call[1].get("config", {}))
        assert config["seed"] == 123
        assert config["use_lord"] is True
        assert config["lord_decay_rate"] == 0.002
        assert "batch_size" in config
        assert "train_steps" in config
        assert "max_formula_len" in config

    def test_engine_default_wandb_project(self):
        """Default wandb_project is 'alphagpt'."""
        import inspect
        from model_core.engine import AlphaEngine
        sig = inspect.signature(AlphaEngine.__init__)
        default = sig.parameters["wandb_project"].default
        assert default == "alphagpt"
