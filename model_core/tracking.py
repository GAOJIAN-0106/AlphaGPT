"""
Experiment tracking for AlphaGPT training runs.

Provides two tracker implementations:
  - DummyTracker: zero-dependency no-op (used when wandb is disabled or unavailable)
  - WandbTracker: wraps Weights & Biases for real experiment tracking

Both implement the same interface (duck typing):
  - log_step(step, metrics_dict)
  - log_summary(metrics_dict)
  - log_artifact(name, filepath)
  - finish()

Use create_tracker() factory to get the appropriate tracker.
"""


class DummyTracker:
    """No-op tracker that silently discards all logging calls.

    Zero external dependencies. Used as fallback when wandb is disabled
    or not installed.
    """

    def log_step(self, step, metrics_dict):
        return None

    def log_summary(self, metrics_dict):
        return None

    def log_artifact(self, name, filepath):
        return None

    def finish(self):
        return None


class WandbTracker:
    """Tracker that logs to Weights & Biases.

    Wraps wandb.init, wandb.log, wandb.summary, and wandb.Artifact.
    Requires the ``wandb`` package to be installed.
    """

    def __init__(self, project, name, config, group=None):
        import wandb
        self._wandb = wandb

        init_kwargs = dict(project=project, name=name, config=config)
        if group is not None:
            init_kwargs["group"] = group

        self._run = wandb.init(**init_kwargs)

    def log_step(self, step, metrics_dict):
        self._wandb.log(metrics_dict, step=step)

    def log_summary(self, metrics_dict):
        self._run.summary.update(metrics_dict)

    def log_artifact(self, name, filepath):
        artifact = self._wandb.Artifact(name, type="model")
        artifact.add_file(filepath)
        self._wandb.log_artifact(artifact)

    def finish(self):
        self._wandb.finish()


def create_tracker(use_wandb=False, project="alphagpt", name=None, config=None,
                   group=None):
    """Factory: return the appropriate tracker based on settings.

    If *use_wandb* is ``True`` and the ``wandb`` package is importable, a
    :class:`WandbTracker` is returned.  Otherwise a :class:`DummyTracker`
    is returned.
    """
    if not use_wandb:
        return DummyTracker()

    try:
        import wandb  # noqa: F401
        if wandb is None:
            raise ImportError("wandb is None (mocked as missing)")
    except ImportError:
        return DummyTracker()

    return WandbTracker(
        project=project,
        name=name,
        config=config or {},
        group=group,
    )
