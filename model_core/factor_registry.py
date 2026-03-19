"""
Factor Registry — pluggable, frequency-aware factor management.

Each factor is a subclass of FactorBase that declares its own frequency,
data requirements, and compute() logic. The registry groups factors by
frequency and provides aligned feature tensors for each frequency band.

Usage:
    registry = FactorRegistry()
    registry.auto_discover()   # finds all FactorBase subclasses
    registry.compute_all(raw_dict, freq='1d')  # returns [N, F_daily, T]
"""

import torch
from abc import ABC, abstractmethod
from collections import defaultdict


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class FactorBase(ABC):
    """Base class for all pluggable factors.

    Subclasses must define:
        name:       str — unique identifier (e.g. 'PINDYCK_VOL')
        frequency:  str — native frequency ('1d', '1h', '5m', '1m')
        data_keys:  list[str] — required keys in raw_dict

    And implement:
        compute(raw_dict) -> Tensor[N, T]
    """

    name: str = ''
    frequency: str = '1d'
    data_keys: list = []
    description: str = ''
    category: str = 'unknown'

    @abstractmethod
    def compute(self, raw_dict: dict) -> torch.Tensor:
        """Compute factor values.

        Args:
            raw_dict: dict of tensors, keys depend on data_keys.
        Returns:
            Tensor of shape [N, T] (num_assets, time_steps).
        """
        ...

    def check_requirements(self, raw_dict: dict) -> bool:
        """Check if raw_dict contains all required keys."""
        return all(k in raw_dict for k in self.data_keys)

    @staticmethod
    def robust_norm(t: torch.Tensor) -> torch.Tensor:
        """Robust MAD normalization, clamped to [-5, 5]."""
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        return torch.clamp((t - median) / mad, -5.0, 5.0)

    @staticmethod
    def rolling_std(x: torch.Tensor, window: int) -> torch.Tensor:
        """Rolling standard deviation over dim=1 (time)."""
        pad = torch.zeros(x.shape[0], window - 1, device=x.device)
        x_pad = torch.cat([pad, x], dim=1)
        unfolded = x_pad.unfold(1, window, 1)  # [N, T, window]
        return unfolded.std(dim=-1)

    @staticmethod
    def rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
        """Rolling mean over dim=1 (time)."""
        pad = torch.zeros(x.shape[0], window - 1, device=x.device)
        x_pad = torch.cat([pad, x], dim=1)
        return x_pad.unfold(1, window, 1).mean(dim=-1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class FactorRegistry:
    """Central registry for all factors, grouped by frequency."""

    def __init__(self):
        self._factors: dict[str, FactorBase] = {}  # name → instance
        self._by_freq: dict[str, list[str]] = defaultdict(list)

    def register(self, factor: FactorBase):
        """Register a factor instance."""
        if factor.name in self._factors:
            raise ValueError(f"Factor '{factor.name}' already registered")
        self._factors[factor.name] = factor
        self._by_freq[factor.frequency].append(factor.name)

    def get(self, name: str) -> FactorBase:
        return self._factors[name]

    def list_factors(self, freq: str = None) -> list[str]:
        """List factor names, optionally filtered by frequency."""
        if freq:
            return list(self._by_freq.get(freq, []))
        return list(self._factors.keys())

    def list_frequencies(self) -> list[str]:
        return list(self._by_freq.keys())

    def compute_group(self, raw_dict: dict, freq: str) -> tuple:
        """Compute all factors for a given frequency.

        Returns:
            (tensor [N, F, T], names list[str])
        """
        names = self._by_freq.get(freq, [])
        if not names:
            return None, []

        results = []
        valid_names = []
        for name in names:
            factor = self._factors[name]
            if not factor.check_requirements(raw_dict):
                continue
            val = factor.compute(raw_dict)  # [N, T]
            results.append(val)
            valid_names.append(name)

        if not results:
            return None, []

        return torch.stack(results, dim=1), valid_names  # [N, F, T]

    def auto_discover(self):
        """Auto-register all FactorBase subclasses that have been imported."""
        for cls in FactorBase.__subclasses__():
            if cls.name:
                self.register(cls())


# ---------------------------------------------------------------------------
# Global registry instance
# ---------------------------------------------------------------------------

_global_registry = None


def get_registry() -> FactorRegistry:
    """Get or create the global factor registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = FactorRegistry()
    return _global_registry
