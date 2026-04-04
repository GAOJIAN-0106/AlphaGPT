"""
Regime Detection Module for AlphaGPT.

Two complementary approaches:
  1. VolatilityRegime: Continuous position scaling based on realized
     cross-sectional volatility (replaces binary Sharpe threshold).
  2. HMMRegimeDetector: Hidden Markov Model that infers 2-3 latent
     market states from observable features, enabling predictive
     (not just reactive) regime detection.

Both produce a regime_scale in [0.3, 1.2] and a regime_label string.
"""

import math
import numpy as np
import torch
from typing import Optional


# ────────────────────────────────────────────────────────────────────
# 方案2: Continuous Volatility Scaling
# ────────────────────────────────────────────────────────────────────

class VolatilityRegime:
    """Continuous regime scaling based on cross-sectional volatility
    and return-based risk metrics.

    Instead of a binary (1.0 / 0.5) switch, produces a smooth scale
    factor that increases exposure in calm markets and decreases it
    during stress — without waiting for 20 days of losses.

    Observable features used:
      - cross_vol: std of returns across products (截面波动率)
      - avg_abs_ret: mean |return| across products (平均绝对收益)
      - cross_corr: average pairwise correlation (截面相关性)
    """

    def __init__(self, vol_window: int = 20, baseline_window: int = 120,
                 min_scale: float = 0.3, max_scale: float = 1.2):
        self.vol_window = vol_window
        self.baseline_window = baseline_window
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Rolling history for baseline estimation
        self._vol_history = []

    def compute_features(self, returns: np.ndarray) -> dict:
        """Extract regime features from return matrix.

        Args:
            returns: [N, T] array of per-product returns.

        Returns:
            dict with cross_vol, avg_abs_ret, cross_corr, vol_ratio.
        """
        N, T = returns.shape
        if T < 2:
            return {'cross_vol': 0.0, 'avg_abs_ret': 0.0,
                    'cross_corr': 0.0, 'vol_ratio': 1.0}

        # Recent window
        w = min(self.vol_window, T)
        recent = returns[:, -w:]

        # 1. Cross-sectional volatility (per-day std across products, then avg)
        daily_cross_std = np.std(recent, axis=0)  # [w]
        cross_vol = float(np.mean(daily_cross_std))

        # 2. Average absolute return
        avg_abs_ret = float(np.mean(np.abs(recent)))

        # 3. Cross-sectional correlation (avg pairwise corr, sampled for speed)
        cross_corr = 0.0
        if N >= 2 and w >= 5:
            # Use correlation matrix of recent returns
            # Each row is a product, columns are time steps
            corr_matrix = np.corrcoef(recent)
            if not np.any(np.isnan(corr_matrix)):
                # Average off-diagonal correlation
                mask = ~np.eye(N, dtype=bool)
                cross_corr = float(np.mean(corr_matrix[mask]))

        # 4. Volatility ratio vs baseline
        self._vol_history.append(cross_vol)
        if len(self._vol_history) > self.baseline_window:
            self._vol_history = self._vol_history[-self.baseline_window:]

        baseline_vol = np.mean(self._vol_history) if self._vol_history else cross_vol
        vol_ratio = cross_vol / max(baseline_vol, 1e-10)

        return {
            'cross_vol': cross_vol,
            'avg_abs_ret': avg_abs_ret,
            'cross_corr': cross_corr,
            'vol_ratio': vol_ratio,
        }

    def get_scale(self, returns: np.ndarray) -> tuple:
        """Compute continuous regime scale.

        Args:
            returns: [N, T] return matrix.

        Returns:
            (scale, features_dict, label)
            scale: float in [min_scale, max_scale]
            label: 'calm' / 'normal' / 'stress' / 'crisis'
        """
        feat = self.compute_features(returns)
        vol_ratio = feat['vol_ratio']
        cross_corr = feat['cross_corr']

        # Base scale: inverse of volatility ratio
        # vol_ratio=1 → scale=1, vol_ratio=2 → scale≈0.5
        base_scale = 1.0 / max(vol_ratio, 0.5)

        # Correlation penalty: high correlation → systemic risk → scale down more
        # cross_corr > 0.5 is unusual and indicates stress
        corr_penalty = max(0.0, cross_corr - 0.3) * 0.5  # 0 to ~0.35

        scale = base_scale - corr_penalty
        scale = np.clip(scale, self.min_scale, self.max_scale)

        # Label
        if vol_ratio < 0.7 and cross_corr < 0.3:
            label = 'calm'
        elif vol_ratio < 1.3 and cross_corr < 0.5:
            label = 'normal'
        elif vol_ratio < 2.0:
            label = 'stress'
        else:
            label = 'crisis'

        feat['scale'] = float(scale)
        feat['label'] = label
        return float(scale), feat, label

    def to_dict(self):
        return {
            'vol_window': self.vol_window,
            'baseline_window': self.baseline_window,
            'min_scale': self.min_scale,
            'max_scale': self.max_scale,
            'vol_history': [float(v) for v in self._vol_history],
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(
            vol_window=d['vol_window'],
            baseline_window=d['baseline_window'],
            min_scale=d['min_scale'],
            max_scale=d['max_scale'],
        )
        obj._vol_history = d.get('vol_history', [])
        return obj


# ────────────────────────────────────────────────────────────────────
# 方案1: HMM Regime Detection
# ────────────────────────────────────────────────────────────────────

class HMMRegimeDetector:
    """Hidden Markov Model for market regime identification.

    Infers 2-3 latent states from observable market features:
      - cross_vol: cross-sectional volatility
      - cross_corr: cross-sectional correlation
      - avg_abs_ret: average absolute return
      - vol_change: rate of change of volatility

    States are automatically labeled after fitting:
      State with lowest vol  → 'calm'    (因子分化期, 正常加仓)
      State with mid vol     → 'trending' (趋势活跃期, 动量因子加权)
      State with highest vol → 'crisis'  (系统性风险, 大幅降仓)

    The detector maintains a rolling buffer of features and refits
    periodically (default every 20 days) to adapt to structural changes.
    """

    def __init__(self, n_states: int = 3, vol_window: int = 20,
                 refit_interval: int = 20, min_history: int = 60,
                 min_scale: float = 0.3, max_scale: float = 1.2):
        self.n_states = n_states
        self.vol_window = vol_window
        self.refit_interval = refit_interval
        self.min_history = min_history
        self.min_scale = min_scale
        self.max_scale = max_scale

        self._model = None
        self._fitted = False
        self._feature_buffer = []  # list of feature vectors
        self._steps_since_fit = 0
        self._state_scales = {}     # state_idx → scale
        self._state_labels = {}     # state_idx → label
        self._vol_regime = VolatilityRegime(vol_window=vol_window)

        # Current state tracking
        self._current_state = None
        self._state_probs = None

    def _extract_features(self, returns: np.ndarray) -> Optional[np.ndarray]:
        """Extract HMM observation features from return matrix.

        Args:
            returns: [N, T] return matrix

        Returns:
            Feature vector [4] or None if insufficient data.
        """
        feat = self._vol_regime.compute_features(returns)

        cross_vol = feat['cross_vol']
        cross_corr = feat['cross_corr']
        avg_abs_ret = feat['avg_abs_ret']

        # Vol change (momentum of volatility)
        if len(self._feature_buffer) >= 1:
            prev_vol = self._feature_buffer[-1][0]
            vol_change = (cross_vol - prev_vol) / max(prev_vol, 1e-10)
        else:
            vol_change = 0.0

        return np.array([cross_vol, cross_corr, avg_abs_ret, vol_change],
                        dtype=np.float64)

    def _fit_hmm(self):
        """Fit or refit the HMM on accumulated feature history."""
        if len(self._feature_buffer) < self.min_history:
            return False

        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            return False

        X = np.array(self._feature_buffer)

        # Normalize features for HMM stability
        self._feat_mean = X.mean(axis=0)
        self._feat_std = X.std(axis=0) + 1e-10
        X_norm = (X - self._feat_mean) / self._feat_std

        try:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type='diag',
                n_iter=100,
                random_state=42,
                verbose=False,
            )
            model.fit(X_norm)
            self._model = model
            self._fitted = True
            self._assign_state_labels(X_norm)
            return True
        except Exception:
            return False

    def _assign_state_labels(self, X_norm: np.ndarray):
        """Label states by their mean volatility level.

        Lowest vol → calm, highest vol → crisis, middle → trending.
        """
        # Decode most likely states for the training data
        states = self._model.predict(X_norm)

        # Compute mean cross_vol (feature 0) per state
        state_vols = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                # Use original (un-normalized) feature 0 = cross_vol
                original_feats = X_norm[mask] * self._feat_std + self._feat_mean
                state_vols[s] = float(np.mean(original_feats[:, 0]))
            else:
                state_vols[s] = 0.0

        # Sort states by volatility
        sorted_states = sorted(state_vols.keys(), key=lambda s: state_vols[s])

        if self.n_states == 2:
            labels = ['calm', 'crisis']
            scales = [1.1, 0.5]
        else:
            labels = ['calm', 'trending', 'crisis']
            scales = [1.15, 0.9, 0.55]

        for rank, state_idx in enumerate(sorted_states):
            self._state_labels[state_idx] = labels[rank]
            self._state_scales[state_idx] = scales[rank]

    def update(self, returns: np.ndarray) -> tuple:
        """Process new return data and update regime estimate.

        Args:
            returns: [N, T] return matrix (full history up to now)

        Returns:
            (scale, info_dict, label)
        """
        feat_vec = self._extract_features(returns)
        if feat_vec is None:
            return 1.0, {}, 'unknown'

        self._feature_buffer.append(feat_vec)
        self._steps_since_fit += 1

        # Fit or refit HMM periodically
        if not self._fitted or self._steps_since_fit >= self.refit_interval:
            if self._fit_hmm():
                self._steps_since_fit = 0

        # If HMM not fitted yet, fall back to VolatilityRegime
        if not self._fitted:
            return self._vol_regime.get_scale(returns)

        # Predict current state
        x = feat_vec.reshape(1, -1)
        x_norm = (x - self._feat_mean) / self._feat_std

        try:
            state = int(self._model.predict(x_norm)[0])
            probs = self._model.predict_proba(x_norm)[0]
        except Exception:
            return self._vol_regime.get_scale(returns)

        self._current_state = state
        self._state_probs = probs

        label = self._state_labels.get(state, 'unknown')
        base_scale = self._state_scales.get(state, 1.0)

        # Blend with volatility regime for smoothness
        vol_scale, vol_feat, _ = self._vol_regime.get_scale(returns)

        # Weighted blend: use sqrt(confidence) to dampen HMM's overconfidence
        confidence = float(probs[state])
        hmm_weight = min(confidence * 0.6, 0.8)  # cap HMM influence at 80%
        blended_scale = hmm_weight * base_scale + (1 - hmm_weight) * vol_scale
        blended_scale = np.clip(blended_scale, self.min_scale, self.max_scale)

        info = {
            'hmm_state': state,
            'hmm_label': label,
            'hmm_confidence': confidence,
            'state_probs': {self._state_labels.get(s, f's{s}'): float(probs[s])
                           for s in range(self.n_states)},
            'scale': float(blended_scale),
            'vol_scale': vol_scale,
            'hmm_scale': base_scale,
            **vol_feat,
        }

        return float(blended_scale), info, label

    def to_dict(self):
        d = {
            'n_states': self.n_states,
            'vol_window': self.vol_window,
            'refit_interval': self.refit_interval,
            'min_history': self.min_history,
            'min_scale': self.min_scale,
            'max_scale': self.max_scale,
            'feature_buffer': [f.tolist() for f in self._feature_buffer],
            'vol_regime': self._vol_regime.to_dict(),
            'fitted': self._fitted,
            'steps_since_fit': self._steps_since_fit,
            'state_scales': {str(k): v for k, v in self._state_scales.items()},
            'state_labels': {str(k): v for k, v in self._state_labels.items()},
        }
        if self._fitted:
            d['feat_mean'] = self._feat_mean.tolist()
            d['feat_std'] = self._feat_std.tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        obj = cls(
            n_states=d['n_states'],
            vol_window=d['vol_window'],
            refit_interval=d['refit_interval'],
            min_history=d['min_history'],
            min_scale=d['min_scale'],
            max_scale=d['max_scale'],
        )
        obj._feature_buffer = [np.array(f) for f in d.get('feature_buffer', [])]
        obj._vol_regime = VolatilityRegime.from_dict(d['vol_regime'])
        obj._fitted = d.get('fitted', False)
        obj._steps_since_fit = d.get('steps_since_fit', 0)
        obj._state_scales = {int(k): v for k, v in d.get('state_scales', {}).items()}
        obj._state_labels = {int(k): v for k, v in d.get('state_labels', {}).items()}
        if obj._fitted and 'feat_mean' in d:
            obj._feat_mean = np.array(d['feat_mean'])
            obj._feat_std = np.array(d['feat_std'])
            # Refit HMM from buffer
            obj._fit_hmm()
        return obj


# ────────────────────────────────────────────────────────────────────
# Unified interface
# ────────────────────────────────────────────────────────────────────

def get_returns_from_raw(raw_data: dict, window: Optional[int] = None) -> np.ndarray:
    """Extract return matrix from raw_data_cache.

    Args:
        raw_data: dict with 'close' tensor [N, T]
        window: if set, only use last `window` time steps

    Returns:
        [N, T-1] numpy array of simple returns
    """
    close = raw_data.get('close')
    if close is None:
        # Try target_ret directly
        ret = raw_data.get('target_ret')
        if ret is not None:
            if isinstance(ret, torch.Tensor):
                ret = ret.detach().cpu().numpy()
            if window:
                ret = ret[:, -window:]
            return ret
        return np.zeros((1, 1))

    if isinstance(close, torch.Tensor):
        close = close.detach().cpu().numpy()

    if close.ndim == 1:
        close = close.reshape(1, -1)

    # Simple returns
    returns = np.diff(close, axis=1) / (np.abs(close[:, :-1]) + 1e-10)
    returns = np.clip(returns, -0.2, 0.2)  # Clamp extreme returns

    if window:
        returns = returns[:, -window:]

    return returns
