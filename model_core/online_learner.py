"""
Online Learning Module for AlphaGPT.

Adapts ensemble formula weights in real-time based on rolling recent
performance — analogous to Jane Street's daily model weight updates.

Two update strategies:
  - 'ema': Original EMA + softmax + linear blend (legacy)
  - 'mwu': Multiplicative Weights Update with O(√(T ln N)) regret bound

Key idea: instead of fixed equal weights, track each formula's recent
PnL/Sharpe and upweight formulas performing well in the current regime.
"""

import math
import torch
import numpy as np

from .ensemble import FormulaEnsemble
from .vm import StackVM
from .backtest import MemeBacktest


class OnlineLearner:
    """
    Wraps a FormulaEnsemble and adapts its weights over time based on
    rolling realized performance.

    Walk-forward process (per period):
      1. Predict signal using current adaptive weights
      2. Observe realized returns for the period
      3. Compute each formula's individual PnL for that period
      4. Update weights: upweight recent winners, downweight losers
      5. Repeat for next period

    Weight update uses exponential moving average of per-formula Sharpe,
    with softmax normalization and minimum weight floor.
    """

    def __init__(self, ensemble, lookback_window=20, learning_rate=0.3,
                 decay=0.95, min_weight=0.05, strategy='mwu', eta=None):
        """
        Args:
            ensemble: FormulaEnsemble instance to adapt
            lookback_window: number of recent periods for performance eval
            learning_rate: blending rate (EMA strategy only)
            decay: exponential decay (EMA strategy only)
            min_weight: floor per formula to prevent total collapse
            strategy: 'mwu' (Multiplicative Weights Update) or 'ema' (legacy)
            eta: MWU learning rate. If None, auto-set to sqrt(ln(N) / T_est)
                 where T_est = lookback_window * 10 as initial estimate.
        """
        self.ensemble = ensemble
        self.vm = StackVM()
        self.bt = MemeBacktest(position_mode='rank')
        self.n = ensemble.num_formulas

        self.lookback_window = lookback_window
        self.learning_rate = learning_rate
        self.decay = decay
        self.min_weight = min_weight
        self.strategy = strategy

        # MWU learning rate: theory-optimal η = sqrt(ln(N) / T)
        if eta is not None:
            self.eta = eta
        else:
            T_est = max(lookback_window * 10, 100)
            self.eta = math.sqrt(math.log(max(self.n, 2)) / T_est)

        # Current adaptive weights (start from ensemble's original weights)
        self._weights = np.array(ensemble.weights.tolist(), dtype=np.float64)

        # Rolling performance: list of per-formula PnL for each update period
        self._perf_history = []  # list of np arrays, each shape [n]

        # Weight snapshots over time
        self._weight_history = []

        # MWU cumulative regret tracking
        self._mwu_step = 0

    def _compute_formula_pnls(self, feat_tensor, raw_data, target_ret):
        """Compute each formula's PnL for one period."""
        formula_pnls = np.zeros(self.n)
        for i, formula in enumerate(self.ensemble.formulas):
            res = self.vm.execute(formula, feat_tensor)
            if res is None or res.std() < 1e-8:
                formula_pnls[i] = 0.0
                continue
            liquidity = raw_data['liquidity']
            is_safe = (liquidity > self.bt.min_liq).float()
            position = self.bt.compute_position(res, is_safe)
            pnl = (position * target_ret).mean().item()
            formula_pnls[i] = pnl
        return formula_pnls

    def update(self, feat_tensor, raw_data, target_ret):
        """
        Update ensemble weights based on observed returns for one period.

        Dispatches to MWU or EMA strategy based on self.strategy.

        Args:
            feat_tensor: [N, F, T] features for this period
            raw_data: dict with 'liquidity' tensor for this period
            target_ret: [N, T] realized returns for this period

        Returns:
            dict with 'weights', 'formula_pnls', 'period_sharpes'
        """
        formula_pnls = self._compute_formula_pnls(
            feat_tensor, raw_data, target_ret)

        self._perf_history.append(formula_pnls.copy())
        if len(self._perf_history) > self.lookback_window:
            self._perf_history = self._perf_history[-self.lookback_window:]

        if self.strategy == 'mwu':
            return self._update_mwu(formula_pnls)
        else:
            return self._update_ema(formula_pnls)

    def _update_mwu(self, formula_pnls):
        """
        Multiplicative Weights Update.

        Core rule:  w_{t+1,i} = w_{t,i} * (1 + η * gain_i)
        where gain_i is the normalized PnL of expert i.

        Regret bound: R_T ≤ 2√(T ln N) for η = √(ln N / T).
        """
        self._mwu_step += 1

        # Normalize PnL to [-1, 1] for numerical stability
        max_abs = np.abs(formula_pnls).max()
        if max_abs > 1e-10:
            gains = formula_pnls / max_abs
        else:
            # No signal this period, keep weights unchanged
            self._weight_history.append(self._weights.copy())
            return {
                'weights': self._weights.copy(),
                'formula_pnls': formula_pnls,
                'period_sharpes': formula_pnls,
            }

        # Multiplicative update: upweight winners, downweight losers
        # w_i *= (1 + η * gain_i)  — gain > 0 means profit
        new_weights = self._weights * (1.0 + self.eta * gains)

        # Clamp to prevent negative weights (can happen if η * gain < -1)
        new_weights = np.maximum(new_weights, 1e-10)

        # Apply minimum weight floor before normalizing
        new_weights = np.maximum(new_weights, self.min_weight)
        new_weights = new_weights / new_weights.sum()

        self._weights = new_weights
        self._weight_history.append(self._weights.copy())

        return {
            'weights': self._weights.copy(),
            'formula_pnls': formula_pnls,
            'period_sharpes': gains,
        }

    def _update_ema(self, formula_pnls):
        """Legacy EMA + softmax + linear blend update."""
        n_periods = len(self._perf_history)
        scores = np.zeros(self.n)
        total_decay_weight = 0.0

        for t_idx, pnl in enumerate(self._perf_history):
            age = n_periods - 1 - t_idx
            w = self.decay ** age
            scores += w * pnl
            total_decay_weight += w

        if total_decay_weight > 0:
            scores /= total_decay_weight

        scores_shifted = scores - scores.max()
        exp_scores = np.exp(scores_shifted / max(np.std(scores) + 1e-8, 1e-6))
        target_weights = exp_scores / exp_scores.sum()

        new_weights = (1 - self.learning_rate) * self._weights + \
                      self.learning_rate * target_weights

        for _ in range(10):
            clamped = np.maximum(new_weights, self.min_weight)
            clamped = clamped / clamped.sum()
            if np.all(clamped >= self.min_weight - 1e-10):
                break
            new_weights = clamped
        new_weights = clamped

        self._weights = new_weights
        self._weight_history.append(self._weights.copy())

        return {
            'weights': self._weights.copy(),
            'formula_pnls': formula_pnls,
            'period_sharpes': scores,
        }

    def predict(self, feat_tensor):
        """
        Generate ensemble signal using current adaptive weights.

        Args:
            feat_tensor: [N, F, T] feature tensor

        Returns:
            Aggregated signal tensor [N, T], or None if all fail.
        """
        signals = []
        valid_weights = []

        for i, formula in enumerate(self.ensemble.formulas):
            res = self.vm.execute(formula, feat_tensor)
            if res is None or res.std() < 1e-8:
                continue
            signals.append(res)
            valid_weights.append(self._weights[i])

        if not signals:
            return None

        stacked = torch.stack(signals, dim=0)
        w = torch.tensor(valid_weights, dtype=torch.float32,
                         device=stacked.device)
        w = w / w.sum()
        weighted = (stacked * w.view(-1, 1, 1)).sum(dim=0)
        return weighted

    def run_online(self, feat_tensor, raw_data, target_ret,
                   period_length=1, warmup_periods=20):
        """
        Run full walk-forward online learning simulation.

        Splits data into consecutive periods of `period_length` steps,
        uses first `warmup_periods` with equal weights, then adapts.

        Args:
            feat_tensor: [N, F, T_total] full feature tensor
            raw_data: dict with 'liquidity' [N, T_total]
            target_ret: [N, T_total] full return series
            period_length: steps per update period (1 = daily)
            warmup_periods: periods before adaptation starts

        Returns:
            dict with:
              'online_pnl': array of per-period portfolio PnL
              'static_pnl': array of per-period PnL with fixed weights
              'weight_history': list of weight arrays over time
              'online_sharpe': annualized Sharpe of online strategy
              'static_sharpe': annualized Sharpe of static strategy
              'improvement': online_sharpe - static_sharpe
        """
        T_total = feat_tensor.shape[2]
        n_periods = T_total // period_length

        # Reset state for fresh simulation
        self._weights = np.ones(self.n) / self.n
        self._perf_history = []
        self._weight_history = []

        online_pnls = []
        static_pnls = []
        weight_history = []

        # Static ensemble for comparison (fixed equal weights)
        static_weights = np.ones(self.n) / self.n

        for p in range(n_periods):
            t_start = p * period_length
            t_end = t_start + period_length

            # Slice this period's data
            p_feat = feat_tensor[:, :, t_start:t_end]
            p_raw = {k: (v[:, t_start:t_end] if v.dim() > 1 else v[t_start:t_end])
                     for k, v in raw_data.items()}
            p_ret = target_ret[:, t_start:t_end]

            # --- Online prediction PnL ---
            online_signal = self.predict(p_feat)
            if online_signal is not None:
                liquidity = p_raw['liquidity']
                is_safe = (liquidity > self.bt.min_liq).float()
                position = self.bt.compute_position(online_signal, is_safe)
                online_pnl = (position * p_ret).mean().item()
            else:
                online_pnl = 0.0
            online_pnls.append(online_pnl)

            # --- Static prediction PnL ---
            static_signal = self._predict_with_weights(p_feat, static_weights)
            if static_signal is not None:
                liquidity = p_raw['liquidity']
                is_safe = (liquidity > self.bt.min_liq).float()
                position = self.bt.compute_position(static_signal, is_safe)
                static_pnl = (position * p_ret).mean().item()
            else:
                static_pnl = 0.0
            static_pnls.append(static_pnl)

            # --- Weight update (skip during warmup) ---
            if p < warmup_periods:
                # Record equal weights during warmup but still track perf
                weight_history.append(self._weights.copy())
                # Track performance but don't update weights
                formula_pnls = np.zeros(self.n)
                for i, formula in enumerate(self.ensemble.formulas):
                    res = self.vm.execute(formula, p_feat)
                    if res is not None and res.std() >= 1e-8:
                        liq = p_raw['liquidity']
                        is_safe_f = (liq > self.bt.min_liq).float()
                        pos = self.bt.compute_position(res, is_safe_f)
                        formula_pnls[i] = (pos * p_ret).mean().item()
                self._perf_history.append(formula_pnls)
                if len(self._perf_history) > self.lookback_window:
                    self._perf_history = self._perf_history[-self.lookback_window:]
            else:
                self.update(p_feat, p_raw, p_ret)
                weight_history.append(self._weights.copy())

        # Store full history
        self._weight_history = weight_history

        # Compute Sharpe ratios
        online_arr = np.array(online_pnls)
        static_arr = np.array(static_pnls)

        annualization = self.bt.annualization / math.sqrt(max(period_length, 1))

        online_sharpe = (online_arr.mean() / (online_arr.std() + 1e-8)) * annualization
        static_sharpe = (static_arr.mean() / (static_arr.std() + 1e-8)) * annualization

        return {
            'online_pnl': online_pnls,
            'static_pnl': static_pnls,
            'weight_history': weight_history,
            'online_sharpe': float(online_sharpe),
            'static_sharpe': float(static_sharpe),
            'improvement': float(online_sharpe - static_sharpe),
        }

    def _predict_with_weights(self, feat_tensor, weights):
        """Predict using explicit weight array (for static baseline)."""
        signals = []
        valid_weights = []

        for i, formula in enumerate(self.ensemble.formulas):
            res = self.vm.execute(formula, feat_tensor)
            if res is None or res.std() < 1e-8:
                continue
            signals.append(res)
            valid_weights.append(weights[i])

        if not signals:
            return None

        stacked = torch.stack(signals, dim=0)
        w = torch.tensor(valid_weights, dtype=torch.float32,
                         device=stacked.device)
        w = w / w.sum()
        weighted = (stacked * w.view(-1, 1, 1)).sum(dim=0)
        return weighted

    def run_ab_compare(self, feat_tensor, raw_data, target_ret,
                       period_length=1, warmup_periods=20):
        """
        Run MWU vs EMA side-by-side on the same data for A/B comparison.

        Returns:
            dict with 'mwu_sharpe', 'ema_sharpe', 'static_sharpe',
                  'mwu_pnl', 'ema_pnl', 'mwu_weights', 'ema_weights'
        """
        results = {}
        for strat in ('mwu', 'ema'):
            self.strategy = strat
            self._weights = np.ones(self.n) / self.n
            self._perf_history = []
            self._weight_history = []
            self._mwu_step = 0
            r = self.run_online(feat_tensor, raw_data, target_ret,
                                period_length, warmup_periods)
            results[strat] = r

        return {
            'mwu_sharpe': results['mwu']['online_sharpe'],
            'ema_sharpe': results['ema']['online_sharpe'],
            'static_sharpe': results['mwu']['static_sharpe'],
            'mwu_pnl': results['mwu']['online_pnl'],
            'ema_pnl': results['ema']['online_pnl'],
            'mwu_weights': results['mwu']['weight_history'],
            'ema_weights': results['ema']['weight_history'],
        }

    def get_weights(self):
        """Return current adaptive weights as numpy array."""
        return self._weights.copy()

    def get_weight_history(self):
        """Return full weight evolution over time."""
        return [w.copy() for w in self._weight_history]

    def to_dict(self):
        """Serialize to dict for JSON saving."""
        return {
            'ensemble': self.ensemble.to_dict(),
            'weights': self._weights.tolist(),
            'lookback_window': self.lookback_window,
            'learning_rate': self.learning_rate,
            'decay': self.decay,
            'min_weight': self.min_weight,
            'strategy': self.strategy,
            'eta': self.eta,
            'mwu_step': self._mwu_step,
            'weight_history': [w.tolist() for w in self._weight_history],
            'performance_history': [p.tolist() for p in self._perf_history],
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct from saved dict."""
        ensemble = FormulaEnsemble.from_dict(d['ensemble'])
        ol = cls(
            ensemble,
            lookback_window=d['lookback_window'],
            learning_rate=d['learning_rate'],
            decay=d['decay'],
            min_weight=d['min_weight'],
            strategy=d.get('strategy', 'ema'),
            eta=d.get('eta'),
        )
        ol._weights = np.array(d['weights'])
        ol._weight_history = [np.array(w) for w in d['weight_history']]
        ol._perf_history = [np.array(p) for p in d['performance_history']]
        ol._mwu_step = d.get('mwu_step', 0)
        return ol
