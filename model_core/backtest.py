import torch
import math
import numpy as np

from .config import ModelConfig, TimeframeProfile


class MemeBacktest:
    def __init__(self, position_mode='sigmoid_clamp', target_turnover=None,
                 timeframe=None, asset_class=None, rebalance_freq=None):
        profile = TimeframeProfile(
            timeframe or ModelConfig.TIMEFRAME,
            asset_class or ModelConfig.ASSET_CLASS,
        )
        self.trade_size = profile.trade_size
        self.min_liq = profile.min_liq
        self.base_fee = profile.base_fee
        self.position_mode = position_mode
        self.tx_cost_scale = 1.0

        # If rebalancing less frequently than bar frequency (Plan B),
        # use rebalance frequency for annualization and turnover targets.
        # rebalance_freq: 'daily' means rebalance once per day regardless of bar freq.
        if rebalance_freq == 'daily' or (rebalance_freq is None and profile.bars_per_day > 1):
            # Annualize based on daily rebalance, not bar frequency
            self.annualization = math.sqrt(252)
            self.target_turnover = target_turnover if target_turnover is not None else 0.05
            self.lazy_threshold = 0.01
        else:
            self.annualization = profile.annualization
            self.target_turnover = target_turnover if target_turnover is not None else profile.target_turnover
            self.lazy_threshold = profile.lazy_threshold

    # ------------------------------------------------------------------
    # Position sizing methods
    # ------------------------------------------------------------------

    def compute_position(self, factors, is_safe):
        """Compute position based on position_mode.

        Args:
            factors: [N, T] factor values
            is_safe: [N, T] liquidity safety mask (0 or 1)

        Returns:
            position: [N, T] in [0, 1]
        """
        if self.position_mode == 'rank':
            return self._compute_position_rank(factors, is_safe)
        elif self.position_mode == 'zscore':
            return self._compute_position_zscore(factors, is_safe)
        elif self.position_mode == 'quantile':
            return self._compute_position_quantile(factors, is_safe)
        else:  # sigmoid_clamp (original)
            signal = torch.sigmoid(factors)
            return torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe

    def _compute_position_rank(self, factors, is_safe):
        """Cross-sectional rank: top 40% get position, rest zero."""
        N = factors.shape[0]
        if N < 2:
            signal = torch.sigmoid(factors)
            return torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe
        ranks = factors.argsort(dim=0).argsort(dim=0).float()
        rank_pct = ranks / (N - 1)
        # Top 40% gets position linearly from 0 to 1
        position = torch.clamp((rank_pct - 0.6) / 0.4, 0.0, 1.0) * is_safe
        return position

    def _compute_position_zscore(self, factors, is_safe):
        """Cross-sectional z-score → steeper sigmoid."""
        N = factors.shape[0]
        if N < 2:
            signal = torch.sigmoid(factors)
            return torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe
        mean = factors.mean(dim=0, keepdim=True)
        std = factors.std(dim=0, keepdim=True) + 1e-6
        z = (factors - mean) / std
        # Steeper sigmoid for more decisive positions
        signal = torch.sigmoid(z * 1.5)
        position = torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe
        return position

    def _compute_position_quantile(self, factors, is_safe):
        """Discrete quantile buckets: top 10% = 1.0, 10-25% = 0.5, rest = 0."""
        N = factors.shape[0]
        if N < 5:
            signal = torch.sigmoid(factors)
            return torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe
        ranks = factors.argsort(dim=0).argsort(dim=0).float()
        rank_pct = ranks / (N - 1)
        position = torch.zeros_like(factors)
        position = torch.where(rank_pct >= 0.90, torch.ones_like(factors), position)
        position = torch.where(
            (rank_pct >= 0.75) & (rank_pct < 0.90),
            torch.full_like(factors, 0.5),
            position,
        )
        return position * is_safe

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self, factors, raw_data, target_ret,
                 formula_length=None, base_factors=None,
                 return_diagnostics=False, **kwargs):
        """Evaluate a factor signal via simulated backtest.

        Computes portfolio-level Sharpe and MaxDD (aggregate across stocks
        first, then compute metrics on the portfolio PnL series), matching
        the standard industry practice for multi-asset strategies.

        Per-stock metrics (turnover, lazy penalty) are computed per-stock
        then aggregated, since they relate to individual trading activity.

        Args:
            factors: [N, T] factor values (N stocks, T time steps).
            raw_data: Dict with 'liquidity' tensor.
            target_ret: [N, T] forward return tensor.
            formula_length: Optional int for complexity penalty.
            base_factors: Optional tensor for redundancy check.
            return_diagnostics: If True, return (score, mean_ret, diagnostics).

        Returns:
            (final_fitness, mean_ret) by default.
            (final_fitness, mean_ret, diagnostics) when return_diagnostics=True.
        """
        device = factors.device
        liquidity = raw_data['liquidity']

        # --- Handle NaN ---
        factors = torch.nan_to_num(factors, nan=0.0)
        target_ret = torch.nan_to_num(target_ret, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Position sizing (dispatched by mode) ---
        is_safe = (liquidity > self.min_liq).float()
        target_position = self.compute_position(factors, is_safe)

        # --- Daily rebalance: only update position at rebalance points ---
        rebal_mask = raw_data.get('rebalance_mask')
        if rebal_mask is not None and rebal_mask.sum() < rebal_mask.numel():
            # Vectorized: forward-fill position from rebalance points.
            # At rebalance bars, use target_position; otherwise carry forward.
            # Build segment IDs via cumsum of rebalance_mask, then scatter.
            rm = rebal_mask.unsqueeze(0)  # [1, T]
            # Multiply target by mask, then forward-fill zeros
            position = target_position * rm  # only rebalance bars have values
            # Forward fill: replace 0s between rebalance points
            # Use segment-based approach: cumsum of mask gives segment ids
            seg_ids = rebal_mask.cumsum(0).long()  # [T]
            # For each segment, all bars should have the value at segment start
            # Gather the rebalance-point values using segment ids
            rebal_indices = torch.where(rebal_mask > 0.5)[0]  # indices of rebalance bars
            if len(rebal_indices) > 0:
                # Pad segment 0 (before first rebalance) to use first rebalance value
                seg_ids_clamped = torch.clamp(seg_ids - 1, min=0)  # 0-based segment index
                seg_ids_clamped = torch.clamp(seg_ids_clamped, max=len(rebal_indices) - 1)
                # Gather: position at each bar = target_position at its segment's rebalance bar
                position = target_position[:, rebal_indices[seg_ids_clamped]]  # [N, T]
        else:
            position = target_position

        # --- Transaction costs ---
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        total_slippage_one_way = (self.base_fee + impact_slippage) * self.tx_cost_scale

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way

        # --- PnL ---
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost                             # [N, T]

        # --- Portfolio PnL (equal-weight across stocks) ---
        portfolio_pnl = net_pnl.mean(dim=0)                       # [T]

        # --- Portfolio Sharpe (annualized) ---
        port_mean = portfolio_pnl.mean()
        port_std = portfolio_pnl.std() + 1e-8
        portfolio_sharpe = (port_mean / port_std) * self.annualization
        portfolio_sharpe = torch.nan_to_num(portfolio_sharpe, nan=0.0)

        # --- Portfolio MaxDD ---
        cum_pnl = portfolio_pnl.cumsum(dim=0)                     # [T]
        running_max = cum_pnl.cummax(dim=0).values                 # [T]
        portfolio_maxdd = (running_max - cum_pnl).max()            # scalar

        # --- Score: portfolio Sharpe minus portfolio drawdown penalty ---
        drawdown_penalty = portfolio_maxdd * 2.0
        score = portfolio_sharpe - drawdown_penalty

        # --- Complexity penalty ---
        if formula_length is not None:
            complexity_penalty = 0.1 * max(0, formula_length - 6)
            score = score - complexity_penalty

        # --- Factor redundancy penalty (per-stock, then mean) ---
        if base_factors is not None:
            redundancy_penalty = self._redundancy_penalty(factors, base_factors)
            score = score - redundancy_penalty.mean()

        # --- Per-stock turnover penalties (keep per-stock, aggregate to scalar) ---
        # When using rebalance mask, compute daily turnover by dividing total
        # turnover by number of rebalance points (not total bars).
        rebal_mask = raw_data.get('rebalance_mask')
        if rebal_mask is not None and rebal_mask.sum() < rebal_mask.numel():
            n_rebal = max(1, int(rebal_mask.sum().item()))
            avg_turnover_per_stock = turnover.sum(dim=1) / n_rebal  # [N]
        else:
            avg_turnover_per_stock = turnover.mean(dim=1)              # [N]
        target_to = self.target_turnover
        lazy_penalty = torch.where(
            avg_turnover_per_stock < self.lazy_threshold,
            torch.tensor(3.0, device=device),
            torch.where(
                avg_turnover_per_stock < target_to,
                (target_to - avg_turnover_per_stock) / target_to * 2.0,
                torch.tensor(0.0, device=device),
            ),
        )
        score = score - lazy_penalty.mean()

        # --- Turnover bonus (per-stock, capped, then mean) ---
        turnover_bonus = torch.clamp(avg_turnover_per_stock * 5.0, 0.0, 0.5)
        score = score + turnover_bonus.mean()

        # --- Portfolio-level activity check ---
        # active_ratio: average fraction of stocks with position at each time step
        active_ratio = (position > 0.05).float().mean()            # scalar
        if active_ratio.item() < 0.05:
            score = torch.tensor(-10.0, device=device)

        # --- Final score with IC ---
        score = torch.nan_to_num(score, nan=-10.0)
        ic = self._cross_sectional_ic(factors, target_ret)
        ic_clipped = max(-0.3, min(0.3, ic))
        final_fitness = score + ic_clipped * 2.0

        # --- Portfolio cumulative return ---
        mean_ret = portfolio_pnl.sum().item()
        mean_ret = 0.0 if (mean_ret != mean_ret) else mean_ret

        if return_diagnostics:
            # Factor autocorrelation: corr(factor_t, factor_{t-1})
            f_t = factors[:, 1:]
            f_lag = factors[:, :-1]
            f_t_dm = f_t - f_t.mean(dim=1, keepdim=True)
            f_lag_dm = f_lag - f_lag.mean(dim=1, keepdim=True)
            numerator = (f_t_dm * f_lag_dm).sum(dim=1)
            denominator = (
                torch.sqrt((f_t_dm ** 2).sum(dim=1))
                * torch.sqrt((f_lag_dm ** 2).sum(dim=1))
                + 1e-8
            )
            factor_autocorr = (numerator / denominator).mean().item()

            diagnostics = {
                'sharpe': portfolio_sharpe.item(),
                'max_drawdown': portfolio_maxdd.item(),
                'avg_turnover': avg_turnover_per_stock.mean().item(),
                'factor_autocorrelation': factor_autocorr,
                'ic': ic,
            }
            return final_fitness, mean_ret, diagnostics

        return final_fitness, mean_ret

    def evaluate_ensemble(self, ensemble, feat_tensor, raw_data, target_ret,
                          base_factors=None):
        """
        Evaluate a FormulaEnsemble and compare against individual formulas.

        Args:
            ensemble: FormulaEnsemble instance
            feat_tensor: [num_tokens, num_features, time_steps]
            raw_data: dict with 'liquidity' tensor
            target_ret: [num_tokens, time_steps]
            base_factors: optional base factors for redundancy penalty

        Returns:
            dict with:
              'ensemble': diagnostics for ensemble signal
              'individuals': list of diagnostics per formula
              'improvement': sharpe improvement vs best single model
              'variance_reduction': std of individual sharpes
        """
        # Ensemble signal
        ens_signal = ensemble.predict(feat_tensor)
        if ens_signal is None:
            return None

        ens_score, ens_ret, ens_diag = self.evaluate(
            ens_signal, raw_data, target_ret,
            base_factors=base_factors,
            return_diagnostics=True,
        )

        # Individual signals
        individual_diags = []
        individual_results = ensemble.predict_individual(feat_tensor)
        for idx, signal in individual_results:
            _, _, diag = self.evaluate(
                signal, raw_data, target_ret,
                base_factors=base_factors,
                return_diagnostics=True,
            )
            diag['formula_index'] = idx
            individual_diags.append(diag)

        individual_sharpes = [d['sharpe'] for d in individual_diags]

        best_single = max(individual_sharpes) if individual_sharpes else 0.0
        avg_single = float(np.mean(individual_sharpes)) if individual_sharpes else 0.0
        std_single = float(np.std(individual_sharpes)) if individual_sharpes else 0.0

        return {
            'ensemble': ens_diag,
            'ensemble_score': ens_score.item(),
            'individuals': individual_diags,
            'individual_sharpes': individual_sharpes,
            'best_single_sharpe': best_single,
            'avg_single_sharpe': avg_single,
            'sharpe_std': std_single,
            'improvement_vs_best': ens_diag['sharpe'] - best_single,
            'improvement_vs_avg': ens_diag['sharpe'] - avg_single,
        }

    @staticmethod
    def _cross_sectional_ic(factors, target_ret):
        """Rank IC across stocks at each time step — measures cross-sectional selection ability."""
        N = factors.shape[0]
        if N < 2:
            return 0.0
        # No cross-sectional variation → IC is meaningless (avoids spurious rank correlation from ties)
        if factors.std(dim=0).mean() < 1e-6:
            return 0.0
        f_rank = factors.argsort(dim=0).argsort(dim=0).float()
        r_rank = target_ret.argsort(dim=0).argsort(dim=0).float()
        f_dm = f_rank - f_rank.mean(dim=0, keepdim=True)
        r_dm = r_rank - r_rank.mean(dim=0, keepdim=True)
        num = (f_dm * r_dm).sum(dim=0)
        den = torch.sqrt((f_dm**2).sum(dim=0)) * torch.sqrt((r_dm**2).sum(dim=0)) + 1e-8
        ic = torch.nan_to_num(num / den, nan=0.0)
        raw_ic = ic.mean().item()

        # Discount IC for binary/degenerate factors:
        # Count unique values per stock (dim=1), average across stocks
        # A truly binary factor has ~2 unique values per row
        unique_counts = []
        for i in range(min(N, 10)):  # sample up to 10 stocks for speed
            unique_counts.append(factors[i].unique().numel())
        avg_unique = sum(unique_counts) / len(unique_counts) if unique_counts else 100
        if avg_unique < 5:
            # Binary/near-binary factor: discount IC by 70%
            raw_ic *= 0.3
        return raw_ic

    @staticmethod
    def _redundancy_penalty(factors, base_factors):
        """Spearman rank-correlation redundancy penalty.

        Computes correlation between `factors` and each column/slice in
        `base_factors`. If any correlation exceeds 0.7, applies a -2.0 penalty.

        Args:
            factors: [batch, T] current factor values.
            base_factors: [batch, T] or [batch, T, K] existing factor(s).

        Returns:
            Penalty tensor of shape [batch].
        """
        def _rank(t):
            """Convert values to fractional ranks along dim=1."""
            # argsort twice gives ranks
            return t.argsort(dim=1).argsort(dim=1).float()

        factor_rank = _rank(factors)

        if base_factors.dim() == 2:
            base_factors = base_factors.unsqueeze(-1)  # [batch, T, 1]

        penalty = torch.zeros(factors.shape[0], device=factors.device)
        n_base = base_factors.shape[-1]
        for k in range(n_base):
            base_rank = _rank(base_factors[..., k])
            # Pearson on ranks ≈ Spearman
            fr_dm = factor_rank - factor_rank.mean(dim=1, keepdim=True)
            br_dm = base_rank - base_rank.mean(dim=1, keepdim=True)
            num = (fr_dm * br_dm).sum(dim=1)
            den = (
                torch.sqrt((fr_dm ** 2).sum(dim=1))
                * torch.sqrt((br_dm ** 2).sum(dim=1))
                + 1e-8
            )
            corr = num / den  # [batch]
            # Penalize if absolute correlation > 0.7
            penalty = torch.where(corr.abs() > 0.7, penalty + 2.0, penalty)

        return penalty
