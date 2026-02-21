import torch
import math
import numpy as np


class MemeBacktest:
    def __init__(self):
        self.trade_size = 100000.0   # A股：10万元
        self.min_liq = 10000000.0    # A股：成交额>1000万
        self.base_fee = 0.0015       # A股：印花税0.05%+佣金0.025%*2 ≈ 0.15%
        self.annualization = math.sqrt(252)

    def evaluate(self, factors, raw_data, target_ret,
                 formula_length=None, base_factors=None,
                 return_diagnostics=False, **kwargs):
        """Evaluate a factor signal via simulated backtest.

        Args:
            factors: Tensor of factor values [batch, time_steps] or [batch, time_steps, stocks].
            raw_data: Dict with 'liquidity' tensor.
            target_ret: Forward return tensor matching factors shape.
            formula_length: Optional int, number of tokens in the formula.
                            Applies complexity penalty when > 6.
            base_factors: Optional tensor of existing factors for redundancy check.
            return_diagnostics: If True, return (score, mean_ret, diagnostics_dict).

        Returns:
            (final_fitness, mean_ret) by default.
            (final_fitness, mean_ret, diagnostics) when return_diagnostics=True.
        """
        device = factors.device
        liquidity = raw_data['liquidity']

        # --- Handle NaN ---
        factors = torch.nan_to_num(factors, nan=0.0)
        target_ret = torch.nan_to_num(target_ret, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Continuous position sizing ---
        signal = torch.sigmoid(factors)
        is_safe = (liquidity > self.min_liq).float()
        position = torch.clamp(signal - 0.5, 0.0, 0.5) * 2.0 * is_safe

        # --- Transaction costs ---
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        total_slippage_one_way = self.base_fee + impact_slippage

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way

        # --- PnL ---
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost

        # --- Sharpe Ratio (annualized, per-sample in batch) ---
        daily_mean = net_pnl.mean(dim=1)
        daily_std = net_pnl.std(dim=1)
        sharpe = (daily_mean / (daily_std + 1e-8)) * self.annualization
        sharpe = torch.nan_to_num(sharpe, nan=0.0)

        # --- Rolling max drawdown ---
        cum_pnl = net_pnl.cumsum(dim=1)                     # [batch, T]
        running_max = cum_pnl.cummax(dim=1).values           # [batch, T]
        drawdown = running_max - cum_pnl                     # [batch, T]
        max_drawdown = drawdown.max(dim=1).values            # [batch]

        # --- Activity threshold ---
        time_steps = position.shape[1]
        min_trades = max(20, int(0.1 * time_steps))
        activity = (position > 0.05).float().sum(dim=1)

        # --- Base score from Sharpe minus drawdown penalty ---
        # Drawdown penalty: proportional, scaled so a 20% drawdown costs ~2 points
        drawdown_penalty = max_drawdown * 2.0
        score = sharpe - drawdown_penalty

        # --- Complexity penalty ---
        if formula_length is not None:
            complexity_penalty = 0.1 * max(0, formula_length - 6)
            score = score - complexity_penalty

        # --- Factor redundancy penalty ---
        if base_factors is not None:
            redundancy_penalty = self._redundancy_penalty(factors, base_factors)
            score = score - redundancy_penalty

        # --- Turnover floor penalty ---
        avg_turnover_per_sample = turnover.mean(dim=1)
        lazy_penalty = torch.where(
            avg_turnover_per_sample < 0.005,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        score = score - lazy_penalty

        # --- Inactive penalty ---
        score = torch.where(
            activity < min_trades,
            torch.tensor(-10.0, device=device),
            score,
        )

        # --- Final aggregation ---
        score = torch.nan_to_num(score, nan=-10.0)
        per_stock_score = torch.median(score)
        ic = self._cross_sectional_ic(factors, target_ret)
        final_fitness = per_stock_score + ic * 10.0

        cum_ret = net_pnl.sum(dim=1)
        mean_ret = cum_ret.mean().item()
        mean_ret = 0.0 if (mean_ret != mean_ret) else mean_ret  # NaN guard

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
                'sharpe': sharpe.median().item(),
                'max_drawdown': max_drawdown.median().item(),
                'avg_turnover': turnover.mean().item(),
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
        return ic.mean().item()

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
