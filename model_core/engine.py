import gc
import torch
import numpy as np
import pandas as pd
from torch.distributions import Categorical
from tqdm import tqdm
import json
import math

from .config import ModelConfig
from .data_loader import CryptoDataLoader
try:
    from .duckdb_loader import DuckDBDataLoader
except ImportError:
    DuckDBDataLoader = None


def _default_loader_cls():
    """Auto-select data loader based on DATA_SOURCE_MODE."""
    if ModelConfig.ASSET_CLASS == 'futures' and DuckDBDataLoader is not None:
        return DuckDBDataLoader
    return CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MemeBacktest
from .ensemble import FormulaEnsemble
from .online_learner import OnlineLearner
from .tracking import create_tracker
from .temporal_cv import TimeSeriesCV, evaluate_formula_cv
from .factors import FeatureEngineer


def _build_masking_tensors():
    """Build RPN stack effect tensors from dynamic config.

    Features: arity=0 (need nothing on stack), delta=+1 (push one value)
    Operators: arity from ops config, delta = 1 - arity (pop arity, push 1 result)
    """
    n_feat = ModelConfig.get_feature_dim()
    ops = ModelConfig.get_ops_config()
    arities = [cfg[2] for cfg in ops]
    arity = torch.tensor([0] * n_feat + arities)
    delta = torch.tensor([1] * n_feat + [1 - a for a in arities])
    return arity, delta


_ARITY, _STACK_DELTA = _build_masking_tensors()

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5,
                 seed=None, use_wandb=False, wandb_project="alphagpt", wandb_group=None,
                 loader=None):
        """
        Initialize AlphaGPT training engine.

        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
            seed: Random seed for reproducibility (None = no fixed seed)
            use_wandb: Enable Weights & Biases experiment tracking
            wandb_project: W&B project name (default: "alphagpt")
            wandb_group: Optional W&B group name (for ensemble runs)
            loader: Pre-configured data loader (default: CryptoDataLoader)
        """
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if loader is not None:
            self.loader = loader
        elif ModelConfig.ASSET_CLASS == 'futures' and DuckDBDataLoader is not None:
            self.loader = DuckDBDataLoader()
        else:
            self.loader = CryptoDataLoader()
        # Skip load_data() if the loader already has data (e.g. pre-loaded)
        if self.loader.feat_tensor is None:
            self.loader.load_data()

        self.model = AlphaGPT().to(ModelConfig.DEVICE)

        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization

        # Experiment tracking
        tracker_config = {
            "batch_size": ModelConfig.BATCH_SIZE,
            "train_steps": ModelConfig.TRAIN_STEPS,
            "max_formula_len": ModelConfig.get_max_formula_len(),
            "seed": seed,
            "use_lord": use_lord_regularization,
            "lord_decay_rate": lord_decay_rate,
        }
        run_name = f"seed-{seed}" if seed is not None else None
        self.tracker = create_tracker(
            use_wandb=use_wandb,
            project=wandb_project,
            name=run_name,
            config=tracker_config,
            group=wandb_group,
        )

        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = MemeBacktest(position_mode='rank')
        # Separate backtest for OOS evaluation — always uses full tx costs (tx_cost_scale=1.0)
        self.bt_oos = MemeBacktest(position_mode='rank')

        # Cache action masking tensors on device (avoid repeated .to() in inner loop)
        self._arity_dev = _ARITY.to(ModelConfig.DEVICE)
        self._delta_dev = _STACK_DELTA.to(ModelConfig.DEVICE)

        self.best_score = -float('inf')
        self.best_formula = None
        self.best_diagnostics = {}
        self.steps_without_improvement = 0
        self.early_stop_patience = 800

        # Novelty tracking: set of formula skeletons seen so far
        self._seen_skeletons = set()
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': [],
            'train_sharpe': [],
            'test_sharpe': [],
            'test_score': [],
        }

    def _effective_formula_length(self, formula):
        """Count non-zero tokens in formula."""
        return len([t for t in formula if t != 0])

    def _get_base_factors(self, feat_tensor):
        """Extract raw returns (feature 0) as baseline for redundancy check."""
        return feat_tensor[:, 0, :]

    def _formula_skeleton(self, formula):
        """Convert formula to a skeleton: features→'F', ops→op_name. Used for novelty tracking."""
        feat_offset = self.vm.feat_offset
        parts = []
        for t in formula:
            t = int(t)
            if t < feat_offset:
                parts.append('F')
            else:
                parts.append(str(t))
        return tuple(parts)

    def _formula_structure_penalty(self, formula):
        """Penalize degenerate formula structures: repeat tokens and low diversity.

        Returns (penalty, bonus) tuple — both non-negative.
        """
        feat_offset = self.vm.feat_offset
        feat_counts = {}
        op_counts = {}

        for token in formula:
            t = int(token)
            if t < feat_offset:
                feat_counts[t] = feat_counts.get(t, 0) + 1
            else:
                op_counts[t] = op_counts.get(t, 0) + 1

        # Repeat penalty: features used >2 times get -0.5 per extra, ops >3 times get -0.3 per extra
        penalty = 0.0
        for count in feat_counts.values():
            if count > 2:
                penalty += 0.5 * (count - 2)
        for count in op_counts.values():
            if count > 3:
                penalty += 0.3 * (count - 3)

        # Diversity bonus: +0.15 per unique feature used beyond 2, capped at +0.6
        n_unique_feats = len(feat_counts)
        bonus = min(0.6, max(0, n_unique_feats - 2) * 0.15)

        return penalty, bonus

    def _evaluate_formula(self, formula, feat_tensor, raw_data, target_ret, base_factors):
        """Evaluate a formula with shaped rewards for partial failures."""
        # Simulate stack to understand failure mode
        stack_size = 0
        for step_i, token in enumerate(formula):
            token = int(token)
            if token < self.vm.feat_offset:
                stack_size += 1
            elif token in self.vm.arity_map:
                arity = self.vm.arity_map[token]
                if stack_size < arity:
                    # Stack underflow - reward based on progress
                    progress = step_i / len(formula)
                    return -5.0 + progress * 2.0  # range: -5.0 to -3.0
                stack_size = stack_size - arity + 1
            else:
                return -5.0  # Unknown token

        if stack_size != 1:
            # Wrong final stack size
            distance = abs(stack_size - 1)
            return -3.0 - min(distance, 4) * 0.5  # range: -3.5 to -5.0

        # Valid formula - execute
        res = self.vm.execute(formula, feat_tensor)
        if res is None:
            return -4.0  # Runtime error

        std_val = res.std().item()
        if std_val < 1e-4:
            return -2.0  # Truly constant
        if std_val < 0.01:
            # Almost non-constant: gradient signal from -2.0 to -1.0
            return -2.0 + (std_val / 0.01)

        formula_len = self._effective_formula_length(formula)
        score, ret_val = self.bt.evaluate(
            res, raw_data, target_ret,
            formula_length=formula_len,
            base_factors=base_factors,
        )

        # Apply structure penalty and diversity bonus
        struct_penalty, div_bonus = self._formula_structure_penalty(formula)

        # Novelty bonus: +0.3 if this formula skeleton hasn't been seen before
        skeleton = self._formula_skeleton(formula)
        novelty_bonus = 0.0
        if skeleton not in self._seen_skeletons:
            novelty_bonus = 0.3
            self._seen_skeletons.add(skeleton)

        return score.item() - struct_penalty + div_bonus + novelty_bonus

    def train(self):
        print("Starting Alpha Mining with OOS validation...")
        if self.use_lord:
            print(f"  LoRD Regularization enabled")
        print(f"  Train steps: {ModelConfig.TRAIN_STEPS}, Early stop patience: {self.early_stop_patience}")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        total_valid = 0
        total_formulas = 0

        for step in pbar:
            # No tx_cost curriculum — train with full costs for honest optimization

            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

            log_probs = []
            tokens_list = []
            entropies = []

            stack_sizes = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)
            # Track consecutive arity-3 ops per batch item (Phase 3: GATE limiter)
            consec_arity3 = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)
            max_consec_arity3 = 2

            # Pre-compute which tokens are arity-3 (for masking)
            _arity3_mask = (self._arity_dev == 3)  # [vocab]

            _max_len = ModelConfig.get_max_formula_len()
            for step_idx in range(_max_len):
                logits, _, _ = self.model(inp)

                # --- Action Masking ---
                remaining = _max_len - 1 - step_idx
                arity = self._arity_dev
                delta = self._delta_dev

                # For each (batch, token) pair, compute validity
                ss = stack_sizes.unsqueeze(1)  # [bs, 1]
                new_stack = ss + delta.unsqueeze(0)  # [bs, vocab]

                valid = (ss >= arity.unsqueeze(0))  # no underflow
                valid &= (new_stack + remaining >= 1)  # can still reach 1
                valid &= (new_stack - 2 * remaining <= 1)  # can still reduce to 1

                if remaining == 0:
                    valid &= (new_stack == 1)  # last step must end at 1

                # Phase 3: Block arity-3 ops if consecutive limit reached
                arity3_blocked = (consec_arity3 >= max_consec_arity3).unsqueeze(1)  # [bs, 1]
                valid &= ~(arity3_blocked & _arity3_mask.unsqueeze(0))  # [bs, vocab]

                # Apply mask
                logits = logits.masked_fill(~valid, -1e9)

                dist = Categorical(logits=logits)
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

                # Update stack sizes
                stack_sizes = stack_sizes + delta[action]

                # Update consecutive arity-3 counter
                is_arity3 = _arity3_mask[action]  # [bs] bool
                consec_arity3 = torch.where(is_arity3, consec_arity3 + 1, torch.zeros_like(consec_arity3))

            seqs = torch.stack(tokens_list, dim=1)

            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

            base_factors_train = self._get_base_factors(self.loader.train_feat)

            for i in range(bs):
                formula = seqs[i].tolist()
                rewards[i] = self._evaluate_formula(
                    formula, self.loader.train_feat,
                    self.loader.train_raw, self.loader.train_ret,
                    base_factors_train
                )

            # Track formula validity
            valid_count = (rewards > -5.0).sum().item()
            total_valid += valid_count
            total_formulas += bs

            # Handle NaN/Inf rewards
            rewards = torch.nan_to_num(rewards, nan=-5.0, posinf=-5.0, neginf=-5.0)

            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            adv = torch.nan_to_num(adv, nan=0.0)

            # Entropy coefficient decays linearly: 0.10 → 0.03 (higher floor for exploration)
            entropy_coef = max(0.03, 0.10 * (1.0 - step / ModelConfig.TRAIN_STEPS))

            policy_loss = 0
            entropy_bonus = 0
            for t in range(len(log_probs)):
                policy_loss += -log_probs[t] * adv
                entropy_bonus += entropies[t]

            loss = policy_loss.mean() - entropy_coef * entropy_bonus.mean()

            # Skip step if loss is NaN
            if torch.isnan(loss):
                continue

            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()

            # --- OOS evaluation every 50 steps ---
            if step % 50 == 0 and self.best_formula is not None:
                self._evaluate_oos(step, pbar)

            # --- Best formula tracking (on train) ---
            avg_reward = rewards.mean().item()
            best_in_batch = rewards.max().item()
            if best_in_batch > self.best_score:
                best_idx = rewards.argmax().item()
                candidate = seqs[best_idx].tolist()

                # Validate on TEST data before accepting (always full tx costs)
                test_res = self.vm.execute(candidate, self.loader.test_feat)
                if test_res is not None and test_res.std() > 1e-4:
                    formula_len = self._effective_formula_length(candidate)
                    base_factors_test = self._get_base_factors(self.loader.test_feat)
                    test_score, test_ret, test_diag = self.bt_oos.evaluate(
                        test_res, self.loader.test_raw, self.loader.test_ret,
                        formula_length=formula_len,
                        base_factors=base_factors_test,
                        return_diagnostics=True,
                    )

                    # Accept if OOS score is reasonable (not catastrophic)
                    if test_score.item() > -1.0:
                        self.best_score = best_in_batch
                        self.best_formula = candidate
                        self.best_diagnostics = test_diag
                        self.steps_without_improvement = 0
                        tqdm.write(
                            f"[!] New Best: TrainScore {best_in_batch:.2f} | "
                            f"TestSharpe {test_diag['sharpe']:.2f} | "
                            f"MaxDD {test_diag['max_drawdown']:.2%} | "
                            f"Formula {candidate}"
                        )
                    else:
                        self.steps_without_improvement += 1
                else:
                    self.steps_without_improvement += 1
            else:
                self.steps_without_improvement += 1

            # Logging
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'Best': f"{self.best_score:.3f}"}

            # W&B: log every step
            step_metrics = {
                'avg_reward': avg_reward,
                'best_score': self.best_score,
                'valid_rate': valid_count / bs if bs > 0 else 0,
            }

            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
                step_metrics['stable_rank'] = stable_rank

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)

            if step % 100 == 0:
                postfix_dict['Ent'] = f"{entropy_coef:.3f}"
                step_metrics['entropy_coef'] = entropy_coef

            self.tracker.log_step(step, step_metrics)
            pbar.set_postfix(postfix_dict)

            # Early stopping
            if self.steps_without_improvement >= self.early_stop_patience:
                tqdm.write(f"[*] Early stopping at step {step} (no improvement for {self.early_stop_patience} steps)")
                break

        if total_formulas > 0:
            print(f"  Formula validity rate: {total_valid/total_formulas:.1%}")

        self._save_results()

    def _evaluate_oos(self, step, pbar):
        """Run OOS evaluation on the current best formula (always full tx costs)."""
        test_res = self.vm.execute(self.best_formula, self.loader.test_feat)
        if test_res is None or test_res.std() < 1e-4:
            return

        formula_len = self._effective_formula_length(self.best_formula)
        base_factors_test = self._get_base_factors(self.loader.test_feat)

        test_score, test_ret, diag = self.bt_oos.evaluate(
            test_res, self.loader.test_raw, self.loader.test_ret,
            formula_length=formula_len,
            base_factors=base_factors_test,
            return_diagnostics=True,
        )

        self.training_history['test_score'].append(test_score.item())
        self.training_history['test_sharpe'].append(diag['sharpe'])

        oos_metrics = {
            'test_score': test_score.item(),
            'test_sharpe': diag['sharpe'],
            'test_max_drawdown': diag['max_drawdown'],
            'test_avg_turnover': diag['avg_turnover'],
        }

        # Also get train diagnostics (full tx costs)
        train_res = self.vm.execute(self.best_formula, self.loader.train_feat)
        if train_res is not None:
            base_factors_train = self._get_base_factors(self.loader.train_feat)
            _, _, train_diag = self.bt_oos.evaluate(
                train_res, self.loader.train_raw, self.loader.train_ret,
                formula_length=formula_len,
                base_factors=base_factors_train,
                return_diagnostics=True,
            )
            self.training_history['train_sharpe'].append(train_diag['sharpe'])
            oos_metrics['train_sharpe'] = train_diag['sharpe']

            if step % 100 == 0:
                tqdm.write(
                    f"  [OOS] Step {step} | "
                    f"TrainSharpe {train_diag['sharpe']:.2f} | "
                    f"TestSharpe {diag['sharpe']:.2f} | "
                    f"MaxDD {diag['max_drawdown']:.2%} | "
                    f"Turnover {diag['avg_turnover']:.4f}"
                )

        self.tracker.log_step(step, oos_metrics)

    def _save_results(self):
        """Save best formula and training history."""
        result = {
            'formula': self.best_formula,
            'train_score': self.best_score,
            'test_sharpe': self.best_diagnostics.get('sharpe', None),
            'max_drawdown': self.best_diagnostics.get('max_drawdown', None),
            'avg_turnover': self.best_diagnostics.get('avg_turnover', None),
            'factor_autocorrelation': self.best_diagnostics.get('factor_autocorrelation', None),
        }
        with open("best_meme_strategy.json", "w") as f:
            json.dump(result, f, indent=2)

        with open("training_history.json", "w") as f:
            json.dump(self.training_history, f)

        # W&B: log final summary and artifacts
        self.tracker.log_summary({
            'best_train_score': self.best_score,
            'best_test_sharpe': self.best_diagnostics.get('sharpe'),
            'best_max_drawdown': self.best_diagnostics.get('max_drawdown'),
            'best_avg_turnover': self.best_diagnostics.get('avg_turnover'),
            'best_formula': str(self.best_formula),
        })
        self.tracker.log_artifact("best-strategy", "best_meme_strategy.json")
        self.tracker.finish()

        print(f"\nTraining completed!")
        print(f"  Best train score: {self.best_score:.4f}")
        print(f"  Test Sharpe: {self.best_diagnostics.get('sharpe', 'N/A')}")
        print(f"  Max Drawdown: {self.best_diagnostics.get('max_drawdown', 'N/A')}")
        print(f"  Best formula: {self.best_formula}")

        # 年度牛熊周期分析
        self._analyze_cycles()

    def _analyze_cycles(self):
        """Analyze best formula performance across yearly bull/bear cycles."""
        if self.best_formula is None:
            print("\nNo valid formula found, skipping cycle analysis.")
            return

        print(f"\n{'='*72}")
        print(f"{'YEARLY BULL/BEAR CYCLE ANALYSIS':^72}")
        print(f"{'='*72}")

        # Execute on full dataset
        full_res = self.vm.execute(self.best_formula, self.loader.feat_tensor)
        if full_res is None:
            print("Formula failed on full dataset.")
            return

        # Compute daily PnL using bt_oos (always full tx costs)
        liquidity = self.loader.raw_data_cache['liquidity']
        is_safe = (liquidity > self.bt_oos.min_liq).float()
        position = self.bt_oos.compute_position(full_res, is_safe)

        impact = torch.clamp(self.bt_oos.trade_size / (liquidity + 1e-9), 0.0, 0.05)
        total_slip = self.bt_oos.base_fee + impact

        prev_pos = torch.cat([torch.zeros_like(position[:, :1]), position[:, :-1]], dim=1)
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slip

        net_pnl = position * self.loader.target_ret - tx_cost

        # Average across all stocks → daily portfolio PnL
        daily_pnl = net_pnl.mean(dim=0).cpu().numpy()
        daily_turnover = turnover.mean(dim=0).cpu().numpy()

        dates = self.loader.dates
        split_idx = int(len(dates) * self.loader.train_ratio)

        pnl_df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'pnl': daily_pnl,
            'turnover': daily_turnover,
        })
        pnl_df['year'] = pnl_df['date'].dt.year

        annualization = self.bt_oos.annualization

        print(f"{'Year':<6} {'Period':<7} {'Bars':>5} {'Sharpe':>8} {'CumRet':>10} "
              f"{'MaxDD':>10} {'Turnover':>10} {'WinRate':>8}")
        print(f"{'-'*72}")

        yearly_rows = []
        for year, g in pnl_df.groupby('year'):
            pnl = g['pnl'].values
            cum = np.cumsum(pnl)
            sharpe = (np.mean(pnl) / (np.std(pnl) + 1e-8)) * annualization
            cum_ret = cum[-1]
            running_max = np.maximum.accumulate(cum)
            max_dd = np.max(running_max - cum)
            avg_to = np.mean(g['turnover'].values)
            win_rate = np.mean(pnl > 0)

            # Determine train/test
            year_indices = g.index.tolist()
            if year_indices[-1] < split_idx:
                period = "Train"
            elif year_indices[0] >= split_idx:
                period = "Test"
            else:
                period = "Mixed"

            print(f"{year:<6} {period:<7} {len(pnl):>5} {sharpe:>8.2f} {cum_ret:>10.4f} "
                  f"{max_dd:>10.4f} {avg_to:>10.4f} {win_rate:>7.1%}")

            yearly_rows.append({
                'year': int(year), 'period': period, 'days': len(pnl),
                'sharpe': round(float(sharpe), 4), 'cum_ret': round(float(cum_ret), 6),
                'max_drawdown': round(float(max_dd), 6), 'avg_turnover': round(float(avg_to), 6),
                'win_rate': round(float(win_rate), 4),
            })

        # Overall OOS summary
        test_pnl = pnl_df[pnl_df.index >= split_idx]['pnl'].values
        if len(test_pnl) > 0:
            test_cum = np.cumsum(test_pnl)
            test_sharpe = (np.mean(test_pnl) / (np.std(test_pnl) + 1e-8)) * annualization
            test_dd = np.max(np.maximum.accumulate(test_cum) - test_cum)
            test_wr = np.mean(test_pnl > 0)
            print(f"{'-'*72}")
            print(f"{'OOS':.<6} {'Test':<7} {len(test_pnl):>5} {test_sharpe:>8.2f} "
                  f"{test_cum[-1]:>10.4f} {test_dd:>10.4f} {'':>10} {test_wr:>7.1%}")

        # Overall full period
        full_pnl = pnl_df['pnl'].values
        full_cum = np.cumsum(full_pnl)
        full_sharpe = (np.mean(full_pnl) / (np.std(full_pnl) + 1e-8)) * annualization
        full_dd = np.max(np.maximum.accumulate(full_cum) - full_cum)
        full_wr = np.mean(full_pnl > 0)
        print(f"{'ALL':.<6} {'Full':<7} {len(full_pnl):>5} {full_sharpe:>8.2f} "
              f"{full_cum[-1]:>10.4f} {full_dd:>10.4f} {'':>10} {full_wr:>7.1%}")
        print(f"{'='*72}")

        # Cross-check: bt_oos.evaluate on OOS data (per-stock median Sharpe)
        oos_res = self.vm.execute(self.best_formula, self.loader.test_feat)
        if oos_res is not None and oos_res.std() > 1e-4:
            formula_len = self._effective_formula_length(self.best_formula)
            base_test = self._get_base_factors(self.loader.test_feat)
            _, _, oos_diag = self.bt_oos.evaluate(
                oos_res, self.loader.test_raw, self.loader.test_ret,
                formula_length=formula_len, base_factors=base_test,
                return_diagnostics=True,
            )
            print(f"\n  bt_oos.evaluate OOS Sharpe (portfolio): {oos_diag['sharpe']:.4f}")
            if len(test_pnl) > 0:
                print(f"  Portfolio OOS Sharpe (mean-across-stocks):     {test_sharpe:.4f}")

        # Decode formula tokens
        vocab = self.model.vocab
        decoded = [vocab[t] if t < len(vocab) else f"?{t}" for t in self.best_formula]
        decoded_str = " \u2192 ".join(decoded)
        print(f"\nBest Formula (decoded): {decoded_str}")
        print(f"Best Formula (tokens):  {self.best_formula}")

        # Save cycle analysis
        result = json.load(open("best_meme_strategy.json"))
        result['yearly_analysis'] = yearly_rows
        result['formula_decoded'] = decoded
        with open("best_meme_strategy.json", "w") as f:
            json.dump(result, f, indent=2)


    @staticmethod
    def train_ensemble(num_seeds=6, use_lord_regularization=True, lord_decay_rate=1e-3,
                       lord_num_iterations=5, use_wandb=False, wandb_project="alphagpt",
                       loader_cls=None):
        """
        Train multiple models with different random seeds and build an ensemble.

        Each seed produces an independently trained model with its own best formula.
        The ensemble aggregates all valid formulas via equal-weight averaging,
        reducing variance by approximately sqrt(num_seeds).

        Args:
            num_seeds: Number of independent training runs (default: 6)
            use_lord_regularization: LoRD regularization flag
            lord_decay_rate: LoRD decay strength
            lord_num_iterations: Newton-Schulz iterations
            use_wandb: Enable W&B tracking (each seed = separate run in same group)
            wandb_project: W&B project name
            loader_cls: Optional data loader class (default: CryptoDataLoader)

        Returns:
            FormulaEnsemble instance with all valid formulas
        """
        formulas = []
        seed_results = []
        group_name = f"ensemble-{num_seeds}seeds" if use_wandb else None

        # Pre-load data once for the ensemble to share
        _cls = loader_cls if loader_cls is not None else _default_loader_cls()
        shared_loader = _cls()
        shared_loader.load_data()

        for seed_idx in range(num_seeds):
            seed = seed_idx * 42 + 7  # deterministic but varied seeds
            print(f"\n{'='*60}")
            print(f"  ENSEMBLE: Training seed {seed_idx+1}/{num_seeds} (seed={seed})")
            print(f"{'='*60}")

            engine = AlphaEngine(
                use_lord_regularization=use_lord_regularization,
                lord_decay_rate=lord_decay_rate,
                lord_num_iterations=lord_num_iterations,
                seed=seed,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_group=group_name,
                loader=shared_loader,
            )

            # Inject previously found formulas as "seen skeletons" to encourage diversity
            for prev_formula in formulas:
                sk = engine._formula_skeleton(prev_formula)
                engine._seen_skeletons.add(sk)

            engine.train()

            if engine.best_formula is not None:
                formulas.append(engine.best_formula)
                seed_results.append({
                    'seed': seed,
                    'formula': engine.best_formula,
                    'train_score': engine.best_score,
                    'test_sharpe': engine.best_diagnostics.get('sharpe'),
                    'max_drawdown': engine.best_diagnostics.get('max_drawdown'),
                })
                print(f"  -> Seed {seed}: score={engine.best_score:.4f}, "
                      f"sharpe={engine.best_diagnostics.get('sharpe', 'N/A')}")
            else:
                print(f"  -> Seed {seed}: no valid formula found")

            # Free GPU memory between seeds to prevent OOM / segfault
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not formulas:
            print("\n[ERROR] No valid formulas found across all seeds!")
            return None

        ensemble = FormulaEnsemble(formulas, mode='mean')

        # Evaluate ensemble on test data (reuse shared loader)
        loader = shared_loader
        bt = MemeBacktest()
        vm = StackVM()

        ensemble_signal = ensemble.predict(loader.test_feat)
        if ensemble_signal is not None:
            formula_len = max(len([t for t in f if t != 0]) for f in formulas)
            base_factors = loader.test_feat[:, 0, :]
            ens_score, ens_ret, ens_diag = bt.evaluate(
                ensemble_signal, loader.test_raw, loader.test_ret,
                formula_length=formula_len,
                base_factors=base_factors,
                return_diagnostics=True,
            )

            # Also evaluate each individual formula for comparison
            individual_sharpes = []
            for f in formulas:
                res = vm.execute(f, loader.test_feat)
                if res is not None:
                    _, _, diag = bt.evaluate(
                        res, loader.test_raw, loader.test_ret,
                        formula_length=len([t for t in f if t != 0]),
                        base_factors=base_factors,
                        return_diagnostics=True,
                    )
                    individual_sharpes.append(diag['sharpe'])

            print(f"\n{'='*60}")
            print(f"  ENSEMBLE RESULTS ({len(formulas)} formulas)")
            print(f"{'='*60}")
            print(f"  Ensemble Sharpe:     {ens_diag['sharpe']:.4f}")
            print(f"  Ensemble MaxDD:      {ens_diag['max_drawdown']:.4%}")
            print(f"  Ensemble Turnover:   {ens_diag['avg_turnover']:.4f}")
            if individual_sharpes:
                print(f"  Individual Sharpes:  {[f'{s:.4f}' for s in individual_sharpes]}")
                print(f"  Best Single Sharpe:  {max(individual_sharpes):.4f}")
                print(f"  Avg Single Sharpe:   {np.mean(individual_sharpes):.4f}")
                print(f"  Sharpe Std (single): {np.std(individual_sharpes):.4f}")
            print(f"{'='*60}")

            ens_diag_save = ens_diag
        else:
            ens_diag_save = {}

        # Save ensemble result
        result = {
            'ensemble': ensemble.to_dict(),
            'seed_results': seed_results,
            'ensemble_test_sharpe': ens_diag_save.get('sharpe'),
            'ensemble_max_drawdown': ens_diag_save.get('max_drawdown'),
            'ensemble_avg_turnover': ens_diag_save.get('avg_turnover'),
            'num_valid_formulas': len(formulas),
            'num_seeds': num_seeds,
        }
        with open("best_ensemble.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nEnsemble saved to best_ensemble.json")
        return ensemble


    @staticmethod
    def evaluate_with_cv(formula=None, ensemble_path=None, n_splits=5,
                         min_train_pct=0.3, gap=0, mode='expanding',
                         loader_cls=None):
        """
        Evaluate a formula or ensemble across temporal CV folds.

        Provides robust OOS evaluation across multiple time periods instead
        of relying on a single train/test split.

        Args:
            formula: single formula token list (mutually exclusive with ensemble_path)
            ensemble_path: path to ensemble JSON file
            n_splits: number of CV folds
            min_train_pct: minimum training data fraction
            gap: purge gap between train and test
            mode: 'expanding' or 'rolling'
            loader_cls: Optional data loader class (default: CryptoDataLoader)

        Returns:
            dict with CV results
        """
        # Load data
        _cls = loader_cls if loader_cls is not None else _default_loader_cls()
        loader = _cls()
        loader.load_data()

        cv = TimeSeriesCV(n_splits=n_splits, min_train_pct=min_train_pct,
                          gap=gap, mode=mode)

        total_steps = loader.feat_tensor.shape[2]
        folds = cv.split(total_steps)

        print(f"\n{'='*72}")
        print(f"{'TIME SERIES CROSS-VALIDATION':^72}")
        print(f"{'='*72}")
        print(f"  Mode: {mode} | Folds: {n_splits} | Min train: {min_train_pct:.0%} | Gap: {gap}")
        print(f"  Total steps: {total_steps}")
        print()

        # Show fold structure
        print(f"  {'Fold':<6} {'Train':>14} {'Test':>14} {'Train Days':>11} {'Test Days':>10}")
        print(f"  {'-'*58}")
        for i, (ts, te, vs, ve) in enumerate(folds):
            print(f"  {i+1:<6} [{ts:>5}:{te:<5}]  [{vs:>5}:{ve:<5}]  {te-ts:>8}    {ve-vs:>8}")
        print()

        # Determine what to evaluate
        formulas_to_eval = []
        labels = []

        if ensemble_path is not None:
            with open(ensemble_path) as f:
                ens_data = json.load(f)
            for i, sr in enumerate(ens_data.get('seed_results', [])):
                formulas_to_eval.append(sr['formula'])
                labels.append(f"Seed-{sr.get('seed', i)}")
        elif formula is not None:
            formulas_to_eval = [formula]
            labels = ['Formula']
        else:
            # Try loading from best_meme_strategy.json
            try:
                with open("best_meme_strategy.json") as f:
                    data = json.load(f)
                formulas_to_eval = [data['formula']]
                labels = ['BestFormula']
            except (FileNotFoundError, KeyError):
                print("[ERROR] No formula or ensemble provided.")
                return None

        # Evaluate each formula
        all_results = []
        print(f"{'='*72}")
        print(f"  {'Formula':<14} {'Fold':<6} {'Sharpe':>8} {'MaxDD':>10} {'Turnover':>10}")
        print(f"  {'-'*52}")

        for formula_tokens, label in zip(formulas_to_eval, labels):
            cv_result = evaluate_formula_cv(
                formula_tokens, loader.feat_tensor, loader.target_ret,
                loader.raw_data_cache, cv,
            )
            cv_result['label'] = label
            cv_result['formula'] = formula_tokens
            all_results.append(cv_result)

            for fr in cv_result['fold_results']:
                if fr['valid']:
                    print(f"  {label:<14} {fr['fold']+1:<6} {fr['sharpe']:>8.4f} "
                          f"{fr['max_drawdown']:>9.2%} {fr['avg_turnover']:>10.4f}")
                else:
                    print(f"  {label:<14} {fr['fold']+1:<6} {'INVALID':>8}")

        # Summary
        print(f"\n{'='*72}")
        print(f"  {'CROSS-VALIDATION SUMMARY':^68}")
        print(f"{'='*72}")
        print(f"  {'Formula':<14} {'MeanSharpe':>11} {'StdSharpe':>10} {'MeanDD':>10} {'ValidFolds':>11}")
        print(f"  {'-'*60}")

        for r in all_results:
            print(f"  {r['label']:<14} {r['mean_sharpe']:>11.4f} {r['std_sharpe']:>10.4f} "
                  f"{r['mean_max_drawdown']:>9.2%} {r['num_valid_folds']:>8}/{n_splits}")

        # Overall robustness assessment
        if len(all_results) > 1:
            sharpes = [r['mean_sharpe'] for r in all_results if r['num_valid_folds'] > 0]
            stds = [r['std_sharpe'] for r in all_results if r['num_valid_folds'] > 0]
            print(f"\n  Avg CV Sharpe across formulas: {np.mean(sharpes):.4f}")
            print(f"  Avg CV Sharpe StdDev:          {np.mean(stds):.4f}")

            # Find most robust formula (highest mean_sharpe / std_sharpe ratio)
            best_idx = -1
            best_ratio = -float('inf')
            for i, r in enumerate(all_results):
                if r['num_valid_folds'] > 0 and r['std_sharpe'] > 0:
                    ratio = r['mean_sharpe'] / r['std_sharpe']
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_idx = i
            if best_idx >= 0:
                print(f"  Most robust formula: {all_results[best_idx]['label']} "
                      f"(Sharpe/Std = {best_ratio:.2f})")

        print(f"{'='*72}")

        # Compare with single split
        if formulas_to_eval:
            print(f"\n  Comparison: Single 70/30 split vs {n_splits}-fold CV")
            vm = StackVM()
            bt = MemeBacktest()
            split_idx = int(total_steps * 0.7)
            test_feat = loader.feat_tensor[:, :, split_idx:]
            test_ret = loader.target_ret[:, split_idx:]
            test_raw = {k: (v[:, split_idx:] if v.dim() > 1 else v[split_idx:])
                        for k, v in loader.raw_data_cache.items()}

            for formula_tokens, label, cv_res in zip(formulas_to_eval, labels, all_results):
                res = vm.execute(formula_tokens, test_feat)
                if res is not None and res.std() > 1e-8:
                    base = test_feat[:, 0, :]
                    _, _, diag = bt.evaluate(
                        res, test_raw, test_ret,
                        formula_length=len([t for t in formula_tokens if t != 0]),
                        base_factors=base,
                        return_diagnostics=True,
                    )
                    single_sharpe = diag['sharpe']
                    cv_sharpe = cv_res['mean_sharpe']
                    diff = single_sharpe - cv_sharpe
                    print(f"  {label:<14} Single={single_sharpe:.4f}  CV={cv_sharpe:.4f}  "
                          f"Diff={diff:+.4f} {'⚠ overfit risk' if diff > 0.3 else ''}")

        # Save results
        save_data = {
            'cv_config': {
                'n_splits': n_splits, 'min_train_pct': min_train_pct,
                'gap': gap, 'mode': mode,
            },
            'results': [{
                'label': r['label'],
                'formula': r['formula'],
                'mean_sharpe': r['mean_sharpe'],
                'std_sharpe': r['std_sharpe'],
                'mean_max_drawdown': r['mean_max_drawdown'],
                'num_valid_folds': r['num_valid_folds'],
                'fold_results': r['fold_results'],
            } for r in all_results],
        }
        with open("cv_results.json", "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Results saved to cv_results.json")

        return all_results


    @staticmethod
    def evaluate_online(ensemble_path="best_ensemble.json",
                        lookback_window=20, learning_rate=0.3,
                        decay=0.95, min_weight=0.05,
                        period_length=1, warmup_periods=20,
                        loader_cls=None):
        """
        Evaluate an ensemble with online learning (adaptive weight optimization).

        Walks forward through time, updating formula weights based on
        recent realized performance — analogous to Jane Street's daily
        model weight updates.

        Args:
            ensemble_path: path to ensemble JSON file
            lookback_window: rolling window for performance evaluation
            learning_rate: weight adaptation rate (0=static, 1=fully adaptive)
            decay: exponential decay for historical performance
            min_weight: minimum weight per formula
            period_length: steps per update period (1 = daily)
            warmup_periods: periods before adaptation starts
            loader_cls: optional data loader class

        Returns:
            dict with online learning results
        """
        # Load ensemble
        with open(ensemble_path) as f:
            ens_data = json.load(f)
        ensemble = FormulaEnsemble.from_dict(ens_data['ensemble'])

        # Load data
        _cls = loader_cls if loader_cls is not None else _default_loader_cls()
        loader = _cls()
        loader.load_data()

        # Use test data for online learning evaluation
        raw_test = {k: v[:, int(v.shape[1] * loader.train_ratio):]
                    for k, v in loader.raw_data_cache.items()}
        test_feat = loader.test_feat
        test_ret = loader.test_ret

        # Create OnlineLearner
        ol = OnlineLearner(
            ensemble,
            lookback_window=lookback_window,
            learning_rate=learning_rate,
            decay=decay,
            min_weight=min_weight,
        )

        print(f"\n{'='*72}")
        print(f"{'ONLINE LEARNING EVALUATION':^72}")
        print(f"{'='*72}")
        print(f"  Ensemble: {ensemble.num_formulas} formulas")
        print(f"  Lookback: {lookback_window} periods | LR: {learning_rate} | Decay: {decay}")
        print(f"  Period: {period_length} steps | Warmup: {warmup_periods} periods")
        print()

        # Run online simulation
        result = ol.run_online(
            test_feat, raw_test, test_ret,
            period_length=period_length,
            warmup_periods=warmup_periods,
        )

        # Print results
        print(f"  {'Metric':<25} {'Static':>10} {'Online':>10} {'Improvement':>12}")
        print(f"  {'-'*60}")
        print(f"  {'Annualized Sharpe':<25} {result['static_sharpe']:>10.4f} "
              f"{result['online_sharpe']:>10.4f} {result['improvement']:>+12.4f}")

        # Cumulative PnL
        static_cum = sum(result['static_pnl'])
        online_cum = sum(result['online_pnl'])
        print(f"  {'Cumulative PnL':<25} {static_cum:>10.6f} "
              f"{online_cum:>10.6f} {online_cum - static_cum:>+12.6f}")

        # Final weights
        final_w = result['weight_history'][-1] if result['weight_history'] else []
        print(f"\n  Final Adaptive Weights: {[f'{w:.3f}' for w in final_w]}")
        print(f"{'='*72}")

        # Save results
        save_data = {
            'online_sharpe': result['online_sharpe'],
            'static_sharpe': result['static_sharpe'],
            'improvement': result['improvement'],
            'final_weights': final_w.tolist() if hasattr(final_w, 'tolist') else list(final_w),
            'config': {
                'lookback_window': lookback_window,
                'learning_rate': learning_rate,
                'decay': decay,
                'min_weight': min_weight,
                'period_length': period_length,
                'warmup_periods': warmup_periods,
            },
            'online_learner': ol.to_dict(),
        }
        with open("online_learning_results.json", "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Results saved to online_learning_results.json")

        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'ensemble':
        num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        AlphaEngine.train_ensemble(num_seeds=num_seeds)
    elif len(sys.argv) > 1 and sys.argv[1] == 'cv':
        # python -m model_core.engine cv [ensemble_path]
        ens_path = sys.argv[2] if len(sys.argv) > 2 else "best_ensemble.json"
        AlphaEngine.evaluate_with_cv(ensemble_path=ens_path)
    elif len(sys.argv) > 1 and sys.argv[1] == 'cv-single':
        # python -m model_core.engine cv-single
        AlphaEngine.evaluate_with_cv()
    elif len(sys.argv) > 1 and sys.argv[1] == 'online':
        # python -m model_core.engine online [ensemble_path]
        ens_path = sys.argv[2] if len(sys.argv) > 2 else "best_ensemble.json"
        AlphaEngine.evaluate_online(ensemble_path=ens_path)
    else:
        eng = AlphaEngine(use_lord_regularization=True)
        eng.train()
