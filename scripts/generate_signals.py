"""
Daily signal generation for AlphaGPT commodity futures trading.

Loads the trained ensemble, applies MWU online learning, generates
per-product target weights with regime detection.

Usage:
    python scripts/generate_signals.py [--output signals/] [--ensemble best_ensemble.json]
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('MODEL_ASSET_CLASS', 'futures')
os.environ.setdefault('ENABLE_TERM_STRUCTURE', '1')

from model_core.ensemble import FormulaEnsemble
from model_core.online_learner import OnlineLearner
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.duckdb_loader import DuckDBDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('signal_gen')


def load_ensemble(path):
    with open(path) as f:
        data = json.load(f)
    ensemble = FormulaEnsemble.from_dict(data['ensemble'])
    log.info(f"Loaded ensemble: {ensemble.num_formulas} formulas from {path}")
    return ensemble, data


def detect_regime(pnl_history, window=20, threshold=-0.5):
    """Detect adverse regime from recent PnL.

    Returns scale factor: 1.0 = normal, 0.5 = adverse regime.
    """
    if len(pnl_history) < window:
        return 1.0
    recent = np.array(pnl_history[-window:])
    if recent.std() < 1e-10:
        return 1.0
    rolling_sharpe = recent.mean() / (recent.std() + 1e-10) * np.sqrt(252)
    if rolling_sharpe < threshold:
        log.warning(f"Adverse regime detected: 20d Sharpe = {rolling_sharpe:.2f}, reducing to 50%")
        return 0.5
    return 1.0


def update_mwu_weights(ensemble, loader, online_learner, lookback=10):
    """Update MWU weights using recent realized PnL.

    Computes each formula's individual PnL over the last `lookback` days,
    then applies the MWU multiplicative weight update.
    """
    vm = StackVM()
    bt = MemeBacktest(position_mode='rank')

    feat = loader.feat_tensor
    ret = loader.target_ret
    raw = loader.raw_data_cache
    N, F, T = feat.shape

    if T < lookback + 5:
        log.warning("Not enough data for MWU update, skipping")
        return

    # Compute each formula's PnL over last lookback days
    period_feat = feat[:, :, -lookback:]
    period_ret = ret[:, -lookback:]
    period_raw = {k: (v[:, -lookback:] if v.dim() > 1 else v[-lookback:])
                  for k, v in raw.items()}

    formula_pnls = np.zeros(ensemble.num_formulas)
    for i, formula in enumerate(ensemble.formulas):
        res = vm.execute(formula, period_feat)
        if res is None or res.std() < 1e-8:
            continue
        liq = period_raw['liquidity']
        is_safe = (liq > bt.min_liq).float() if liq.dim() > 1 else (liq > bt.min_liq).float()
        pos = bt.compute_position(res, is_safe)
        pnl = (pos * period_ret).mean().item()
        formula_pnls[i] = pnl

    # MWU update
    old_w = online_learner.get_weights()
    result = online_learner.update(period_feat, period_raw, period_ret)
    new_w = result['weights']

    log.info(f"MWU weight update (last {lookback}d PnL):")
    log.info(f"  Formula PnLs: {[f'{p:+.6f}' for p in formula_pnls]}")
    log.info(f"  Old weights:  {[f'{w:.3f}' for w in old_w]}")
    log.info(f"  New weights:  {[f'{w:.3f}' for w in new_w]}")


def generate_signals(ensemble, loader, online_learner=None):
    """Generate per-product target weights."""
    vm = StackVM()
    bt = MemeBacktest(position_mode='rank')

    feat = loader.feat_tensor      # [N, F, T]
    raw = loader.raw_data_cache
    ret = loader.target_ret        # [N, T]
    N, F, T = feat.shape

    # Use last day's features for signal generation
    latest_feat = feat[:, :, -1:]   # [N, F, 1]
    liquidity = raw['liquidity']
    if liquidity.dim() > 1:
        latest_liq = liquidity[:, -1:]
    else:
        latest_liq = liquidity[-1:]

    # Per-formula signals
    formula_signals = []
    formula_names = []
    for i, formula in enumerate(ensemble.formulas):
        res = vm.execute(formula, latest_feat)
        if res is not None and res.std() > 1e-8:
            formula_signals.append(res.squeeze(-1))  # [N]
            formula_names.append(f"seed_{i}")
        else:
            formula_signals.append(torch.zeros(N))
            formula_names.append(f"seed_{i}_dead")

    # Ensemble signal (weighted by MWU if available)
    if online_learner is not None:
        weights = online_learner.get_weights()
    else:
        weights = np.ones(ensemble.num_formulas) / ensemble.num_formulas

    stacked = torch.stack(formula_signals, dim=0)  # [num_formulas, N]
    device = stacked.device
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    ensemble_signal = (stacked * w_tensor.unsqueeze(-1)).sum(dim=0)  # [N]

    # Cross-sectional rank normalization
    ranks = ensemble_signal.argsort().argsort().float()
    rank_norm = (ranks / (N - 1) - 0.5) * 2  # [-1, 1]

    # Regime detection from recent portfolio PnL
    # Use last 20 days of ensemble signal PnL as proxy
    recent_pnls = []
    for t in range(max(0, T - 21), T - 1):
        day_feat = feat[:, :, t:t+1]
        sig = ensemble.predict(day_feat)
        if sig is not None:
            is_safe = torch.ones(N, 1, device=sig.device)
            pos = bt.compute_position(sig, is_safe)
            pnl = (pos * ret[:, t+1:t+2]).mean().item()
            recent_pnls.append(pnl)

    regime_scale = detect_regime(recent_pnls)

    # Build output
    # Get product addresses from the data
    # We need to map indices back to product names
    products = []
    try:
        import duckdb
        con = duckdb.connect(loader.db_path, read_only=True)
        product_df = con.execute("""
            SELECT DISTINCT product_id || '.' || exchange as address
            FROM kline_1min ORDER BY address
        """).fetchdf()
        con.close()
        products = product_df['address'].tolist()[:N]
    except Exception:
        products = [f"product_{i}" for i in range(N)]

    signals = []
    for i in range(N):
        raw_weight = rank_norm[i].item()
        adjusted_weight = raw_weight * regime_scale

        # Per-formula contribution
        contributions = {}
        for j, name in enumerate(formula_names):
            contributions[name] = {
                'signal': formula_signals[j][i].item(),
                'weight': float(weights[j]),
            }

        signals.append({
            'product': products[i] if i < len(products) else f'product_{i}',
            'target_weight': round(adjusted_weight, 6),
            'raw_weight': round(raw_weight, 6),
            'signal_strength': round(abs(raw_weight), 6),
            'direction': 'LONG' if adjusted_weight > 0.01 else ('SHORT' if adjusted_weight < -0.01 else 'FLAT'),
            'regime_scale': regime_scale,
            'formula_contributions': contributions,
        })

    # Sort by signal strength
    signals.sort(key=lambda x: x['signal_strength'], reverse=True)

    output = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'ensemble_weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
        'regime_scale': regime_scale,
        'num_products': N,
        'num_long': sum(1 for s in signals if s['direction'] == 'LONG'),
        'num_short': sum(1 for s in signals if s['direction'] == 'SHORT'),
        'num_flat': sum(1 for s in signals if s['direction'] == 'FLAT'),
        'signals': signals,
    }
    return output


def main():
    parser = argparse.ArgumentParser(description='Generate daily trading signals')
    parser.add_argument('--ensemble', default='best_ensemble.json', help='Ensemble JSON path')
    parser.add_argument('--output', default='signals/', help='Output directory')
    parser.add_argument('--mwu-state', default='signals/mwu_state.json',
                        help='MWU online learner state (auto-created if missing)')
    parser.add_argument('--mwu-eta', type=float, default=0.05, help='MWU learning rate')
    parser.add_argument('--mwu-lookback', type=int, default=10, help='Days for MWU PnL computation')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load ensemble
    ensemble, ens_data = load_ensemble(args.ensemble)

    # Load data
    log.info("Loading data...")
    loader = DuckDBDataLoader(timeframe='1d')
    loader.load_data()

    # Load or create MWU online learner
    if os.path.exists(args.mwu_state):
        with open(args.mwu_state) as f:
            ol_data = json.load(f)
        online_learner = OnlineLearner.from_dict(ol_data)
        log.info(f"Loaded MWU state: strategy={online_learner.strategy}, "
                 f"weights={[f'{w:.3f}' for w in online_learner.get_weights()]}, "
                 f"step={online_learner._mwu_step}")
    else:
        online_learner = OnlineLearner(ensemble, strategy='mwu', eta=args.mwu_eta,
                                        lookback_window=args.mwu_lookback, min_weight=0.02)
        log.info(f"Created new MWU learner: eta={args.mwu_eta}, equal weights")

    # Step 1: Update MWU weights using recent realized PnL
    log.info("Updating MWU weights...")
    update_mwu_weights(ensemble, loader, online_learner, lookback=args.mwu_lookback)

    # Step 2: Generate signals with updated weights
    log.info("Generating signals with MWU weights...")
    output = generate_signals(ensemble, loader, online_learner)

    # Step 3: Save MWU state for next run
    mwu_dict = online_learner.to_dict()
    with open(args.mwu_state, 'w') as f:
        json.dump(mwu_dict, f, indent=2)
    log.info(f"MWU state saved to {args.mwu_state}")

    # Save signals
    date_str = output['date']
    out_path = os.path.join(args.output, f'signals_{date_str}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    latest_path = os.path.join(args.output, 'latest.json')
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"Signals saved: {out_path}")
    log.info(f"  Long: {output['num_long']}, Short: {output['num_short']}, Flat: {output['num_flat']}")
    log.info(f"  Regime scale: {output['regime_scale']}")
    log.info(f"  MWU weights: {[f'{w:.3f}' for w in online_learner.get_weights()]}")

    # Print top signals
    print(f"\nTop 10 signals ({date_str}):")
    print(f"{'Product':<15} {'Direction':<8} {'Weight':>8} {'Strength':>10}")
    print('-' * 45)
    for s in output['signals'][:10]:
        print(f"{s['product']:<15} {s['direction']:<8} {s['target_weight']:>+8.4f} {s['signal_strength']:>10.4f}")


if __name__ == '__main__':
    main()
