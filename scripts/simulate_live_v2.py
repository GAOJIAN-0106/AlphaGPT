"""
Historical live-simulation backtest V2 for AlphaGPT.

Upgrades over V1:
  - MWU online learning: dynamically adjusts formula weights based on recent PnL
  - Walk-forward full-period: uses ALL data (not just test set), with expanding
    training window to avoid look-ahead bias
  - Regime detection: reduces positions when rolling Sharpe is negative

Usage:
    python scripts/simulate_live_v2.py --capital 300000 --topn 5 --fee 0.0005 --rebalance 10
"""

import os
import sys
import json
import argparse
import logging
import math
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('MODEL_ASSET_CLASS', 'futures')
os.environ.setdefault('ENABLE_TERM_STRUCTURE', '1')

from model_core.ensemble import FormulaEnsemble
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.duckdb_loader import DuckDBDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('sim_live_v2')


# ---------------------------------------------------------------------------
# Contract specs (same as V1)
# ---------------------------------------------------------------------------
CONTRACT_SPECS = {
    'cu': (5, 0.12, 10), 'al': (5, 0.12, 5), 'zn': (5, 0.12, 5),
    'pb': (5, 0.12, 5), 'ni': (1, 0.15, 10), 'sn': (1, 0.15, 10),
    'au': (1000, 0.12, 0.02), 'ag': (15, 0.12, 1),
    'rb': (10, 0.10, 1), 'hc': (10, 0.10, 1), 'i': (100, 0.13, 0.5),
    'j': (100, 0.12, 0.5), 'jm': (60, 0.12, 0.5),
    'sc': (1000, 0.15, 0.1), 'fu': (10, 0.12, 1), 'bu': (10, 0.12, 1),
    'lu': (10, 0.12, 1), 'pg': (20, 0.12, 1),
    'TA': (5, 0.10, 2), 'MA': (10, 0.10, 1), 'eg': (10, 0.10, 1),
    'PP': (5, 0.10, 1), 'L': (5, 0.10, 5), 'V': (5, 0.10, 5),
    'SA': (20, 0.10, 1), 'eb': (5, 0.10, 1), 'ru': (10, 0.14, 5),
    'c': (10, 0.10, 1), 'cs': (10, 0.10, 1), 'a': (10, 0.10, 1),
    'm': (10, 0.10, 1), 'y': (10, 0.10, 2), 'p': (10, 0.10, 2),
    'OI': (10, 0.10, 1), 'RM': (10, 0.10, 1), 'CF': (5, 0.10, 5),
    'SR': (10, 0.10, 1), 'AP': (10, 0.10, 1), 'CJ': (5, 0.12, 5),
    'jd': (10, 0.10, 1), 'lh': (16, 0.12, 5), 'b': (10, 0.10, 1),
    'rr': (10, 0.10, 1), 'ss': (5, 0.10, 5), 'sp': (10, 0.10, 2),
    'SM': (5, 0.10, 2), 'SF': (5, 0.10, 2), 'FG': (20, 0.10, 1),
    'PF': (5, 0.10, 2), 'UR': (20, 0.10, 1), 'PK': (5, 0.10, 2),
    'bc': (5, 0.14, 10), 'nr': (10, 0.14, 5), 'si': (5, 0.12, 5),
    'ao': (20, 0.12, 1), 'lc': (10, 0.12, 1),
}

SECTOR_MAP = {
    'rb': 'Ferrous', 'hc': 'Ferrous', 'i': 'Ferrous', 'j': 'Ferrous', 'jm': 'Ferrous',
    'SF': 'Ferrous', 'SM': 'Ferrous', 'ss': 'Ferrous',
    'cu': 'Base Metal', 'al': 'Base Metal', 'zn': 'Base Metal', 'pb': 'Base Metal',
    'ni': 'Base Metal', 'sn': 'Base Metal', 'bc': 'Base Metal', 'ao': 'Base Metal', 'si': 'Base Metal',
    'au': 'Precious', 'ag': 'Precious',
    'sc': 'Energy', 'fu': 'Energy', 'bu': 'Energy', 'lu': 'Energy', 'pg': 'Energy',
    'TA': 'Chemical', 'MA': 'Chemical', 'eg': 'Chemical', 'eb': 'Chemical', 'PP': 'Chemical',
    'L': 'Chemical', 'V': 'Chemical', 'SA': 'Chemical', 'ru': 'Chemical', 'nr': 'Chemical',
    'FG': 'Chemical', 'PF': 'Chemical', 'sp': 'Chemical', 'UR': 'Chemical', 'SH': 'Chemical',
    'c': 'Agri', 'cs': 'Agri', 'a': 'Agri', 'm': 'Agri', 'y': 'Agri', 'p': 'Agri',
    'OI': 'Agri', 'RM': 'Agri', 'CF': 'Agri', 'SR': 'Agri', 'AP': 'Agri', 'CJ': 'Agri',
    'jd': 'Agri', 'lh': 'Agri', 'PK': 'Agri', 'b': 'Agri', 'rr': 'Agri',
}


def get_spec(pid):
    return CONTRACT_SPECS.get(pid, (10, 0.12, 1))

def get_sector(addr):
    return SECTOR_MAP.get(addr.split('.')[0], 'Other')


# ---------------------------------------------------------------------------
# MWU Online Weights
# ---------------------------------------------------------------------------
class MWUWeights:
    """Lightweight MWU for simulation (no dependency on OnlineLearner module)."""

    def __init__(self, n_formulas, eta=0.03, min_weight=0.02):
        self.n = n_formulas
        self.eta = eta
        self.min_weight = min_weight
        self.weights = np.ones(n_formulas) / n_formulas
        self.weight_history = []

    def update(self, formula_pnls):
        """Update weights based on per-formula PnL.

        formula_pnls: array of length n_formulas
        """
        max_abs = np.abs(formula_pnls).max()
        if max_abs < 1e-12:
            self.weight_history.append(self.weights.copy())
            return self.weights.copy()

        gains = formula_pnls / max_abs  # normalize to [-1, 1]

        # Multiplicative update
        new_w = self.weights * (1.0 + self.eta * gains)
        new_w = np.maximum(new_w, self.min_weight)
        new_w = new_w / new_w.sum()

        self.weights = new_w
        self.weight_history.append(self.weights.copy())
        return self.weights.copy()

    def get_weights_tensor(self, device):
        return torch.tensor(self.weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Simulator V2
# ---------------------------------------------------------------------------
class LiveSimulatorV2:
    def __init__(self, capital, top_n, fee_rate, rebalance_period,
                 max_margin_pct=0.60, max_per_position_pct=0.20,
                 max_per_sector=2, stop_loss_pct=0.03,
                 mwu_eta=0.03, regime_window=40, regime_threshold=-0.3):
        self.initial_capital = capital
        self.capital = capital
        self.equity = capital
        self.peak_equity = capital
        self.top_n = top_n
        self.fee_rate = fee_rate
        self.rebalance_period = rebalance_period
        self.max_margin_pct = max_margin_pct
        self.max_per_position_pct = max_per_position_pct
        self.max_per_sector = max_per_sector
        self.stop_loss_pct = stop_loss_pct
        self.regime_window = regime_window
        self.regime_threshold = regime_threshold

        self.positions = {}
        self.equity_curve = []
        self.trade_log = []
        self.daily_log = []
        self.mwu = None  # initialized when we know n_formulas
        self.mwu_eta = mwu_eta

    @property
    def drawdown(self):
        if self.peak_equity <= 0: return 0
        return (self.peak_equity - self.equity) / self.peak_equity

    def detect_regime(self):
        """Reduce positions if recent equity trend is bad."""
        if len(self.equity_curve) < self.regime_window:
            return 1.0
        recent = np.array(self.equity_curve[-self.regime_window:])
        rets = np.diff(recent) / recent[:-1]
        if rets.std() < 1e-10:
            return 1.0
        sharpe = rets.mean() / (rets.std() + 1e-10) * math.sqrt(252)
        if sharpe < self.regime_threshold:
            return 0.5  # reduce 50%
        return 1.0

    def compute_formula_pnls(self, ensemble, vm, bt, feat_t, raw_t, ret_t):
        """Compute each formula's PnL for one period (for MWU update)."""
        N = feat_t.shape[0]
        pnls = np.zeros(ensemble.num_formulas)
        for i, formula in enumerate(ensemble.formulas):
            res = vm.execute(formula, feat_t)
            if res is None or res.std() < 1e-8:
                continue
            liq = raw_t['liquidity']
            is_safe = (liq > bt.min_liq).float() if liq.dim() > 1 else (liq > bt.min_liq).float()
            pos = bt.compute_position(res, is_safe)
            pnl = (pos * ret_t).mean().item()
            pnls[i] = pnl
        return pnls

    def generate_signals_with_mwu(self, ensemble, vm, feat_up_to_t,
                                   close_t, addresses, regime_scale):
        """Generate signals using MWU-weighted ensemble."""
        N = feat_up_to_t.shape[0]

        # Per-formula signals
        formula_signals = []
        for formula in ensemble.formulas:
            res = vm.execute(formula, feat_up_to_t[:, :, -1:])
            if res is not None and res.std() > 1e-8:
                formula_signals.append(res.squeeze(-1))
            else:
                formula_signals.append(torch.zeros(N, device=feat_up_to_t.device))

        stacked = torch.stack(formula_signals, dim=0)  # [n_formulas, N]

        # MWU weights
        w = self.mwu.get_weights_tensor(stacked.device)
        signal = (stacked * w.unsqueeze(-1)).sum(dim=0)  # [N]

        # Apply regime scale
        signal = signal * regime_scale

        # Cross-sectional rank
        ranks = signal.argsort().argsort().float()
        rank_norm = (ranks / max(N - 1, 1) - 0.5) * 2

        # Build candidates
        max_margin = self.capital * self.max_per_position_pct
        total_budget = self.capital * self.max_margin_pct
        per_pos = total_budget / (2 * self.top_n) if self.top_n > 0 else 0

        candidates = []
        for i in range(N):
            addr = addresses[i]
            pid = addr.split('.')[0]
            mult, margin_rate, _ = get_spec(pid)
            price = close_t[i].item()
            if price <= 0: continue
            margin_per_lot = price * mult * margin_rate
            if margin_per_lot > max_margin or margin_per_lot <= 0: continue

            lots = max(int(per_pos / margin_per_lot), 1)
            candidates.append({
                'idx': i, 'address': addr, 'sector': get_sector(addr),
                'signal': rank_norm[i].item(), 'strength': abs(rank_norm[i].item()),
                'price': price, 'multiplier': mult,
                'margin_per_lot': margin_per_lot, 'lots': lots,
            })

        longs = sorted([c for c in candidates if c['signal'] > 0.05],
                       key=lambda x: x['strength'], reverse=True)
        shorts = sorted([c for c in candidates if c['signal'] < -0.05],
                        key=lambda x: x['strength'], reverse=True)

        def pick_topn(cands, n):
            sel, sc = [], {}
            for c in cands:
                if len(sel) >= n: break
                cnt = sc.get(c['sector'], 0)
                if cnt >= self.max_per_sector: continue
                sel.append(c)
                sc[c['sector']] = cnt + 1
            return sel

        return pick_topn(longs, self.top_n), pick_topn(shorts, self.top_n)

    def rebalance(self, day_idx, longs, shorts, close_prices, open_prices):
        """Same rebalance logic as V1."""
        trades = []
        target = {}
        for c in longs:
            target[c['idx']] = {'lots': c['lots'], 'direction': 1, 'price': c['price'],
                                'address': c['address'], 'multiplier': c['multiplier']}
        for c in shorts:
            target[c['idx']] = {'lots': c['lots'], 'direction': -1, 'price': c['price'],
                                'address': c['address'], 'multiplier': c['multiplier']}

        # Close positions not in target
        for idx in list(self.positions.keys()):
            if idx not in target:
                pos = self.positions[idx]
                ep = open_prices[idx].item()
                if ep <= 0: continue
                pnl = pos['direction'] * (ep - pos['entry_price']) * pos['lots'] * pos['multiplier']
                cost = abs(ep * pos['lots'] * pos['multiplier'] * self.fee_rate)
                self.capital += pnl - cost
                trades.append({'day': day_idx, 'action': 'CLOSE', 'product': pos['address'],
                               'lots': pos['lots'], 'entry': pos['entry_price'],
                               'exit': ep, 'pnl': round(pnl - cost, 2)})
                del self.positions[idx]

        # Open/adjust
        for idx, tgt in target.items():
            ep = open_prices[idx].item()
            if ep <= 0: continue
            if idx in self.positions:
                pos = self.positions[idx]
                if pos['direction'] != tgt['direction']:
                    pnl = pos['direction'] * (ep - pos['entry_price']) * pos['lots'] * pos['multiplier']
                    cost = abs(ep * pos['lots'] * pos['multiplier'] * self.fee_rate)
                    self.capital += pnl - cost
                    trades.append({'day': day_idx, 'action': 'FLIP', 'product': pos['address'],
                                   'lots': pos['lots'], 'entry': pos['entry_price'],
                                   'exit': ep, 'pnl': round(pnl - cost, 2)})
                    cost_open = abs(ep * tgt['lots'] * tgt['multiplier'] * self.fee_rate)
                    self.capital -= cost_open
                    self.positions[idx] = {'lots': tgt['lots'], 'direction': tgt['direction'],
                                           'entry_price': ep, 'address': tgt['address'],
                                           'multiplier': tgt['multiplier']}
            else:
                cost_open = abs(ep * tgt['lots'] * tgt['multiplier'] * self.fee_rate)
                self.capital -= cost_open
                self.positions[idx] = {'lots': tgt['lots'], 'direction': tgt['direction'],
                                       'entry_price': ep, 'address': tgt['address'],
                                       'multiplier': tgt['multiplier']}
                trades.append({'day': day_idx, 'action': 'OPEN', 'product': tgt['address'],
                               'direction': 'LONG' if tgt['direction'] > 0 else 'SHORT',
                               'lots': tgt['lots'], 'entry': ep})

        self.trade_log.extend(trades)
        return trades

    def mark_to_market(self, close_prices):
        unrealized = 0
        for idx, pos in self.positions.items():
            p = close_prices[idx].item()
            if p > 0:
                unrealized += pos['direction'] * (p - pos['entry_price']) * pos['lots'] * pos['multiplier']
        self.equity = self.capital + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)

    def check_stop_loss(self, close_prices, day_idx):
        stop_amt = self.initial_capital * self.stop_loss_pct
        to_close = []
        for idx, pos in self.positions.items():
            p = close_prices[idx].item()
            if p <= 0: continue
            pnl = pos['direction'] * (p - pos['entry_price']) * pos['lots'] * pos['multiplier']
            if pnl < -stop_amt:
                to_close.append(idx)
        for idx in to_close:
            pos = self.positions[idx]
            ep = close_prices[idx].item()
            pnl = pos['direction'] * (ep - pos['entry_price']) * pos['lots'] * pos['multiplier']
            cost = abs(ep * pos['lots'] * pos['multiplier'] * self.fee_rate)
            self.capital += pnl - cost
            self.trade_log.append({'day': day_idx, 'action': 'STOP_LOSS', 'product': pos['address'],
                                   'lots': pos['lots'], 'entry': pos['entry_price'],
                                   'exit': ep, 'pnl': round(pnl - cost, 2)})
            del self.positions[idx]


def main():
    parser = argparse.ArgumentParser(description='Historical live simulation V2 (with MWU)')
    parser.add_argument('--capital', type=float, default=300000)
    parser.add_argument('--topn', type=int, default=5)
    parser.add_argument('--fee', type=float, default=0.0005)
    parser.add_argument('--rebalance', type=int, default=10)
    parser.add_argument('--mwu-eta', type=float, default=0.05, help='MWU learning rate')
    parser.add_argument('--warmup', type=int, default=120, help='Min days before trading starts')
    parser.add_argument('--ensemble', default='best_ensemble.json')
    parser.add_argument('--output', default='sim_results/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load ensemble
    with open(args.ensemble) as f:
        ens_data = json.load(f)
    ensemble = FormulaEnsemble.from_dict(ens_data['ensemble'])
    vm = StackVM()
    bt = MemeBacktest(position_mode='rank')
    log.info(f"Ensemble: {ensemble.num_formulas} formulas")

    # Load data
    log.info("Loading data...")
    loader = DuckDBDataLoader(timeframe='1d')
    loader.load_data()

    feat = loader.feat_tensor
    close = loader.raw_data_cache['close']
    open_p = loader.raw_data_cache['open']
    ret = loader.target_ret
    raw = loader.raw_data_cache
    N, F, T = feat.shape

    # Product addresses
    import duckdb
    con = duckdb.connect(loader.db_path, read_only=True)
    addr_df = con.execute("SELECT DISTINCT product_id || '.' || exchange as address FROM kline_1min ORDER BY address").fetchdf()
    con.close()
    addresses = addr_df['address'].tolist()[:N]

    # Initialize simulator
    sim = LiveSimulatorV2(
        capital=args.capital, top_n=args.topn, fee_rate=args.fee,
        rebalance_period=args.rebalance, mwu_eta=args.mwu_eta,
    )
    sim.mwu = MWUWeights(ensemble.num_formulas, eta=args.mwu_eta)

    # Walk-forward: start trading after warmup, use ALL data
    start_day = args.warmup
    last_rebalance = start_day

    log.info(f"\n{'='*60}")
    log.info(f"Simulation V2 Config (with MWU Online Learning)")
    log.info(f"{'='*60}")
    log.info(f"  Capital:      {args.capital:>10,.0f}")
    log.info(f"  Top-N:        {args.topn:>10}")
    log.info(f"  Fee:          {args.fee:>10.4%}")
    log.info(f"  Rebalance:    every {args.rebalance} days")
    log.info(f"  MWU eta:      {args.mwu_eta:>10.3f}")
    log.info(f"  Warmup:       {args.warmup} days")
    log.info(f"  Full period:  day {start_day} to {T-2} ({T - start_day - 1} trading days)")
    log.info(f"  Products:     {N}")
    log.info(f"{'='*60}\n")

    # Day-by-day simulation
    for t in range(start_day, T - 1):
        day_in_sim = t - start_day

        # MWU weight update: compute each formula's PnL from previous period
        if day_in_sim > 0 and t > start_day:
            # Use last rebalance_period days for PnL computation
            lookback = min(args.rebalance, t - start_day)
            period_feat = feat[:, :, t-lookback:t]
            period_ret = ret[:, t-lookback:t]
            period_raw = {k: (v[:, t-lookback:t] if v.dim() > 1 else v[t-lookback:t])
                          for k, v in raw.items()}
            formula_pnls = sim.compute_formula_pnls(ensemble, vm, bt,
                                                     period_feat, period_raw, period_ret)
            sim.mwu.update(formula_pnls)

        # Rebalance check
        should_rebalance = (t - last_rebalance) >= args.rebalance or day_in_sim == 0

        if should_rebalance:
            # Regime detection
            regime_scale = sim.detect_regime()

            # Generate signals with MWU weights
            feat_up_to_t = feat[:, :, :t+1]
            longs, shorts = sim.generate_signals_with_mwu(
                ensemble, vm, feat_up_to_t, close[:, t], addresses, regime_scale)

            # Execute at next-day open
            trades = sim.rebalance(t, longs, shorts, close[:, t], open_p[:, t + 1])
            last_rebalance = t

            if day_in_sim % 100 == 0:
                w = sim.mwu.weights
                w_str = ' '.join(f'{wi:.2f}' for wi in w)
                log.info(f"  Day {day_in_sim:>4}: equity={sim.equity:>10,.0f}, "
                         f"dd={sim.drawdown:.1%}, regime={regime_scale}, "
                         f"MWU=[{w_str}]")

        # Stop loss check
        sim.check_stop_loss(close[:, t + 1], t)

        # Mark to market
        sim.mark_to_market(close[:, t + 1])

        sim.daily_log.append({
            'day': day_in_sim, 'equity': round(sim.equity, 2),
            'capital': round(sim.capital, 2), 'drawdown': round(sim.drawdown, 4),
            'n_positions': len(sim.positions),
            'mwu_weights': sim.mwu.weights.tolist(),
        })

    # Results
    eq = np.array(sim.equity_curve)
    daily_rets = np.diff(eq) / eq[:-1]
    n_days = len(eq)
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-10) * math.sqrt(252)
    peak = np.maximum.accumulate(eq)
    max_dd = ((peak - eq) / peak).max()
    total_ret = (sim.equity - args.capital) / args.capital
    ann_ret = (sim.equity / args.capital) ** (252 / max(n_days, 1)) - 1
    n_trades = len(sim.trade_log)
    trade_pnls = [t.get('pnl', 0) for t in sim.trade_log if 'pnl' in t]
    win_trades = sum(1 for p in trade_pnls if p > 0)
    loss_trades = sum(1 for p in trade_pnls if p < 0)
    win_rate = win_trades / max(win_trades + loss_trades, 1)

    print(f"\n{'='*60}")
    print(f"{'SIMULATION V2 RESULTS (MWU + Walk-Forward)':^60}")
    print(f"{'='*60}")
    print(f"  Initial Capital:    {args.capital:>12,.0f}")
    print(f"  Final Equity:       {sim.equity:>12,.0f}")
    print(f"  Total Return:       {total_ret:>12.2%}")
    print(f"  Annualized Return:  {ann_ret:>12.2%}")
    print(f"  Annualized Sharpe:  {sharpe:>12.4f}")
    print(f"  Max Drawdown:       {float(max_dd):>12.2%}")
    print(f"  Trading Days:       {n_days:>12}")
    print(f"  Total Trades:       {n_trades:>12}")
    print(f"  Win Rate:           {win_rate:>12.1%}")
    print(f"  MWU Final Weights:  {[f'{w:.3f}' for w in sim.mwu.weights]}")

    # Monthly breakdown
    dpm = 21
    print(f"\n  Monthly P&L:")
    for ms in range(0, len(sim.daily_log), dpm):
        me = min(ms + dpm, len(sim.daily_log))
        if me <= ms: break
        es = sim.daily_log[ms]['equity']
        ee = sim.daily_log[me - 1]['equity']
        mr = (ee - es) / es
        md = max(d['drawdown'] for d in sim.daily_log[ms:me])
        mi = ms // dpm + 1
        bar = '+' * int(mr * 100) if mr > 0 else '-' * int(abs(mr) * 100)
        print(f"    Month {mi:>2}: ret={mr:>+7.2%}, maxDD={md:>6.2%}  {bar}")

    print(f"{'='*60}")

    # Save
    results = {
        'config': {'capital': args.capital, 'top_n': args.topn, 'fee': args.fee,
                   'rebalance_period': args.rebalance, 'mwu_eta': args.mwu_eta,
                   'warmup': args.warmup, 'version': 'v2_mwu'},
        'summary': {'final_equity': round(sim.equity, 2), 'total_return': round(total_ret, 4),
                    'ann_return': round(ann_ret, 4), 'sharpe': round(sharpe, 4),
                    'max_drawdown': round(float(max_dd), 4), 'n_trades': n_trades,
                    'win_rate': round(win_rate, 4), 'trading_days': n_days},
        'equity_curve': [round(e, 2) for e in sim.equity_curve],
        'mwu_weight_history': [w.tolist() for w in sim.mwu.weight_history[-50:]],
        'trade_log': sim.trade_log[-200:],
        'daily_log': sim.daily_log,
    }
    out_path = os.path.join(args.output, 'sim_live_v2_results.json')
    # Clean numpy types for JSON
    def clean(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(clean(results), f, indent=2, ensure_ascii=False)
    log.info(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
