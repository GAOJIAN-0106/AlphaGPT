"""
Historical live-simulation backtest for AlphaGPT.

Replays history day-by-day with the EXACT same pipeline as live trading:
  1. Generate signals from ensemble (using only data available up to that day)
  2. Apply risk filter (Top-N small capital mode)
  3. Execute at next-day open price
  4. Track P&L, drawdown, margin usage, trade log

This is the most realistic backtest — no look-ahead bias, realistic costs.

Usage:
    python scripts/simulate_live.py --capital 300000 --topn 3 --fee 0.0003
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
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('MODEL_ASSET_CLASS', 'futures')
os.environ.setdefault('ENABLE_TERM_STRUCTURE', '1')

from model_core.ensemble import FormulaEnsemble
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.duckdb_loader import DuckDBDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('sim_live')


# ---------------------------------------------------------------------------
# Margin and contract specs
# ---------------------------------------------------------------------------
CONTRACT_SPECS = {
    # product_id: (multiplier, margin_rate, tick_size)
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
    'rb': '黑色', 'hc': '黑色', 'i': '黑色', 'j': '黑色', 'jm': '黑色',
    'SF': '黑色', 'SM': '黑色', 'ss': '黑色',
    'cu': '有色', 'al': '有色', 'zn': '有色', 'pb': '有色', 'ni': '有色',
    'sn': '有色', 'bc': '有色', 'ao': '有色', 'si': '有色',
    'au': '贵金属', 'ag': '贵金属',
    'sc': '能源', 'fu': '能源', 'bu': '能源', 'lu': '能源', 'pg': '能源',
    'TA': '化工', 'MA': '化工', 'eg': '化工', 'eb': '化工', 'PP': '化工',
    'L': '化工', 'V': '化工', 'SA': '化工', 'ru': '化工', 'FG': '化工',
    'PF': '化工', 'sp': '化工', 'UR': '化工', 'nr': '化工',
    'c': '农产品', 'cs': '农产品', 'a': '农产品', 'm': '农产品',
    'y': '农产品', 'p': '农产品', 'OI': '农产品', 'RM': '农产品',
    'CF': '农产品', 'SR': '农产品', 'AP': '农产品', 'CJ': '农产品',
    'jd': '农产品', 'lh': '农产品', 'PK': '农产品', 'b': '农产品',
    'rr': '农产品',
}


def get_spec(product_id):
    return CONTRACT_SPECS.get(product_id, (10, 0.12, 1))


def get_sector(address):
    return SECTOR_MAP.get(address.split('.')[0], '其他')


# ---------------------------------------------------------------------------
# Portfolio simulator
# ---------------------------------------------------------------------------
class LiveSimulator:
    def __init__(self, capital, top_n, fee_rate, max_margin_pct=0.60,
                 max_per_position_pct=0.20, max_per_sector=2, stop_loss_pct=0.03,
                 rebalance_period=20):
        self.initial_capital = capital
        self.capital = capital
        self.equity = capital
        self.peak_equity = capital
        self.top_n = top_n
        self.fee_rate = fee_rate
        self.max_margin_pct = max_margin_pct
        self.max_per_position_pct = max_per_position_pct
        self.max_per_sector = max_per_sector
        self.stop_loss_pct = stop_loss_pct
        self.rebalance_period = rebalance_period

        # Current holdings: {product_idx: {'lots': int, 'direction': 1/-1, 'entry_price': float}}
        self.positions = {}

        # Tracking
        self.equity_curve = []
        self.trade_log = []
        self.daily_log = []

    @property
    def drawdown(self):
        if self.peak_equity <= 0:
            return 0
        return (self.peak_equity - self.equity) / self.peak_equity

    def generate_topn_signals(self, ensemble, vm, feat_up_to_t, close_prices_t,
                               product_addresses):
        """Generate Top-N signals using data up to day t (no look-ahead)."""
        N = feat_up_to_t.shape[0]

        # Ensemble signal on latest features
        formula_signals = []
        for formula in ensemble.formulas:
            res = vm.execute(formula, feat_up_to_t[:, :, -1:])
            if res is not None and res.std() > 1e-8:
                formula_signals.append(res.squeeze(-1))
            else:
                formula_signals.append(torch.zeros(N, device=feat_up_to_t.device))

        stacked = torch.stack(formula_signals, dim=0)
        weights = torch.ones(len(formula_signals), device=stacked.device) / len(formula_signals)
        signal = (stacked * weights.unsqueeze(-1)).sum(dim=0)  # [N]

        # Cross-sectional rank
        ranks = signal.argsort().argsort().float()
        rank_norm = (ranks / max(N - 1, 1) - 0.5) * 2  # [-1, 1]

        # Build candidate list with margin check
        max_margin = self.capital * self.max_per_position_pct
        total_margin_budget = self.capital * self.max_margin_pct
        per_pos_budget = total_margin_budget / (2 * self.top_n) if self.top_n > 0 else 0

        candidates = []
        for i in range(N):
            addr = product_addresses[i]
            pid = addr.split('.')[0]
            mult, margin_rate, _ = get_spec(pid)
            price = close_prices_t[i].item()
            if price <= 0:
                continue
            margin_per_lot = price * mult * margin_rate
            if margin_per_lot > max_margin or margin_per_lot <= 0:
                continue

            max_lots = max(int(per_pos_budget / margin_per_lot), 1)
            actual_margin = max_lots * margin_per_lot

            candidates.append({
                'idx': i,
                'address': addr,
                'sector': get_sector(addr),
                'signal': rank_norm[i].item(),
                'strength': abs(rank_norm[i].item()),
                'price': price,
                'multiplier': mult,
                'margin_per_lot': margin_per_lot,
                'lots': max_lots,
                'margin': actual_margin,
            })

        # Sort by strength
        longs = sorted([c for c in candidates if c['signal'] > 0.05],
                       key=lambda x: x['strength'], reverse=True)
        shorts = sorted([c for c in candidates if c['signal'] < -0.05],
                        key=lambda x: x['strength'], reverse=True)

        # Pick Top-N with sector diversification
        def pick_topn(cands, n):
            selected = []
            sector_count = {}
            for c in cands:
                if len(selected) >= n:
                    break
                sc = sector_count.get(c['sector'], 0)
                if sc >= self.max_per_sector:
                    continue
                selected.append(c)
                sector_count[c['sector']] = sc + 1
            return selected

        return pick_topn(longs, self.top_n), pick_topn(shorts, self.top_n)

    def rebalance(self, day_idx, longs, shorts, close_prices, open_prices):
        """Execute rebalance: close old positions, open new ones."""
        trades_today = []

        # Target positions
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
                exit_price = open_prices[idx].item()
                if exit_price <= 0:
                    continue
                pnl = pos['direction'] * (exit_price - pos['entry_price']) * \
                      pos['lots'] * pos['multiplier']
                cost = abs(exit_price * pos['lots'] * pos['multiplier'] * self.fee_rate)
                net_pnl = pnl - cost
                self.capital += net_pnl
                trades_today.append({
                    'day': day_idx,
                    'action': 'CLOSE',
                    'product': pos['address'],
                    'lots': pos['lots'],
                    'entry': pos['entry_price'],
                    'exit': exit_price,
                    'pnl': round(net_pnl, 2),
                })
                del self.positions[idx]

        # Open/adjust positions in target
        for idx, tgt in target.items():
            entry_price = open_prices[idx].item()
            if entry_price <= 0:
                continue

            if idx in self.positions:
                # Already holding — check if direction changed
                pos = self.positions[idx]
                if pos['direction'] != tgt['direction']:
                    # Close and reopen
                    pnl = pos['direction'] * (entry_price - pos['entry_price']) * \
                          pos['lots'] * pos['multiplier']
                    cost = abs(entry_price * pos['lots'] * pos['multiplier'] * self.fee_rate)
                    self.capital += pnl - cost
                    trades_today.append({
                        'day': day_idx, 'action': 'FLIP',
                        'product': pos['address'], 'lots': pos['lots'],
                        'entry': pos['entry_price'], 'exit': entry_price,
                        'pnl': round(pnl - cost, 2),
                    })
                    # Open new direction
                    cost_open = abs(entry_price * tgt['lots'] * tgt['multiplier'] * self.fee_rate)
                    self.capital -= cost_open
                    self.positions[idx] = {
                        'lots': tgt['lots'], 'direction': tgt['direction'],
                        'entry_price': entry_price, 'address': tgt['address'],
                        'multiplier': tgt['multiplier'],
                    }
                # Same direction, maybe different lots — keep simple, no adjustment
            else:
                # New position
                cost_open = abs(entry_price * tgt['lots'] * tgt['multiplier'] * self.fee_rate)
                self.capital -= cost_open
                self.positions[idx] = {
                    'lots': tgt['lots'], 'direction': tgt['direction'],
                    'entry_price': entry_price, 'address': tgt['address'],
                    'multiplier': tgt['multiplier'],
                }
                trades_today.append({
                    'day': day_idx, 'action': 'OPEN',
                    'product': tgt['address'],
                    'direction': 'LONG' if tgt['direction'] > 0 else 'SHORT',
                    'lots': tgt['lots'], 'entry': entry_price,
                })

        self.trade_log.extend(trades_today)
        return trades_today

    def mark_to_market(self, close_prices):
        """Compute equity = capital + unrealized P&L."""
        unrealized = 0
        for idx, pos in self.positions.items():
            price = close_prices[idx].item()
            if price > 0:
                unrealized += pos['direction'] * (price - pos['entry_price']) * \
                             pos['lots'] * pos['multiplier']
        self.equity = self.capital + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)

    def check_stop_loss(self, close_prices, open_prices, day_idx):
        """Check per-position stop loss."""
        stop_amount = self.initial_capital * self.stop_loss_pct
        to_close = []
        for idx, pos in self.positions.items():
            price = close_prices[idx].item()
            if price <= 0:
                continue
            pnl = pos['direction'] * (price - pos['entry_price']) * \
                  pos['lots'] * pos['multiplier']
            if pnl < -stop_amount:
                to_close.append(idx)

        for idx in to_close:
            pos = self.positions[idx]
            exit_price = close_prices[idx].item()
            pnl = pos['direction'] * (exit_price - pos['entry_price']) * \
                  pos['lots'] * pos['multiplier']
            cost = abs(exit_price * pos['lots'] * pos['multiplier'] * self.fee_rate)
            self.capital += pnl - cost
            self.trade_log.append({
                'day': day_idx, 'action': 'STOP_LOSS',
                'product': pos['address'], 'lots': pos['lots'],
                'entry': pos['entry_price'], 'exit': exit_price,
                'pnl': round(pnl - cost, 2),
            })
            del self.positions[idx]


def main():
    parser = argparse.ArgumentParser(description='Historical live simulation')
    parser.add_argument('--capital', type=float, default=300000)
    parser.add_argument('--topn', type=int, default=3)
    parser.add_argument('--fee', type=float, default=0.0003, help='Fee rate per trade')
    parser.add_argument('--rebalance', type=int, default=20, help='Rebalance period (days)')
    parser.add_argument('--ensemble', default='best_ensemble.json')
    parser.add_argument('--output', default='sim_results/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load ensemble
    with open(args.ensemble) as f:
        ens_data = json.load(f)
    ensemble = FormulaEnsemble.from_dict(ens_data['ensemble'])
    vm = StackVM()
    log.info(f"Loaded ensemble: {ensemble.num_formulas} formulas")

    # Load data
    log.info("Loading data...")
    loader = DuckDBDataLoader(timeframe='1d')
    loader.load_data()

    feat = loader.feat_tensor       # [N, F, T]
    close = loader.raw_data_cache['close']   # [N, T]
    open_p = loader.raw_data_cache['open']   # [N, T]
    N, F, T = feat.shape

    # Get product addresses
    import duckdb
    con = duckdb.connect(loader.db_path, read_only=True)
    products_df = con.execute("""
        SELECT DISTINCT product_id || '.' || exchange as address
        FROM kline_1min ORDER BY address
    """).fetchdf()
    con.close()
    addresses = products_df['address'].tolist()[:N]

    # Simulation parameters
    # Use test period only (last 30% of data)
    split_idx = int(T * 0.7)
    warmup = 60  # need 60 days of history for factors

    sim = LiveSimulator(
        capital=args.capital, top_n=args.topn, fee_rate=args.fee,
        rebalance_period=args.rebalance,
    )

    log.info(f"\nSimulation Config:")
    log.info(f"  Capital:    {args.capital:>12,.0f}")
    log.info(f"  Top-N:      {args.topn:>12}")
    log.info(f"  Fee:        {args.fee:>12.4%}")
    log.info(f"  Rebalance:  every {args.rebalance} days")
    log.info(f"  Period:     day {split_idx} to {T-1} ({T - split_idx} days)")
    log.info(f"  Products:   {N}")

    # Day-by-day simulation
    log.info(f"\nRunning simulation...")
    last_rebalance = split_idx

    for t in range(split_idx, T - 1):
        day_in_sim = t - split_idx

        # Rebalance check
        should_rebalance = (t - last_rebalance) >= args.rebalance or day_in_sim == 0

        if should_rebalance:
            # Generate signals using data up to day t (inclusive)
            feat_up_to_t = feat[:, :, :t+1]
            close_t = close[:, t]

            longs, shorts = sim.generate_topn_signals(
                ensemble, vm, feat_up_to_t, close_t, addresses)

            # Execute at next day's open
            open_next = open_p[:, t + 1]
            trades = sim.rebalance(t, longs, shorts, close[:, t], open_next)
            last_rebalance = t

            if day_in_sim % 50 == 0:
                n_trades = len(trades)
                log.info(f"  Day {day_in_sim:>4}: rebalance, {n_trades} trades, "
                         f"equity={sim.equity:>12,.0f}, dd={sim.drawdown:.2%}")

        # Check stop loss
        sim.check_stop_loss(close[:, t + 1], open_p[:, t + 1], t)

        # Mark to market at close
        sim.mark_to_market(close[:, t + 1])

        # Daily log
        sim.daily_log.append({
            'day': day_in_sim,
            'equity': round(sim.equity, 2),
            'capital': round(sim.capital, 2),
            'drawdown': round(sim.drawdown, 4),
            'n_positions': len(sim.positions),
        })

    # Results
    equity_arr = np.array(sim.equity_curve)
    daily_rets = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-10) * math.sqrt(252)
    max_dd = sim.drawdown
    total_ret = (sim.equity - args.capital) / args.capital
    n_trades = len(sim.trade_log)
    win_trades = len([t for t in sim.trade_log if t.get('pnl', 0) > 0])
    loss_trades = len([t for t in sim.trade_log if t.get('pnl', 0) < 0])

    # Find max drawdown from equity curve
    peak = np.maximum.accumulate(equity_arr)
    dd_series = (peak - equity_arr) / peak
    max_dd_val = dd_series.max()

    print(f"\n{'='*60}")
    print(f"{'LIVE SIMULATION RESULTS':^60}")
    print(f"{'='*60}")
    print(f"  Initial Capital:    {args.capital:>12,.0f}")
    print(f"  Final Equity:       {sim.equity:>12,.0f}")
    print(f"  Total Return:       {total_ret:>12.2%}")
    print(f"  Annualized Sharpe:  {sharpe:>12.4f}")
    print(f"  Max Drawdown:       {max_dd_val:>12.2%}")
    print(f"  Trading Days:       {len(sim.equity_curve):>12}")
    print(f"  Total Trades:       {n_trades:>12}")
    print(f"  Win/Loss:           {win_trades}/{loss_trades}")
    win_rate = win_trades / max(win_trades + loss_trades, 1)
    print(f"  Win Rate:           {win_rate:>12.1%}")
    print(f"  Rebalance Period:   every {args.rebalance} days")

    # Monthly breakdown
    print(f"\n  Monthly P&L:")
    days_per_month = 21
    for m_start in range(0, len(sim.daily_log), days_per_month):
        m_end = min(m_start + days_per_month, len(sim.daily_log))
        if m_end <= m_start:
            break
        eq_start = sim.daily_log[m_start]['equity']
        eq_end = sim.daily_log[m_end - 1]['equity']
        m_ret = (eq_end - eq_start) / eq_start
        m_dd = max(d['drawdown'] for d in sim.daily_log[m_start:m_end])
        month_idx = m_start // days_per_month + 1
        bar = '+' * int(m_ret * 100) if m_ret > 0 else '-' * int(abs(m_ret) * 100)
        print(f"    Month {month_idx:>2}: ret={m_ret:>+7.2%}, maxDD={m_dd:>6.2%}  {bar}")

    print(f"{'='*60}")

    # Save results
    results = {
        'config': {
            'capital': args.capital, 'top_n': args.topn,
            'fee': args.fee, 'rebalance_period': args.rebalance,
        },
        'summary': {
            'final_equity': round(sim.equity, 2),
            'total_return': round(total_ret, 4),
            'sharpe': round(sharpe, 4),
            'max_drawdown': round(float(max_dd_val), 4),
            'n_trades': n_trades,
            'win_rate': round(win_rate, 4),
        },
        'equity_curve': [round(e, 2) for e in sim.equity_curve],
        'trade_log': sim.trade_log[-100:],  # last 100 trades
        'daily_log': sim.daily_log,
    }
    out_path = os.path.join(args.output, 'sim_live_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
