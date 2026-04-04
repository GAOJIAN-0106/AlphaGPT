"""
Historical replay backtest — exact reproduction of the daily pipeline.

For each trading day, loads data ONLY up to that date (no lookahead bias),
then runs the same signal generation + risk filter logic as the live pipeline.

Usage:
    python scripts/backtest_replay.py --start 2026-01-02 --end 2026-03-27
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MODEL_ASSET_CLASS', 'futures')
os.environ.setdefault('ENABLE_TERM_STRUCTURE', '1')

from model_core.ensemble import FormulaEnsemble
from model_core.online_learner import OnlineLearner
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.config import ModelConfig
from model_core.regime_detector import HMMRegimeDetector, get_returns_from_raw

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger('backtest_replay')

import duckdb

# ─── Sector map (same as risk_filter.py) ────────────────────────────
SECTOR_MAP = {
    'rb': '黑色', 'hc': '黑色', 'i': '黑色', 'j': '黑色', 'jm': '黑色',
    'SF': '黑色', 'SM': '黑色', 'ss': '黑色',
    'cu': '有色', 'al': '有色', 'zn': '有色', 'pb': '有色', 'ni': '有色',
    'sn': '有色', 'bc': '有色', 'ao': '有色', 'si': '有色',
    'au': '贵金属', 'ag': '贵金属',
    'sc': '能源', 'fu': '能源', 'bu': '能源', 'lu': '能源', 'pg': '能源',
    'TA': '化工', 'MA': '化工', 'eg': '化工', 'eb': '化工', 'pp': '化工',
    'l': '化工', 'v': '化工', 'SA': '化工', 'UR': '化工', 'FG': '化工',
    'PF': '化工', 'ru': '化工', 'sp': '化工', 'nr': '化工', 'SH': '化工',
    'br': '化工', 'PX': '化工',
    'c': '农产品', 'cs': '农产品', 'a': '农产品', 'm': '农产品', 'y': '农产品',
    'p': '农产品', 'OI': '农产品', 'RM': '农产品', 'CF': '农产品', 'SR': '农产品',
    'CJ': '农产品', 'AP': '农产品', 'jd': '农产品', 'lh': '农产品', 'PK': '农产品',
    'CY': '农产品', 'rr': '农产品', 'WH': '农产品', 'PM': '农产品', 'RI': '农产品',
    'RS': '农产品', 'JR': '农产品', 'LR': '农产品', 'b': '农产品',
}

ALIAS = {'ppF.DCE': 'pp.DCE', 'lF.DCE': 'l.DCE', 'vF.DCE': 'v.DCE'}


def get_sector(product):
    return SECTOR_MAP.get(product.split('.')[0], '其他')


def pick_topn_diversified(candidates, n, max_per_sector=2):
    selected = []
    sector_count = {}
    for c in candidates:
        sec = get_sector(c['product'])
        if sector_count.get(sec, 0) >= max_per_sector:
            continue
        selected.append(c)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(selected) >= n:
            break
    return selected


def apply_topn_filter(signals, capital=300000, topn=5):
    longs = sorted([s for s in signals if s['target_weight'] > 0.01],
                   key=lambda x: x['signal_strength'], reverse=True)
    shorts = sorted([s for s in signals if s['target_weight'] < -0.01],
                    key=lambda x: x['signal_strength'], reverse=True)
    return pick_topn_diversified(longs, topn) + pick_topn_diversified(shorts, topn)


class ReplayDataLoader:
    """Efficient loader for replay backtesting.

    Loads all raw data once, then for each day builds tensors using only
    data up to that date — no lookahead bias.
    """

    def __init__(self, db_path, product_ids):
        self.db_path = db_path
        self.product_ids = product_ids

        # Load all raw data once
        print("Loading raw data (one-time)...")
        from model_core.duckdb_loader import DuckDBDataLoader
        self._full_loader = DuckDBDataLoader(
            timeframe='1d', db_path=db_path, products=product_ids)
        self._full_loader.load_data()

        # Store the full tensors and dates
        self.full_feat = self._full_loader.feat_tensor       # [N, F, T]
        self.full_ret = self._full_loader.target_ret          # [N, T]
        self.full_raw = self._full_loader.raw_data_cache
        self.products = self._get_products()

        # Date list
        self.dates = []
        for d in self._full_loader.dates:
            if hasattr(d, 'strftime'):
                self.dates.append(d.strftime('%Y-%m-%d'))
            else:
                self.dates.append(str(d)[:10])

        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        N, F, T = self.full_feat.shape
        print(f"Full data: {N} products × {F} features × {T} days")
        print(f"Date range: {self.dates[0]} → {self.dates[-1]}")

    def _get_products(self):
        plist = ", ".join(f"'{p}'" for p in self.product_ids)
        con = duckdb.connect(self.db_path, read_only=True)
        pdf = con.execute(f"""
            SELECT DISTINCT product_id || '.' || exchange as address
            FROM kline_1min WHERE product_id IN ({plist})
            ORDER BY address
        """).fetchdf()
        con.close()
        N = self.full_feat.shape[0]
        return pdf['address'].tolist()[:N]

    def get_data_up_to(self, date_str):
        """Return feat, ret, raw sliced up to and including date_str.

        This is the key function that prevents lookahead bias:
        features are computed from the full time series but we only
        return data up to the given date.

        Because features like rolling means/stds are computed causally
        (they only look backward), slicing the time dimension is equivalent
        to recomputing from scratch with data up to that date.
        """
        idx = self.date_to_idx.get(date_str)
        if idx is None:
            return None, None, None, None
        t_end = idx + 1  # inclusive

        feat = self.full_feat[:, :, :t_end]
        ret = self.full_ret[:, :t_end]
        raw = {}
        for k, v in self.full_raw.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 2 and v.shape[1] >= t_end:
                    raw[k] = v[:, :t_end]
                elif v.dim() == 1 and v.shape[0] >= t_end:
                    raw[k] = v[:t_end]
                else:
                    raw[k] = v
            else:
                raw[k] = v

        return feat, ret, raw, self.products


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2026-01-02')
    parser.add_argument('--end', default='2026-03-27')
    parser.add_argument('--ensemble', default='best_ensemble.json')
    parser.add_argument('--capital', type=float, default=300000)
    parser.add_argument('--topn', type=int, default=5)
    parser.add_argument('--output', default='backtest_results.json')
    parser.add_argument('--regime-mode', default='hmm_topn',
                        choices=['baseline', 'legacy', 'hmm_scale', 'hmm_topn',
                                 'hmm_weighted', 'hmm_factor_mix', 'hmm_topn_scale'],
                        help='Regime strategy: baseline/legacy/hmm_scale/hmm_topn/'
                             'hmm_weighted/hmm_factor_mix/hmm_topn_scale')
    args = parser.parse_args()

    db_path = os.environ.get('DUCKDB_PATH',
                             os.path.expanduser('~/quant/tick_data/kline_1min.duckdb'))

    # Load ensemble
    with open(args.ensemble) as f:
        ens_data = json.load(f)
    ensemble = FormulaEnsemble.from_dict(ens_data['ensemble'])
    print(f"Ensemble: {ensemble.num_formulas} formulas")

    # Get live product list
    live_sig_file = os.path.join(os.path.dirname(__file__), '..', 'signals', 'signals_2026-03-25.json')
    with open(live_sig_file) as f:
        live_data = json.load(f)
    live_pids = sorted(set(s['product'].split('.')[0] for s in live_data['signals']))
    print(f"Products: {len(live_pids)}")

    # Initialize replay loader (one-time full load)
    loader = ReplayDataLoader(db_path, live_pids)

    # Get trading days in range
    all_dates = loader.dates
    trading_days = [d for d in all_dates if args.start <= d <= args.end]
    print(f"Trading days: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")

    # Initialize MWU
    online_learner = OnlineLearner(ensemble, strategy='mwu', eta=0.05,
                                   lookback_window=10, min_weight=0.02)

    # Initialize regime detector
    regime_detector = None
    use_hmm = args.regime_mode.startswith('hmm')
    if use_hmm:
        regime_detector = HMMRegimeDetector(n_states=3, vol_window=20, refit_interval=20)
    print(f"Regime mode: {args.regime_mode}")

    # Load close prices for PnL (next-day return)
    plist = ", ".join(f"'{p}'" for p in live_pids)
    con = duckdb.connect(db_path, read_only=True)
    price_df = con.execute(f"""
        WITH daily_oi AS (
            SELECT product_id, exchange, symbol,
                   DATE_TRUNC('day', datetime) as day,
                   AVG(close_oi) as avg_oi
            FROM kline_1min
            WHERE product_id IN ({plist})
              AND datetime >= '{args.start}'::DATE - INTERVAL '5 days'
              AND datetime <= '{args.end}'::DATE + INTERVAL '2 days'
            GROUP BY product_id, exchange, symbol, day
        ),
        main_contracts AS (
            SELECT day, product_id, exchange, symbol,
                   ROW_NUMBER() OVER (PARTITION BY product_id, day ORDER BY avg_oi DESC) as rn
            FROM daily_oi
        )
        SELECT mc.product_id || '.' || mc.exchange as address, mc.day as time,
               LAST(k.close ORDER BY k.datetime) as close
        FROM main_contracts mc
        JOIN kline_1min k ON k.symbol = mc.symbol AND DATE_TRUNC('day', k.datetime) = mc.day
        WHERE mc.rn = 1
        GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchdf()
    con.close()

    price_df['time'] = pd.to_datetime(price_df['time'])
    price_df = price_df.sort_values(['address', 'time'])
    ret_data = {}
    for addr, grp in price_df.groupby('address'):
        grp = grp.sort_values('time')
        for i in range(len(grp) - 1):
            d = grp.iloc[i]['time'].strftime('%Y-%m-%d')
            ret_data[(addr, d)] = grp.iloc[i + 1]['close'] / grp.iloc[i]['close'] - 1

    # ─── Main replay loop ───────────────────────────────────────────
    vm = StackVM()
    bt = MemeBacktest(position_mode='rank')

    results = []
    cumul = 0

    print(f"\n{'Date':>12} {'PnL(bps)':>10} {'Cumul':>10} {'Rgm':>5} {'Label':>10} {'Positions'}")
    print('─' * 85)

    for day_str in trading_days:
        feat, ret, raw, products = loader.get_data_up_to(day_str)
        if feat is None:
            continue

        N, F, T_now = feat.shape
        if T_now < 30:
            continue

        # 1. Update MWU weights (last 10 days)
        if T_now > 15:
            lookback = min(10, T_now - 5)
            p_feat = feat[:, :, -lookback:]
            p_ret = ret[:, -lookback:]
            p_raw = {k: (v[:, -lookback:] if isinstance(v, torch.Tensor) and v.dim() == 2 else
                         (v[-lookback:] if isinstance(v, torch.Tensor) and v.dim() == 1 else v))
                     for k, v in raw.items()}
            try:
                online_learner.update(p_feat, p_raw, p_ret)
            except Exception:
                pass

        # 2. Per-formula signals (latest day only)
        formula_signals = []
        for formula in ensemble.formulas:
            res = vm.execute(formula, feat[:, :, -1:])
            if res is not None and res.std() > 1e-8:
                # Flatten to [N]: take last time step if extra dims exist
                while res.dim() > 1:
                    res = res[:, -1]  # Take last column (not squeeze, which fails on size>1)
                if res.shape[0] != N:
                    res = res[:N]
                formula_signals.append(res)
            else:
                formula_signals.append(torch.zeros(N))

        # 3. Regime detection
        regime_scale = 1.0
        regime_label = 'none'

        if args.regime_mode == 'legacy':
            recent_pnls = []
            for t in range(max(0, T_now - 21), T_now - 1):
                sig = ensemble.predict(feat[:, :, t:t + 1])
                if sig is not None:
                    is_safe = torch.ones(N, 1, device=sig.device)
                    pos = bt.compute_position(sig, is_safe)
                    if t + 1 < T_now:
                        pnl = (pos * ret[:, t + 1:t + 2]).mean().item()
                        recent_pnls.append(pnl)
            if len(recent_pnls) >= 20:
                recent = np.array(recent_pnls[-20:])
                rolling_sharpe = recent.mean() / (recent.std() + 1e-10) * np.sqrt(252)
                regime_scale = 0.5 if rolling_sharpe < -0.5 else 1.0
            regime_label = 'normal' if regime_scale == 1.0 else 'adverse'
        elif use_hmm:
            returns = get_returns_from_raw(raw)
            regime_scale, _info, regime_label = regime_detector.update(returns)

        # 4. Formula weights (possibly adjusted by hmm_factor_mix)
        weights = online_learner.get_weights().copy()

        if args.regime_mode == 'hmm_factor_mix' and regime_label == 'crisis':
            n_f = len(formula_signals)
            formula_vols = np.zeros(n_f)
            lkb = min(20, T_now - 5)
            if lkb > 5:
                for i, formula in enumerate(ensemble.formulas):
                    res = vm.execute(formula, feat[:, :, -lkb:])
                    if res is not None:
                        formula_vols[i] = float(res.mean(dim=0).std())
                    else:
                        formula_vols[i] = 1.0
                if formula_vols.max() > 1e-10:
                    inv_vol = 1.0 / (formula_vols + 1e-10)
                    inv_vol_w = inv_vol / inv_vol.sum()
                    weights = 0.6 * inv_vol_w + 0.4 * weights
                    weights = weights / weights.sum()

        # 5. Ensemble signal
        stacked = torch.stack(formula_signals, dim=0)
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=stacked.device)
        ensemble_signal = (stacked * w_tensor.unsqueeze(-1)).sum(dim=0)

        # Cross-sectional rank → [-1, 1]
        ranks = ensemble_signal.argsort().argsort().float()
        rank_norm = (ranks / (N - 1) - 0.5) * 2

        # 6. Build signals
        signals = []
        for i in range(N):
            raw_w = rank_norm[i].item()
            signals.append({
                'product': products[i] if i < len(products) else f'p_{i}',
                'target_weight': round(raw_w, 6),
                'raw_weight': round(raw_w, 6),
                'signal_strength': round(abs(raw_w), 6),
                'direction': 'LONG' if raw_w > 0.01 else ('SHORT' if raw_w < -0.01 else 'FLAT'),
            })

        # 7. TopN — adjust by regime for hmm_topn / hmm_topn_scale
        if args.regime_mode in ('hmm_topn', 'hmm_topn_scale'):
            if regime_label == 'crisis':
                effective_topn = max(2, int(args.topn * regime_scale))
            elif regime_label == 'calm':
                effective_topn = args.topn + 2
            else:
                effective_topn = args.topn
        else:
            effective_topn = args.topn

        selected = apply_topn_filter(signals, capital=args.capital, topn=effective_topn)

        # 8. Compute next-day PnL
        n_pos = len(selected)
        day_pnl = 0
        matched = 0

        if args.regime_mode == 'hmm_weighted':
            # Signal-strength weighted + regime scale
            total_str = sum(abs(p['target_weight']) for p in selected)
            if total_str < 1e-10:
                total_str = 1.0
            for pos in selected:
                prod = ALIAS.get(pos['product'], pos['product'])
                r = ret_data.get((prod, day_str))
                if r is not None and np.isfinite(r):
                    sign = 1 if pos['direction'] == 'LONG' else -1
                    pw = abs(pos['target_weight']) / total_str * regime_scale
                    day_pnl += sign * r * pw
                    matched += 1
        elif args.regime_mode in ('hmm_scale', 'hmm_topn_scale', 'legacy'):
            # Equal weight × regime_scale
            for pos in selected:
                prod = ALIAS.get(pos['product'], pos['product'])
                r = ret_data.get((prod, day_str))
                if r is not None and np.isfinite(r):
                    sign = 1 if pos['direction'] == 'LONG' else -1
                    day_pnl += sign * r * regime_scale / max(n_pos, 1)
                    matched += 1
        else:
            # baseline / hmm_topn / hmm_factor_mix: equal weight, no scale multiplier
            for pos in selected:
                prod = ALIAS.get(pos['product'], pos['product'])
                r = ret_data.get((prod, day_str))
                if r is not None and np.isfinite(r):
                    sign = 1 if pos['direction'] == 'LONG' else -1
                    day_pnl += sign * r / max(n_pos, 1)
                    matched += 1

        day_bps = day_pnl * 10000
        cumul += day_bps

        longs = [p['product'] for p in selected if p['direction'] == 'LONG']
        shorts = [p['product'] for p in selected if p['direction'] == 'SHORT']
        pos_str = f"L:{','.join(longs[:3])} S:{','.join(shorts[:3])}"

        print(f"{day_str:>12} {day_bps:>+10.2f} {cumul:>+10.2f} {regime_scale:>5.2f} {regime_label:>10} {pos_str}")

        results.append({
            'date': day_str,
            'pnl_bps': round(day_bps, 4),
            'cumul_bps': round(cumul, 4),
            'regime_scale': regime_scale,
            'regime_label': regime_label,
            'n_positions': n_pos,
            'matched': matched,
            'mwu_weights': weights.tolist() if isinstance(weights, np.ndarray) else list(weights),
            'longs': longs,
            'shorts': shorts,
        })

    # ─── Summary ────────────────────────────────────────────────────
    pnl_arr = np.array([r['pnl_bps'] for r in results])
    valid = pnl_arr[pnl_arr != 0]
    n_v = len(valid)

    print(f"\n{'='*70}")
    print(f"  回测汇总 ({args.start} → {args.end}, {len(results)} 交易日, {n_v} 有效)")
    print(f"{'='*70}")

    if n_v > 1:
        avg = valid.mean()
        std = valid.std()
        sharpe = avg / (std + 1e-9) * np.sqrt(252)
        win = (valid > 0).sum()
        cumul_arr = np.cumsum(pnl_arr)
        max_dd = (cumul_arr - np.maximum.accumulate(cumul_arr)).min()

        print(f"  累计收益:    {pnl_arr.sum():+.2f} bps ({pnl_arr.sum()/10000*100:+.4f}%)")
        print(f"  日均收益:    {avg:+.2f} bps")
        print(f"  日均波动:    {std:.2f} bps")
        print(f"  年化Sharpe:  {sharpe:.2f}")
        print(f"  胜率:        {win}/{n_v} ({win/n_v*100:.0f}%)")
        print(f"  最大回撤:    {max_dd:.2f} bps")
        print(f"  最好一天:    {valid.max():+.2f} bps")
        print(f"  最差一天:    {valid.min():+.2f} bps")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == '__main__':
    main()
