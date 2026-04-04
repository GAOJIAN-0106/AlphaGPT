"""
Walk-Forward Replay with Event-Driven Updates.

Updates only when ensemble performance decays, not on a fixed schedule:
  - Monitor: 60-day rolling Sharpe, checked daily
  - Trigger: Sharpe < 0 for 10 consecutive days
  - Level 1 (Sharpe 0 ~ -0.5):  MWU only (automatic, no action needed)
  - Level 2 (Sharpe -0.5 ~ -1.5): swap 1 factor + replace 1 worst formula
  - Level 3 (Sharpe < -1.5):      swap 2 factors + replace all Sharpe < 0 formulas
  - Factor count stays fixed at 20

Usage:
    python scripts/wfr_quarterly.py --start 2025-04-01 --end 2026-03-27
"""

import os
import sys
import gc
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
from model_core.duckdb_loader import DuckDBDataLoader
from model_core.features_v2 import FEATURES_V3_FULL, FEATURES_V3_LIST

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('wfr_event')

import duckdb

# ─── Constants ──────────────────────────────────────────────────────
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
FIXED_FACTOR_COUNT = 20


def get_sector(product):
    return SECTOR_MAP.get(product.split('.')[0], '其他')


def pick_topn(signals, n=5, max_per_sector=2):
    def sel(cands):
        picked, sc = [], {}
        for c in cands:
            s = get_sector(c['product'])
            if sc.get(s, 0) >= max_per_sector:
                continue
            picked.append(c)
            sc[s] = sc.get(s, 0) + 1
            if len(picked) >= n:
                break
        return picked
    longs = sorted([s for s in signals if s['tw'] > 0.01], key=lambda x: -abs(x['tw']))
    shorts = sorted([s for s in signals if s['tw'] < -0.01], key=lambda x: -abs(x['tw']))
    return sel(longs) + sel(shorts)


# ═══════════════════════════════════════════════════════════════════════
# IC Screening + Factor Update
# ═══════════════════════════════════════════════════════════════════════

def compute_all_ic(feat_tensor, ret_tensor, feature_names, lookback_months=6):
    """Compute IC and ICIR for all features."""
    feat = feat_tensor.cpu()
    ret = ret_tensor.cpu()
    N, F, T = feat.shape
    t_start = max(0, T - lookback_months * 22)

    results = {}
    for f_idx in range(min(F, len(feature_names))):
        daily_ics = []
        for t in range(t_start, T - 1):
            x = feat[:, f_idx, t].numpy()
            y = ret[:, t].numpy()
            mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-10)
            if mask.sum() > 10:
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
                if np.isfinite(corr):
                    daily_ics.append(corr)
        if len(daily_ics) >= 20:
            mean_ic = np.mean(daily_ics)
            icir = abs(mean_ic) / (np.std(daily_ics) + 1e-10)
            results[feature_names[f_idx]] = (f_idx, mean_ic, icir)
    return results


def swap_factors(current_names, all_ic, n_swap):
    """Swap n_swap factors: remove worst, add best new. Total stays at 20."""
    if n_swap == 0:
        return current_names, [], []

    # Current factors ranked by ICIR (worst first)
    current_icirs = [(n, all_ic[n][2] if n in all_ic else 0.0) for n in current_names]
    current_icirs.sort(key=lambda x: x[1])

    # New candidates (not in current, best first)
    candidates = [(n, all_ic[n][2]) for n in all_ic if n not in current_names]
    candidates.sort(key=lambda x: -x[1])

    n_actual = min(n_swap, len(candidates))
    remove = [current_icirs[i][0] for i in range(n_actual)]
    add = [candidates[i][0] for i in range(n_actual)]

    changes = []
    for rm, ad in zip(remove, add):
        changes.append(f"  SWAP: -{rm}(ICIR={dict(current_icirs)[rm]:.3f}) "
                       f"+{ad}(ICIR={dict(candidates)[ad]:.3f})")

    new_names = [n for n in current_names if n not in remove] + add
    new_indices = []
    for n in new_names:
        if n in all_ic:
            new_indices.append(all_ic[n][0])
        elif n in FEATURES_V3_FULL:
            new_indices.append(FEATURES_V3_FULL.index(n))

    return new_names, new_indices, changes


# ═══════════════════════════════════════════════════════════════════════
# Formula Evaluation + Replacement
# ═══════════════════════════════════════════════════════════════════════

def evaluate_formulas(ensemble, feat, ret, raw, feat_dim):
    """Evaluate each formula's Sharpe on last 60 days."""
    vm = StackVM(feat_offset=feat_dim)
    bt = MemeBacktest(position_mode='rank')
    N, F, T = feat.shape
    eval_len = min(60, T - 10)
    eval_feat = feat[:, :, -eval_len:]
    eval_ret = ret[:, -eval_len:]
    eval_liq = raw.get('liquidity', torch.ones(N, T))
    if eval_liq.dim() == 2:
        eval_liq = eval_liq[:, -eval_len:]

    sharpes = []
    for formula in ensemble.formulas:
        res = vm.execute(formula, eval_feat)
        if res is not None and res.std() > 1e-8:
            while res.dim() > 2:
                res = res[:, :, -1]
            is_safe = (eval_liq > bt.min_liq).float() if eval_liq.dim() == 2 else torch.ones_like(eval_ret)
            pos = bt.compute_position(res, is_safe)
            daily_pnl = (pos * eval_ret).mean(dim=0)
            sharpes.append(float(daily_pnl.mean() / (daily_pnl.std() + 1e-10) * np.sqrt(252)))
        else:
            sharpes.append(-999.0)
    return sharpes


def search_replacement_formulas(n_replace, db_path, product_ids, end_date,
                                feature_names, feature_indices):
    """Search for n_replace new formulas. Uses GPU for training."""
    import model_core.features_v2 as fv2
    import model_core.config as mc_config
    from model_core.engine import AlphaEngine

    orig_list = fv2.FEATURES_V3_LIST[:]
    fv2.FEATURES_V3_LIST = list(feature_names)

    # Force GPU for ensemble search (main loop uses CPU for WFR)
    orig_device = mc_config.ModelConfig.DEVICE
    if torch.cuda.is_available():
        mc_config.ModelConfig.DEVICE = 'cuda'

    live_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_ensemble.json')
    import shutil
    backup = live_path + '.wfr_backup'
    if os.path.exists(live_path):
        shutil.copy2(live_path, backup)

    try:
        loader = DuckDBDataLoader(
            timeframe='1d', db_path=db_path,
            products=product_ids, end_date=end_date,
            skip_ic_screening=True)
        loader.load_data()

        if loader.feat_tensor.shape[1] > len(feature_indices):
            idx_t = torch.tensor(feature_indices)
            loader.feat_tensor = loader.feat_tensor[:, idx_t, :]
            T = loader.feat_tensor.shape[2]
            split = int(T * loader.train_ratio)
            loader.train_feat = loader.feat_tensor[:, :, :split]
            loader.test_feat = loader.feat_tensor[:, :, split:]
            loader.train_ret = loader.target_ret[:, :split]
            loader.test_ret = loader.target_ret[:, split:]
            loader.train_raw = {k: v[:, :split] if isinstance(v, torch.Tensor) and v.dim() > 1 else v
                                for k, v in loader.raw_data_cache.items()}
            loader.test_raw = {k: v[:, split:] if isinstance(v, torch.Tensor) and v.dim() > 1 else v
                               for k, v in loader.raw_data_cache.items()}

        class PreloadedLoader:
            def __init__(self_pl):
                for attr in dir(loader):
                    if not attr.startswith('_'):
                        val = getattr(loader, attr, None)
                        if val is not None and not callable(val):
                            try:
                                setattr(self_pl, attr, val)
                            except:
                                pass
                self_pl.train_ratio = getattr(loader, 'train_ratio', 0.7)
                self_pl.db_path = db_path
                self_pl.timeframe = '1d'
            def load_data(self_pl):
                pass

        new_ens = AlphaEngine.train_ensemble(num_seeds=n_replace, loader_cls=PreloadedLoader)
        return new_ens.formulas if new_ens else []
    except Exception as e:
        log.error(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        fv2.FEATURES_V3_LIST = orig_list
        mc_config.ModelConfig.DEVICE = orig_device
        if os.path.exists(backup):
            import shutil
            shutil.copy2(backup, live_path)
            os.remove(backup)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Event-Driven Update Logic
# ═══════════════════════════════════════════════════════════════════════

def execute_update(level, current_ensemble, current_names, current_indices,
                   db_path, live_pids, update_date, feat_dim):
    """Execute an update at the given level. Returns (new_ensemble, new_names, new_indices, log_lines)."""
    log_lines = [f"UPDATE TRIGGERED at {update_date} (Level {level})"]

    # ─── IC screening ───────────────────────────────────────────
    import model_core.features_v2 as fv2
    orig_list = fv2.FEATURES_V3_LIST[:]
    fv2.FEATURES_V3_LIST = list(FEATURES_V3_FULL)
    try:
        ic_loader = DuckDBDataLoader(
            timeframe='1d', db_path=db_path,
            products=live_pids, end_date=update_date,
            skip_ic_screening=True)
        ic_loader.load_data()
    finally:
        fv2.FEATURES_V3_LIST = orig_list

    all_feat_names = list(FEATURES_V3_FULL[:ic_loader.feat_tensor.shape[1]])
    all_ic = compute_all_ic(ic_loader.feat_tensor, ic_loader.target_ret, all_feat_names)

    # ─── Factor swap ────────────────────────────────────────────
    n_swap = 1 if level == 2 else 2  # Level 2: 1 swap, Level 3: 2 swaps
    new_names, new_indices, swap_changes = swap_factors(current_names, all_ic, n_swap)
    log_lines.extend(swap_changes if swap_changes else ["  No factor swaps needed"])

    # ─── Formula evaluation + replacement ────────────────────────
    # Slice IC loader to current factors for evaluation
    import model_core.config as mc_config
    orig_device = mc_config.ModelConfig.DEVICE
    mc_config.ModelConfig.DEVICE = 'cpu'

    if ic_loader.feat_tensor.shape[1] > len(new_indices):
        eval_feat = ic_loader.feat_tensor[:, torch.tensor(new_indices), :]
    else:
        eval_feat = ic_loader.feat_tensor
    eval_ret = ic_loader.target_ret

    sharpes = evaluate_formulas(current_ensemble, eval_feat, eval_ret,
                                ic_loader.raw_data_cache, feat_dim)
    log_lines.append(f"  Formula Sharpes: {[f'{s:.2f}' for s in sharpes]}")

    mc_config.ModelConfig.DEVICE = orig_device
    del ic_loader
    gc.collect()

    # Decide how many to replace
    if level == 2:
        # Replace only the single worst
        worst_idx = int(np.argmin(sharpes))
        if sharpes[worst_idx] < 0:
            n_replace = 1
            replace_indices = [worst_idx]
        else:
            n_replace = 0
            replace_indices = []
    else:  # level 3
        # Replace all with Sharpe < 0
        replace_indices = [i for i, s in enumerate(sharpes) if s < 0]
        n_replace = len(replace_indices)

    new_ensemble = current_ensemble
    if n_replace > 0:
        log_lines.append(f"  Replacing {n_replace} formulas: indices {replace_indices}")
        new_formulas_list = search_replacement_formulas(
            n_replace, db_path, live_pids, update_date,
            new_names, new_indices)

        if new_formulas_list:
            formula_list = list(current_ensemble.formulas)
            for i, new_f in zip(replace_indices[:len(new_formulas_list)], new_formulas_list):
                formula_list[i] = new_f
                log_lines.append(f"  Replaced F{i}")
            new_ensemble = FormulaEnsemble(formula_list, mode='mean')
        else:
            log_lines.append("  Search failed, keeping old formulas")
    else:
        log_lines.append("  All formulas OK, no replacement")

    return new_ensemble, new_names, new_indices, log_lines


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2025-04-01')
    parser.add_argument('--end', default='2026-03-27')
    parser.add_argument('--capital', type=float, default=300000)
    parser.add_argument('--topn', type=int, default=5)
    parser.add_argument('--sharpe-window', type=int, default=60,
                        help='Rolling Sharpe lookback (trading days)')
    parser.add_argument('--trigger-days', type=int, default=10,
                        help='Consecutive days Sharpe < 0 before triggering update')
    parser.add_argument('--cooldown-days', type=int, default=30,
                        help='Min days between updates')
    parser.add_argument('--output', default='wfr_quarterly_results.json')
    args = parser.parse_args()

    db_path = os.environ.get('DUCKDB_PATH',
                             os.path.expanduser('~/quant/tick_data/kline_1min.duckdb'))

    # Product list
    live_sig_file = os.path.join(os.path.dirname(__file__), '..', 'signals', 'signals_2026-03-25.json')
    with open(live_sig_file) as f:
        live_data = json.load(f)
    live_pids = sorted(set(s['product'].split('.')[0] for s in live_data['signals']))

    # Price data for PnL
    plist = ", ".join(f"'{p}'" for p in live_pids)
    con = duckdb.connect(db_path, read_only=True)
    price_df = con.execute(f"""
        WITH doi AS (SELECT product_id, exchange, symbol, DATE_TRUNC('day', datetime) as day,
                     AVG(close_oi) as avg_oi FROM kline_1min
                     WHERE product_id IN ({plist})
                       AND datetime >= '{args.start}'::DATE - INTERVAL '5 days'
                       AND datetime <= '{args.end}'::DATE + INTERVAL '2 days'
                     GROUP BY 1,2,3,4),
        mc AS (SELECT day, product_id, exchange, symbol,
               ROW_NUMBER() OVER (PARTITION BY product_id, day ORDER BY avg_oi DESC) as rn FROM doi)
        SELECT mc.product_id || '.' || mc.exchange as addr, mc.day as time,
               LAST(k.close ORDER BY k.datetime) as close
        FROM mc JOIN kline_1min k ON k.symbol = mc.symbol AND DATE_TRUNC('day', k.datetime) = mc.day
        WHERE mc.rn = 1 GROUP BY 1,2 ORDER BY 1,2
    """).fetchdf()
    con.close()
    price_df['time'] = pd.to_datetime(price_df['time'])
    ret_data = {}
    for addr, grp in price_df.groupby('addr'):
        grp = grp.sort_values('time')
        for i in range(len(grp) - 1):
            ret_data[(addr, grp.iloc[i]['time'].strftime('%Y-%m-%d'))] = \
                grp.iloc[i + 1]['close'] / grp.iloc[i]['close'] - 1

    # ─── Load full data once ────────────────────────────────────
    log.info("Loading data...")
    import model_core.config as mc_config
    mc_config.ModelConfig.DEVICE = 'cpu'
    loader = DuckDBDataLoader(timeframe='1d', db_path=db_path, products=live_pids)
    loader.load_data()

    N = loader.feat_tensor.shape[0]
    feat_full = loader.feat_tensor       # [N, 20, T]
    ret_full = loader.target_ret         # [N, T]
    raw_full = loader.raw_data_cache

    dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
             for d in loader.dates]
    date_to_idx = {d: i for i, d in enumerate(dates)}

    con = duckdb.connect(db_path, read_only=True)
    pdf = con.execute(f"SELECT DISTINCT product_id || '.' || exchange as address FROM kline_1min WHERE product_id IN ({plist}) ORDER BY address").fetchdf()
    con.close()
    products = pdf['address'].tolist()[:N]

    trading_days = [d for d in dates if args.start <= d <= args.end]
    log.info(f"Data: {feat_full.shape}, trading days: {len(trading_days)}")

    # ─── Initialize ─────────────────────────────────────────────
    ens_data = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'best_ensemble.json')))
    current_ensemble = FormulaEnsemble.from_dict(ens_data['ensemble'])
    current_names = list(FEATURES_V3_LIST)
    current_indices = [FEATURES_V3_FULL.index(n) for n in current_names if n in FEATURES_V3_FULL]
    feat_dim = FIXED_FACTOR_COUNT

    mc_config.ModelConfig.get_feature_dim = staticmethod(lambda: feat_dim)
    online_learner = OnlineLearner(current_ensemble, strategy='mwu', eta=0.05,
                                   lookback_window=10, min_weight=0.02)

    # ─── Main daily loop ────────────────────────────────────────
    vm = StackVM(feat_offset=feat_dim)
    results = []
    pnl_history = []  # For rolling Sharpe computation
    consecutive_bad = 0  # Days with rolling Sharpe < 0
    last_update_day = -999  # Day index of last update (for cooldown)
    update_events = []
    cumul = 0

    print(f"\n{'Date':>12} {'PnL':>10} {'Cumul':>10} {'RollSharpe':>12} {'Bad':>4} {'Event'}")
    print('─' * 65)

    for day_str in trading_days:
        idx = date_to_idx.get(day_str)
        if idx is None or idx < 30:
            continue

        # ─── MWU update ─────────────────────────────────────────
        lkb = min(10, idx - 5)
        try:
            p_raw = {k: (v[:, idx-lkb:idx] if isinstance(v, torch.Tensor) and v.dim() == 2
                         else (v[idx-lkb:idx] if isinstance(v, torch.Tensor) and v.dim() == 1
                               else v))
                     for k, v in raw_full.items()}
            online_learner.update(feat_full[:, :, idx-lkb:idx], p_raw,
                                  ret_full[:, idx-lkb:idx])
        except:
            pass

        # ─── Per-formula signals ─────────────────────────────────
        formula_signals = []
        for formula in current_ensemble.formulas:
            res = vm.execute(formula, feat_full[:, :, idx:idx+1])
            if res is not None and res.std() > 1e-8:
                while res.dim() > 1:
                    res = res[:, -1]
                formula_signals.append(res[:N])
            else:
                formula_signals.append(torch.zeros(N, device=feat_full.device))

        weights = online_learner.get_weights()
        stacked = torch.stack(formula_signals, dim=0)
        w_t = torch.tensor(weights, dtype=torch.float32, device=stacked.device)
        sig = (stacked * w_t.unsqueeze(-1)).sum(dim=0)

        ranks = sig.argsort().argsort().float()
        rank_norm = (ranks / (N - 1) - 0.5) * 2

        # ─── TopN + PnL ─────────────────────────────────────────
        signals = [{'product': products[i] if i < len(products) else f'p_{i}',
                     'tw': rank_norm[i].item()} for i in range(N)]
        selected = pick_topn(signals, n=args.topn)

        n_pos = len(selected)
        day_pnl = 0
        for pos in selected:
            prod = ALIAS.get(pos['product'], pos['product'])
            r = ret_data.get((prod, day_str))
            if r is not None and np.isfinite(r):
                day_pnl += (1 if pos['tw'] > 0 else -1) * r / n_pos

        day_bps = day_pnl * 10000
        cumul += day_bps
        pnl_history.append(day_bps)

        # ─── Rolling Sharpe ──────────────────────────────────────
        event_str = ""
        roll_sharpe = 0.0
        if len(pnl_history) >= args.sharpe_window:
            recent = np.array(pnl_history[-args.sharpe_window:])
            roll_sharpe = recent.mean() / (recent.std() + 1e-10) * np.sqrt(252)

            if roll_sharpe < 0:
                consecutive_bad += 1
            else:
                consecutive_bad = 0

            # Check trigger
            day_idx_in_loop = len(results)
            if (consecutive_bad >= args.trigger_days and
                    day_idx_in_loop - last_update_day >= args.cooldown_days):

                # Determine level
                if roll_sharpe < -1.5:
                    level = 3
                elif roll_sharpe < -0.5:
                    level = 2
                else:
                    level = 1  # MWU handles this, no action

                if level >= 2:
                    event_str = f"⚡ L{level} UPDATE"
                    log.info(f"\n{'!'*60}")
                    log.info(f"  {event_str} at {day_str} (Sharpe={roll_sharpe:.2f}, "
                             f"bad_days={consecutive_bad})")
                    log.info(f"{'!'*60}")

                    current_ensemble, current_names, current_indices, log_lines = \
                        execute_update(level, current_ensemble, current_names,
                                       current_indices, db_path, live_pids,
                                       day_str, feat_dim)

                    for line in log_lines:
                        log.info(line)

                    # Reload data with new factors (re-slice feat_full)
                    # Since we keep factor count at 20, just reload
                    import model_core.features_v2 as fv2
                    orig_list = fv2.FEATURES_V3_LIST[:]
                    mc_config.ModelConfig.DEVICE = 'cpu'
                    fv2.FEATURES_V3_LIST = list(current_names)
                    try:
                        new_loader = DuckDBDataLoader(
                            timeframe='1d', db_path=db_path, products=live_pids)
                        new_loader.load_data()
                        feat_full = new_loader.feat_tensor
                        ret_full = new_loader.target_ret
                        raw_full = new_loader.raw_data_cache
                    finally:
                        fv2.FEATURES_V3_LIST = orig_list

                    # Reset VM and OnlineLearner with new ensemble
                    vm = StackVM(feat_offset=feat_dim)
                    online_learner = OnlineLearner(
                        current_ensemble, strategy='mwu', eta=0.05,
                        lookback_window=10, min_weight=0.02)

                    last_update_day = day_idx_in_loop
                    consecutive_bad = 0

                    update_events.append({
                        'date': day_str,
                        'level': level,
                        'roll_sharpe': round(roll_sharpe, 3),
                        'changes': log_lines,
                    })

        results.append({'date': day_str, 'pnl_bps': round(day_bps, 4)})

        # Print (every 20 days or on events)
        if len(results) % 20 == 0 or event_str:
            print(f"{day_str:>12} {day_bps:>+10.2f} {cumul:>+10.2f} {roll_sharpe:>+12.2f} "
                  f"{consecutive_bad:>4} {event_str}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    pnl_arr = np.array([r['pnl_bps'] for r in results])
    valid = pnl_arr[pnl_arr != 0]

    print(f"\n{'='*65}")
    print(f"  WFR Event-Driven ({args.start} → {args.end})")
    print(f"{'='*65}")

    if len(valid) > 1:
        cumul_arr = np.cumsum(pnl_arr)
        max_dd = (cumul_arr - np.maximum.accumulate(cumul_arr)).min()
        sharpe = valid.mean() / (valid.std() + 1e-9) * np.sqrt(252)
        print(f"  累计收益:    {pnl_arr.sum():+.2f} bps ({pnl_arr.sum()/10000*100:+.4f}%)")
        print(f"  日均收益:    {valid.mean():+.2f} bps")
        print(f"  年化Sharpe:  {sharpe:.2f}")
        print(f"  胜率:        {(valid>0).sum()}/{len(valid)} ({(valid>0).sum()/len(valid)*100:.0f}%)")
        print(f"  最大回撤:    {max_dd:.2f} bps")
        print(f"  更新次数:    {len(update_events)}")

    if update_events:
        print(f"\n  更新事件:")
        for evt in update_events:
            print(f"    {evt['date']}: Level {evt['level']}, Sharpe={evt['roll_sharpe']}")

    output = {
        'config': vars(args),
        'update_events': update_events,
        'daily_results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == '__main__':
    main()
