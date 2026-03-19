"""
Risk management filter for AlphaGPT commodity futures trading.

Takes raw signals from generate_signals.py and applies risk controls:
  1. Single product max position (15%)
  2. Portfolio drawdown stop (5% reduce, 8% full stop)
  3. Daily loss limit (2%)
  4. Sector concentration limit (40%)
  5. Volatility targeting (15% annualized)
  6. Correlation dedup (|corr| > 0.8 same direction)

Usage:
    python scripts/risk_filter.py [--input signals/latest.json] [--capital 1000000]
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('risk_filter')


# ---------------------------------------------------------------------------
# Commodity sector mapping (Chinese futures)
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    # 黑色系
    'rb': '黑色', 'hc': '黑色', 'i': '黑色', 'j': '黑色', 'jm': '黑色',
    'SF': '黑色', 'SM': '黑色', 'ss': '黑色',
    # 有色金属
    'cu': '有色', 'al': '有色', 'zn': '有色', 'pb': '有色', 'ni': '有色',
    'sn': '有色', 'bc': '有色', 'ao': '有色', 'si': '有色',
    # 贵金属
    'au': '贵金属', 'ag': '贵金属',
    # 能源化工
    'sc': '能源', 'fu': '能源', 'bu': '能源', 'lu': '能源', 'pg': '能源',
    'TA': '化工', 'MA': '化工', 'eg': '化工', 'eb': '化工', 'PP': '化工',
    'L': '化工', 'V': '化工', 'SA': '化工', 'UR': '化工', 'FG': '化工',
    'PF': '化工', 'nr': '化工', 'ru': '化工', 'sp': '化工', 'PX': '化工',
    'SH': '化工', 'BR': '化工',
    # 农产品
    'c': '农产品', 'cs': '农产品', 'a': '农产品', 'm': '农产品', 'y': '农产品',
    'p': '农产品', 'OI': '农产品', 'RM': '农产品', 'b': '农产品',
    'CF': '农产品', 'SR': '农产品', 'CJ': '农产品', 'AP': '农产品',
    'jd': '农产品', 'lh': '农产品', 'PK': '农产品', 'CY': '农产品',
    'rr': '农产品', 'WH': '农产品', 'PM': '农产品', 'RI': '农产品',
    'RS': '农产品', 'JR': '农产品', 'LR': '农产品',
}


def get_sector(product_address):
    """Extract sector from product address like 'cu.SHFE'."""
    product_id = product_address.split('.')[0]
    return SECTOR_MAP.get(product_id, '其他')


# ---------------------------------------------------------------------------
# Portfolio state tracker
# ---------------------------------------------------------------------------
@dataclass
class PortfolioState:
    """Tracks portfolio PnL, drawdown, and daily performance."""
    capital: float = 1_000_000.0
    peak_equity: float = 1_000_000.0
    current_equity: float = 1_000_000.0
    daily_pnl: float = 0.0
    positions: dict = field(default_factory=dict)
    pnl_history: list = field(default_factory=list)

    @property
    def drawdown(self):
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def daily_return(self):
        if self.capital <= 0:
            return 0.0
        return self.daily_pnl / self.capital

    def update(self, daily_pnl):
        self.daily_pnl = daily_pnl
        self.current_equity += daily_pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)
        self.pnl_history.append(daily_pnl)

    def to_dict(self):
        return {
            'capital': self.capital,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'drawdown': f"{self.drawdown:.2%}",
            'daily_pnl': self.daily_pnl,
            'n_days': len(self.pnl_history),
        }

    @classmethod
    def load(cls, path):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            state = cls(capital=data['capital'])
            state.peak_equity = data['peak_equity']
            state.current_equity = data['current_equity']
            state.daily_pnl = data.get('daily_pnl', 0)
            state.pnl_history = data.get('pnl_history', [])
            return state
        return cls()

    def save(self, path):
        data = {
            'capital': self.capital,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'daily_pnl': self.daily_pnl,
            'pnl_history': self.pnl_history[-252:],  # keep last year
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Risk filter
# ---------------------------------------------------------------------------
class RiskFilter:
    def __init__(self, capital=1_000_000,
                 max_position_pct=0.15,
                 max_drawdown_reduce=0.05,
                 max_drawdown_stop=0.08,
                 max_daily_loss=0.02,
                 max_sector_pct=0.40,
                 target_vol=0.15,
                 max_corr=0.80):
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.max_drawdown_reduce = max_drawdown_reduce
        self.max_drawdown_stop = max_drawdown_stop
        self.max_daily_loss = max_daily_loss
        self.max_sector_pct = max_sector_pct
        self.target_vol = target_vol
        self.max_corr = max_corr
        self.adjustments = []

    def _log_adj(self, product, rule, before, after, reason):
        self.adjustments.append({
            'product': product,
            'rule': rule,
            'before': round(before, 6),
            'after': round(after, 6),
            'reason': reason,
        })

    def apply(self, signals, portfolio_state):
        """Apply all risk filters. Returns adjusted signals list."""
        self.adjustments = []

        # Extract weights
        products = [s['product'] for s in signals]
        weights = np.array([s['target_weight'] for s in signals])

        # Rule 1: Drawdown stop
        weights = self._check_drawdown(weights, portfolio_state)

        # Rule 2: Daily loss limit
        weights = self._check_daily_loss(weights, portfolio_state)

        # Rule 3: Single position limit
        weights = self._check_position_limit(weights, products)

        # Rule 4: Sector concentration
        weights = self._check_sector_concentration(weights, products)

        # Rule 5: Volatility targeting
        weights = self._check_volatility(weights, portfolio_state)

        # Rule 6: Correlation dedup
        weights = self._check_correlation(weights, products, signals)

        # Rebuild signals
        adjusted = []
        for i, s in enumerate(signals):
            adj = dict(s)
            adj['target_weight'] = round(float(weights[i]), 6)
            adj['direction'] = ('LONG' if weights[i] > 0.01
                                else 'SHORT' if weights[i] < -0.01
                                else 'FLAT')
            adjusted.append(adj)

        return adjusted

    def _check_drawdown(self, weights, state):
        dd = state.drawdown
        if dd >= self.max_drawdown_stop:
            self._log_adj('ALL', 'drawdown_stop',
                          float(np.abs(weights).sum()),
                          0.0,
                          f"Drawdown {dd:.1%} >= {self.max_drawdown_stop:.0%}, FULL STOP")
            return np.zeros_like(weights)

        if dd >= self.max_drawdown_reduce:
            scale = 0.5
            self._log_adj('ALL', 'drawdown_reduce',
                          float(np.abs(weights).sum()),
                          float(np.abs(weights).sum() * scale),
                          f"Drawdown {dd:.1%} >= {self.max_drawdown_reduce:.0%}, reducing 50%")
            return weights * scale

        return weights

    def _check_daily_loss(self, weights, state):
        if abs(state.daily_return) > self.max_daily_loss and state.daily_pnl < 0:
            self._log_adj('ALL', 'daily_loss_stop',
                          float(np.abs(weights).sum()),
                          0.0,
                          f"Daily loss {state.daily_return:.1%} > {self.max_daily_loss:.0%}")
            return np.zeros_like(weights)
        return weights

    def _check_position_limit(self, weights, products):
        for i, (w, p) in enumerate(zip(weights, products)):
            if abs(w) > self.max_position_pct:
                clamped = np.sign(w) * self.max_position_pct
                self._log_adj(p, 'position_limit', w, clamped,
                              f"|weight| {abs(w):.1%} > {self.max_position_pct:.0%}")
                weights[i] = clamped
        return weights

    def _check_sector_concentration(self, weights, products):
        # Compute sector exposure
        sector_exposure = {}
        sector_indices = {}
        for i, p in enumerate(products):
            sector = get_sector(p)
            sector_exposure.setdefault(sector, 0.0)
            sector_indices.setdefault(sector, [])
            sector_exposure[sector] += abs(weights[i])
            sector_indices[sector].append(i)

        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_pct:
                scale = self.max_sector_pct / exposure
                for idx in sector_indices[sector]:
                    old_w = weights[idx]
                    weights[idx] *= scale
                    if abs(old_w - weights[idx]) > 1e-6:
                        self._log_adj(products[idx], 'sector_limit',
                                      old_w, weights[idx],
                                      f"{sector} exposure {exposure:.1%} > {self.max_sector_pct:.0%}")
        return weights

    def _check_volatility(self, weights, state):
        # Estimate portfolio volatility from recent PnL
        if len(state.pnl_history) < 20:
            return weights  # not enough history

        recent_rets = np.array(state.pnl_history[-60:]) / (state.capital + 1e-9)
        realized_vol = np.std(recent_rets) * np.sqrt(252)

        if realized_vol > 1e-6 and realized_vol > self.target_vol * 1.5:
            scale = self.target_vol / realized_vol
            self._log_adj('ALL', 'vol_target',
                          float(np.abs(weights).sum()),
                          float(np.abs(weights).sum() * scale),
                          f"Realized vol {realized_vol:.1%} > target {self.target_vol:.0%} × 1.5")
            return weights * scale
        return weights

    def _check_correlation(self, weights, products, signals):
        # Simple correlation check: same sector + same direction → reduce weaker
        n = len(weights)
        for i in range(n):
            if abs(weights[i]) < 0.01:
                continue
            for j in range(i + 1, n):
                if abs(weights[j]) < 0.01:
                    continue
                # Same sector and same direction?
                si = get_sector(products[i])
                sj = get_sector(products[j])
                same_dir = np.sign(weights[i]) == np.sign(weights[j])

                if si == sj and same_dir:
                    # Check signal strength
                    str_i = signals[i].get('signal_strength', abs(weights[i]))
                    str_j = signals[j].get('signal_strength', abs(weights[j]))

                    # Reduce the weaker one by 50%
                    if str_i < str_j:
                        old = weights[i]
                        weights[i] *= 0.5
                        self._log_adj(products[i], 'corr_dedup',
                                      old, weights[i],
                                      f"Same sector ({si}) & direction as {products[j]}")
                    else:
                        old = weights[j]
                        weights[j] *= 0.5
                        self._log_adj(products[j], 'corr_dedup',
                                      old, weights[j],
                                      f"Same sector ({sj}) & direction as {products[i]}")
        return weights


# ---------------------------------------------------------------------------
# Small capital Top-N filter
# ---------------------------------------------------------------------------

# Approximate margin per lot (元) — conservative estimates
# Format: product_id → (margin_per_lot, min_daily_volume)
MARGIN_TABLE = {
    # 贵金属 (high margin)
    'au': (72000, 50000), 'ag': (13500, 200000),
    # 有色
    'cu': (42000, 80000), 'al': (12000, 150000), 'zn': (15000, 80000),
    'pb': (10000, 20000), 'ni': (15000, 100000), 'sn': (15000, 30000),
    'bc': (45000, 30000), 'ao': (10000, 50000), 'si': (8000, 100000),
    # 黑色
    'rb': (3500, 500000), 'hc': (3500, 200000), 'i': (5000, 300000),
    'j': (12000, 100000), 'jm': (8000, 100000), 'SF': (5000, 50000),
    'SM': (5000, 50000), 'ss': (5000, 80000),
    # 能源
    'sc': (50000, 80000), 'fu': (3000, 100000), 'bu': (3000, 200000),
    'lu': (6000, 50000), 'pg': (4000, 100000),
    # 化工
    'TA': (3000, 300000), 'MA': (2500, 200000), 'eg': (3500, 200000),
    'eb': (5000, 100000), 'PP': (4000, 150000), 'L': (4000, 100000),
    'V': (3000, 80000), 'SA': (3000, 200000), 'UR': (3000, 100000),
    'FG': (3000, 150000), 'PF': (3000, 50000), 'ru': (15000, 200000),
    'nr': (10000, 30000), 'SH': (5000, 50000), 'PX': (5000, 30000),
    'sp': (3000, 50000),
    # 农产品
    'c': (2000, 200000), 'cs': (2000, 100000), 'a': (4000, 100000),
    'm': (2500, 300000), 'y': (5000, 200000), 'p': (4000, 200000),
    'OI': (3000, 80000), 'RM': (2500, 150000), 'CF': (5000, 100000),
    'SR': (4000, 100000), 'CJ': (5000, 50000), 'AP': (5000, 80000),
    'jd': (3000, 30000), 'lh': (15000, 50000), 'PK': (5000, 30000),
    'b': (3000, 20000), 'rr': (2000, 10000),
}


class SmallCapFilter:
    """Top-N filter for small capital accounts (e.g., 30万).

    Strategy:
    1. Filter out products where 1 lot margin > max_margin_pct of capital
    2. Filter out illiquid products (daily volume < threshold)
    3. Pick Top-N long + Top-N short by signal strength
    4. Ensure sector diversification (max 2 products per sector)
    5. Size positions by equal risk (inverse volatility weighting)
    6. Apply hard stop-loss per product (3% of total capital)
    """

    def __init__(self, capital, top_n=3, max_margin_pct=0.20,
                 max_total_margin_pct=0.60, max_per_sector=2,
                 stop_loss_pct=0.03):
        self.capital = capital
        self.top_n = top_n
        self.max_margin_pct = max_margin_pct
        self.max_total_margin_pct = max_total_margin_pct
        self.max_per_sector = max_per_sector
        self.stop_loss_pct = stop_loss_pct
        self.adjustments = []

    def _log_adj(self, product, rule, reason):
        self.adjustments.append({
            'product': product, 'rule': rule, 'reason': reason,
        })

    def _get_margin(self, product):
        pid = product.split('.')[0]
        return MARGIN_TABLE.get(pid, (20000, 50000))

    def apply(self, signals, portfolio_state):
        """Select Top-N long + Top-N short from eligible products."""
        self.adjustments = []

        # Step 1: Filter by margin affordability
        max_margin = self.capital * self.max_margin_pct
        eligible = []
        for s in signals:
            margin, min_vol = self._get_margin(s['product'])
            if margin > max_margin:
                self._log_adj(s['product'], 'margin_too_high',
                              f"Margin {margin:,.0f} > {max_margin:,.0f} ({self.max_margin_pct:.0%} of capital)")
                continue
            s['_margin'] = margin
            s['_min_vol'] = min_vol
            eligible.append(s)

        log.info(f"Margin filter: {len(signals)} → {len(eligible)} eligible")

        # Step 2: Split into long and short candidates
        longs = sorted([s for s in eligible if s.get('raw_weight', s['target_weight']) > 0.01],
                       key=lambda x: x.get('signal_strength', abs(x['target_weight'])),
                       reverse=True)
        shorts = sorted([s for s in eligible if s.get('raw_weight', s['target_weight']) < -0.01],
                        key=lambda x: x.get('signal_strength', abs(x['target_weight'])),
                        reverse=True)

        # Step 3: Pick Top-N with sector diversification
        selected_long = self._pick_topn_diversified(longs, self.top_n, 'LONG')
        selected_short = self._pick_topn_diversified(shorts, self.top_n, 'SHORT')

        all_selected = selected_long + selected_short

        # Step 4: Size by equal capital allocation
        if not all_selected:
            return [dict(s, target_weight=0.0, direction='FLAT') for s in signals]

        total_margin_budget = self.capital * self.max_total_margin_pct
        per_position_budget = total_margin_budget / len(all_selected)

        for s in all_selected:
            margin = s['_margin']
            # How many lots can we afford?
            max_lots = int(per_position_budget / margin)
            max_lots = max(max_lots, 1)  # at least 1 lot
            actual_margin = max_lots * margin

            # Weight = actual_margin / capital
            weight = actual_margin / self.capital
            direction = 1.0 if s in selected_long else -1.0
            s['target_weight'] = round(direction * weight, 6)
            s['direction'] = 'LONG' if direction > 0 else 'SHORT'
            s['lots'] = max_lots
            s['margin_used'] = actual_margin
            s['stop_loss'] = round(self.stop_loss_pct * self.capital, 2)

        # Build full output (set non-selected to FLAT)
        selected_products = {s['product'] for s in all_selected}
        result = []
        for s in signals:
            if s['product'] in selected_products:
                # Find the adjusted version
                for sel in all_selected:
                    if sel['product'] == s['product']:
                        result.append(sel)
                        break
            else:
                flat = dict(s)
                flat['target_weight'] = 0.0
                flat['direction'] = 'FLAT'
                flat['lots'] = 0
                flat['margin_used'] = 0
                result.append(flat)

        return result

    def _pick_topn_diversified(self, candidates, n, side):
        """Pick top N candidates with sector diversification."""
        selected = []
        sector_count = {}

        for s in candidates:
            if len(selected) >= n:
                break
            sector = get_sector(s['product'])
            count = sector_count.get(sector, 0)

            if count >= self.max_per_sector:
                self._log_adj(s['product'], 'sector_diversity',
                              f"{side}: {sector} already has {count} products")
                continue

            selected.append(s)
            sector_count[sector] = count + 1

        return selected


def main():
    parser = argparse.ArgumentParser(description='Apply risk filters to trading signals')
    parser.add_argument('--input', default='signals/latest.json', help='Input signals JSON')
    parser.add_argument('--output', default=None, help='Output path (default: overwrite input)')
    parser.add_argument('--capital', type=float, default=1_000_000, help='Account capital')
    parser.add_argument('--state', default='signals/portfolio_state.json', help='Portfolio state file')
    parser.add_argument('--mode', choices=['full', 'topn'], default='full',
                        help='full=all products, topn=small capital Top-N')
    parser.add_argument('--topn', type=int, default=3, help='Top-N per side (topn mode)')
    args = parser.parse_args()

    # Auto-detect mode: if capital < 500k, suggest topn
    if args.capital < 500_000 and args.mode == 'full':
        log.warning(f"Capital {args.capital:,.0f} < 500k, auto-switching to --mode topn")
        args.mode = 'topn'

    # Load signals
    with open(args.input) as f:
        signal_data = json.load(f)
    signals = signal_data['signals']
    log.info(f"Loaded {len(signals)} signals from {args.input}")

    # Load portfolio state
    state = PortfolioState.load(args.state)
    state.capital = args.capital
    log.info(f"Portfolio: equity={state.current_equity:,.0f}, drawdown={state.drawdown:.2%}")

    # Apply risk filters
    if args.mode == 'topn':
        rf = SmallCapFilter(capital=args.capital, top_n=args.topn)
        adjusted = rf.apply(signals, state)
    else:
        rf = RiskFilter(capital=args.capital)
        adjusted = rf.apply(signals, state)

    # Summary
    n_adjusted = len(rf.adjustments)
    n_long = sum(1 for s in adjusted if s['direction'] == 'LONG')
    n_short = sum(1 for s in adjusted if s['direction'] == 'SHORT')
    n_flat = sum(1 for s in adjusted if s['direction'] == 'FLAT')

    total_gross = sum(abs(s['target_weight']) for s in adjusted)

    log.info(f"Risk adjustments: {n_adjusted}")
    for adj in rf.adjustments:
        if 'before' in adj:
            log.info(f"  [{adj['rule']}] {adj['product']}: "
                     f"{adj['before']:+.4f} → {adj['after']:+.4f} ({adj['reason']})")
        else:
            log.info(f"  [{adj['rule']}] {adj['product']}: {adj['reason']}")

    log.info(f"After filter: Long={n_long}, Short={n_short}, Flat={n_flat}, "
             f"Gross exposure={total_gross:.2%}")

    # Update signal data
    signal_data['signals'] = adjusted
    signal_data['risk_filter'] = {
        'timestamp': datetime.now().isoformat(),
        'n_adjustments': n_adjusted,
        'adjustments': rf.adjustments,
        'portfolio_state': state.to_dict(),
        'gross_exposure': round(total_gross, 4),
    }
    signal_data['num_long'] = n_long
    signal_data['num_short'] = n_short
    signal_data['num_flat'] = n_flat

    # Save
    out_path = args.output or args.input
    with open(out_path, 'w') as f:
        json.dump(signal_data, f, indent=2, ensure_ascii=False)
    log.info(f"Saved to {out_path}")

    # Save portfolio state
    os.makedirs(os.path.dirname(args.state) or '.', exist_ok=True)
    state.save(args.state)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'RISK FILTER SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"  Capital:         {args.capital:>12,.0f}")
    print(f"  Drawdown:        {state.drawdown:>12.2%}")
    print(f"  Gross exposure:  {total_gross:>12.2%}")
    print(f"  Adjustments:     {n_adjusted:>12d}")
    print(f"  Long/Short/Flat: {n_long}/{n_short}/{n_flat}")

    if rf.adjustments:
        print(f"\n  Risk adjustments:")
        for adj in rf.adjustments[:20]:
            if 'before' in adj:
                print(f"    [{adj['rule']:<16}] {adj['product']:<15} "
                      f"{adj['before']:+.4f} → {adj['after']:+.4f}")
            else:
                print(f"    [{adj['rule']:<16}] {adj['product']:<15} {adj['reason']}")
        if len(rf.adjustments) > 20:
            print(f"    ... and {len(rf.adjustments) - 20} more")

    # Mode-specific output
    if args.mode == 'topn':
        active = [s for s in adjusted if s.get('lots', 0) > 0]
        total_margin = sum(s.get('margin_used', 0) for s in active)
        print(f"\n  === TOP-N POSITIONS ({len(active)} active) ===")
        print(f"  Total margin: {total_margin:>10,.0f} ({total_margin/args.capital:.0%} of capital)")
        print(f"  Per-position stop: {args.capital * 0.03:>10,.0f} ({3}%)")
        print(f"\n  {'Product':<15} {'Sector':<8} {'Dir':<6} {'Lots':>5} {'Margin':>10} {'Weight':>8} {'StopLoss':>10}")
        print(f"  {'-'*68}")
        for s in sorted(active, key=lambda x: abs(x['target_weight']), reverse=True):
            sector = get_sector(s['product'])
            print(f"  {s['product']:<15} {sector:<8} {s['direction']:<6} "
                  f"{s.get('lots', 0):>5} {s.get('margin_used', 0):>10,.0f} "
                  f"{s['target_weight']:>+8.4f} {s.get('stop_loss', 0):>10,.0f}")
    else:
        print(f"\n  Top 10 positions (after filter):")
        top = sorted(adjusted, key=lambda x: abs(x['target_weight']), reverse=True)[:10]
        print(f"  {'Product':<15} {'Sector':<8} {'Direction':<8} {'Weight':>8}")
        print(f"  {'-'*42}")
        for s in top:
            sector = get_sector(s['product'])
            print(f"  {s['product']:<15} {sector:<8} {s['direction']:<8} {s['target_weight']:>+8.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
