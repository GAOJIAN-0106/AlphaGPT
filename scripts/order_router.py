"""
Order router for AlphaGPT — executes signals via TqSdk.

Reads signals from risk_filter output, compares with current positions,
and uses TargetPosTask to adjust to target positions.

Usage:
    # Simulation mode (default, uses TqKq)
    python scripts/order_router.py --input signals/latest.json --mode sim

    # Live mode (requires TqAccount with broker config)
    python scripts/order_router.py --input signals/latest.json --mode live \
        --broker "H海通期货" --account "123456" --password "123456"

    # Dry-run mode (print orders without executing)
    python scripts/order_router.py --input signals/latest.json --mode dry
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('order_router')

try:
    from tqsdk import TqApi, TqAuth, TqKq, TqAccount, TargetPosTask
except ImportError:
    log.error("tqsdk not installed. Run: pip install tqsdk")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Contract symbol mapping: product address → TqSdk symbol
# ---------------------------------------------------------------------------
# TqSdk uses "EXCHANGE.symbol" format with main-contract suffix "@"
# e.g., "SHFE.cu2604" for specific contract, "KQ.m@SHFE.cu" for main contract

def address_to_tq_symbol(address):
    """Convert 'cu.SHFE' to 'KQ.m@SHFE.cu' (main contract)."""
    parts = address.split('.')
    if len(parts) != 2:
        return None
    product_id, exchange = parts
    return f"KQ.m@{exchange}.{product_id}"


# ---------------------------------------------------------------------------
# Order router
# ---------------------------------------------------------------------------
class OrderRouter:
    def __init__(self, api, account=None, price_mode='ACTIVE', timeout=60):
        """
        Args:
            api: TqApi instance
            account: account object (for multi-account mode)
            price_mode: 'ACTIVE' (chase price) or 'PASSIVE' (queue)
            timeout: seconds to wait for all targets to fill
        """
        self.api = api
        self.account = account
        self.price_mode = price_mode
        self.timeout = timeout
        self._tasks = {}  # symbol → TargetPosTask

    def get_current_positions(self, symbols):
        """Get current position for each symbol."""
        positions = {}
        for sym in symbols:
            pos = self.api.get_position(sym, account=self.account)
            positions[sym] = {
                'long': pos.pos_long,
                'short': pos.pos_short,
                'net': pos.pos_long - pos.pos_short,
            }
        return positions

    def get_account_info(self):
        """Get account funds info."""
        acc = self.api.get_account(account=self.account)
        return {
            'balance': acc.balance,
            'available': acc.available,
            'margin': acc.margin,
            'float_profit': acc.float_profit,
            'risk_ratio': acc.risk_ratio,
        }

    def execute_signals(self, signals):
        """Execute a list of signal dicts from risk_filter output.

        Each signal has: product, direction, lots, target_weight
        """
        orders = []
        active_signals = [s for s in signals if s.get('lots', 0) > 0]

        if not active_signals:
            log.info("No active signals to execute")
            return orders

        # Map signals to TqSdk symbols
        for s in active_signals:
            sym = address_to_tq_symbol(s['product'])
            if sym is None:
                log.warning(f"Cannot map {s['product']} to TqSdk symbol, skipping")
                continue

            direction = s['direction']
            lots = s.get('lots', 0)
            target_net = lots if direction == 'LONG' else -lots

            orders.append({
                'product': s['product'],
                'symbol': sym,
                'direction': direction,
                'target_net': target_net,
                'lots': lots,
                'weight': s.get('target_weight', 0),
            })

        if not orders:
            return orders

        # Get current positions
        log.info(f"Preparing {len(orders)} position targets...")
        symbols = [o['symbol'] for o in orders]
        current = self.get_current_positions(symbols)

        # Log current vs target
        log.info(f"\n  {'Product':<15} {'Symbol':<25} {'Current':>8} {'Target':>8} {'Delta':>8}")
        log.info(f"  {'-'*70}")
        for o in orders:
            cur = current.get(o['symbol'], {}).get('net', 0)
            tgt = o['target_net']
            delta = tgt - cur
            o['current_net'] = cur
            o['delta'] = delta
            marker = ' ← no change' if delta == 0 else ''
            log.info(f"  {o['product']:<15} {o['symbol']:<25} {cur:>8} {tgt:>8} {delta:>+8}{marker}")

        # Filter out no-change orders
        orders_to_exec = [o for o in orders if o['delta'] != 0]
        if not orders_to_exec:
            log.info("All positions already at target, nothing to do")
            return orders

        # Create TargetPosTask for each symbol
        log.info(f"\nExecuting {len(orders_to_exec)} position changes...")
        for o in orders_to_exec:
            sym = o['symbol']
            if sym not in self._tasks:
                kwargs = {'price': self.price_mode}
                if self.account:
                    kwargs['account'] = self.account
                self._tasks[sym] = TargetPosTask(self.api, sym, **kwargs)

            self._tasks[sym].set_target_volume(o['target_net'])
            log.info(f"  {o['product']}: set target = {o['target_net']}")

        # Also close positions for products NOT in signal list
        # (if we previously had positions that are now FLAT)
        all_products_in_signal = {address_to_tq_symbol(s['product'])
                                  for s in signals if s.get('direction') != 'FLAT'}
        for sym, task in self._tasks.items():
            if sym not in all_products_in_signal:
                task.set_target_volume(0)
                log.info(f"  Closing position: {sym} → 0")

        # Wait for fills
        log.info(f"\nWaiting for fills (timeout={self.timeout}s)...")
        start = time.time()
        while time.time() - start < self.timeout:
            self.api.wait_update()
            # Check if all tasks are finished
            all_done = all(task.is_finished() for task in self._tasks.values())
            if all_done:
                log.info("All targets reached!")
                break
        else:
            log.warning(f"Timeout after {self.timeout}s, some orders may be unfilled")

        # Log final positions
        log.info("\nFinal positions:")
        for o in orders:
            pos = self.api.get_position(o['symbol'], account=self.account)
            net = pos.pos_long - pos.pos_short
            o['final_net'] = net
            status = '✓' if net == o['target_net'] else '✗'
            log.info(f"  {status} {o['product']:<15} target={o['target_net']:>4} actual={net:>4}")

        return orders

    def close_all(self):
        """Emergency: close all positions."""
        log.warning("CLOSING ALL POSITIONS")
        for sym, task in self._tasks.items():
            task.set_target_volume(0)

        start = time.time()
        while time.time() - start < 30:
            self.api.wait_update()
            if all(task.is_finished() for task in self._tasks.values()):
                break

        log.info("All positions closed")


def main():
    parser = argparse.ArgumentParser(description='Execute trading signals via TqSdk')
    parser.add_argument('--input', default='signals/latest.json', help='Signal JSON')
    parser.add_argument('--mode', choices=['sim', 'live', 'dry'], default='sim',
                        help='sim=TqKq simulation, live=real account, dry=print only')
    parser.add_argument('--broker', default='', help='Broker name (live mode)')
    parser.add_argument('--td-account', default='', help='Trading account (live mode)')
    parser.add_argument('--td-password', default='', help='Trading password (live mode)')
    parser.add_argument('--timeout', type=int, default=60, help='Order fill timeout')
    args = parser.parse_args()

    # Load signals
    with open(args.input) as f:
        signal_data = json.load(f)
    signals = signal_data['signals']
    active = [s for s in signals if s.get('lots', 0) > 0]
    log.info(f"Loaded {len(active)} active signals from {args.input}")

    # Dry run: just print what would happen
    if args.mode == 'dry':
        print(f"\n{'='*60}")
        print(f"{'DRY RUN — Orders that would be placed':^60}")
        print(f"{'='*60}")
        print(f"\n  {'Product':<15} {'TqSdk Symbol':<25} {'Dir':<6} {'Lots':>5}")
        print(f"  {'-'*55}")
        for s in active:
            sym = address_to_tq_symbol(s['product'])
            print(f"  {s['product']:<15} {sym or 'UNKNOWN':<25} {s['direction']:<6} {s.get('lots', 0):>5}")
        print(f"\n  Total: {len(active)} positions")
        print(f"{'='*60}")
        return

    # TqSdk auth
    tq_user = os.getenv('TIANQIN_USER', '')
    tq_pass = os.getenv('TIANQIN_PASSWORD', '')
    if not tq_user or not tq_pass:
        log.error("TIANQIN_USER and TIANQIN_PASSWORD not set in .env")
        sys.exit(1)

    auth = TqAuth(tq_user, tq_pass)

    # Create account
    account = None
    if args.mode == 'sim':
        log.info("Mode: TqKq simulation (快期模拟盘)")
        account = TqKq()
        api = TqApi(account, auth=auth)
    elif args.mode == 'live':
        if not args.broker or not args.td_account:
            log.error("Live mode requires --broker, --td-account, --td-password")
            sys.exit(1)
        log.info(f"Mode: LIVE trading ({args.broker})")
        log.warning("⚠ REAL MONEY MODE — orders will be executed!")
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            log.info("Aborted.")
            return
        account = TqAccount(args.broker, args.td_account, args.td_password)
        api = TqApi(account, auth=auth)

    try:
        # Wait for initial data
        api.wait_update()

        # Print account info
        router = OrderRouter(api, account=account, timeout=args.timeout)
        acc_info = router.get_account_info()
        log.info(f"\nAccount: balance={acc_info['balance']:,.2f}, "
                 f"available={acc_info['available']:,.2f}, "
                 f"margin={acc_info['margin']:,.2f}, "
                 f"risk_ratio={acc_info['risk_ratio']:.2%}")

        # Execute
        results = router.execute_signals(signals)

        # Save execution log
        exec_log = {
            'timestamp': datetime.now().isoformat(),
            'mode': args.mode,
            'account_info': acc_info,
            'orders': [{k: v for k, v in o.items() if k != 'symbol'} for o in results],
            'n_executed': len([o for o in results if o.get('delta', 0) != 0]),
        }
        log_path = os.path.join(os.path.dirname(args.input), 'execution_log.json')
        with open(log_path, 'w') as f:
            json.dump(exec_log, f, indent=2, ensure_ascii=False)
        log.info(f"\nExecution log saved to {log_path}")

    finally:
        api.close()
        log.info("API closed")


if __name__ == '__main__':
    main()
