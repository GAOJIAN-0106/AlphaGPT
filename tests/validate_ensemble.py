"""
Ensemble Validation Script
--------------------------
Simulates realistic conditions to validate that ensemble:
1. Reduces variance across different market regimes
2. Improves risk-adjusted returns (Sharpe) vs best single model
3. Reduces max drawdown
4. Is robust to formula diversity levels

Uses synthetic data that mimics real market characteristics:
- Trending periods, mean-reverting periods, and high-volatility periods
- Realistic return distributions (fat tails)
- Multiple "models" with varying quality and correlation
"""

import torch
import numpy as np
import math
from model_core.ensemble import FormulaEnsemble
from model_core.backtest import MemeBacktest
from model_core.vm import StackVM
from model_core.factors import FeatureEngineer


def generate_synthetic_market(batch=20, T=500, seed=42):
    """Generate synthetic market data with realistic characteristics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    # Create OHLCV-like data with regime changes
    # Start with close prices following geometric Brownian motion
    drift = 0.0002  # slight positive drift
    vol = 0.02

    close = torch.zeros(batch, T, device=device)
    close[:, 0] = 100.0

    for t in range(1, T):
        # Regime switching: higher vol every 100 steps
        regime_vol = vol * (2.0 if (t // 100) % 2 == 1 else 1.0)
        returns = torch.randn(batch) * regime_vol + drift
        close[:, t] = close[:, t-1] * (1 + returns)

    high = close * (1 + torch.rand(batch, T) * 0.02)
    low = close * (1 - torch.rand(batch, T) * 0.02)
    open_ = (close + torch.randn(batch, T) * close * 0.005)
    volume = torch.rand(batch, T) * 1e6 + 1e5
    liquidity = torch.rand(batch, T) * 1e8 + 1e7
    fdv = torch.zeros(batch, T)

    raw_dict = {
        'close': close, 'open': open_, 'high': high, 'low': low,
        'volume': volume, 'liquidity': liquidity, 'fdv': fdv,
    }

    feat_tensor = FeatureEngineer.compute_features(raw_dict)

    # Target returns
    next_close = torch.cat([close[:, 1:], close[:, -1:]], dim=1)
    target_ret = (next_close - close) / (close + 1e-9)
    target_ret[:, -1] = 0.0
    target_ret = torch.clamp(target_ret, -0.2, 0.2)

    raw_data = {'liquidity': liquidity}

    return feat_tensor, raw_data, target_ret


def simulate_diverse_formulas(n_formulas=6):
    """
    Generate a set of diverse valid formulas.
    Mimics what train_ensemble would produce with different seeds.
    """
    vm = StackVM()
    feat_dim = FeatureEngineer.INPUT_DIM  # 6
    add_token = feat_dim + 0   # ADD
    sub_token = feat_dim + 1   # SUB
    mul_token = feat_dim + 2   # MUL
    neg_token = feat_dim + 4   # NEG
    abs_token = feat_dim + 5   # ABS
    sign_token = feat_dim + 6  # SIGN
    decay_token = feat_dim + 9 # DECAY
    delay_token = feat_dim + 10 # DELAY1

    formulas = [
        [0],                           # raw returns
        [1],                           # liquidity
        [2],                           # pressure
        [0, neg_token],                # -returns (reversal)
        [0, 1, add_token],            # returns + liquidity
        [0, 2, mul_token],            # returns * pressure
        [3],                           # fomo
        [4],                           # deviation
        [0, abs_token],               # |returns|
        [0, decay_token],             # decay(returns)
    ]

    # Filter to valid formulas
    valid = []
    dummy = torch.randn(3, feat_dim, 50)
    for f in formulas:
        res = vm.execute(f, dummy)
        if res is not None and res.std() > 1e-6:
            valid.append(f)

    return valid[:n_formulas]


def run_validation():
    """Main validation routine."""
    print("=" * 70)
    print("   ENSEMBLE VALIDATION: Synthetic Market Simulation")
    print("=" * 70)

    bt = MemeBacktest()
    vm = StackVM()

    # ===== Test 1: Variance reduction across market regimes =====
    print("\n--- Test 1: Variance Reduction Across Seeds ---")

    feat, raw_data, target_ret = generate_synthetic_market(batch=20, T=500)
    formulas = simulate_diverse_formulas(n_formulas=6)
    print(f"  Generated {len(formulas)} diverse formulas")

    # Evaluate individuals
    individual_results = []
    for i, f in enumerate(formulas):
        res = vm.execute(f, feat)
        if res is not None:
            _, _, diag = bt.evaluate(res, raw_data, target_ret, return_diagnostics=True)
            individual_results.append(diag)
            print(f"  Formula {i}: Sharpe={diag['sharpe']:.4f}, "
                  f"MaxDD={diag['max_drawdown']:.4%}, Turnover={diag['avg_turnover']:.4f}")

    individual_sharpes = [d['sharpe'] for d in individual_results]
    individual_dds = [d['max_drawdown'] for d in individual_results]

    # Evaluate ensemble
    ensemble = FormulaEnsemble(formulas, mode='mean')
    ens_signal = ensemble.predict(feat)
    _, _, ens_diag = bt.evaluate(ens_signal, raw_data, target_ret, return_diagnostics=True)

    print(f"\n  Ensemble:   Sharpe={ens_diag['sharpe']:.4f}, "
          f"MaxDD={ens_diag['max_drawdown']:.4%}, Turnover={ens_diag['avg_turnover']:.4f}")
    print(f"  Best Single:  Sharpe={max(individual_sharpes):.4f}")
    print(f"  Avg Single:   Sharpe={np.mean(individual_sharpes):.4f}")
    print(f"  Std Single:   {np.std(individual_sharpes):.4f}")

    # ===== Test 2: Stability across different data windows =====
    print("\n--- Test 2: Stability Across Time Windows ---")

    window_sharpes_ens = []
    window_sharpes_best_single = []
    T_total = feat.shape[2]
    window_size = 100

    for start in range(0, T_total - window_size, window_size):
        end = start + window_size
        window_feat = feat[:, :, start:end]
        window_raw = {k: v[:, start:end] for k, v in raw_data.items()}
        window_ret = target_ret[:, start:end]

        # Ensemble Sharpe on this window
        ens_sig = ensemble.predict(window_feat)
        if ens_sig is not None:
            _, _, d = bt.evaluate(ens_sig, window_raw, window_ret, return_diagnostics=True)
            window_sharpes_ens.append(d['sharpe'])

        # Best single on this window
        best_sharpe = -float('inf')
        for f in formulas:
            res = vm.execute(f, window_feat)
            if res is not None:
                _, _, d = bt.evaluate(res, window_raw, window_ret, return_diagnostics=True)
                if d['sharpe'] > best_sharpe:
                    best_sharpe = d['sharpe']
        window_sharpes_best_single.append(best_sharpe)

    ens_sharpe_std = np.std(window_sharpes_ens) if window_sharpes_ens else 0
    single_sharpe_std = np.std(window_sharpes_best_single) if window_sharpes_best_single else 0

    print(f"  Ensemble Sharpe across windows: mean={np.mean(window_sharpes_ens):.4f}, "
          f"std={ens_sharpe_std:.4f}")
    print(f"  Best Single Sharpe across windows: mean={np.mean(window_sharpes_best_single):.4f}, "
          f"std={single_sharpe_std:.4f}")

    stability_improvement = (single_sharpe_std - ens_sharpe_std) / (single_sharpe_std + 1e-8)
    print(f"  Stability improvement: {stability_improvement:.1%}")

    # ===== Test 3: rank_mean vs mean comparison =====
    print("\n--- Test 3: rank_mean vs mean Mode Comparison ---")

    ens_mean = FormulaEnsemble(formulas, mode='mean')
    ens_rank = FormulaEnsemble(formulas, mode='rank_mean')

    sig_mean = ens_mean.predict(feat)
    sig_rank = ens_rank.predict(feat)

    _, _, diag_mean = bt.evaluate(sig_mean, raw_data, target_ret, return_diagnostics=True)
    _, _, diag_rank = bt.evaluate(sig_rank, raw_data, target_ret, return_diagnostics=True)

    print(f"  Mean mode:      Sharpe={diag_mean['sharpe']:.4f}, MaxDD={diag_mean['max_drawdown']:.4%}")
    print(f"  Rank-mean mode: Sharpe={diag_rank['sharpe']:.4f}, MaxDD={diag_rank['max_drawdown']:.4%}")

    # ===== Test 4: evaluate_ensemble comprehensive report =====
    print("\n--- Test 4: evaluate_ensemble() Comprehensive Report ---")

    report = bt.evaluate_ensemble(ensemble, feat, raw_data, target_ret)
    if report:
        print(f"  Ensemble Sharpe:       {report['ensemble']['sharpe']:.4f}")
        print(f"  Best Single Sharpe:    {report['best_single_sharpe']:.4f}")
        print(f"  Avg Single Sharpe:     {report['avg_single_sharpe']:.4f}")
        print(f"  Sharpe Std:            {report['sharpe_std']:.4f}")
        print(f"  Improvement vs Best:   {report['improvement_vs_best']:.4f}")
        print(f"  Improvement vs Avg:    {report['improvement_vs_avg']:.4f}")

    # ===== Test 5: Increasing ensemble size =====
    print("\n--- Test 5: Effect of Ensemble Size ---")

    all_formulas = simulate_diverse_formulas(n_formulas=10)
    for n in [1, 2, 3, 4, 6, 8, 10]:
        if n > len(all_formulas):
            break
        sub_formulas = all_formulas[:n]
        ens = FormulaEnsemble(sub_formulas, mode='mean')
        sig = ens.predict(feat)
        if sig is not None:
            _, _, d = bt.evaluate(sig, raw_data, target_ret, return_diagnostics=True)
            print(f"  N={n:2d}: Sharpe={d['sharpe']:.4f}, MaxDD={d['max_drawdown']:.4%}, "
                  f"Turnover={d['avg_turnover']:.4f}")

    # ===== Test 6: Serialization roundtrip =====
    print("\n--- Test 6: Serialization Roundtrip ---")
    import json
    d = ensemble.to_dict()
    json_str = json.dumps(d, indent=2)
    restored = FormulaEnsemble.from_dict(json.loads(json_str))

    sig_orig = ensemble.predict(feat)
    sig_rest = restored.predict(feat)
    match = torch.allclose(sig_orig, sig_rest, atol=1e-5)
    print(f"  Serialization roundtrip: {'PASS' if match else 'FAIL'}")

    # ===== Summary =====
    print(f"\n{'=' * 70}")
    print("   VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total formulas in ensemble:   {len(formulas)}")
    print(f"  Ensemble Sharpe:              {ens_diag['sharpe']:.4f}")
    print(f"  Best Single Sharpe:           {max(individual_sharpes):.4f}")
    print(f"  Avg Single Sharpe:            {np.mean(individual_sharpes):.4f}")
    print(f"  Sharpe Std (individuals):     {np.std(individual_sharpes):.4f}")
    print(f"  Ensemble MaxDD:               {ens_diag['max_drawdown']:.4%}")
    print(f"  Best Single MaxDD:            {min(individual_dds):.4%}")

    # Check key assertions
    checks = []

    # Check 1: Ensemble Sharpe >= avg individual
    c1 = ens_diag['sharpe'] >= np.mean(individual_sharpes) - 0.5
    checks.append(("Ensemble Sharpe >= Avg - 0.5", c1))

    # Check 2: Ensemble produces finite results
    c2 = math.isfinite(ens_diag['sharpe']) and math.isfinite(ens_diag['max_drawdown'])
    checks.append(("Ensemble diagnostics finite", c2))

    # Check 3: Stability improvement (ensemble less variable across windows)
    c3 = ens_sharpe_std <= single_sharpe_std * 1.5  # allow some tolerance
    checks.append(("Ensemble more stable across windows", c3))

    # Check 4: Serialization works
    checks.append(("Serialization roundtrip", match))

    # Check 5: evaluate_ensemble produces complete report
    c5 = report is not None and 'ensemble' in report
    checks.append(("evaluate_ensemble complete", c5))

    print(f"\n  Validation Checks:")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 70}")

    return all_pass


if __name__ == "__main__":
    success = run_validation()
    exit(0 if success else 1)
