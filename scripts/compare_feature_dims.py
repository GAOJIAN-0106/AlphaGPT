"""
A/B comparison: 18-dim (V2) vs 21-dim (V3 with term structure) features.

Trains 3 seeds each (faster than full 6-seed ensemble) and compares:
  - Best single Sharpe
  - Ensemble Sharpe
  - IC of new term-structure factors

Usage:
    python scripts/compare_feature_dims.py
"""

import os
import sys
import json
import gc
import torch
import numpy as np

sys.path.insert(0, ".")

os.environ['MODEL_ASSET_CLASS'] = 'futures'


def train_and_evaluate(enable_ts, num_seeds=3, label=""):
    """Train ensemble with given feature config and return results."""
    os.environ['ENABLE_TERM_STRUCTURE'] = '1' if enable_ts else '0'

    # Force reload of config and modules
    import importlib
    import model_core.config
    importlib.reload(model_core.config)
    import model_core.vm
    importlib.reload(model_core.vm)

    from model_core.config import ModelConfig
    from model_core.engine import AlphaEngine, _build_masking_tensors
    import model_core.engine as eng_mod

    # Rebuild masking tensors for new feature dim
    eng_mod._ARITY, eng_mod._STACK_DELTA = _build_masking_tensors()

    feat_dim = ModelConfig.get_feature_dim()
    print(f"\n{'#'*70}")
    print(f"  {label}: feature_dim={feat_dim}, seeds={num_seeds}")
    print(f"{'#'*70}")

    ensemble = AlphaEngine.train_ensemble(
        num_seeds=num_seeds,
        use_lord_regularization=True,
        lord_decay_rate=1e-3,
    )

    # Load the saved result
    with open("best_ensemble.json") as f:
        result = json.load(f)

    # Rename output file to avoid overwrite
    suffix = "21dim" if enable_ts else "18dim"
    out_path = f"best_ensemble_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_path}")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    num_seeds = 3  # Use 3 seeds for faster comparison

    # --- 18-dim baseline ---
    result_18 = train_and_evaluate(
        enable_ts=False, num_seeds=num_seeds, label="BASELINE (18-dim V2)")

    # --- 21-dim with term structure ---
    result_21 = train_and_evaluate(
        enable_ts=True, num_seeds=num_seeds, label="NEW (21-dim V3 + term structure)")

    # --- Comparison ---
    print(f"\n{'='*70}")
    print(f"{'FEATURE DIMENSION A/B COMPARISON':^70}")
    print(f"{'='*70}")

    def extract_metrics(result):
        sharpes = [sr.get('test_sharpe', 0) or 0
                   for sr in result.get('seed_results', [])]
        return {
            'ensemble_sharpe': result.get('ensemble_test_sharpe', 0) or 0,
            'ensemble_maxdd': result.get('ensemble_max_drawdown', 0) or 0,
            'ensemble_turnover': result.get('ensemble_avg_turnover', 0) or 0,
            'best_single': max(sharpes) if sharpes else 0,
            'avg_single': np.mean(sharpes) if sharpes else 0,
            'num_valid': result.get('num_valid_formulas', 0),
        }

    m18 = extract_metrics(result_18)
    m21 = extract_metrics(result_21)

    print(f"\n  {'Metric':<25} {'18-dim (V2)':>12} {'21-dim (V3)':>12} {'Delta':>10}")
    print(f"  {'-'*62}")

    for key, label in [
        ('ensemble_sharpe', 'Ensemble Sharpe'),
        ('best_single', 'Best Single Sharpe'),
        ('avg_single', 'Avg Single Sharpe'),
        ('ensemble_maxdd', 'Max Drawdown'),
        ('ensemble_turnover', 'Avg Turnover'),
        ('num_valid', 'Valid Formulas'),
    ]:
        v18 = m18[key]
        v21 = m21[key]
        delta = v21 - v18
        if key == 'ensemble_maxdd':
            print(f"  {label:<25} {v18:>11.2%} {v21:>11.2%} {delta:>+9.2%}")
        elif key == 'num_valid':
            print(f"  {label:<25} {v18:>12d} {v21:>12d} {delta:>+10d}")
        else:
            print(f"  {label:<25} {v18:>12.4f} {v21:>12.4f} {delta:>+10.4f}")

    # Verdict
    print(f"\n  {'VERDICT':}")
    sharpe_delta = m21['ensemble_sharpe'] - m18['ensemble_sharpe']
    if sharpe_delta > 0.1:
        print(f"  21-dim WINS by {sharpe_delta:+.4f} Sharpe — term structure adds value")
    elif sharpe_delta < -0.1:
        print(f"  18-dim WINS by {-sharpe_delta:+.4f} Sharpe — term structure may add noise")
    else:
        print(f"  INCONCLUSIVE (delta={sharpe_delta:+.4f}) — need more seeds for statistical power")

    print(f"{'='*70}")

    # Save comparison
    comparison = {
        '18dim': m18, '21dim': m21,
        'sharpe_delta': sharpe_delta,
        'num_seeds': num_seeds,
    }
    with open("feature_dim_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Comparison saved to feature_dim_comparison.json")


if __name__ == "__main__":
    main()
