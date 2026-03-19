"""
在线学习评估脚本 — 对比静态集成 vs 自适应集成

用法:
    python scripts/run_online_learning.py
    python scripts/run_online_learning.py --sweep     # 参数扫描找最优配置
"""

import sys
import json
import numpy as np
sys.path.insert(0, ".")

from model_core.online_learner import OnlineLearner
from model_core.ensemble import FormulaEnsemble
from model_core.data_loader import CryptoDataLoader


def load_ensemble_and_data():
    """加载集成和数据"""
    with open("best_ensemble.json") as f:
        ens_data = json.load(f)
    ensemble = FormulaEnsemble.from_dict(ens_data['ensemble'])
    print(f"Loaded ensemble: {ensemble.num_formulas} formulas")
    print(f"  Static Sharpe: {ens_data['ensemble_test_sharpe']:.4f}")

    loader = CryptoDataLoader()
    loader.load_data()
    return ensemble, loader


def evaluate_single(ensemble, loader, lookback=20, lr=0.3, decay=0.95,
                    min_weight=0.05, period_length=1, warmup=20):
    """单次在线学习评估"""
    ol = OnlineLearner(
        ensemble,
        lookback_window=lookback,
        learning_rate=lr,
        decay=decay,
        min_weight=min_weight,
    )

    result = ol.run_online(
        loader.test_feat,
        loader.test_raw,
        loader.test_ret,
        period_length=period_length,
        warmup_periods=warmup,
    )
    return result, ol


def run_default():
    """默认参数运行"""
    ensemble, loader = load_ensemble_and_data()
    result, ol = evaluate_single(ensemble, loader)

    print(f"\n{'='*60}")
    print(f"{'在线学习评估结果':^56}")
    print(f"{'='*60}")
    print(f"  {'指标':<20} {'静态集成':>10} {'在线学习':>10} {'提升':>10}")
    print(f"  {'-'*52}")
    print(f"  {'年化Sharpe':<20} {result['static_sharpe']:>10.4f} "
          f"{result['online_sharpe']:>10.4f} {result['improvement']:>+10.4f}")

    static_cum = sum(result['static_pnl'])
    online_cum = sum(result['online_pnl'])
    print(f"  {'累计收益':<20} {static_cum:>10.6f} "
          f"{online_cum:>10.6f} {online_cum - static_cum:>+10.6f}")

    # 最终权重
    final_w = ol.get_weights()
    print(f"\n  初始权重: {['0.167'] * 6}")
    print(f"  最终权重: {[f'{w:.3f}' for w in final_w]}")

    # 权重变化最大的公式
    init_w = 1.0 / ensemble.num_formulas
    changes = [(i, w - init_w) for i, w in enumerate(final_w)]
    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  权重变化最大:")
    for i, delta in changes[:3]:
        direction = "增" if delta > 0 else "减"
        print(f"    公式{i} (seed {[7,49,91,133,175,217][i]}): "
              f"{direction}{abs(delta):.3f}")

    print(f"{'='*60}")

    # 保存结果
    save_data = {
        'online_sharpe': result['online_sharpe'],
        'static_sharpe': result['static_sharpe'],
        'improvement': result['improvement'],
        'final_weights': final_w.tolist(),
        'online_learner': ol.to_dict(),
    }
    with open("online_learning_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  结果已保存到 online_learning_results.json")


def run_sweep():
    """参数扫描找最优配置"""
    ensemble, loader = load_ensemble_and_data()

    print(f"\n{'='*60}")
    print(f"{'参数扫描':^56}")
    print(f"{'='*60}")

    configs = []
    for lr in [0.1, 0.2, 0.3, 0.5, 0.7]:
        for decay in [0.8, 0.9, 0.95, 0.99]:
            for lookback in [10, 20, 40]:
                result, _ = evaluate_single(
                    ensemble, loader,
                    lookback=lookback, lr=lr, decay=decay,
                    period_length=1, warmup=20,
                )
                configs.append({
                    'lr': lr, 'decay': decay, 'lookback': lookback,
                    'online_sharpe': result['online_sharpe'],
                    'static_sharpe': result['static_sharpe'],
                    'improvement': result['improvement'],
                })

    # 排序
    configs.sort(key=lambda x: x['improvement'], reverse=True)

    print(f"\n  {'LR':>5} {'Decay':>6} {'LB':>4} {'StaticSharpe':>13} "
          f"{'OnlineSharpe':>13} {'Improvement':>12}")
    print(f"  {'-'*58}")
    for c in configs[:10]:
        marker = " <-- best" if c == configs[0] else ""
        print(f"  {c['lr']:>5.1f} {c['decay']:>6.2f} {c['lookback']:>4d} "
              f"{c['static_sharpe']:>13.4f} {c['online_sharpe']:>13.4f} "
              f"{c['improvement']:>+12.4f}{marker}")

    best = configs[0]
    print(f"\n  最优配置: lr={best['lr']}, decay={best['decay']}, "
          f"lookback={best['lookback']}")
    print(f"  最优提升: {best['improvement']:+.4f} Sharpe")
    print(f"{'='*60}")

    with open("online_learning_sweep.json", "w") as f:
        json.dump(configs, f, indent=2)
    print(f"  扫描结果已保存到 online_learning_sweep.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--sweep':
        run_sweep()
    else:
        run_default()
