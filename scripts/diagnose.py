"""
诊断脚本：逐层排查训练表现差的根因
Layer 1: 数据质量
Layer 2: 因子信号质量
Layer 3: 回测机制是否合理
Layer 4: RL 搜索效率
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import math

# ========== Layer 1: 数据质量 ==========
print("=" * 72)
print("LAYER 1: 数据质量诊断")
print("=" * 72)

from model_core.duckdb_loader import DuckDBDataLoader
loader = DuckDBDataLoader()
loader.load_data()

feat = loader.feat_tensor  # [N, 12, T]
raw = loader.raw_data_cache
target_ret = loader.target_ret
N, F, T = feat.shape
print(f"  品种数: {N}, 特征数: {F}, 时间步: {T}")
print(f"  训练步数: {loader.train_feat.shape[2]}, 测试步数: {loader.test_feat.shape[2]}")

# 检查收益率分布
ret = raw['close'][:, 1:] / (raw['close'][:, :-1] + 1e-9) - 1
ret_flat = ret.flatten().cpu().numpy()
ret_flat = ret_flat[np.isfinite(ret_flat)]
print(f"\n  日收益率统计:")
print(f"    均值:   {np.mean(ret_flat):.6f}")
print(f"    中位数: {np.median(ret_flat):.6f}")
print(f"    标准差: {np.std(ret_flat):.6f}")
print(f"    偏度:   {float(torch.tensor(ret_flat).float().mean() / (torch.tensor(ret_flat).float().std() + 1e-8)):.4f}")
print(f"    最大:   {np.max(ret_flat):.4f}")
print(f"    最小:   {np.min(ret_flat):.4f}")
print(f"    >0 占比: {np.mean(ret_flat > 0):.2%}")

# 截面相关性（品种间同步性）
close = raw['close']
ret_matrix = close[:, 1:] / (close[:, :-1] + 1e-9) - 1  # [N, T-1]
# 取最后200天看品种间相关性
sample = ret_matrix[:, -200:].cpu().numpy()
# 处理 NaN/Inf
sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
cross_corr = np.corrcoef(sample)
cross_corr = np.nan_to_num(cross_corr, nan=0.0)
upper_tri = cross_corr[np.triu_indices_from(cross_corr, k=1)]
print(f"\n  品种间收益相关性 (最近200天):")
print(f"    平均相关: {np.mean(upper_tri):.4f}")
print(f"    中位相关: {np.median(upper_tri):.4f}")
print(f"    >0.5 占比: {np.mean(upper_tri > 0.5):.2%}")
print(f"    >0.8 占比: {np.mean(upper_tri > 0.8):.2%}")

# target_ret 分布
tr = target_ret.flatten().cpu().numpy()
tr = tr[~np.isnan(tr)]
print(f"\n  target_ret (模型用的收益率):")
print(f"    均值:   {np.mean(tr):.6f}")
print(f"    标准差: {np.std(tr):.6f}")
print(f"    被 clamp 到 0 的比例: {np.mean(np.abs(tr) < 1e-8):.2%}")

# 检查 liquidity 分布
liq = raw['liquidity'].flatten().cpu().numpy()
print(f"\n  流动性 (close_oi) 分布:")
print(f"    均值:   {np.mean(liq):.0f}")
print(f"    中位数: {np.median(liq):.0f}")
print(f"    为 0 的比例: {np.mean(liq == 0):.2%}")
print(f"    < 100 的比例: {np.mean(liq < 100):.2%}")

# ========== Layer 2: 因子信号质量 ==========
print(f"\n{'=' * 72}")
print("LAYER 2: 因子信号质量诊断")
print("=" * 72)

feature_names = ['RET', 'LIQ', 'PRESSURE', 'FOMO', 'DEV', 'LOG_VOL',
                 'VOL_CLUSTER', 'MOM_REV', 'REL_STR', 'HL_RANGE', 'CLOSE_POS', 'VOL_TREND']

# 单因子 IC 测试（截面 Rank IC）
print(f"\n  单因子截面 Rank IC（全样本）:")
print(f"  {'因子':<14} {'Mean IC':>9} {'Std IC':>9} {'IC/Std':>9} {'>0 占比':>9}")
print(f"  {'-' * 55}")

for i, name in enumerate(feature_names):
    factor = feat[:, i, :]  # [N, T]
    ret_next = target_ret  # [N, T]

    # 每个时间步计算截面 Rank IC (手动 Spearman)
    ics = []
    for t in range(factor.shape[1]):
        f_col = factor[:, t].cpu().numpy()
        r_col = ret_next[:, t].cpu().numpy()

        mask = np.isfinite(f_col) & np.isfinite(r_col)
        if mask.sum() < 5:
            continue
        f_col = f_col[mask]
        r_col = r_col[mask]

        # 手动 Spearman: rank then Pearson
        f_rank = np.argsort(np.argsort(f_col)).astype(float)
        r_rank = np.argsort(np.argsort(r_col)).astype(float)
        f_dm = f_rank - f_rank.mean()
        r_dm = r_rank - r_rank.mean()
        denom = np.sqrt((f_dm**2).sum() * (r_dm**2).sum())
        if denom > 1e-8:
            corr = (f_dm * r_dm).sum() / denom
            ics.append(corr)

    if ics:
        ics = np.array(ics)
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        icir = mean_ic / (std_ic + 1e-8)
        pos_pct = np.mean(ics > 0)
        print(f"  {name:<14} {mean_ic:>9.4f} {std_ic:>9.4f} {icir:>9.4f} {pos_pct:>8.1%}")

# 单因子时序自相关
print(f"\n  单因子时序自相关 (lag-1):")
print(f"  {'因子':<14} {'Mean AC':>9}")
print(f"  {'-' * 25}")
for i, name in enumerate(feature_names):
    factor = feat[:, i, :]
    ac = torch.corrcoef(torch.stack([factor[:, 1:].flatten(), factor[:, :-1].flatten()]))[0, 1].item()
    print(f"  {name:<14} {ac:>9.4f}")


# ========== Layer 3: 回测机制诊断 ==========
print(f"\n{'=' * 72}")
print("LAYER 3: 回测机制诊断")
print("=" * 72)

from model_core.backtest import MemeBacktest
from model_core.vm import StackVM
from model_core.config import TimeframeProfile, ModelConfig

profile = TimeframeProfile(ModelConfig.TIMEFRAME, ModelConfig.ASSET_CLASS)
print(f"  TimeframeProfile: timeframe={ModelConfig.TIMEFRAME}, asset={ModelConfig.ASSET_CLASS}")
print(f"    annualization:    {profile.annualization:.2f}")
print(f"    base_fee:         {profile.base_fee}")
print(f"    min_liq:          {profile.min_liq}")
print(f"    trade_size:       {profile.trade_size}")
print(f"    target_turnover:  {profile.target_turnover:.6f}")
print(f"    ret_clamp:        {profile.ret_clamp}")

bt = MemeBacktest(position_mode='rank')
liq = raw['liquidity']
is_safe = (liq > bt.min_liq).float()
safe_pct = is_safe.mean().item()
print(f"\n  流动性安全率: {safe_pct:.2%} (> min_liq={bt.min_liq})")

# 测试：用纯 RET 因子做多 top 40% 的表现
print(f"\n  基准测试: 纯收益率动量 (RET) top-40% 做多:")
ret_factor = feat[:, 0, :]  # RET
position = bt.compute_position(ret_factor, is_safe)

impact = torch.clamp(bt.trade_size / (liq + 1e-9), 0.0, 0.05)
total_slip = bt.base_fee + impact
prev_pos = torch.cat([torch.zeros_like(position[:, :1]), position[:, :-1]], dim=1)
turnover = torch.abs(position - prev_pos)
tx_cost = turnover * total_slip
net_pnl = position * target_ret - tx_cost

daily_pnl = net_pnl.mean(dim=0).cpu().numpy()
sharpe = (np.mean(daily_pnl) / (np.std(daily_pnl) + 1e-8)) * profile.annualization
print(f"    Sharpe (long-only top40%): {sharpe:.4f}")
print(f"    累计 PnL: {np.sum(daily_pnl):.6f}")
print(f"    平均日 turnover: {turnover.mean().item():.6f}")
print(f"    平均日 tx_cost: {tx_cost.mean().item():.6f}")
print(f"    平均日 gross_pnl: {(position * target_ret).mean().item():.6f}")

# 测试：纯 RET 因子做多空（top 20% 做多，bottom 20% 做空）
print(f"\n  对照: 纯 RET 因子做多空 (long top20% - short bottom20%):")
ranks = ret_factor.argsort(dim=0).argsort(dim=0).float()
rank_pct = ranks / (N - 1)
long_pos = torch.clamp((rank_pct - 0.8) / 0.2, 0.0, 1.0) * is_safe
short_pos = torch.clamp((0.2 - rank_pct) / 0.2, 0.0, 1.0) * is_safe
ls_position = long_pos - short_pos

prev_ls = torch.cat([torch.zeros_like(ls_position[:, :1]), ls_position[:, :-1]], dim=1)
ls_turnover = torch.abs(ls_position - prev_ls)
ls_tx = ls_turnover * total_slip
ls_net = ls_position * target_ret - ls_tx
ls_daily = ls_net.mean(dim=0).cpu().numpy()
ls_sharpe = (np.mean(ls_daily) / (np.std(ls_daily) + 1e-8)) * profile.annualization
print(f"    Sharpe (long-short): {ls_sharpe:.4f}")
print(f"    累计 PnL: {np.sum(ls_daily):.6f}")

# 测试: 完美前瞻因子 (用未来收益做排序) 的上限
print(f"\n  天花板测试: 完美前瞻因子 (用未来收益排序):")
perfect_factor = target_ret  # 用真实未来收益做因子
perfect_pos = bt.compute_position(perfect_factor, is_safe)
prev_pp = torch.cat([torch.zeros_like(perfect_pos[:, :1]), perfect_pos[:, :-1]], dim=1)
pp_turnover = torch.abs(perfect_pos - prev_pp)
pp_tx = pp_turnover * total_slip
pp_net = perfect_pos * target_ret - pp_tx
pp_daily = pp_net.mean(dim=0).cpu().numpy()
pp_sharpe = (np.mean(pp_daily) / (np.std(pp_daily) + 1e-8)) * profile.annualization
print(f"    Sharpe (做多 perfect): {pp_sharpe:.4f}")
print(f"    累计 PnL: {np.sum(pp_daily):.6f}")

# 完美前瞻 + 多空
pp_long = torch.clamp((perfect_factor.argsort(dim=0).argsort(dim=0).float() / (N - 1) - 0.8) / 0.2, 0.0, 1.0) * is_safe
pp_short = torch.clamp((0.2 - perfect_factor.argsort(dim=0).argsort(dim=0).float() / (N - 1)) / 0.2, 0.0, 1.0) * is_safe
pp_ls = pp_long - pp_short
prev_ppls = torch.cat([torch.zeros_like(pp_ls[:, :1]), pp_ls[:, :-1]], dim=1)
ppls_to = torch.abs(pp_ls - prev_ppls)
ppls_tx = ppls_to * total_slip
ppls_net = pp_ls * target_ret - ppls_tx
ppls_daily = ppls_net.mean(dim=0).cpu().numpy()
ppls_sharpe = (np.mean(ppls_daily) / (np.std(ppls_daily) + 1e-8)) * profile.annualization
print(f"    Sharpe (多空 perfect): {ppls_sharpe:.4f}")
print(f"    累计 PnL: {np.sum(ppls_daily):.6f}")


# ========== Layer 4: RL 搜索效率 ==========
print(f"\n{'=' * 72}")
print("LAYER 4: RL 搜索效率诊断")
print("=" * 72)

vm = StackVM()

# 随机采样 1000 条合法公式，看 reward 分布
print(f"\n  随机合法公式 reward 分布 (1000 条):")
from model_core.factors import FeatureEngineer
from model_core.ops import OPS_CONFIG
import random

n_features = FeatureEngineer.INPUT_DIM
n_ops = len(OPS_CONFIG)
vocab_size = n_features + n_ops
op_arities = [cfg[2] for cfg in OPS_CONFIG]

random.seed(42)
valid_formulas = []
max_len = 12

attempts = 0
while len(valid_formulas) < 1000 and attempts < 50000:
    attempts += 1
    stack_size = 0
    formula = []
    for step in range(max_len):
        remaining = max_len - 1 - step
        candidates = []
        for token in range(vocab_size):
            if token < n_features:
                arity = 0
                delta = 1
            else:
                arity = op_arities[token - n_features]
                delta = 1 - arity
            new_stack = stack_size + delta
            if stack_size < arity:
                continue
            if new_stack + remaining < 1:
                continue
            if new_stack - 2 * remaining > 1:
                continue
            if remaining == 0 and new_stack != 1:
                continue
            candidates.append(token)
        if not candidates:
            break
        token = random.choice(candidates)
        formula.append(token)
        if token < n_features:
            stack_size += 1
        else:
            stack_size = stack_size - op_arities[token - n_features] + 1

    if stack_size == 1 and len(formula) == max_len:
        valid_formulas.append(formula)

print(f"  生成 {len(valid_formulas)} 条合法公式 (尝试 {attempts} 次, 成功率 {len(valid_formulas)/attempts:.1%})")

# 评估每条公式
train_feat = loader.train_feat
train_raw = loader.train_raw
train_ret = loader.train_ret
base_factors = train_feat[:, 0, :]

rewards = []
valid_exec = 0
for formula in valid_formulas[:500]:
    res = vm.execute(formula, train_feat)
    if res is not None and res.std() > 1e-4:
        valid_exec += 1
        score, _ = bt.evaluate(res, train_raw, train_ret,
                               formula_length=len([t for t in formula if t != 0]),
                               base_factors=base_factors)
        rewards.append(score.item())
    else:
        rewards.append(-5.0)

rewards = np.array(rewards)
print(f"  有效执行率: {valid_exec}/{min(500, len(valid_formulas))} = {valid_exec/min(500, len(valid_formulas)):.1%}")
print(f"  Reward 分布:")
print(f"    均值:     {np.mean(rewards):.4f}")
print(f"    中位数:   {np.median(rewards):.4f}")
print(f"    最大:     {np.max(rewards):.4f}")
print(f"    > -3 占比: {np.mean(rewards > -3):.2%}")
print(f"    > 0 占比:  {np.mean(rewards > 0):.2%}")

# Reward 分布直方图
bins = [-10, -5, -4, -3, -2, -1, 0, 1, 2, 5]
hist, _ = np.histogram(rewards, bins=bins)
print(f"\n  Reward 直方图:")
for i in range(len(bins) - 1):
    bar = "█" * (hist[i] * 50 // max(hist.max(), 1))
    print(f"    [{bins[i]:>4}, {bins[i+1]:>3}): {hist[i]:>4} {bar}")

print(f"\n{'=' * 72}")
print("诊断完成")
print("=" * 72)
