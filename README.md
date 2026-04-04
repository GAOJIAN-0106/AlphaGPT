# AlphaGPT

基于 Transformer 的自动因子挖掘量化交易系统。生成可解释的因子公式，通过强化学习回测优化，支持商品期货、A 股等多市场部署。

> Fork from [imbue-bit/AlphaGPT](https://github.com/imbue-bit/AlphaGPT)

## 最新性能指标

**商品期货 20 因子集成策略（2025-04 → 2026-03，Walk-Forward Replay）**

| 指标 | 固定 Baseline | 事件驱动更新 |
|------|:---:|:---:|
| 累计收益 | +43.5% | **+47.9%** |
| 年化 Sharpe | 3.58 | **3.79** |
| 胜率 | 59% | **60%** |
| 最大回撤 | -6.09% | **-5.95%** |
| 更新次数 | 0 | 0 |

**3 年回测（2023-06 → 2026-03，事件驱动更新）**

| 指标 | 值 |
|------|-----|
| 累计收益 | +73.2% |
| 年化 Sharpe | 2.45 |
| 胜率 | 55% |
| 最大回撤 | -6.35% |
| 更新触发次数 | 1 次（2023-12-16，Level 3） |

## 系统架构

```
数据层                    因子层                   模型层                    交易层
─────                    ─────                   ─────                    ─────
DuckDB (1min K线)   →  78 候选因子            →  6 公式 Ensemble      →  Daily Pipeline
天勤 EDB (基本面)   →  IC 筛选 → 20 因子     →  MWU 在线学习         →  TopN 风控
会员持仓排名        →  ICIR>0.05 + 去相关     →  事件驱动更新         →  Dry/Sim/Live
```

### 因子体系（78 → 20）

| 类别 | 数量 | 示例 |
|------|------|------|
| 日内微观结构 | 5 | TWAP_DEV, CLOSE_POS, SMART_MONEY |
| 波动率 | 4 | INTRADAY_CVAR, VOL_TREND, VOL_SKEW, IVOL |
| 期限结构 | 2 | TERM_SPREAD, BASIS_MOM_LOG |
| 行为/情绪 | 2 | PROSPECT_TK, SALIENCE_RET |
| 价格形态 | 2 | HL_RANGE, REL_STR |
| 流动性 | 1 | AMIHUD_ILLIQ |
| 订单流 | 2 | LARGE_ORDER, NET_SUPPORT_VOL |
| 基本面 | 1 | TGD |
| 隔夜 | 1 | AB_NIGHT_REV |

### 事件驱动更新（WFR）

```
每天：计算 60 日滚动 Sharpe

Sharpe > 0     → 不做任何改变（策略健康）
连续 10 天 < 0 → 触发更新：
  Level 2 (Sharpe -0.5 ~ -1.5): 换 1 个因子 + 替换 1 个最差公式
  Level 3 (Sharpe < -1.5):      换 2 个因子 + 替换所有 Sharpe<0 的公式

因子总数始终保持 20 个（token 映射不变，好公式继续有效）
```

## 更新日志

### 2026-04-04

**基本面因子体系**
- 新增 EDB 数据接入：天勤经济数据库 625 个指标，覆盖仓单、库存、现货价、成本等
- 新增 15+ 基本面/持仓因子：WAREHOUSE_CHG、INVENTORY_MOM、MEMBER_LS、LONG_CONC_STD、LS_STRENGTH 等
- 新增 EDB ID → 期货品种映射配置（`model_core/edb_config.py`）
- 新增数据拉取脚本：`fetch_member_positions.py`、`fetch_edb_warehouse.py`、`fetch_edb_inventory.py`

**20 因子优化**
- 基于 ICIR 衰减分析确定最优因子数 = 20（平均 ICIR 0.134，第 28 名后骤降）
- IC 筛选标准统一：ICIR > 0.05 + 层次聚类去相关（|corr| > 0.7 合并）
- `config.get_feature_dim()` 改为动态读取 `FEATURES_V3_LIST` 长度

**事件驱动 WFR 回测**
- 新增 `scripts/wfr_quarterly.py`：事件驱动更新回测
- 对比三种方案：全量季度更新（Sharpe 0.70） < 渐进季度更新（1.86） < 事件驱动（3.79）
- 3 年回测验证：807 个交易日仅触发 1 次更新，年化 Sharpe 2.45

**Walk-Forward Replay 回测框架**
- 新增 `scripts/backtest_replay.py`：逐日前推，与 live pipeline 完全一致
- 支持 HMM regime 检测、MWU 在线学习、TopN 风控
- 新增 `scripts/update_duckdb.py`：增量更新 DuckDB 行情数据

**EWMA / BREAKOUT 算子**
- 新增 EWMA5、EWMA10、BREAKOUT 算子到 OPS_CONFIG_EXTENDED
- Ensemble 搜索可自动发现并使用新算子

**Bug 修复**
- 修复 VM 输出 squeeze 无限循环（`[N, 2]` shape 时 `squeeze(-1)` 不降维）
- 修复 `factor_registry.compute_group()` 品种数不一致时的 tensor 对齐
- 修复 `ensemble.predict()` 多公式 shape 不一致的 stack 报错
- 修复 DuckDB loader 的 product filter SQL 在子 CTE 中引用错误别名

**清理**
- 删除 `simulate_live.py`、`simulate_live_v2.py`（被 `backtest_replay.py` 替代）
- 删除旧回测结果文件，更新 `.gitignore`

### 2025-02-22

**A 股数据源支持**
- 新增 TushareProvider，支持 HS300 / ZZ500 / ZZ1000 股票池
- DataManager 引入工厂模式，`--mode astock/solana` 切换数据源
- 新增 `.env.example` 环境变量模板

**模型核心增强**
- 新增 FormulaEnsemble 集成策略（加权平均 / 排名均值）
- 新增 TimeSeriesCV 时序交叉验证（expanding / rolling window）
- 新增 WandbTracker 实验追踪
- AlphaEngine 引入 LoRD 正则化和 reward shaping 优化
- MemeBacktest 增加分年分析、因子自相关等高级指标

**测试与文档**
- 新增测试套件覆盖 reward / ensemble / temporal CV / tracking
- 新增 Jupyter 学习指南

## 快速开始

```bash
git clone https://github.com/GAOJIAN-0106/AlphaGPT.git
cd AlphaGPT
pip install -r requirements.txt
cp .env.example .env  # 编辑填入数据库密码、API Key
```

## License

[Apache License 2.0](LICENSE)
