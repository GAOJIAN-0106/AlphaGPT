# AlphaGPT

基于 Transformer 的自动因子挖掘量化交易系统。生成可解释的因子公式，通过强化学习回测优化，支持商品期货、A 股等多市场部署。

> Fork from [imbue-bit/AlphaGPT](https://github.com/imbue-bit/AlphaGPT)

## 当前状态(2026-09)

> **目前没有经过验证的可交易 alpha。实盘决策:GO FLAT(不交易)。**

本页此前展示的回测指标(累计收益 +43.5% / +47.9% / +73.2%,年化 Sharpe 3.58 / 3.79 / 2.45 等)**已全部撤回**。它们来自后来查实被污染的管线,不代表任何可实现的表现:

- **换月幻影(2026-06)**:对按持仓量拼接的主力连续序列跨换月做差分,把任何真实仓位都赚不到的月间价差记成了收益;旗舰回测窗口约 85% 的"alpha"来自这里。后续版本已改为同合约收益 + 比例后复权。
- **归一化前视(2026-08)**:因子用全历史中位数 / MAD 做标准化,决策日用到了未来统计量;仅这一项就让生产组合的回放虚增约 0.38 Sharpe。后续版本已改为锚定式扩展窗口(只用决策日及以前的数据)。
- 另有交易日轴(夜盘被切出"幽灵周六")、期限结构查询缺少截止日期等一系列前视与数据问题。

> ⚠️ **本仓库公开的代码停留在 2026-04-04,不包含上述修复,仍带有这些问题。请勿依据它的回测结果做任何决策。** 修复在后续版本中完成,尚未公开。

修复后(未公开版本),所有新假设都先写预注册协议、冻结代码,再在诚实数据上净成本跑一次,判决写死、不事后调参。迄今结果:

| 预注册书(净成本) | 净 Sharpe | 判决 |
|---|:---:|:---:|
| LARGE_ORDER 单因子截面书(2026-08-25) | +0.72(5 年中 3 年为正) | FAIL |
| HURST 路由的日频反转书(2026-08-28) | −0.18 | FAIL |
| TSMOM × BOCPD 三臂书,主判决臂 = 波动率目标化 12 个月时序动量(2026-09-25) | +0.46(5 年中 3 年为正) | FAIL |

审查宪章、问题账本、预注册协议与结果尚未公开。

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

> ⚠️ 后续实测中它每次触发都拉低了回测表现,且从未在实盘路径上运行过;后续(未公开)版本已默认关闭。本仓库公开代码中它仍默认开启。

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

> ⚠️ **更正(2026-06 / 2026-08)**:本条中的回测与因子数字(Sharpe 0.70 / 1.86 / 3.79 / 2.45、平均 ICIR 0.134)产自后来查实的换月幻影与归一化前视管线,已撤回,不代表可实现表现。原文保留作为历史记录;现状见上方"当前状态"。

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
