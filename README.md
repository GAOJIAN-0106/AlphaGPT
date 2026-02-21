# AlphaGPT

基于 Transformer 的自动因子挖掘量化交易系统。生成可解释的因子公式，通过强化学习回测优化，部署到 Solana meme 币和 A 股市场。

> Fork from [imbue-bit/AlphaGPT](https://github.com/imbue-bit/AlphaGPT)

## 最新性能指标

**最优单策略（ZZ1000 A 股，2020-2026）**

| 指标 | 值 |
|------|-----|
| 测试集 Sharpe | 0.576 |
| 最大回撤 | 5.43% |
| 因子自相关 | 0.913 |
| 平均换手率 | 1.07% |

**分年表现**

| 年份 | 阶段 | Sharpe | 累计收益 | 最大回撤 | 胜率 |
|------|------|--------|----------|----------|------|
| 2020 | Train | 1.08 | 3.88% | 2.94% | 52.3% |
| 2021 | Train | 2.17 | 3.43% | 1.23% | 59.3% |
| 2022 | Train | 0.09 | 0.26% | 3.22% | 54.6% |
| 2023 | Train | 0.18 | 0.39% | 2.62% | 48.4% |
| 2024 | Mixed | 0.39 | 1.44% | 3.51% | 49.2% |
| 2025 | Test | 1.29 | 1.79% | 1.13% | 60.1% |
| 2026 | Test | 2.41 | 0.11% | 0.08% | 50.0% |

**集成策略（6 公式等权）**

| 指标 | 值 |
|------|-----|
| 测试集 Sharpe | 0.510 |
| 最大回撤 | 4.76% |
| 平均换手率 | 1.03% |

**5-Fold 时序交叉验证（最优种子 Seed-7）**

| Fold | Sharpe | 最大回撤 |
|------|--------|----------|
| 0 | 0.082 | 3.91% |
| 1 | 0.140 | 2.67% |
| 2 | -0.158 | 3.90% |
| 3 | 0.401 | 2.69% |
| 4 | 0.682 | 1.65% |
| **均值** | **0.229** | **2.96%** |

## 更新日志

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
