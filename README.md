<h1 align="center">llm-trading</h1>
<p align="center">基于 Nautilus Trader 与大语言模型的自动化加密交易实验项目</p>
<p align="center">
  <img src="https://img.shields.io/github/license/Thyrs1/llm-trading?style=for-the-badge&color=blue" alt="license badge">
  <img src="https://img.shields.io/github/issues/Thyrs1/llm-trading?style=for-the-badge&color=red" alt="issues badge">
  <img src="https://img.shields.io/github/contributors/Thyrs1/llm-trading?style=for-the-badge&color=cyan" alt="contributors badge">
</p>

## 项目简介
- 使用大语言模型（LLM）对市场进行多维情境分析，并驱动 Nautilus Trader 执行交易。
- 支持实盘（`--mode live`）与官方回测（`--mode backtest`）两种模式。
- 内置风险控制、遥测记录、触发器管理等组件，可追踪策略行为并生成报告。
- 项目仍处在概念验证阶段，欢迎贡献改进与测试反馈。

## 目录结构
```
├── main.py                    # 统一启动入口
├── nautilus_bot/
│   ├── orchestrator.py        # 构建 AI、风险、遥测与 Nautilus 节点
│   ├── strategy/llm_strategy.py # LLM 决策策略
│   ├── telemetry.py / risk.py # 遥测与风险控制服务
│   └── utils/                 # 触发器等辅助工具
├── config.toml.example        # 配置模板（请复制为 config.toml 并修改）
├── requirements.txt           # 依赖列表
├── data/                      # 运行期数据目录（默认空）
└── docs/                      # 设计说明与补充文档
```

## 快速开始
```bash
# 1. 克隆仓库并进入目录
git clone https://github.com/Thyrs1/llm-trading.git
cd llm-trading

# 2. 创建/启用虚拟环境（示例：Python venv）
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 基于模板复制配置文件
cp config.toml.example config.toml
#   按需填写 Binance API、AI 服务、策略参数等

# 5. 启动回测或实盘
python main.py --mode backtest   # 离线验证，将在 reports/ 输出指标
python main.py --mode live       # 连接 Binance 与 AI 服务（需有效密钥）
```
> 提示：首次回测会自动下载所需 K 线到 `data/catalog/`，确保网络畅通。

## 配置要点
- `binance.*`：填写现货/合约 API Key、Secret、Base URL 及保证金模式。
- `ai.*`：配置 LLM 调用信息（如 Base URL、模型名称、情绪分析开关等）。
- `strategy.*`：指定交易合约（例如 `BTCUSDT-PERP.BINANCE`）、时间框架与分析冷却时间。
- `backtest.*`：设置回测时间区间与 catalog 路径，建议与实盘参数保持一致。

## 运行模式
| 模式 | 命令 | 说明 |
|------|------|------|
| 实盘 | `python main.py --mode live` | 启动 Nautilus TradingNode，实时订阅 Binance 数据并执行 AI 决策。需提前测试配置，谨慎使用真实资金。 |
| 回测 | `python main.py --mode backtest` | 使用 Nautilus BacktestNode，读取 `data/catalog/` 数据并生成报告（`reports/<timestamp>/`）。 |

## 测试与验证
- 回测结果是主要的回归手段：检查生成的 `metrics.csv`、`report.md`、`equity_curve.csv` 等文件。
- 如需编写额外测试，可在新建的 `tests/` 目录中使用 `pytest`；命名遵循 `test_*.py`。
- 提交前可执行 `python -m compileall nautilus_bot` 进行快速语法检查。

## 开发与贡献
- 请阅读《[Repository Guidelines](AGENTS.md)》了解代码风格、提交流程与安全提示。
- Commit 信息使用祈使句、聚焦单一改动，并在正文注明测试命令及结果。
- Pull Request 应包含：改动摘要、验证步骤、必要的截图或日志，以及相关 Issue 链接。

## 常见问题
- **订阅合约失败**：确认 `strategy.instrument_id` 与 `binance.account_type` 设定一致，且 API Key 具备正确权限。
- **不知道可用的 Instrument ID？** 运行 `python tools/list_instruments.py --limit 20` 查看指定账户类型下的合约列表。
- **情绪分析报错**：缺少 PyTorch/FinBERT 时，可在 `config.toml` 中设置 `ai.enable_sentiment = false`。
- **目录污染**：运行后产生的报告与 catalog 已被 `.gitignore` 忽略；如需保留结果，请自行备份。

---
> ⚠️ 风险提示：本项目仅作技术实验，不构成任何投资建议。请在模拟或小额环境中充分验证策略后再考虑实盘部署。
