# Nautilus Trader 重构方案概览

## 背景与目标
- 现有机器人基于自研循环驱动，缺乏 Nautilus 事件引擎的高并发优势与异步调度能力。
- 用户要求全面迁移至 Nautilus Trader，统一使用其 TradingNode / BacktestNode 管理行情、执行与策略生命周期。
- 本次重构需要在保持 AI 决策与数据库遥测能力的同时，完成以下目标：
  - 引入 Nautilus Trader 的 Engine/Strategy 框架承载交易生命周期。
  - 重构配置、风险管理、日志、数据库等横切关注点，统一到新架构中。

## 新架构总览
```
├── nautilus_bot/
│   ├── __init__.py
│   ├── config.py              # 配置数据类 + 环境加载
│   ├── orchestrator.py        # 构建 TradingNode / BacktestNode，调度策略与依赖
│   ├── ai_service.py          # LLM 与情绪分析封装
│   ├── telemetry.py           # SQLite 遥测、事件记录
│   ├── risk.py                # 风险状态跟踪与统一接口
│   ├── strategy/
│   │   ├── __init__.py
│   │   └── llm_strategy.py    # 基于 Nautilus Strategy 的 LLM 交易逻辑
│   └── utils/
│       └── triggers.py        # 动态触发器与指标计算
├── main.py                    # 统一入口（live/backtest）
├── requirements.txt           # 新增 nautilus-trader、pydantic 等依赖
└── docs/                      # 设计与迁移文档
```

## 模块职责说明
- `nautilus_bot.config.BotSettings`：集中化配置，支持 `.env`/环境变量覆盖；与 Nautilus `LiveConfig` 对接。
- `nautilus_bot.ai_service`：封装原 `ai_processor` 中的 LLM 调用、FinBERT 情绪分析、提示词构造，暴露同步/异步接口给策略。
- `nautilus_bot.telemetry`：整合原 `database_manager` 功能，改用事件驱动写库，同时给 Nautilus `Event` 的监听器使用。
- `nautilus_bot.strategy.llm_strategy.LLMStrategy`：继承 `Strategy`，在 `on_bar` 钩子内调用 AI 服务、风险管理，并向 Engine 发出下单或调整请求。
- `nautilus_bot.risk.RiskController`：统一管理日内回撤、连亏等限制，作为策略依赖注入。
- `nautilus_bot.orchestrator`：根据运行模式（live/backtest）构建 TradingNode / BacktestNode，加载配置、注册策略与服务，并暴露 `run_live()`/`run_backtest()`。

## 迁移现状
- 旧版脚本（如 `advanced_backtester.py`、`execution_manager.py` 等）已从主分支移除。
- AI、风控、遥测已完全由 `nautilus_bot` 包内模块承担，不再依赖兼容层。
- BacktestNode 运行前会自动调用 Binance 公共 REST 接口拉取 K 线，写入 Parquet catalog 并复用在后续回测中，无需额外脚本。
- 入口命令统一为 `python main.py --mode live` 或 `--mode backtest`，均使用 Nautilus 官方节点执行。
