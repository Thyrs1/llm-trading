# Repository Guidelines

## 项目结构与模块分工
- `nautilus_bot/`：核心代码。`strategy/llm_strategy.py` 负责多标的 LLM 策略逻辑，`runtime.py` 负责装配 AI、风险与 Nautilus 节点，`telemetry.py`、`risk.py`、`utils/` 提供配套服务。
- `main.py`：统一入口，支持 `--mode live`（实盘）与 `--mode backtest`（回测）。
- `data/`：运行期生成的 K 线目录（默认空，保留 `.gitkeep`）。
- `docs/`：设计文档与配置说明。

## 构建、运行与开发命令
- `python -m venv .venv && source .venv/bin/activate`：初始化/启用虚拟环境。
- `pip install -r requirements.txt`：安装依赖。
- `python main.py --mode live`：连接 Binance 与 AI 服务启动实盘；需先配置 `config.toml`。
- `python main.py --mode backtest`：运行 Nautilus 官方回测并在 `reports/` 输出指标。
- `python -m compileall nautilus_bot`：快速语法校验。

## 代码风格与命名约定
- Python 3.12+，四空格缩进，遵循 PEP 8；核心类使用 `CamelCase`，内部函数/变量使用 `snake_case`。
- 尽量使用类型注解；和 Nautilus 交互时保持枚举、ID（如 `BTCUSDT-PERP.BINANCE`）格式一致。
- 运行产生的资源统一落在 `.gitignore` 已忽略的目录（如 `reports/`、`data/catalog/`）。

## 测试与验证指引
- 回测视作基础回归测试：`python main.py --mode backtest`。
- 如需单元/集成测试，可在新增的 `tests/` 目录内采用 `pytest` 规范命名（`test_xxx.py`）。
- 提交前检查输出报告与日志，确认无关键告警（如未加载合约或 AI 失败）。

## 提交与 PR 规范
- Commit 使用祈使句且聚焦单一改动，例如 `Add instrument preload before subscription`。
- 提交说明中引用关联 Issue（如 `Refs #123`），列出主要变更与验证命令。
- PR 必须包含：变更摘要、测试记录（命令+结果）、配置调整说明、必要的截图或日志片段。

## 安全与配置提示
- 切勿提交密钥、`config.toml`、AI 记忆文件；如需分享模板请参考 `config.toml.example`。
- Binance 实盘需确认 API Key 权限与保证金模式（`account_type`）。
- 若无 GPU 或未安装 PyTorch，可在 `config.toml` 中设置 `ai.enable_sentiment = false`，避免 FinBERT 加载失败。
