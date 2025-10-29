from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore


@dataclass(slots=True)
class BinanceClientSettings:
    """Binance 交易节点配置。"""

    api_key: str = ""
    api_secret: str = ""
    account_type: str = "USDT_FUTURES"
    base_http_url: Optional[str] = None
    base_ws_url: Optional[str] = None
    is_us: bool = False
    instrument_provider: "InstrumentProviderSettings" = field(default_factory=lambda: InstrumentProviderSettings())


@dataclass(slots=True)
class InstrumentProviderSettings:
    """行情合约加载配置。"""

    load_all: bool = False
    load_ids: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    filter_callable: Optional[str] = None
    log_warnings: bool = True


@dataclass(slots=True)
class AISettings:
    """AI 与情绪分析服务配置。"""

    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    enable_sentiment: bool = True
    rss_feeds: List[str] = field(
        default_factory=lambda: [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
        ]
    )


@dataclass(slots=True)
class RiskSettings:
    """风险约束。"""

    max_risk_per_trade: float = 0.02
    max_daily_drawdown_pct: float = 5.0
    max_consecutive_losses: int = 3
    trading_pause_seconds: int = 3_600


@dataclass(slots=True)
class InstrumentSettings:
    """单个标的的策略参数。"""

    instrument_id: str = "BTCUSDT-PERP.BINANCE"
    bar_type: str = "BTCUSDT-PERP.BINANCE-5-MINUTE-LAST-INTERNAL"
    binance_symbol: str = "BTCUSDT"
    binance_interval: str = "5m"
    bar_history: int = 720
    min_history: int = 120
    trade_size: float = 0.0
    order_id_tag: str = "LLM"
    analysis_cooldown_secs: int = 300
    default_leverage: float = 10.0
    max_leverage: float = 50.0


@dataclass(slots=True)
class StrategySettings:
    """策略与行情参数。"""

    instrument_id: str = "BTCUSDT-PERP.BINANCE"
    bar_type: str = "BTCUSDT-PERP.BINANCE-5-MINUTE-LAST-INTERNAL"
    binance_symbol: str = "BTCUSDT"
    binance_interval: str = "5m"
    bar_history: int = 720
    min_history: int = 120
    trade_size: float = 0.0
    order_id_tag: str = "LLM"
    analysis_cooldown_secs: int = 300
    default_leverage: float = 10.0
    max_leverage: float = 50.0
    instruments: List[InstrumentSettings] = field(default_factory=list)
    force_initial_analysis: bool = False


@dataclass(slots=True)
class TelemetrySettings:
    """数据库与日志配置。"""

    database_path: Path = Path("trading_data.db")


@dataclass(slots=True)
class BacktestSettings:
    """回测运行配置。"""

    catalog_path: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    trader_id: str = "LLM-001"
    venue_name: str = "BINANCE"
    oms_type: str = "HEDGING"
    account_type: str = "MARGIN"
    base_currency: str = "USDT"
    starting_balances: List[str] = field(default_factory=lambda: ["100000 USDT"])


@dataclass(slots=True)
class BotSettings:
    """机器人整体配置载体。"""

    binance: BinanceClientSettings = field(default_factory=BinanceClientSettings)
    ai: AISettings = field(default_factory=AISettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    telemetry: TelemetrySettings = field(default_factory=TelemetrySettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)


def _load_from_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".toml" and tomllib is not None:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"不支持的配置文件格式: {path}")


def load_settings(config_path: Optional[Path] = None) -> BotSettings:
    """
    生成 BotSettings：默认值 < 配置文件 < 环境变量。
    """

    settings = BotSettings()

    candidate_paths: List[Path] = []
    if config_path is not None:
        candidate_paths.append(Path(config_path))
    else:
        candidate_paths.extend([Path("config.toml"), Path("config.json")])

    for path in candidate_paths:
        if not path.exists():
            continue
        data = _load_from_file(path)
        _apply_file_overrides(settings, data)
        break

    _apply_env_overrides(settings)
    return settings


def _apply_file_overrides(settings: BotSettings, data: Dict[str, Any]) -> None:
    binance_cfg = data.get("binance", {})
    settings.binance.api_key = binance_cfg.get("api_key", settings.binance.api_key)
    settings.binance.api_secret = binance_cfg.get("api_secret", settings.binance.api_secret)
    settings.binance.account_type = binance_cfg.get("account_type", settings.binance.account_type)
    settings.binance.base_http_url = binance_cfg.get("base_http_url", settings.binance.base_http_url)
    settings.binance.base_ws_url = binance_cfg.get("base_ws_url", settings.binance.base_ws_url)
    settings.binance.is_us = binance_cfg.get("is_us", settings.binance.is_us)

    data_clients_cfg = data.get("data_clients", {})
    binance_data_cfg = data_clients_cfg.get("binance", {})
    provider_cfg = binance_cfg.get("instrument_provider", {}) or binance_data_cfg.get(
        "instrument_provider",
        {},
    )
    if provider_cfg:
        settings.binance.instrument_provider.load_all = provider_cfg.get(
            "load_all",
            settings.binance.instrument_provider.load_all,
        )
        load_ids_cfg = provider_cfg.get("load_ids")
        if isinstance(load_ids_cfg, list):
            settings.binance.instrument_provider.load_ids = [str(item) for item in load_ids_cfg]
        filters_cfg = provider_cfg.get("filters")
        if isinstance(filters_cfg, dict):
            settings.binance.instrument_provider.filters = filters_cfg
        if "filter_callable" in provider_cfg:
            settings.binance.instrument_provider.filter_callable = provider_cfg.get("filter_callable")
        if "log_warnings" in provider_cfg:
            settings.binance.instrument_provider.log_warnings = bool(provider_cfg.get("log_warnings"))

    ai_cfg = data.get("ai", {})
    settings.ai.base_url = ai_cfg.get("base_url", settings.ai.base_url)
    settings.ai.api_key = ai_cfg.get("api_key", settings.ai.api_key)
    settings.ai.model = ai_cfg.get("model", settings.ai.model)
    settings.ai.enable_sentiment = ai_cfg.get("enable_sentiment", settings.ai.enable_sentiment)
    feeds = ai_cfg.get("rss_feeds")
    if isinstance(feeds, list):
        settings.ai.rss_feeds = [str(feed) for feed in feeds]

    risk_cfg = data.get("risk", {})
    settings.risk.max_risk_per_trade = risk_cfg.get("max_risk_per_trade", settings.risk.max_risk_per_trade)
    settings.risk.max_daily_drawdown_pct = risk_cfg.get(
        "max_daily_drawdown_pct",
        settings.risk.max_daily_drawdown_pct,
    )
    settings.risk.max_consecutive_losses = risk_cfg.get(
        "max_consecutive_losses",
        settings.risk.max_consecutive_losses,
    )
    settings.risk.trading_pause_seconds = risk_cfg.get(
        "trading_pause_seconds",
        settings.risk.trading_pause_seconds,
    )

    strat_cfg = data.get("strategy", {})
    settings.strategy.instrument_id = strat_cfg.get("instrument_id", settings.strategy.instrument_id)
    settings.strategy.bar_type = strat_cfg.get("bar_type", settings.strategy.bar_type)
    settings.strategy.bar_history = strat_cfg.get("bar_history", settings.strategy.bar_history)
    settings.strategy.min_history = strat_cfg.get("min_history", settings.strategy.min_history)
    settings.strategy.trade_size = strat_cfg.get("trade_size", settings.strategy.trade_size)
    settings.strategy.order_id_tag = strat_cfg.get("order_id_tag", settings.strategy.order_id_tag)
    settings.strategy.analysis_cooldown_secs = strat_cfg.get(
        "analysis_cooldown_secs",
        settings.strategy.analysis_cooldown_secs,
    )
    settings.strategy.default_leverage = strat_cfg.get(
        "default_leverage",
        settings.strategy.default_leverage,
    )
    settings.strategy.max_leverage = strat_cfg.get(
        "max_leverage",
        settings.strategy.max_leverage,
    )
    settings.strategy.binance_symbol = strat_cfg.get("binance_symbol", settings.strategy.binance_symbol)
    settings.strategy.binance_interval = strat_cfg.get("binance_interval", settings.strategy.binance_interval)
    if "force_initial_analysis" in strat_cfg:
        settings.strategy.force_initial_analysis = bool(strat_cfg.get("force_initial_analysis"))

    base_instrument = InstrumentSettings(
        instrument_id=settings.strategy.instrument_id,
        bar_type=settings.strategy.bar_type,
        binance_symbol=settings.strategy.binance_symbol,
        binance_interval=settings.strategy.binance_interval,
        bar_history=settings.strategy.bar_history,
        min_history=settings.strategy.min_history,
        trade_size=settings.strategy.trade_size,
        order_id_tag=settings.strategy.order_id_tag,
        analysis_cooldown_secs=settings.strategy.analysis_cooldown_secs,
        default_leverage=settings.strategy.default_leverage,
        max_leverage=settings.strategy.max_leverage,
    )
    instruments_cfg = strat_cfg.get("instruments")
    if isinstance(instruments_cfg, list) and instruments_cfg:
        settings.strategy.instruments = [
            _instrument_from_dict(item, base_instrument)
            for item in instruments_cfg
            if isinstance(item, dict)
        ]

    telemetry_cfg = data.get("telemetry", {})
    db_path = telemetry_cfg.get("database_path")
    if db_path:
        settings.telemetry.database_path = Path(db_path)

    backtest_cfg = data.get("backtest", {})
    settings.backtest.catalog_path = backtest_cfg.get("catalog_path", settings.backtest.catalog_path)
    settings.backtest.start = backtest_cfg.get("start", settings.backtest.start)
    settings.backtest.end = backtest_cfg.get("end", settings.backtest.end)
    settings.backtest.trader_id = backtest_cfg.get("trader_id", settings.backtest.trader_id)
    settings.backtest.venue_name = backtest_cfg.get("venue_name", settings.backtest.venue_name)
    settings.backtest.oms_type = backtest_cfg.get("oms_type", settings.backtest.oms_type)
    settings.backtest.account_type = backtest_cfg.get("account_type", settings.backtest.account_type)
    settings.backtest.base_currency = backtest_cfg.get("base_currency", settings.backtest.base_currency)
    starting_balances_cfg = backtest_cfg.get("starting_balances")
    if isinstance(starting_balances_cfg, list) and starting_balances_cfg:
        settings.backtest.starting_balances = [str(item) for item in starting_balances_cfg]


def _instrument_from_dict(data: Dict[str, Any], defaults: InstrumentSettings) -> InstrumentSettings:
    return InstrumentSettings(
        instrument_id=str(data.get("instrument_id", defaults.instrument_id)),
        bar_type=str(data.get("bar_type", defaults.bar_type)),
        binance_symbol=str(data.get("binance_symbol", defaults.binance_symbol)),
        binance_interval=str(data.get("binance_interval", defaults.binance_interval)),
        bar_history=int(data.get("bar_history", defaults.bar_history)),
        min_history=int(data.get("min_history", defaults.min_history)),
        trade_size=float(data.get("trade_size", defaults.trade_size)),
        order_id_tag=str(data.get("order_id_tag", defaults.order_id_tag)),
        analysis_cooldown_secs=int(
            data.get("analysis_cooldown_secs", defaults.analysis_cooldown_secs)
        ),
        default_leverage=float(data.get("default_leverage", defaults.default_leverage)),
        max_leverage=float(data.get("max_leverage", defaults.max_leverage)),
    )


def _apply_env_overrides(settings: BotSettings) -> None:
    env = os.environ

    settings.binance.api_key = env.get("BOT_BINANCE_API_KEY", settings.binance.api_key)
    settings.binance.api_secret = env.get("BOT_BINANCE_API_SECRET", settings.binance.api_secret)
    settings.binance.account_type = env.get("BOT_BINANCE_ACCOUNT_TYPE", settings.binance.account_type)
    settings.binance.base_http_url = env.get("BOT_BINANCE_BASE_HTTP_URL", settings.binance.base_http_url)
    settings.binance.base_ws_url = env.get("BOT_BINANCE_BASE_WS_URL", settings.binance.base_ws_url)
    is_us = env.get("BOT_BINANCE_IS_US")
    if is_us is not None:
        settings.binance.is_us = is_us.lower() in {"1", "true", "yes"}

    settings.ai.base_url = env.get("BOT_AI_BASE_URL", settings.ai.base_url)
    settings.ai.api_key = env.get("BOT_AI_API_KEY", settings.ai.api_key)
    settings.ai.model = env.get("BOT_AI_MODEL", settings.ai.model)
    enable_sentiment = env.get("BOT_AI_ENABLE_SENTIMENT")
    if enable_sentiment is not None:
        settings.ai.enable_sentiment = enable_sentiment.lower() in {"1", "true", "yes"}
    feeds_raw = env.get("BOT_AI_RSS_FEEDS")
    if feeds_raw:
        settings.ai.rss_feeds = [feed.strip() for feed in feeds_raw.split(",") if feed.strip()]

    risk_pct = env.get("BOT_RISK_MAX_PER_TRADE")
    if risk_pct:
        settings.risk.max_risk_per_trade = float(risk_pct)
    dd = env.get("BOT_RISK_MAX_DRAWDOWN")
    if dd:
        settings.risk.max_daily_drawdown_pct = float(dd)
    losses = env.get("BOT_RISK_MAX_CONSECUTIVE_LOSSES")
    if losses:
        settings.risk.max_consecutive_losses = int(losses)
    pause = env.get("BOT_RISK_PAUSE_SECONDS")
    if pause:
        settings.risk.trading_pause_seconds = int(pause)

    instrument_id = env.get("BOT_STRATEGY_INSTRUMENT_ID")
    if instrument_id:
        settings.strategy.instrument_id = instrument_id
    bar_type = env.get("BOT_STRATEGY_BAR_TYPE")
    if bar_type:
        settings.strategy.bar_type = bar_type
    bar_history = env.get("BOT_STRATEGY_BAR_HISTORY")
    if bar_history:
        settings.strategy.bar_history = int(bar_history)
    min_history = env.get("BOT_STRATEGY_MIN_HISTORY")
    if min_history:
        settings.strategy.min_history = int(min_history)
    trade_size = env.get("BOT_STRATEGY_TRADE_SIZE")
    if trade_size:
        settings.strategy.trade_size = float(trade_size)
    order_tag = env.get("BOT_STRATEGY_ORDER_TAG")
    if order_tag:
        settings.strategy.order_id_tag = order_tag
    cooldown = env.get("BOT_STRATEGY_ANALYSIS_COOLDOWN")
    if cooldown:
        settings.strategy.analysis_cooldown_secs = int(cooldown)
    binance_symbol = env.get("BOT_STRATEGY_BINANCE_SYMBOL")
    if binance_symbol:
        settings.strategy.binance_symbol = binance_symbol
    binance_interval = env.get("BOT_STRATEGY_BINANCE_INTERVAL")
    if binance_interval:
        settings.strategy.binance_interval = binance_interval
    default_leverage = env.get("BOT_STRATEGY_DEFAULT_LEVERAGE")
    if default_leverage:
        settings.strategy.default_leverage = float(default_leverage)
    max_leverage = env.get("BOT_STRATEGY_MAX_LEVERAGE")
    if max_leverage:
        settings.strategy.max_leverage = float(max_leverage)

    db_path = env.get("BOT_DB_PATH")
    if db_path:
        settings.telemetry.database_path = Path(db_path)

    catalog = env.get("BOT_BACKTEST_CATALOG")
    if catalog:
        settings.backtest.catalog_path = catalog
    start = env.get("BOT_BACKTEST_START")
    if start:
        settings.backtest.start = start
    end = env.get("BOT_BACKTEST_END")
    if end:
        settings.backtest.end = end
    trader_id = env.get("BOT_BACKTEST_TRADER_ID")
    if trader_id:
        settings.backtest.trader_id = trader_id
    venue_name = env.get("BOT_BACKTEST_VENUE_NAME")
    if venue_name:
        settings.backtest.venue_name = venue_name
    oms_type = env.get("BOT_BACKTEST_OMS_TYPE")
    if oms_type:
        settings.backtest.oms_type = oms_type
    account_type = env.get("BOT_BACKTEST_ACCOUNT_TYPE")
    if account_type:
        settings.backtest.account_type = account_type
    base_currency = env.get("BOT_BACKTEST_BASE_CURRENCY")
    if base_currency:
        settings.backtest.base_currency = base_currency
    balances_raw = env.get("BOT_BACKTEST_STARTING_BALANCES")
    if balances_raw:
        settings.backtest.starting_balances = [
            balance.strip() for balance in balances_raw.split(",") if balance.strip()
        ]
