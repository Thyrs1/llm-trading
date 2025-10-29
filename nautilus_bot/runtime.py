from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from nautilus_bot.ai_service import AIService
from nautilus_bot.config import BotSettings, InstrumentSettings, load_settings
from nautilus_bot.data.downloader import ensure_catalog_data
from nautilus_bot.risk import RiskController
from nautilus_bot.strategy.llm_strategy import InstrumentSpec, LLMStrategy, LLMStrategyConfig
from nautilus_bot.telemetry import TelemetryStore

try:  # pragma: no cover
    from nautilus_trader.adapters.binance import (
        BINANCE,
        BinanceLiveDataClientFactory,
        BinanceLiveExecClientFactory,
    )
    from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
    from nautilus_trader.adapters.binance.config import (
        BinanceDataClientConfig,
        BinanceExecClientConfig,
    )
    from nautilus_trader.adapters.binance.factories import (
        get_cached_binance_futures_instrument_provider,
        get_cached_binance_http_client,
    )
    from nautilus_trader.common.component import LiveClock
    from nautilus_trader.common.config import InstrumentProviderConfig
    from nautilus_trader.backtest.node import BacktestNode
    from nautilus_trader.config import (
        BacktestDataConfig,
        BacktestEngineConfig,
        BacktestRunConfig,
        BacktestVenueConfig,
        ImportableStrategyConfig,
        TradingNodeConfig,
    )
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
    from nautilus_trader.model.instruments import Instrument
except ImportError:  # pragma: no cover
    BINANCE = "BINANCE"  # type: ignore[assignment]
    BinanceLiveDataClientFactory = None  # type: ignore[assignment]
    BinanceLiveExecClientFactory = None  # type: ignore[assignment]
    BinanceAccountType = None  # type: ignore[assignment]
    BinanceDataClientConfig = None  # type: ignore[assignment]
    BinanceExecClientConfig = None  # type: ignore[assignment]
    InstrumentProviderConfig = None  # type: ignore[assignment]
    TradingNode = None  # type: ignore[assignment]
    TradingNodeConfig = None  # type: ignore[assignment]
    BacktestNode = None  # type: ignore[assignment]
    BacktestRunConfig = None  # type: ignore[assignment]
    BacktestEngineConfig = None  # type: ignore[assignment]
    BacktestVenueConfig = None  # type: ignore[assignment]
    BacktestDataConfig = None  # type: ignore[assignment]
    ImportableStrategyConfig = None  # type: ignore[assignment]
    InstrumentId = None  # type: ignore[assignment]
    Instrument = None  # type: ignore[assignment]
    TraderId = None  # type: ignore[assignment]
    Bar = None  # type: ignore[assignment]
    get_cached_binance_http_client = None  # type: ignore[assignment]
    get_cached_binance_futures_instrument_provider = None  # type: ignore[assignment]
    LiveClock = None  # type: ignore[assignment]


async def run_live(settings: BotSettings) -> None:
    """
    启动实时交易节点并注册多标的 LLM 策略。
    """

    if TradingNode is None or TradingNodeConfig is None:
        raise RuntimeError("未检测到 Nautilus Trader，请安装 `nautilus-trader` 包后重试。")

    specs = _instrument_specs(settings)
    if not specs:
        raise ValueError("未配置策略合约，请在 config.toml 的 strategy.instruments 中指定。")

    ai_service = AIService(settings)
    risk = RiskController(settings.risk)
    telemetry = TelemetryStore(settings)

    node = await _build_trading_node(settings, specs)
    strategy = _build_strategy(settings, specs, ai_service, risk, telemetry)
    trader = node.trader
    trader.add_strategy(strategy)
    trader.start_strategy(strategy.id)
    telemetry.log("🤖 已注册多标的 LLM Strategy。", "SYSTEM")

    vitals_task = asyncio.create_task(_poll_account_vitals(node, telemetry, risk))
    telemetry.log("🚀 Nautilus TradingNode 已就绪，开始运行。")
    try:
        await node.run_async()
    except asyncio.CancelledError:
        telemetry.log("⚠️ 事件循环被取消，TradingNode 即将停止。", "SYSTEM")
    finally:
        vitals_task.cancel()
        with suppress(asyncio.CancelledError):
            await vitals_task
        telemetry.log("🛑 TradingNode 已停止。")


def run_backtest(settings: BotSettings) -> List[Dict[str, Any]]:
    """
    使用 Nautilus 官方回测框架运行策略，并返回结果列表。
    """

    if BacktestNode is None or BacktestRunConfig is None:
        raise RuntimeError("未检测到 Nautilus Trader Backtest 模块。")

    specs = _instrument_specs(settings)
    if not specs:
        raise ValueError("未配置策略合约，请在 config.toml 的 strategy.instruments 中指定。")

    run_config = _build_backtest_config(settings, specs)
    node = BacktestNode(configs=[run_config])
    telemetry = TelemetryStore(settings)
    telemetry.log("🧪 启动 BacktestNode ...", "SYSTEM")
    results = node.run()
    telemetry.log(f"✅ 回测完成，共 {len(results)} 项结果。", "SYSTEM")
    return [getattr(result, "__dict__", {}) for result in results]


def load_bot_settings(config_path: Optional[str]) -> BotSettings:
    """
    从指定路径加载 BotSettings；若路径为空则使用默认文件。
    """

    path = Path(config_path) if config_path else None
    return load_settings(path)


async def _build_trading_node(settings: BotSettings, specs: Sequence[InstrumentSpec]) -> TradingNode:
    assert TradingNode is not None and TradingNodeConfig is not None
    assert BinanceLiveDataClientFactory is not None and BinanceLiveExecClientFactory is not None
    assert BinanceDataClientConfig is not None and BinanceExecClientConfig is not None

    account_type = _resolve_account_type(settings.binance.account_type)
    provider_config = _build_instrument_provider_config(settings, specs)

    data_config = BinanceDataClientConfig(
        api_key=settings.binance.api_key or None,
        api_secret=settings.binance.api_secret or None,
        account_type=account_type,
        base_url_http=settings.binance.base_http_url or None,
        base_url_ws=settings.binance.base_ws_url or None,
        us=settings.binance.is_us,
        instrument_provider=provider_config,
    )
    exec_config = BinanceExecClientConfig(
        api_key=settings.binance.api_key or None,
        api_secret=settings.binance.api_secret or None,
        account_type=account_type,
        base_url_http=settings.binance.base_http_url or None,
        base_url_ws=settings.binance.base_ws_url or None,
        us=settings.binance.is_us,
        instrument_provider=provider_config,
    )

    node_config = TradingNodeConfig(
        data_clients={BINANCE: data_config},
        exec_clients={BINANCE: exec_config},
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)
    node.build()

    await _warm_instrument_caches(settings, specs, node, provider_config)
    return node


def _build_strategy(
    settings: BotSettings,
    specs: Sequence[InstrumentSpec],
    ai_service: AIService,
    risk: RiskController,
    telemetry: TelemetryStore,
) -> LLMStrategy:
    initial_balance = _initial_balance_hint(settings)
    strategy_config = LLMStrategyConfig(
        instruments=list(specs),
        initial_equity=initial_balance,
        initial_available_margin=initial_balance,
    )
    return LLMStrategy(
        config=strategy_config,
        ai_service=ai_service,
        risk_controller=risk,
        telemetry=telemetry,
    )


def _instrument_specs(settings: BotSettings) -> List[InstrumentSpec]:
    instruments = settings.strategy.instruments or [
        InstrumentSettings(
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
    ]

    used_tags: set[str] = set()
    normalized: List[InstrumentSettings] = []
    for idx, target in enumerate(instruments):
        tag = target.order_id_tag or f"LLM{idx:02d}"
        base = tag
        suffix = 1
        while tag in used_tags:
            tag = f"{base}{suffix:02d}"
            suffix += 1
        used_tags.add(tag)
        if tag != target.order_id_tag:
            target = replace(target, order_id_tag=tag)
        normalized.append(target)

    return [
        InstrumentSpec(
            instrument_id=item.instrument_id,
            bar_type=item.bar_type,
            trade_size=item.trade_size,
            min_history=item.min_history,
            bar_history=item.bar_history,
            analysis_cooldown_secs=item.analysis_cooldown_secs,
            order_id_tag=item.order_id_tag,
            default_leverage=item.default_leverage,
            max_leverage=item.max_leverage,
            binance_symbol=item.binance_symbol,
            binance_interval=item.binance_interval,
        )
        for item in normalized
    ]


def _build_instrument_provider_config(
    settings: BotSettings,
    specs: Sequence[InstrumentSpec],
) -> InstrumentProviderConfig:
    assert InstrumentProviderConfig is not None  # for mypy
    provider_settings = settings.binance.instrument_provider

    load_ids: set[Any] = {
        InstrumentId.from_str(spec.instrument_id) if InstrumentId is not None else spec.instrument_id
        for spec in specs
    }

    if provider_settings.load_ids:
        for raw_id in provider_settings.load_ids:
            try:
                converted = InstrumentId.from_str(raw_id) if InstrumentId is not None else raw_id
                load_ids.add(converted)
            except Exception:
                pass

    configured_load_all = provider_settings.load_all
    load_ids_frozen = frozenset(load_ids) if load_ids else None
    # 若用户未显式要求加载全部且指定了目标合约，则优先按需加载
    load_all = configured_load_all and not load_ids

    return InstrumentProviderConfig(
        load_all=load_all,
        load_ids=load_ids_frozen,
        filters=provider_settings.filters or None,
        filter_callable=provider_settings.filter_callable or None,
        log_warnings=provider_settings.log_warnings,
    )


def _resolve_account_type(value: str) -> BinanceAccountType:
    if BinanceAccountType is None:
        raise RuntimeError("未检测到 BinanceAccountType 枚举。")
    mapping = {
        "SPOT": BinanceAccountType.SPOT,
        "MARGIN": BinanceAccountType.MARGIN,
        "ISOLATED_MARGIN": BinanceAccountType.ISOLATED_MARGIN,
        "USDT_FUTURE": BinanceAccountType.USDT_FUTURES,
        "USDT_FUTURES": BinanceAccountType.USDT_FUTURES,
        "COIN_FUTURE": BinanceAccountType.COIN_FUTURES,
        "COIN_FUTURES": BinanceAccountType.COIN_FUTURES,
    }
    if not value:
        return BinanceAccountType.USDT_FUTURES
    key = value.strip().upper()
    if key in mapping:
        return mapping[key]
    if key not in BinanceAccountType.__members__:
        raise ValueError(f"不支持的 Binance 账户类型：{value}")
    return BinanceAccountType[key]


async def _warm_instrument_caches(
    settings: BotSettings,
    specs: Sequence[InstrumentSpec],
    node: TradingNode,
    provider_config: InstrumentProviderConfig,
) -> None:
    if (
        get_cached_binance_http_client is None
        or get_cached_binance_futures_instrument_provider is None
        or InstrumentId is None
        or Instrument is None
        or LiveClock is None
    ):
        return
    if not settings.binance.api_key or not settings.binance.api_secret:
        return

    account_type = _resolve_account_type(settings.binance.account_type)
    clock = getattr(getattr(node, "kernel", None), "clock", None) or LiveClock()
    client = get_cached_binance_http_client(
        clock=clock,
        account_type=account_type,
        api_key=settings.binance.api_key,
        api_secret=settings.binance.api_secret,
        base_url=settings.binance.base_http_url or None,
        is_us=settings.binance.is_us,
    )
    provider = get_cached_binance_futures_instrument_provider(
        client=client,
        clock=clock,
        account_type=account_type,
        config=provider_config,
        venue=Venue(str(BINANCE)),
    )
    ids = [InstrumentId.from_str(spec.instrument_id) for spec in specs]
    try:
        await provider.initialize(reload=True)
    except Exception:
        return

    for inst_id in ids:
        instrument = provider.find(inst_id)
        if instrument is None:
            continue
        if hasattr(node, "cache"):
            try:
                node.cache.add_instrument(instrument)  # type: ignore[attr-defined]
            except Exception:
                pass
        trader = getattr(node, "trader", None)
        if trader is not None and hasattr(trader, "cache"):
            try:
                trader.cache.add_instrument(instrument)  # type: ignore[attr-defined]
            except Exception:
                pass


async def _poll_account_vitals(node: TradingNode, telemetry: TelemetryStore, risk: RiskController) -> None:
    while True:
        try:
            portfolio = node.trader.portfolio
            total_equity = float(getattr(portfolio, "net_asset_value", 0.0))
            available = float(getattr(portfolio, "available_balance", total_equity))
            telemetry.update_account_vitals(total_equity, available)
            risk.refresh_daily_state(total_equity, datetime.now(timezone.utc))
        except Exception:
            pass
        await asyncio.sleep(5)


def _initial_balance_hint(settings: BotSettings) -> float:
    balances = getattr(settings.backtest, "starting_balances", []) or []
    for entry in balances:
        if not entry:
            continue
        token = entry.replace(",", " ").split()[0]
        try:
            return float(token)
        except ValueError:
            continue
    return 0.0


def _build_backtest_config(settings: BotSettings, specs: Sequence[InstrumentSpec]) -> BacktestRunConfig:
    assert BacktestRunConfig is not None
    assert BacktestEngineConfig is not None
    assert BacktestVenueConfig is not None
    assert BacktestDataConfig is not None
    assert ImportableStrategyConfig is not None
    assert TraderId is not None
    if not settings.backtest.catalog_path:
        raise ValueError("backtest.catalog_path 未配置，无法运行回测。")
    if not settings.backtest.start or not settings.backtest.end:
        raise ValueError("backtest.start 与 backtest.end 必须配置，以便自动下载数据。")

    initial_balance = _initial_balance_hint(settings)
    strategy_dict = {
        "instruments": [_spec_to_dict(spec) for spec in specs],
        "initial_equity": initial_balance,
        "initial_available_margin": initial_balance,
    }
    importable = ImportableStrategyConfig(
        strategy_path="nautilus_bot.strategy.llm_strategy:LLMStrategy",
        config_path="nautilus_bot.strategy.llm_strategy:LLMStrategyConfig",
        config=strategy_dict,
    )

    data_configs: List[BacktestDataConfig] = []
    for spec in specs:
        catalog_dir = ensure_catalog_data(
            catalog_path=Path(settings.backtest.catalog_path),
            instrument_id=spec.instrument_id,
            bar_type=spec.bar_type,
            symbol=spec.binance_symbol or spec.instrument_id.split("-")[0],
            interval=spec.binance_interval or "5m",
            start=datetime.fromisoformat(settings.backtest.start.replace("Z", "+00:00")),
            end=datetime.fromisoformat(settings.backtest.end.replace("Z", "+00:00")),
        )
        data_configs.append(
            BacktestDataConfig(
                catalog_path=str(catalog_dir),
                data_cls=Bar,
                instrument_id=spec.instrument_id,
                start_time=settings.backtest.start,
                end_time=settings.backtest.end,
            )
        )

    engine_config = BacktestEngineConfig(
        trader_id=TraderId(settings.backtest.trader_id),
        strategies=[importable],
    )

    venue_config = BacktestVenueConfig(
        name=settings.backtest.venue_name,
        oms_type=settings.backtest.oms_type,
        account_type=settings.backtest.account_type,
        starting_balances=settings.backtest.starting_balances,
        base_currency=settings.backtest.base_currency,
    )

    return BacktestRunConfig(
        engine=engine_config,
        venues=[venue_config],
        data=data_configs,
        start=settings.backtest.start,
        end=settings.backtest.end,
    )


def _spec_to_dict(spec: InstrumentSpec) -> Dict[str, Any]:
    return {
        "instrument_id": spec.instrument_id,
        "bar_type": spec.bar_type,
        "trade_size": spec.trade_size,
        "min_history": spec.min_history,
        "bar_history": spec.bar_history,
        "analysis_cooldown_secs": spec.analysis_cooldown_secs,
        "order_id_tag": spec.order_id_tag,
        "default_leverage": spec.default_leverage,
        "max_leverage": spec.max_leverage,
        "binance_symbol": spec.binance_symbol,
        "binance_interval": spec.binance_interval,
    }
