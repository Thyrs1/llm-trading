from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from contextlib import suppress
from dataclasses import replace

from nautilus_bot.ai_service import AIService
from nautilus_bot.config import BotSettings, InstrumentSettings, load_settings
from nautilus_bot.data.downloader import ensure_catalog_data
from nautilus_bot.risk import RiskController
from nautilus_bot.strategy.llm_strategy import LLMStrategy, LLMStrategyConfig
from nautilus_bot.telemetry import TelemetryStore

try:  # pragma: no cover
    from nautilus_trader.adapters.binance import (
        BINANCE,
        BinanceLiveDataClientFactory,
        BinanceLiveExecClientFactory,
        BinanceFuturesInstrumentProvider,
    )
    from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
    from nautilus_trader.adapters.binance.config import (
        BinanceDataClientConfig,
        BinanceExecClientConfig,
    )
    from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
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


class TradingOrchestrator:
    """
    负责构建 Nautilus TradingNode / BacktestNode，
    统一承载 AI、风控、遥测等能力。
    """

    def __init__(self, settings: BotSettings):
        self.settings = settings
        self.telemetry = TelemetryStore(settings)
        self.ai_service = AIService(settings)
        self.risk = RiskController(settings.risk)
        self._strategy_targets_cache: Optional[List[InstrumentSettings]] = None

    # ------------------------------------------------------------------ #
    # Live Trading
    # ------------------------------------------------------------------ #

    def run_live(self) -> None:
        """启动实时节点，并在事件循环中运行。"""

        if TradingNode is None or TradingNodeConfig is None:
            raise RuntimeError("未检测到 Nautilus Trader，请安装 `nautilus-trader` 包后重试。")

        asyncio.run(self._run_live_async())

    async def _run_live_async(self) -> None:
        node = self._build_trading_node()
        self._register_strategy(node)

        vitals_task = asyncio.create_task(self._poll_account_vitals(node))
        self.telemetry.log("🚀 Nautilus TradingNode 已就绪，开始运行。")
        try:
            await asyncio.to_thread(node.run)
        finally:
            vitals_task.cancel()
            with suppress(asyncio.CancelledError):
                await vitals_task
            self.telemetry.log("🛑 TradingNode 已停止。")

    def _build_trading_node(self) -> TradingNode:
        assert TradingNode is not None and TradingNodeConfig is not None
        assert BinanceLiveDataClientFactory is not None and BinanceLiveExecClientFactory is not None
        assert BinanceDataClientConfig is not None and BinanceExecClientConfig is not None

        targets = self._strategy_targets()
        account_type = self._resolve_account_type(self.settings.binance.account_type)
        instrument_ids = frozenset(
            InstrumentId.from_str(target.instrument_id) if InstrumentId is not None else target.instrument_id
            for target in targets
        )
        assert InstrumentProviderConfig is not None  # for type checkers
        instrument_provider = InstrumentProviderConfig(load_ids=instrument_ids, load_all=False)

        data_config = BinanceDataClientConfig(
            api_key=self.settings.binance.api_key or None,
            api_secret=self.settings.binance.api_secret or None,
            account_type=account_type,
            base_url_http=self.settings.binance.base_http_url or None,
            base_url_ws=self.settings.binance.base_ws_url or None,
            us=self.settings.binance.is_us,
            instrument_provider=instrument_provider,
        )
        exec_config = BinanceExecClientConfig(
            api_key=self.settings.binance.api_key or None,
            api_secret=self.settings.binance.api_secret or None,
            account_type=account_type,
            base_url_http=self.settings.binance.base_http_url or None,
            base_url_ws=self.settings.binance.base_ws_url or None,
            us=self.settings.binance.is_us,
            instrument_provider=instrument_provider,
        )

        config = TradingNodeConfig(
            data_clients={BINANCE: data_config},
            exec_clients={BINANCE: exec_config},
        )

        node = TradingNode(config=config)
        node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
        node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)
        node.build()
        return node

    def _resolve_account_type(self, value: str) -> "BinanceAccountType":
        if BinanceAccountType is None:  # pragma: no cover
            raise RuntimeError("BinanceAccountType 未定义，无法解析账户类型。")
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

    def _fetch_instrument_metadata(self, instrument_id: str) -> Optional["Instrument"]:
        if (
            BinanceFuturesInstrumentProvider is None
            or BinanceHttpClient is None
            or InstrumentId is None
            or Instrument is None
        ):
            return None
        if not self.settings.binance.api_key or not self.settings.binance.api_secret:
            return None
        try:
            clock = LiveClock()
            client = BinanceHttpClient(
                clock=clock,
                api_key=self.settings.binance.api_key,
                api_secret=self.settings.binance.api_secret,
                base_url=self.settings.binance.base_http_url or "https://fapi.binance.com",
            )
            provider = BinanceFuturesInstrumentProvider(
                client=client,
                clock=clock,
                account_type=self._resolve_account_type(self.settings.binance.account_type),
                config=InstrumentProviderConfig(load_all=False),
                venue=Venue(str(BINANCE)),
            )
            inst_id = InstrumentId.from_str(instrument_id)
            provider.load_ids([inst_id])
            instrument = provider.find(inst_id)
            return instrument
        except Exception as exc:  # pragma: no cover - 网络或权限问题
            self.telemetry.log(f"⚠️ 合约元数据加载失败：{exc}", instrument_id)
            return None

    def _preload_instrument(self, node: TradingNode, target: InstrumentSettings) -> None:
        if not hasattr(node, "trader") or Instrument is None:
            return
        instrument = self._fetch_instrument_metadata(target.instrument_id)
        if instrument is None:
            return
        try:
            node.trader.cache.add_instrument(instrument)
        except Exception:
            pass

    def _strategy_targets(self) -> List[InstrumentSettings]:
        if self._strategy_targets_cache is not None:
            return self._strategy_targets_cache
        if self.settings.strategy.instruments:
            raw_targets = self.settings.strategy.instruments
        else:
            raw_targets = [
                InstrumentSettings(
                    instrument_id=self.settings.strategy.instrument_id,
                    bar_type=self.settings.strategy.bar_type,
                    binance_symbol=self.settings.strategy.binance_symbol,
                    binance_interval=self.settings.strategy.binance_interval,
                    bar_history=self.settings.strategy.bar_history,
                    min_history=self.settings.strategy.min_history,
                    trade_size=self.settings.strategy.trade_size,
                    order_id_tag=self.settings.strategy.order_id_tag,
                    analysis_cooldown_secs=self.settings.strategy.analysis_cooldown_secs,
                )
            ]
        used_tags: set[str] = set()
        normalized: List[InstrumentSettings] = []
        for index, target in enumerate(raw_targets):
            tag = target.order_id_tag or f"LLM{index:02d}"
            base = tag
            suffix = 1
            while tag in used_tags:
                tag = f"{base}{suffix:02d}"
                suffix += 1
            used_tags.add(tag)
            if tag != target.order_id_tag:
                target = replace(target, order_id_tag=tag)
            normalized.append(target)
        self._strategy_targets_cache = normalized
        return normalized

    def _register_strategy(self, node: TradingNode) -> None:
        trader = node.trader
        for target in self._strategy_targets():
            self._preload_instrument(node, target)
            strategy_config = LLMStrategyConfig(
                instrument_id=target.instrument_id,
                bar_type=target.bar_type,
                trade_size=target.trade_size,
                min_history=target.min_history,
                bar_history=target.bar_history,
                analysis_cooldown_secs=target.analysis_cooldown_secs,
                order_id_tag=target.order_id_tag,
            )
            strategy = LLMStrategy(
                config=strategy_config,
                ai_service=self.ai_service,
                risk_controller=self.risk,
                telemetry=self.telemetry,
            )
            trader.add_strategy(strategy)
            trader.start_strategy(strategy.id)
            self.telemetry.log(
                f"🤖 已注册策略 {strategy.id.value} 监控 {target.instrument_id}",
                str(target.instrument_id),
            )

    async def _poll_account_vitals(self, node: TradingNode) -> None:
        while True:
            try:
                portfolio = node.trader.portfolio
                total_equity = float(getattr(portfolio, "net_asset_value", 0.0))
                available = float(getattr(portfolio, "available_balance", total_equity))
                self.telemetry.update_account_vitals(total_equity, available)
                self.risk.refresh_daily_state(total_equity, datetime.now(timezone.utc))
            except Exception:
                pass
            await asyncio.sleep(5)

    # ------------------------------------------------------------------ #
    # Backtest
    # ------------------------------------------------------------------ #

    def run_backtest(self) -> List[Dict[str, Any]]:
        """运行官方 BacktestNode，并返回结果列表。"""

        if BacktestNode is None or BacktestRunConfig is None:
            raise RuntimeError("未检测到 Nautilus Trader Backtest 模块。")

        run_config = self._build_backtest_config()
        node = BacktestNode(configs=[run_config])
        self.telemetry.log("🧪 启动 BacktestNode ...")
        results = node.run()
        self.telemetry.log(f"✅ 回测完成，共 {len(results)} 项结果。")
        try:
            report_paths = self._export_backtest_reports(results)
            self.telemetry.log(
                f"📑 回测报告已生成：{report_paths['overview']}，指标 CSV：{report_paths['metrics']}。",
                "SYSTEM",
            )
        except Exception as exc:  # noqa: BLE001
            self.telemetry.log(f"⚠️ 回测报告生成失败：{exc}", "SYSTEM")
        return results

    def _export_backtest_reports(self, results: List[Any]) -> Dict[str, Path]:
        trades_df = self._load_trades_dataframe()
        timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        reports_root = Path(self.settings.telemetry.database_path).resolve().parent / "reports"
        run_dir = reports_root / timestamp_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        trades_path = run_dir / "trades.csv"
        equity_path = run_dir / "equity_curve.csv"
        metrics_path = run_dir / "metrics.csv"
        overview_path = run_dir / "report.md"
        stats_path = run_dir / "nautilus_stats.json"

        metrics = self._compute_backtest_metrics(trades_df, results)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(metrics_path, index=False)

        if trades_df.empty:
            trades_df.to_csv(trades_path, index=False)
            equity_path.touch()
        else:
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], errors="coerce")
            trades_df = trades_df.sort_values("timestamp")
            trades_df["trigger_reason"] = trades_df["reasoning"].str.extract(r"触发：([^；]+)")
            trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()

            equity_df = trades_df[["timestamp", "cumulative_pnl"]].rename(columns={"cumulative_pnl": "equity"})
            equity_df.to_csv(equity_path, index=False)

            export_df = trades_df.copy()
            try:
                export_df["timestamp"] = export_df["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (TypeError, AttributeError):
                export_df["timestamp"] = export_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            export_df.to_csv(trades_path, index=False)

        if results:
            stats_payload = {
                "stats_pnls": getattr(results[0], "stats_pnls", {}),
                "stats_returns": getattr(results[0], "stats_returns", {}),
            }
            with stats_path.open("w", encoding="utf-8") as fp:
                json.dump(stats_payload, fp, ensure_ascii=False, indent=2)
        else:
            stats_path.touch()

        overview_lines = [
            "# 回测报告",
            "",
            f"- 生成时间：{timestamp_tag}",
            f"- 总交易数：{metrics['total_trades']}",
            f"- 盈利笔数：{metrics['winning_trades']}",
            f"- 胜率：{metrics['win_rate_pct']:.2f}%",
            f"- 总盈亏：{metrics['total_pnl']:.4f}",
            f"- 最大回撤：{metrics['max_drawdown']:.4f}",
            f"- 触发成交笔数：{metrics['trigger_hits']}",
            f"- 触发命中率：{metrics['trigger_hit_rate_pct']:.2f}%",
            f"- 回测耗时（秒）：{metrics['backtest_elapsed_sec']:.2f}",
            "",
            "## 指标概览",
        ]
        for key, value in metrics.items():
            if key.endswith("_pct") or key in {
                "total_trades",
                "winning_trades",
                "trigger_hits",
                "total_orders",
                "total_positions",
                "iterations",
                "win_rate",
                "trigger_hit_rate",
            }:
                continue
            overview_lines.append(f"- {key}: {value}")
        overview_lines.extend(
            [
                "",
                "## 数据文件",
                f"- 交易记录：`{trades_path.name}`",
                f"- 权益曲线：`{equity_path.name}`",
                f"- 指标表格：`{metrics_path.name}`",
                f"- Nautilus 原始统计：`{stats_path.name}`",
            ],
        )
        with overview_path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(overview_lines))

        return {
            "trades": trades_path,
            "equity": equity_path,
            "metrics": metrics_path,
            "overview": overview_path,
            "stats": stats_path,
        }

    def _load_trades_dataframe(self) -> pd.DataFrame:
        db_path = Path(self.settings.telemetry.database_path)
        if not db_path.exists():
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "side",
                    "entry_price",
                    "exit_price",
                    "quantity",
                    "pnl",
                    "pnl_pct",
                    "reasoning",
                ],
            )
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(
                """
                SELECT timestamp, symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct, reasoning
                FROM trades
                ORDER BY timestamp
                """,
                conn,
            )
        finally:
            conn.close()
        for column in ["entry_price", "exit_price", "quantity", "pnl", "pnl_pct"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        return df

    def _compute_backtest_metrics(self, trades_df: pd.DataFrame, results: List[Any]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "total_trades": int(trades_df.shape[0]),
            "winning_trades": 0,
            "win_rate": 0.0,
            "win_rate_pct": 0.0,
            "total_pnl": float(trades_df["pnl"].sum()) if "pnl" in trades_df else 0.0,
            "average_pnl": 0.0,
            "max_drawdown": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "trigger_hits": 0,
            "trigger_hit_rate": 0.0,
            "trigger_hit_rate_pct": 0.0,
            "backtest_elapsed_sec": 0.0,
            "total_orders": 0,
            "total_positions": 0,
            "iterations": 0,
        }

        if not trades_df.empty and "pnl" in trades_df:
            wins = (trades_df["pnl"] > 0).sum()
            metrics["winning_trades"] = int(wins)
            metrics["win_rate"] = wins / trades_df.shape[0]
            metrics["win_rate_pct"] = metrics["win_rate"] * 100
            metrics["average_pnl"] = float(trades_df["pnl"].mean())
            metrics["best_trade"] = float(trades_df["pnl"].max())
            metrics["worst_trade"] = float(trades_df["pnl"].min())

            cumulative = trades_df["pnl"].cumsum()
            drawdown = cumulative - cumulative.cummax()
            metrics["max_drawdown"] = float(drawdown.min())

            trigger_series = trades_df["reasoning"].str.extract(r"触发：([^；]+)")
            trigger_hits = trigger_series[0].fillna("").str.strip().ne("")
            metrics["trigger_hits"] = int(trigger_hits.sum())
            metrics["trigger_hit_rate"] = trigger_hits.mean()
            metrics["trigger_hit_rate_pct"] = metrics["trigger_hit_rate"] * 100

        if results:
            primary = results[0]
            metrics["backtest_elapsed_sec"] = float(getattr(primary, "elapsed_time", 0.0) or 0.0)
            metrics["total_orders"] = int(getattr(primary, "total_orders", 0) or 0)
            metrics["total_positions"] = int(getattr(primary, "total_positions", 0) or 0)
            metrics["iterations"] = int(getattr(primary, "iterations", 0) or 0)

        return metrics
    def _build_backtest_config(self) -> BacktestRunConfig:
        assert BacktestRunConfig is not None
        assert BacktestEngineConfig is not None
        assert BacktestVenueConfig is not None
        assert BacktestDataConfig is not None
        assert ImportableStrategyConfig is not None
        assert TraderId is not None
        if not self.settings.backtest.catalog_path:
            raise ValueError("backtest.catalog_path 未配置，无法运行回测。")

        importable_configs: List[ImportableStrategyConfig] = []
        data_configs: List[BacktestDataConfig] = []

        for target in self._strategy_targets():
            strategy_dict = {
                "instrument_id": target.instrument_id,
                "bar_type": target.bar_type,
                "trade_size": target.trade_size,
                "min_history": target.min_history,
                "bar_history": target.bar_history,
                "analysis_cooldown_secs": target.analysis_cooldown_secs,
                "order_id_tag": target.order_id_tag,
            }
            importable_configs.append(
                ImportableStrategyConfig(
                    strategy_path="nautilus_bot.strategy.llm_strategy:LLMStrategy",
                    config_path="nautilus_bot.strategy.llm_strategy:LLMStrategyConfig",
                    config=strategy_dict,
                )
            )

            catalog_dir = ensure_catalog_data(
                catalog_path=Path(self.settings.backtest.catalog_path or "./data/catalog"),
                instrument_id=target.instrument_id,
                bar_type=target.bar_type,
                symbol=target.binance_symbol,
                interval=target.binance_interval,
                start=datetime.fromisoformat(self.settings.backtest.start.replace("Z", "+00:00")),
                end=datetime.fromisoformat(self.settings.backtest.end.replace("Z", "+00:00")),
            )

            data_configs.append(
                BacktestDataConfig(
                    catalog_path=str(catalog_dir),
                    data_cls=Bar,
                    instrument_id=target.instrument_id,
                    start_time=self.settings.backtest.start,
                    end_time=self.settings.backtest.end,
                )
            )

        engine_config = BacktestEngineConfig(
            trader_id=TraderId(self.settings.backtest.trader_id),
            strategies=importable_configs,
        )

        venue_config = BacktestVenueConfig(
            name="BINANCE",
            oms_type="HEDGING",
            account_type="MARGIN",
            starting_balances=["100000 USDT"],
            base_currency="USDT",
        )

        if not self.settings.backtest.start or not self.settings.backtest.end:
            raise ValueError("backtest.start and backtest.end must be configured for automatic download.")

        return BacktestRunConfig(
            engine=engine_config,
            venues=[venue_config],
            data=data_configs,
            start=self.settings.backtest.start,
            end=self.settings.backtest.end,
        )


def build_orchestrator(config_path: Optional[str] = None) -> TradingOrchestrator:
    settings = load_settings(Path(config_path) if config_path else None)
    return TradingOrchestrator(settings)
