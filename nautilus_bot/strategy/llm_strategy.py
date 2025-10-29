from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional

import json
import time

import pandas as pd
from msgspec import Struct
from types import SimpleNamespace

from nautilus_bot.ai_service import AIService, DecisionPayload
from nautilus_bot.config import BotSettings, load_settings
from nautilus_bot.risk import RiskController
from nautilus_bot.telemetry import PositionRecord, TelemetryStore
from nautilus_bot.types import (
    AIDecision,
    DecisionAction,
    MarketSnapshot,
    StrategyState,
    TradeLifecycle,
)
from nautilus_bot.utils.triggers import TriggerManager

try:  # pragma: no cover
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import BarAggregation, OrderSide, TimeInForce
    from nautilus_trader.model.events.order import OrderCanceled, OrderFilled, OrderRejected
    from nautilus_trader.model.events.position import PositionClosed
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.trading.strategy import Strategy
except ImportError:  # pragma: no cover

    class StrategyConfig(Struct):  # type: ignore[misc]
        pass

    class Strategy:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            self.clock = type("Clock", (), {"utc_now": lambda self: datetime.now(timezone.utc)})()

        def log(self, msg: str) -> None:
            print(msg)

    class InstrumentId:  # type: ignore[misc]
        @staticmethod
        def from_str(value: str) -> str:
            return value

    class BarType:  # type: ignore[misc]
        @staticmethod
        def from_str(value: str) -> str:
            return value

    class BarAggregation:  # type: ignore[misc]
        SECOND = "SECOND"
        MINUTE = "MINUTE"
        HOUR = "HOUR"
        DAY = "DAY"

    class OrderSide:  # type: ignore[misc]
        BUY = "BUY"
        SELL = "SELL"

    class TimeInForce:  # type: ignore[misc]
        FOK = "FOK"

    class Bar:  # type: ignore[misc]
        open: float
        high: float
        low: float
        close: float
        volume: float
        end_time: datetime

    class OrderFilled:  # type: ignore[misc]
        instrument_id: str
        last_px: float
        last_qty: float
        order_side: str
        ts_event: Optional[int]
        client_order_id: Optional[str]
        venue_order_id: Optional[str]

    class OrderCanceled:  # type: ignore[misc]
        instrument_id: str
        client_order_id: Optional[str]
        ts_event: Optional[int]

    class OrderRejected:  # type: ignore[misc]
        instrument_id: str
        client_order_id: Optional[str]
        reason: str
        ts_event: Optional[int]

    class PositionClosed:  # type: ignore[misc]
        instrument_id: str
        realized_pnl: float
        realized_return: float
        ts_event: Optional[int]
        avg_px_close: float
        last_px: float
        closing_order_id: Optional[str]
        ts_closed: Optional[int]

    class Instrument:  # type: ignore[misc]
        id: str

    class Quantity:  # type: ignore[misc]
        def __init__(self, value: float) -> None:
            self._value = float(value)

        @staticmethod
        def from_f64(value: float) -> "Quantity":
            return Quantity(value)

        @staticmethod
        def from_str(value: str) -> "Quantity":
            return Quantity(float(value))

        @staticmethod
        def from_int(value: int) -> "Quantity":
            return Quantity(float(value))

        def as_double(self) -> float:
            return self._value

        def __float__(self) -> float:
            return self._value


class InstrumentSpec(Struct, kw_only=True, frozen=True):
    instrument_id: str
    bar_type: str
    trade_size: float
    min_history: int = 120
    bar_history: int = 720
    analysis_cooldown_secs: int = 300
    order_id_tag: str = "LLM"
    default_leverage: float = 10.0
    max_leverage: float = 50.0
    binance_symbol: Optional[str] = None
    binance_interval: Optional[str] = None


class LLMStrategyConfig(StrategyConfig, Struct, kw_only=True, frozen=True):
    instruments: List[InstrumentSpec]
    initial_equity: float = 0.0
    initial_available_margin: float = 0.0
    force_initial_analysis: bool = False


@dataclass(slots=True)
class InstrumentContext:
    """承载单个合约的运行时状态。"""

    config: InstrumentSpec
    _instrument_id: "InstrumentId"
    _bar_type: "BarType"
    _history: Deque[dict]
    _context_history: Deque[str]
    _state: StrategyState
    _trigger_manager: TriggerManager
    _last_analysis_ts: float
    _trade_size: Decimal
    _default_leverage: float
    _max_leverage: float
    _initial_equity: float
    _initial_available_margin: float
    _trade: TradeLifecycle
    _latest_context: Dict[str, Any]
    _pending_trigger_reason: str
    _last_price: float
    _instrument_ready: bool
    _bars_subscribed: bool
    _last_effective_leverage: float
    _last_risk_fraction: float
    _last_target_notional: float
    _bootstrap_requested: bool
    _initial_analysis_done: bool
    _history_log_flag: bool


class LLMStrategy(Strategy):
    """基于 Nautilus 事件引擎的 LLM 决策策略。"""

    _CONTEXT_FIELDS: frozenset[str] = frozenset(
        {
            "_instrument_id",
            "_bar_type",
            "_history",
            "_context_history",
            "_state",
            "_trigger_manager",
            "_last_analysis_ts",
            "_trade_size",
            "_default_leverage",
            "_max_leverage",
            "_initial_equity",
            "_initial_available_margin",
            "_trade",
            "_latest_context",
            "_pending_trigger_reason",
            "_last_price",
            "_instrument_ready",
            "_bars_subscribed",
            "_last_effective_leverage",
            "_last_risk_fraction",
            "_last_target_notional",
            "_bootstrap_requested",
            "_initial_analysis_done",
            "_history_log_flag",
        }
    )

    def __init__(
        self,
        config: LLMStrategyConfig,
        ai_service: Optional[AIService] = None,
        risk_controller: Optional[RiskController] = None,
        telemetry: Optional[TelemetryStore] = None,
    ) -> None:
        super().__init__(config=config)

        if ai_service is None or risk_controller is None or telemetry is None:
            fallback_settings: BotSettings = load_settings()
            ai_service = ai_service or AIService(fallback_settings)
            risk_controller = risk_controller or RiskController(fallback_settings.risk)
            telemetry = telemetry or TelemetryStore(fallback_settings)

        object.__setattr__(self, "ai_service", ai_service)
        object.__setattr__(self, "risk", risk_controller)
        object.__setattr__(self, "telemetry", telemetry)
        object.__setattr__(self, "_force_initial_analysis", bool(config.force_initial_analysis))

        specs = self._normalize_specs(config.instruments)
        if not specs:
            raise ValueError("LLMStrategy 至少需要一个交易合约配置。")

        initial_equity = max(float(config.initial_equity), 0.0)
        initial_available = max(float(config.initial_available_margin), 0.0)

        contexts: Dict[str, InstrumentContext] = {}
        contexts_by_bar_type: Dict[str, InstrumentContext] = {}
        for spec in specs:
            ctx = self._create_context(
                spec=spec,
                initial_equity=initial_equity,
                initial_available=initial_available,
            )
            contexts[str(ctx._instrument_id)] = ctx
            bar_type_key = self._normalize_bar_type_key(ctx._bar_type)
            contexts_by_bar_type[bar_type_key] = ctx

        object.__setattr__(self, "_contexts", contexts)
        object.__setattr__(self, "_contexts_by_bar_type", contexts_by_bar_type)
        object.__setattr__(self, "_active_ctx", None)

    @staticmethod
    def _normalize_specs(instruments: Iterable[InstrumentSpec]) -> List[InstrumentSpec]:
        specs = list(instruments or [])
        if not specs:
            return []

        normalized: List[InstrumentSpec] = []
        used_tags: Dict[str, None] = {}
        for spec in specs:
            tag = spec.order_id_tag or "LLM"
            base = tag
            suffix = 1
            while tag in used_tags:
                tag = f"{base}{suffix:02d}"
                suffix += 1
            used_tags[tag] = None
            if tag != spec.order_id_tag:
                spec = InstrumentSpec(
                    instrument_id=spec.instrument_id,
                    bar_type=spec.bar_type,
                    trade_size=spec.trade_size,
                    min_history=spec.min_history,
                    bar_history=spec.bar_history,
                    analysis_cooldown_secs=spec.analysis_cooldown_secs,
                    order_id_tag=tag,
                    default_leverage=spec.default_leverage,
                    max_leverage=spec.max_leverage,
                    binance_symbol=spec.binance_symbol,
                    binance_interval=spec.binance_interval,
                )
            normalized.append(spec)
        return normalized

    def _create_context(
        self,
        spec: InstrumentSpec,
        initial_equity: float,
        initial_available: float,
    ) -> InstrumentContext:
        instrument_id = InstrumentId.from_str(spec.instrument_id)
        bar_type = BarType.from_str(spec.bar_type)
        default_leverage = max(1.0, float(spec.default_leverage))
        max_leverage = max(default_leverage, float(spec.max_leverage))
        return InstrumentContext(
            config=spec,
            _instrument_id=instrument_id,
            _bar_type=bar_type,
            _history=deque(maxlen=spec.bar_history),
            _context_history=deque(maxlen=12),
            _state=StrategyState(),
            _trigger_manager=TriggerManager(symbol=str(instrument_id)),
            _last_analysis_ts=0.0,
            _trade_size=Decimal(str(spec.trade_size)),
            _default_leverage=default_leverage,
            _max_leverage=max_leverage,
            _initial_equity=initial_equity,
            _initial_available_margin=initial_available,
            _trade=TradeLifecycle(symbol=str(instrument_id)),
            _latest_context={},
            _pending_trigger_reason="",
            _last_price=0.0,
            _instrument_ready=False,
            _bars_subscribed=False,
            _last_effective_leverage=default_leverage,
            _last_risk_fraction=self.risk.settings.max_risk_per_trade,
            _last_target_notional=0.0,
            _bootstrap_requested=False,
            _initial_analysis_done=False,
            _history_log_flag=False,
        )

    def _get_context(self, instrument_id: Any, bar_type: Any = None) -> Optional[InstrumentContext]:
        key = None if instrument_id is None else str(instrument_id)
        if key is not None:
            ctx = self._contexts.get(key)
            if ctx is not None:
                return ctx
        lookup = None
        if bar_type is not None:
            lookup = self._normalize_bar_type_key(bar_type)
            ctx = self._contexts_by_bar_type.get(lookup)
            if ctx is not None:
                return ctx
        # 调试日志便于排查为什么历史数据没有归入上下文
        symbol = str(instrument_id or lookup or bar_type)
        self.telemetry.log(
            f"⚙️ 未找到上下文：instrument_id={instrument_id}, bar_type={bar_type}",
            symbol,
        )
        return None

    @staticmethod
    def _normalize_bar_type_key(bar_type: Any) -> str:
        try:
            return str(getattr(bar_type, "value", bar_type))
        except Exception:
            return str(bar_type)

    @contextmanager
    def _activate_context(self, context: InstrumentContext) -> Iterator[InstrumentContext]:
        prev = self._active_ctx
        object.__setattr__(self, "_active_ctx", context)
        try:
            yield context
        finally:
            object.__setattr__(self, "_active_ctx", prev)

    def __getattr__(self, name: str) -> Any:
        if name in self._CONTEXT_FIELDS and self._active_ctx is not None:
            return getattr(self._active_ctx, name)
        raise AttributeError(f"{type(self).__name__} 对象不存在属性 {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._CONTEXT_FIELDS:
            active = object.__getattribute__(self, "_active_ctx")
            if active is None:
                raise AttributeError(
                    f"未激活上下文时无法写入属性 {name!r}，请在 _activate_context 内操作。",
                )
            setattr(active, name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def _instrument_config(self) -> InstrumentSpec:
        if self._active_ctx is None:
            raise RuntimeError("当前调用不在具体合约上下文内。")
        return self._active_ctx.config

    # ------------------------------------------------------------------ #
    # Nautilus 生命周期钩子
    # ------------------------------------------------------------------ #

    def on_start(self) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        self.telemetry.log("🚀 LLM 策略启动", "SYSTEM")
        try:
            self.ai_service.initialize()
        except Exception as exc:  # noqa: BLE001
            self.telemetry.log(f"⚠️ AI 服务初始化失败：{exc}", "SYSTEM")

        for context in self._contexts.values():
            with self._activate_context(context):
                self._start_context()

    def on_instrument(self, instrument: Instrument) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        ctx = self._get_context(getattr(instrument, "id", instrument))
        if ctx is None:
            return
        with self._activate_context(ctx):
            self._handle_instrument_ready(instrument)

    def _start_context(self) -> None:
        self.telemetry.log("🚀 LLM 策略启动", str(self._instrument_id))
        instrument = self._fetch_cached_instrument()
        if instrument is not None:
            self._handle_instrument_ready(instrument)
            return
        self.telemetry.log("⌛ 合约尚未出现在缓存中，等待交易所回调。", str(self._instrument_id))

    def on_historical_data(self, data: Any) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        bars: List["Bar"] = []
        if isinstance(data, Bar):
            bars = [data]
        else:
            raw = getattr(data, "data", None)
            if raw is None:
                try:
                    raw = list(data)  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    raw = None

            if raw is not None:
                try:
                    iterator = list(raw)
                except TypeError:
                    iterator = []
                except Exception:  # noqa: BLE001
                    iterator = []

                for item in iterator:
                    if isinstance(item, Bar) or hasattr(item, "instrument_id"):
                        bars.append(item)
        if not bars:
            self.telemetry.log("🛰️ 历史数据无法解析，忽略", str(getattr(data, "instrument_id", "HIST")))
            return

        first = bars[0]
        instrument_id = getattr(first, "instrument_id", None)
        bar_type = getattr(first, "bar_type", None)
        ctx = self._get_context(instrument_id or getattr(first, "symbol", None), bar_type)
        if ctx is None:
            return

        bars.sort(key=lambda bar: getattr(bar, "end_time", self.clock.utc_now()))
        with self._activate_context(ctx):
            self.telemetry.log(
                f"🛰️ 历史片段解包 {len(bars)} 条", str(self._instrument_id)
            )
            for bar in bars:
                self._append_bar(bar)
            self._bootstrap_requested = False
            self.telemetry.log(
                f"📚 历史 K 线补齐：新增 {len(bars)} 根，现有缓存 {len(self._history)} 根。",
                str(self._instrument_id),
            )
            if (
                self._force_initial_analysis
                and not self._initial_analysis_done
                and len(self._history) >= self._instrument_config.min_history
            ):
                last_bar = bars[-1]
                self._maybe_trigger_analysis(last_bar, "启动强制分析", force=True)
                self._initial_analysis_done = True

    def on_data(self, data: Any) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        is_hist = bool(getattr(data, "is_historical", False))
        data_type = type(data).__name__
        sample_id = getattr(data, "instrument_id", None) or getattr(data, "symbol", None)
        if is_hist:
            self.telemetry.log(
                f"🛰️ 历史数据载荷：{data_type}, instrument={sample_id}",
                str(sample_id or "HIST"),
            )
            self.on_historical_data(data)
            return
        super().on_data(data)  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # 外部注入历史数据（REST 预热）
    # ------------------------------------------------------------------ #

    def ingest_external_history(
        self,
        history: Dict[str, List[dict]],
        *,
        trigger_analysis: bool = False,
        trigger_reason: str = "启动强制分析",
    ) -> None:
        """供 runtime 在 TradingNode 启动前手动注入历史 K 线。"""

        if not history:
            return

        if trigger_analysis:
            self._ensure_services_ready()

        for instrument_id, rows in history.items():
            ctx = self._contexts.get(instrument_id)
            if ctx is None:
                continue
            with self._activate_context(ctx):
                ctx._history.clear()
                parsed = []
                for row in sorted(rows, key=lambda item: item.get("date")):
                    date = row.get("date")
                    if not isinstance(date, datetime):
                        continue
                    parsed.append(
                        {
                            "date": date,
                            "open": float(row.get("open", 0.0)),
                            "high": float(row.get("high", 0.0)),
                            "low": float(row.get("low", 0.0)),
                            "close": float(row.get("close", 0.0)),
                            "volume": float(row.get("volume", 0.0)),
                        }
                    )
                for item in parsed[-ctx.config.bar_history :]:
                    ctx._history.append(item)
                if ctx._history:
                    ctx._last_price = ctx._history[-1]["close"]
                ctx._history_log_flag = len(ctx._history) >= ctx.config.min_history
                self.telemetry.log(
                    f"🪣 外部注入历史：{len(parsed)} 条，缓存 {len(ctx._history)} 条。",
                    str(ctx._instrument_id),
                )

                if (
                    trigger_analysis
                    and not ctx._initial_analysis_done
                    and len(ctx._history) >= ctx.config.min_history
                ):
                    last_row = ctx._history[-1]
                    fake_bar = SimpleNamespace(
                        end_time=last_row["date"],
                        close=last_row["close"],
                        is_historical=True,
                    )
                    self._maybe_trigger_analysis(fake_bar, trigger_reason, force=True)
                    ctx._initial_analysis_done = True

    def _ensure_services_ready(self) -> None:
        ai = getattr(self, "ai_service", None)
        if ai is None:
            return
        already_ready = getattr(ai, "_initialized", False)
        if already_ready:
            return
        try:
            ai.initialize()
        except Exception as exc:  # noqa: BLE001
            self.telemetry.log(f"⚠️ AI 服务初始化失败：{exc}", "SYSTEM")

    def _fetch_cached_instrument(self) -> Optional[Instrument]:
        try:
            cache = getattr(self, "cache", None)
            if cache is not None:
                instrument = cache.instrument(self._instrument_id)
                if instrument is not None:
                    return instrument
        except Exception:
            pass
        return None

    def _handle_instrument_ready(self, instrument: Instrument) -> None:
        if self._instrument_ready:
            return
        try:
            # 直接写入策略缓存，确保后续订阅命令能够找到合约
            self.cache.add_instrument(instrument)  # type: ignore[attr-defined]
        except Exception:
            pass
        self._instrument_ready = True
        self.telemetry.log("📦 已加载交易合约信息。", str(self._instrument_id))
        self._bootstrap_history()
        self._subscribe_market_data()

    def _bootstrap_history(self) -> None:
        if self._bootstrap_requested:
            return

        min_history = max(int(self._instrument_config.min_history), 0)
        if min_history <= 0:
            return

        interval_seconds = self._bar_interval_seconds()
        if interval_seconds is None:
            return

        end = self.clock.utc_now()
        start = end - timedelta(seconds=interval_seconds * (min_history + 5))

        try:
            self._bootstrap_requested = True
            self.request_bars(  # type: ignore[attr-defined]
                self._bar_type,
                start=start,
                end=end,
                limit=min_history,
                update_catalog=False,
            )
            self.telemetry.log(
                (
                    f"📥 已请求历史 K 线以填充上下文："
                    f"{min_history} 根，窗口 {start.isoformat()} ~ {end.isoformat()}"
                ),
                str(self._instrument_id),
            )
        except AttributeError:
            self._bootstrap_requested = False
        except Exception as exc:  # noqa: BLE001
            self._bootstrap_requested = False
            self.telemetry.log(f"⚠️ 历史 K 线请求失败：{exc}", str(self._instrument_id))

    def _bar_interval_seconds(self) -> Optional[int]:
        spec = getattr(self._bar_type, "bar_spec", None)
        if spec is None:
            # 某些外部 BarType（如 Binance EXTERNAL）不会携带 bar_spec，此时尝试根据配置推断间隔
            hint = getattr(self._instrument_config, "binance_interval", None)
            seconds = self._interval_hint_to_seconds(hint)
            if seconds is not None:
                return seconds
            return self._infer_interval_from_bar_type(str(self._bar_type))

        step = getattr(spec, "step", None)
        aggregation = getattr(spec, "aggregation", None)
        if step is None or aggregation is None:
            return None

        try:
            step_value = int(step)
        except Exception:
            return None

        agg_name = getattr(aggregation, "name", None) or getattr(aggregation, "value", None)
        if agg_name is None:
            agg_name = str(aggregation)
        agg_name = str(agg_name).upper()

        base = {
            "SECOND": 1,
            "MINUTE": 60,
            "HOUR": 3600,
            "DAY": 86400,
        }.get(agg_name)
        if base is None:
            return None
        return step_value * base

    @staticmethod
    def _interval_hint_to_seconds(value: Optional[str]) -> Optional[int]:
        if not value:
            return None
        text = value.strip().lower()
        if not text:
            return None

        suffix_map = {
            "s": 1,
            "sec": 1,
            "secs": 1,
            "second": 1,
            "seconds": 1,
            "m": 60,
            "min": 60,
            "mins": 60,
            "minute": 60,
            "minutes": 60,
            "h": 3600,
            "hr": 3600,
            "hour": 3600,
            "hours": 3600,
            "d": 86400,
            "day": 86400,
            "days": 86400,
            "w": 604800,
            "week": 604800,
            "weeks": 604800,
        }

        for suffix, multiplier in suffix_map.items():
            if text.endswith(suffix):
                number_part = text[: -len(suffix)].strip()
                try:
                    value_num = float(number_part)
                except ValueError:
                    continue
                if value_num <= 0:
                    continue
                return int(value_num * multiplier)

        # 像 "5m"、"1h" 一类可能未被上方捕获（因短后缀重叠），单独尝试解析
        if text[-1:] in {"s", "m", "h", "d", "w"}:
            try:
                value_num = float(text[:-1])
            except ValueError:
                return None
            multiplier = {
                "s": 1,
                "m": 60,
                "h": 3600,
                "d": 86400,
                "w": 604800,
            }[text[-1:]]
            if value_num > 0:
                return int(value_num * multiplier)
        return None

    @staticmethod
    def _infer_interval_from_bar_type(name: str) -> Optional[int]:
        if not name:
            return None
        parts = name.replace("_", "-").split("-")
        for idx, part in enumerate(parts):
            if not part.isdigit():
                continue
            if idx + 1 >= len(parts):
                continue
            unit = parts[idx + 1].upper()
            try:
                value_num = int(part)
            except ValueError:
                continue
            if value_num <= 0:
                continue
            multiplier = {
                "SECOND": 1,
                "SECONDS": 1,
                "SEC": 1,
                "MINUTE": 60,
                "MINUTES": 60,
                "MIN": 60,
                "HOUR": 3600,
                "HOURS": 3600,
                "DAY": 86400,
                "DAYS": 86400,
            }.get(unit)
            if multiplier is not None:
                return value_num * multiplier
        return None

    def _subscribe_market_data(self) -> None:
        if self._bars_subscribed:
            return
        deadline = time.time() + 10.0
        while time.time() < deadline:
            instrument = self._fetch_cached_instrument()
            if instrument is not None:
                break
            time.sleep(0.2)

        try:
            self.subscribe_bars(self._bar_type)  # type: ignore[attr-defined]
        except AttributeError:
            # 本地降级模式，仅用于静态检查
            return
        except Exception as exc:  # noqa: BLE001
            self.telemetry.log(f"⚠️ 行情订阅失败：{exc}", str(self._instrument_id))
            return
        self._bars_subscribed = True
        self.telemetry.log("📡 已订阅 5m 行情。", str(self._instrument_id))

    def on_bar(self, bar: Bar) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        ctx = self._get_context(getattr(bar, "instrument_id", None), getattr(bar, "bar_type", None))
        if ctx is None:
            return
        with self._activate_context(ctx):
            self._on_bar_context(bar)

    def _on_bar_context(self, bar: Bar) -> None:
        self._append_bar(bar)
        is_historical = bool(getattr(bar, "is_historical", False))
        if is_historical:
            if (
                self._force_initial_analysis
                and not self._initial_analysis_done
                and len(self._history) >= self._instrument_config.min_history
            ):
                self.telemetry.log(
                    (
                        f"🧭 首轮分析触发：历史样本 {len(self._history)} 根，"
                        "即将执行启动强制分析。"
                    ),
                    str(self._instrument_id),
                )
                self._maybe_trigger_analysis(bar, "启动强制分析", force=True)
                self._initial_analysis_done = True
            return

        if (
            self._force_initial_analysis
            and not self._initial_analysis_done
            and len(self._history) >= self._instrument_config.min_history
        ):
            self.telemetry.log(
                (
                    f"🧭 首轮分析触发：实时样本 {len(self._history)} 根，"
                    "即将执行启动强制分析。"
                ),
                str(self._instrument_id),
            )
            self._maybe_trigger_analysis(bar, "启动强制分析", force=True)
            self._initial_analysis_done = True
            return

        self._maybe_trigger_analysis(bar, "定期分析", force=False)

    def _maybe_trigger_analysis(self, bar: Bar, reason_hint: str, *, force: bool) -> None:
        if len(self._history) < self._instrument_config.min_history:
            return

        now_ts = bar.end_time.timestamp() if hasattr(bar, "end_time") else self.clock.utc_now().timestamp()
        reason_text = reason_hint if force else "定期分析"

        if not force:
            if now_ts - self._last_analysis_ts < self._instrument_config.analysis_cooldown_secs:
                should, reason = self._trigger_manager.should_analyze(
                    self._history_df,
                    self._instrument_config.analysis_cooldown_secs,
                )
                if not should:
                    self._pending_trigger_reason = ""
                    return
                reason_text = reason or "触发器命中"
        self._pending_trigger_reason = reason_text
        if reason_text:
            self.telemetry.log(f"📊 触发分析原因：{reason_text}", str(self._instrument_id))
        self._last_analysis_ts = now_ts

        active_triggers = [self._serialize_trigger(trigger) for trigger in self._trigger_manager.triggers]
        snapshot = MarketSnapshot(
            instrument_id=str(self._instrument_id),
            timeframe=self._instrument_config.bar_type,
            current_price=float(getattr(bar, "close", 0.0)),
            ohlcv=self._history_df.copy(),
            metadata={
                "active_triggers": active_triggers,
                "pending_trigger_reason": self._pending_trigger_reason,
            },
        )
        position_text = self._describe_position()
        context_summary = "\n".join(self._context_history) or "无历史上下文。"
        vitals = self._current_account_vitals()

        payload = self.ai_service.request_decision(
            snapshot=snapshot,
            position_text=position_text,
            context_summary=context_summary,
            live_equity=vitals.get("total_equity", 0.0),
            available_margin=vitals.get("available_margin", 0.0),
            active_triggers=active_triggers,
            trigger_reason=self._pending_trigger_reason,
        )

        self._handle_decision(payload, vitals)
        self._update_context(payload)
        self._update_bot_state(bar, payload)

    # ------------------------------------------------------------------ #
    # 决策及执行
    # ------------------------------------------------------------------ #

    def _handle_decision(self, payload: DecisionPayload, account_vitals: Dict[str, float]) -> None:
        decision = payload.decision
        if decision is None:
            self.telemetry.log("⚠️ AI 未返回有效决策，保持观望。", str(self._instrument_id))
            return

        self._state.last_decision = decision
        self.telemetry.log(f"🤖 AI 决策：{decision.action} | 原因：{decision.reasoning}", str(self._instrument_id))

        if decision.action == DecisionAction.WAIT:
            self._update_triggers(decision)
            return
        if decision.action == DecisionAction.OPEN:
            self._open_position(decision, account_vitals, payload)
            return
        if decision.action == DecisionAction.CLOSE:
            self._close_position(decision, payload)
            return
        if decision.action == DecisionAction.MODIFY:
            self.telemetry.log(
                f"ℹ️ 调整指令：SL={decision.new_stop_loss}, TP={decision.new_take_profit}",
                str(self._instrument_id),
            )

    def _open_position(
        self,
        decision: AIDecision,
        account_vitals: Dict[str, float],
        payload: DecisionPayload,
    ) -> None:
        if self._state.has_position or self._state.pending_side:
            self.telemetry.log("⚠️ 已持仓，忽略开仓信号。", str(self._instrument_id))
            return

        equity = account_vitals.get("total_equity", 0.0)
        now = self.clock.utc_now()
        if not self.risk.can_open_new_trade(equity, now, now.timestamp()):
            self.telemetry.log("⛔ 风控限制，禁止开仓。", str(self._instrument_id))
            return

        side = (decision.side or "LONG").upper()
        order_side = OrderSide.BUY if side == "LONG" else OrderSide.SELL

        try:
            price_hint = decision.entry_price or self._last_price or 0.0
            quantity = self._resolve_order_quantity(
                price=price_hint,
                equity=equity,
                available_margin=account_vitals.get("available_margin", equity),
                decision=decision,
            )
        except Exception as exc:
            self.telemetry.log(f"❌ 构造交易数量失败：{exc}", str(self._instrument_id))
            return

        try:
            order = self.order_factory.market(  # type: ignore[attr-defined]
                instrument_id=self._instrument_id,
                order_side=order_side,
                quantity=quantity,
                time_in_force=TimeInForce.FOK,
            )
            self.submit_order(order)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            self.telemetry.log(f"❌ 下单失败：{exc}", str(self._instrument_id))
            return

        order_id = getattr(order, "client_order_id", None) or getattr(order, "id", None)
        self._state.pending_side = side
        self._state.pending_quantity = self._quantity_as_float(quantity)
        self._state.open_order_id = order_id
        self._state.last_trigger_reason = self._pending_trigger_reason
        self._state.is_closing = False
        self._trade = TradeLifecycle(symbol=str(self._instrument_id))
        self._trade.side = side
        self._trade.quantity = self._quantity_as_float(quantity)
        self._trade.reasoning = decision.reasoning
        self._trade.trigger_reason = self._pending_trigger_reason
        self._trade.context_notes = list(self._context_history)[-5:]
        self._trade.triggers = [self._serialize_trigger(trigger) for trigger in decision.triggers]
        self._trade.sentiment = payload.sentiment
        self._trade.market_regime = payload.market_regime
        self._trade.news_digest = payload.news_digest
        self._trade.leverage = self._last_effective_leverage
        self._trade.risk_fraction = self._last_risk_fraction
        self._trade.notional = self._last_target_notional
        if decision.entry_price is not None:
            self._trade.entry_price = decision.entry_price
        self._trade.entry_order_id = order_id
        self.telemetry.log(
            (
                f"✅ 已提交 {side} 市价单，数量 {self._quantity_as_float(quantity):.6f}，"
                f"杠杆 {self._last_effective_leverage:.2f}x，名义≈{self._last_target_notional:.2f} USDT"
            ),
            str(self._instrument_id),
        )
        self._update_triggers(decision)

    def _resolve_order_quantity(
        self,
        price: float,
        equity: float,
        available_margin: float,
        decision: AIDecision,
    ) -> Quantity:
        """
        根据风险预算、杠杆与合约精度生成下单数量。
        """

        instrument: Optional[Instrument] = None
        try:
            instrument = self.instrument(self._instrument_id)  # type: ignore[attr-defined]
        except Exception:
            instrument = None

        instrument_min_qty = 0.0
        if instrument is not None:
            try:
                min_quantity = getattr(instrument, "min_quantity", None)
                if min_quantity is not None:
                    instrument_min_qty = max(self._quantity_as_float(min_quantity), 0.0)
            except Exception:
                instrument_min_qty = 0.0

        effective_price = price if price > 0 else self._last_price
        if effective_price <= 0 and instrument is not None:
            try:
                effective_price = float(getattr(instrument, "last_price", 0.0))  # type: ignore[arg-type]
            except Exception:
                effective_price = 0.0
        effective_price = max(effective_price, 0.0)
        risk_cap = max(self.risk.settings.max_risk_per_trade, 0.0)
        decision_risk = decision.risk_percent
        if decision_risk is not None and decision_risk > 0:
            normalized = decision_risk / 100.0 if decision_risk > 1 else decision_risk
            risk_cap = min(risk_cap, max(normalized, 0.0))
        risk_cap = min(risk_cap, 1.0)

        margin_budget = max(equity * risk_cap, 0.0)
        if available_margin > 0:
            margin_budget = min(margin_budget, available_margin) if margin_budget > 0 else available_margin

        desired_leverage = decision.leverage or self._default_leverage
        if desired_leverage is None or desired_leverage <= 0:
            desired_leverage = self._default_leverage
        effective_leverage = min(self._max_leverage, max(float(desired_leverage), 1.0))

        target_qty = 0.0
        explicit_qty = decision.quantity if decision.quantity and decision.quantity > 0 else None
        if explicit_qty is not None:
            target_qty = float(explicit_qty)
        elif decision.notional and decision.notional > 0 and effective_price > 0:
            target_qty = float(decision.notional) / effective_price

        max_qty_from_risk = 0.0
        if effective_price > 0 and margin_budget > 0:
            max_qty_from_risk = (margin_budget * effective_leverage) / effective_price
            if target_qty <= 0:
                target_qty = max_qty_from_risk

        if max_qty_from_risk > 0 and target_qty > max_qty_from_risk > 0:
            self.telemetry.log(
                f"⚠️ AI 请求仓位超出风险上限，按风险预算缩减至 {max_qty_from_risk:.6f}",
                str(self._instrument_id),
            )
            target_qty = max_qty_from_risk

        if effective_price > 0 and effective_leverage > 0 and available_margin > 0 and target_qty > 0:
            required_margin = (target_qty * effective_price) / effective_leverage
            if required_margin > available_margin and required_margin > 0:
                scale = available_margin / required_margin
                target_qty *= max(scale, 0.0)
                self.telemetry.log(
                    f"⚠️ 可用保证金不足，缩减手数至 {target_qty:.6f}",
                    str(self._instrument_id),
                )

        if target_qty <= 0:
            fallback_qty = float(self._trade_size) if self._trade_size > 0 else instrument_min_qty
            if fallback_qty > 0:
                target_qty = fallback_qty
                self.telemetry.log(
                    f"⚠️ AI 未返回仓位规模，使用回退数量 {target_qty:.6f}",
                    str(self._instrument_id),
                )
            else:
                raise RuntimeError("AI 未提供仓位规模且无法推导最小下单量。")

        required_margin = 0.0
        if effective_price > 0 and effective_leverage > 0:
            required_margin = (target_qty * effective_price) / effective_leverage

        risk_fraction = required_margin / equity if equity > 0 else 0.0
        capped_fraction = min(risk_fraction, risk_cap) if risk_cap > 0 else risk_fraction
        self._last_risk_fraction = max(capped_fraction, 0.0)
        self._last_effective_leverage = effective_leverage
        self._last_target_notional = target_qty * effective_price if effective_price > 0 else 0.0

        if instrument is not None:
            try:
                quantity = instrument.make_qty(float(target_qty))  # type: ignore[attr-defined]
                if quantity is not None and self._quantity_as_float(quantity) > 0:
                    return quantity
            except Exception as exc:
                self.telemetry.log(
                    f"⚠️ 合约精度转换失败，改用通用解析：{exc}",
                    str(self._instrument_id),
                )

        normalized = Decimal(str(target_qty)).normalize()
        size_str = format(normalized, "f")
        try:
            return Quantity.from_str(size_str)  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(f"无法将数量 {size_str} 转换为 Quantity") from exc

    def _close_position(self, decision: AIDecision, payload: DecisionPayload) -> None:
        if not self._state.has_position:
            self.telemetry.log("⚠️ 当前无持仓，忽略平仓信号。", str(self._instrument_id))
            return

        try:
            # 使用 close_all_positions 支持通过 InstrumentId 直接平掉该品种的所有仓位
            self.close_all_positions(instrument_id=self._instrument_id)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            self.telemetry.log(f"❌ 平仓指令失败：{exc}", str(self._instrument_id))
            return

        self.telemetry.log("💰 已提交平仓指令。", str(self._instrument_id))
        self._state.is_closing = True
        self._state.last_trigger_reason = self._pending_trigger_reason or decision.reasoning
        self._trade.exit_reason = decision.reasoning
        serialized_triggers = [self._serialize_trigger(trigger) for trigger in decision.triggers]
        if serialized_triggers:
            self._trade.triggers = serialized_triggers
        self._trade.sentiment = payload.sentiment
        self._trade.market_regime = payload.market_regime
        if payload.news_digest:
            self._trade.news_digest = payload.news_digest
        self._trade.context_notes.extend(self._latest_context.get("context_notes", [])[-3:])
        self._trigger_manager.clear()

    # ------------------------------------------------------------------ #
    # 订单与成交事件
    # ------------------------------------------------------------------ #

    def on_order_filled(self, event: OrderFilled) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        ctx = self._get_context(getattr(event, "instrument_id", None))
        if ctx is None:
            return
        with self._activate_context(ctx):
            self._handle_order_filled(event)

    def _handle_order_filled(self, event: OrderFilled) -> None:
        price = self._safe_float(getattr(event, "last_px", None))
        quantity = abs(self._safe_float(getattr(event, "last_qty", None)))
        if quantity <= 0:
            return

        timestamp = self._event_time(getattr(event, "ts_event", None))
        order_id = getattr(event, "client_order_id", None) or getattr(event, "venue_order_id", None)
        order_side = self._normalize_side(getattr(event, "order_side", None))
        fill_record = {
            "time": timestamp.isoformat(),
            "price": price,
            "quantity": quantity,
            "side": order_side,
        }
        self._trade.fills.append(fill_record)
        self._last_price = price

        if self._trade.entry_time is None:
            prev_filled = self._trade.entry_filled
            self._trade.entry_filled += quantity
            if prev_filled > 0:
                weighted = (self._trade.entry_price or price) * prev_filled + price * quantity
                self._trade.entry_price = weighted / self._trade.entry_filled
            else:
                self._trade.entry_price = self._trade.entry_price or price
                self._trade.entry_time = timestamp
            self._trade.entry_order_id = self._trade.entry_order_id or order_id
            self._trade.quantity = max(self._trade.entry_filled, self._trade.quantity)

            self._state.has_position = True
            self._state.pending_side = None
            self._state.pending_quantity = 0.0
            self._state.position_side = "LONG" if order_side == "BUY" else "SHORT"
            self._state.position_size = self._state.position_size + quantity if prev_filled else quantity
            self._state.entry_price = self._trade.entry_price
            self.telemetry.log(
                f"📈 开仓成交：{self._state.position_side} 数量 {quantity:.6f}，价格 {price:.6f}",
                str(self._instrument_id),
            )
        else:
            self._trade.exit_time = timestamp
            self._trade.exit_price = price
            self._trade.exit_order_id = order_id
            self._state.position_size = max(self._state.position_size - quantity, 0.0)
            self.telemetry.log(
                f"📉 平仓成交：数量 {quantity:.6f}，价格 {price:.6f}",
                str(self._instrument_id),
            )

        self._push_bot_state()

    def on_order_canceled(self, event: OrderCanceled) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        ctx = self._get_context(getattr(event, "instrument_id", None))
        if ctx is None:
            return
        with self._activate_context(ctx):
            if self._state.is_closing:
                self.telemetry.log("⚠️ 平仓订单被取消，保持持仓等待。", str(self._instrument_id))
                self._state.is_closing = False
                return

            if self._trade.entry_time is None:
                self.telemetry.log("⚠️ 开仓订单被取消。", str(self._instrument_id))
                self._reset_pending_order()

    def on_order_rejected(self, event: OrderRejected) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        ctx = self._get_context(getattr(event, "instrument_id", None))
        if ctx is None:
            return
        with self._activate_context(ctx):
            reason = getattr(event, "reason", "未知原因")
            if self._state.is_closing:
                self.telemetry.log(f"❌ 平仓订单被拒绝：{reason}", str(self._instrument_id))
                self._state.is_closing = False
                return

            self.telemetry.log(f"❌ 开仓订单被拒绝：{reason}", str(self._instrument_id))
            self._reset_pending_order()

    def on_position_closed(self, event: PositionClosed) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        ctx = self._get_context(getattr(event, "instrument_id", None))
        if ctx is None:
            return
        with self._activate_context(ctx):
            self.telemetry.log("✅ 仓位已完全关闭。", str(self._instrument_id))
            close_price = self._safe_float(getattr(event, "avg_px_close", None)) or self._safe_float(
                getattr(event, "last_px", None),
            )
            close_time = self._event_time(getattr(event, "ts_closed", None) or getattr(event, "ts_event", None))
            pnl = self._safe_float(getattr(event, "realized_pnl", None))
            pnl_pct = self._safe_float(getattr(event, "realized_return", None))

            self._trade.exit_price = close_price or self._trade.exit_price or self._last_price
            self._trade.exit_time = close_time
            self._trade.realized_pnl = pnl
            self._trade.realized_return = pnl_pct
            self._trade.exit_order_id = getattr(event, "closing_order_id", None) or self._trade.exit_order_id

            self._finalize_trade()
            self._state = StrategyState()
            self._trigger_manager.clear()
            self._push_bot_state()

    # ------------------------------------------------------------------ #
    # 状态与上下文更新

    # ------------------------------------------------------------------ #
    # 状态与上下文更新
    # ------------------------------------------------------------------ #

    def _append_bar(self, bar: Bar) -> None:
        timestamp = getattr(bar, "end_time", self.clock.utc_now())
        row = {
            "date": pd.Timestamp(timestamp).to_pydatetime(),
            "open": float(getattr(bar, "open", 0.0)),
            "high": float(getattr(bar, "high", 0.0)),
            "low": float(getattr(bar, "low", 0.0)),
            "close": float(getattr(bar, "close", 0.0)),
            "volume": float(getattr(bar, "volume", 0.0)),
        }
        self._history.append(row)
        self._last_price = row["close"]
        if not self._history_log_flag:
            prefix = "历史" if getattr(bar, "is_historical", False) else "实时"
            self.telemetry.log(
                f"🧮 {prefix} K 线追加：当前缓存 {len(self._history)} / {self._instrument_config.min_history}",
                str(self._instrument_id),
            )
            if len(self._history) >= self._instrument_config.min_history:
                self._history_log_flag = True

    @property
    def _history_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self._history)
        if df.empty:
            return df
        return df.set_index("date")

    def _describe_position(self) -> str:
        if not self._state.has_position:
            return "FLAT - 未持仓"
        return (
            f"持有 {self._state.position_side} 仓位，数量 {self._state.position_size:.4f}"
            f"，入场价 {self._state.entry_price or 0:.4f}"
        )

    def _update_triggers(self, decision: AIDecision) -> None:
        triggers = decision.triggers
        timeout = decision.trigger_timeout
        if triggers:
            self._trigger_manager.update(triggers, timeout)
            self.telemetry.log(f"🔔 安装 {len(triggers)} 个触发器。", str(self._instrument_id))
        else:
            self._trigger_manager.clear()

    def _update_context(self, payload: DecisionPayload) -> None:
        if payload.context_update:
            self._context_history.append(json.dumps(payload.context_update, ensure_ascii=False))
        context_notes = list(self._context_history)[-5:]
        self._latest_context.update(
            {
                "market_context": payload.context_update,
                "raw_response": payload.raw_response,
                "chain_of_thought": payload.chain_of_thought,
                "news_digest": payload.news_digest,
                "market_regime": payload.market_regime,
                "sentiment": payload.sentiment,
                "context_notes": context_notes,
            },
        )

    def _update_bot_state(self, bar: Bar, payload: DecisionPayload) -> None:
        price = float(getattr(bar, "close", 0.0))
        context = {
            "market_context": self._latest_context.get("market_context", payload.context_update),
            "last_ai_response": self._latest_context.get("raw_response", payload.raw_response),
            "last_sentiment_score": self._latest_context.get("sentiment", payload.sentiment),
            "last_known_price": price,
            "chain_of_thought": self._latest_context.get("chain_of_thought", payload.chain_of_thought),
            "active_triggers": [self._serialize_trigger(trigger) for trigger in self._trigger_manager.triggers],
            "trigger_reason": self._pending_trigger_reason,
            "news_digest": self._latest_context.get("news_digest", payload.news_digest),
            "market_regime": self._latest_context.get("market_regime", payload.market_regime),
        }
        context["context_notes"] = self._latest_context.get("context_notes", [])
        self._latest_context = context
        position = PositionRecord(
            side=self._state.position_side,
            entry_price=self._state.entry_price,
            quantity=self._state.position_size,
            unrealized_pnl=self._compute_unrealized_pnl(price),
        )
        self.telemetry.update_bot_state(
            symbol=str(self._instrument_id),
            is_in_position=self._state.has_position,
            position=position,
            context=context,
        )
        try:
            self.telemetry.write_dashboard_snapshot(self.risk.status)
        except Exception:
            pass

    @staticmethod
    def _serialize_trigger(trigger: Any) -> Dict[str, Any]:
        """将触发器对象安全转换为可序列化字典。"""

        if is_dataclass(trigger):
            return asdict(trigger)
        if isinstance(trigger, dict):
            return trigger
        if hasattr(trigger, "to_dict"):
            try:
                return trigger.to_dict()  # type: ignore[return-value]
            except Exception:
                pass
        if hasattr(trigger, "__dict__"):
            return {key: value for key, value in vars(trigger).items() if not key.startswith("_")}
        return {"repr": repr(trigger)}

    @staticmethod
    def _quantity_as_float(quantity: Any) -> float:
        """提取 Quantity 数值表示，失败时回退为 0."""

        if quantity is None:
            return 0.0
        if isinstance(quantity, (int, float)):
            return float(quantity)
        for attr in ("as_f64", "as_double", "to_double", "to_float"):
            if hasattr(quantity, attr):
                try:
                    return float(getattr(quantity, attr)())
                except Exception:
                    continue
        try:
            return float(quantity)
        except Exception:
            return 0.0

    def _current_account_vitals(self) -> Dict[str, float]:
        try:
            total_equity = float(self.portfolio.net_asset_value)  # type: ignore[attr-defined]
        except Exception:
            total_equity = 0.0
        try:
            available = float(self.portfolio.available_balance)  # type: ignore[attr-defined]
        except Exception:
            available = total_equity
        if total_equity <= 0.0 and self._initial_equity > 0.0:
            total_equity = self._initial_equity
        else:
            self._initial_equity = max(total_equity, self._initial_equity)
        if available <= 0.0:
            if self._initial_available_margin > 0.0:
                available = self._initial_available_margin
            elif total_equity > 0.0:
                available = total_equity
        else:
            self._initial_available_margin = max(available, self._initial_available_margin)
        return {"total_equity": total_equity, "available_margin": available}

    def _push_bot_state(self) -> None:
        context = dict(self._latest_context)
        context.setdefault("last_known_price", self._last_price)
        context.setdefault(
            "active_triggers",
            [self._serialize_trigger(trigger) for trigger in self._trigger_manager.triggers],
        )
        context.setdefault("trigger_reason", self._state.last_trigger_reason)
        position = PositionRecord(
            side=self._state.position_side,
            entry_price=self._state.entry_price,
            quantity=self._state.position_size,
            unrealized_pnl=self._compute_unrealized_pnl(self._last_price),
        )
        self.telemetry.update_bot_state(
            symbol=str(self._instrument_id),
            is_in_position=self._state.has_position,
            position=position,
            context=context,
        )

    def _reset_pending_order(self) -> None:
        self._state.pending_side = None
        self._state.pending_quantity = 0.0
        self._state.open_order_id = None
        self._state.last_trigger_reason = ""
        if not self._state.has_position:
            self._trade = TradeLifecycle(symbol=str(self._instrument_id))
        self._push_bot_state()

    @staticmethod
    def _safe_float(value: Any) -> float:
        if value is None:
            return 0.0
        if hasattr(value, "as_double"):
            try:
                return float(value.as_double())
            except Exception:
                pass
        if hasattr(value, "value"):
            try:
                return float(value.value)
            except Exception:
                pass
        try:
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_side(side: Any) -> str:
        if side is None:
            return ""
        if hasattr(side, "name"):
            return str(side.name).upper()
        if hasattr(side, "value"):
            return str(side.value).upper()
        return str(side).upper()

    def _event_time(self, ts_ns: Optional[int]) -> datetime:
        if not ts_ns:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _compute_unrealized_pnl(self, price: float) -> float:
        if not self._state.has_position or not self._state.entry_price:
            return 0.0
        direction = 1.0 if (self._state.position_side or "").upper() == "LONG" else -1.0
        return (price - (self._state.entry_price or 0.0)) * self._state.position_size * direction

    def _finalize_trade(self) -> None:
        if self._trade.entry_time is None:
            self._reset_pending_order()
            return

        exit_time = self._trade.exit_time or datetime.now(timezone.utc)
        exit_price = self._trade.exit_price or self._last_price
        pnl = self._trade.realized_pnl
        if pnl is None and self._trade.entry_price is not None:
            direction = 1.0 if (self._trade.side or "LONG").upper() == "LONG" else -1.0
            pnl = (exit_price - self._trade.entry_price) * self._trade.quantity * direction
        pnl = pnl or 0.0

        pnl_pct = self._trade.realized_return
        if pnl_pct is None and self._trade.entry_price:
            pnl_pct = (exit_price - self._trade.entry_price) / self._trade.entry_price * 100.0
        elif pnl_pct is not None:
            pnl_pct = pnl_pct * 100 if abs(pnl_pct) <= 1 else pnl_pct

        summary = self._compose_trade_summary(self._trade, pnl, pnl_pct or 0.0)
        self.telemetry.record_trade(
            symbol=self._trade.symbol,
            side=self._trade.side or "",
            entry_price=self._trade.entry_price or 0.0,
            exit_price=exit_price,
            quantity=self._trade.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct or 0.0,
            reasoning=summary,
        )
        self.risk.record_trade_result(pnl, exit_time.timestamp())
        if summary:
            self.ai_service.summarize_trade(summary, self._trade.symbol)
        self.telemetry.log(f"📝 交易完成：{summary}", self._trade.symbol)

        self._state.has_position = False
        self._state.pending_side = None
        self._state.pending_quantity = 0.0
        self._state.open_order_id = None
        self._state.is_closing = False
        self._push_bot_state()
        self._trade = TradeLifecycle(symbol=str(self._instrument_id))

    def _compose_trade_summary(self, trade: TradeLifecycle, pnl: float, pnl_pct: float) -> str:
        parts: List[str] = []
        if trade.side:
            parts.append(f"方向 {trade.side}")
        parts.append(f"手数 {trade.quantity:.6f}")
        if trade.entry_price is not None and trade.entry_time is not None:
            parts.append(
                f"开仓 {trade.entry_price:.6f} @ {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
        if trade.exit_price is not None and trade.exit_time is not None:
            parts.append(
                f"平仓 {trade.exit_price:.6f} @ {trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
        if trade.leverage:
            parts.append(f"杠杆 {trade.leverage:.2f}x")
        if trade.risk_fraction:
            parts.append(f"风险占用 {trade.risk_fraction * 100:.2f}%")
        if trade.notional:
            parts.append(f"名义 {trade.notional:.2f}")
        parts.append(f"PnL {pnl:.4f} ({pnl_pct:.2f}%)")
        if trade.trigger_reason:
            parts.append(f"触发：{trade.trigger_reason}")
        if trade.reasoning:
            parts.append(f"决策要点：{trade.reasoning}")
        if trade.exit_reason:
            parts.append(f"离场依据：{trade.exit_reason}")
        if trade.sentiment is not None:
            parts.append(f"情绪分：{trade.sentiment:.2f}")
        if trade.market_regime:
            parts.append(f"市场状态：{trade.market_regime}")
        if trade.context_notes:
            parts.append(f"上下文：{' | '.join(trade.context_notes[-2:])}")
        return "；".join(parts)
