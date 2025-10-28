from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Deque, Dict, List, Optional

import json

import pandas as pd
from msgspec import Struct

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
    from nautilus_trader.model.enums import OrderSide, TimeInForce
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
        @staticmethod
        def from_f64(value: float) -> float:
            return value


class LLMStrategyConfig(StrategyConfig, Struct, kw_only=True, frozen=True):
    instrument_id: str
    bar_type: str
    trade_size: float
    min_history: int = 120
    bar_history: int = 720
    analysis_cooldown_secs: int = 300
    order_id_tag: str = "LLM"


class LLMStrategy(Strategy):
    """基于 Nautilus 事件引擎的 LLM 决策策略。"""

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

        self.ai_service = ai_service
        self.risk = risk_controller
        self.telemetry = telemetry

        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type = BarType.from_str(config.bar_type)
        self._history: Deque[dict] = deque(maxlen=config.bar_history)
        self._context_history: Deque[str] = deque(maxlen=12)
        self._state: StrategyState = StrategyState()
        self._trigger_manager = TriggerManager(symbol=str(self._instrument_id))
        self._last_analysis_ts: float = 0.0
        self._trade_size = Decimal(str(config.trade_size))
        self._trade = TradeLifecycle(symbol=str(self._instrument_id))
        self._latest_context: Dict[str, Any] = {}
        self._pending_trigger_reason: str = ""
        self._last_price: float = 0.0
        self._instrument_ready: bool = False
        self._bars_subscribed: bool = False

    # ------------------------------------------------------------------ #
    # Nautilus 生命周期钩子
    # ------------------------------------------------------------------ #

    def on_start(self) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        self.telemetry.log("🚀 LLM 策略启动", str(self._instrument_id))
        try:
            self.ai_service.initialize()
        except Exception as exc:  # noqa: BLE001
            self.telemetry.log(f"⚠️ AI 服务初始化失败：{exc}", str(self._instrument_id))
        try:
            instrument = self.request_instrument(self._instrument_id)  # type: ignore[attr-defined]
            if instrument is not None:
                self._handle_instrument_ready(instrument)  # type: ignore[arg-type]
        except AttributeError:
            # 本地降级模式仅用于静态检查
            pass
        except Exception as exc:  # noqa: BLE001
            self.telemetry.log(f"⚠️ 请求合约信息失败：{exc}", str(self._instrument_id))

    def on_instrument(self, instrument: Instrument) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        if str(getattr(instrument, "id", instrument)) != str(self._instrument_id):
            return
        self._handle_instrument_ready(instrument)

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
        self._subscribe_market_data()

    def _subscribe_market_data(self) -> None:
        if self._bars_subscribed:
            return
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
        self._append_bar(bar)
        if len(self._history) < self.config.min_history:
            return

        now_ts = bar.end_time.timestamp() if hasattr(bar, "end_time") else self.clock.utc_now().timestamp()
        reason_text = "定期分析"
        if now_ts - self._last_analysis_ts < self.config.analysis_cooldown_secs:
            should, reason = self._trigger_manager.should_analyze(self._history_df, self.config.analysis_cooldown_secs)
            if not should:
                self._pending_trigger_reason = ""
                return
            reason_text = reason or "触发器命中"
        self._pending_trigger_reason = reason_text
        if reason_text:
            self.telemetry.log(f"📊 触发分析原因：{reason_text}", str(self._instrument_id))
        self._last_analysis_ts = now_ts

        snapshot = MarketSnapshot(
            instrument_id=str(self._instrument_id),
            timeframe=self.config.bar_type,
            current_price=float(getattr(bar, "close", 0.0)),
            ohlcv=self._history_df.copy(),
        )
        position_text = self._describe_position()
        context_summary = "\n".join(self._context_history) or "无历史上下文。"
        vitals = self._current_account_vitals()

        payload = self.ai_service.request_decision(
            snapshot=snapshot,
            position_text=position_text,
            context_summary=context_summary,
            live_equity=vitals.get("total_equity", 0.0),
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

        quantity = None
        try:
            instrument = self.instrument(self._instrument_id)  # type: ignore[attr-defined]
        except Exception:
            instrument = None

        if instrument is not None:
            try:
                quantity = instrument.make_qty(float(self._trade_size))  # type: ignore[attr-defined]
            except Exception:
                quantity = None

        if quantity is None:
            try:
                quantity = Quantity.from_f64(float(self._trade_size))  # type: ignore[attr-defined]
            except Exception:
                quantity = float(self._trade_size)

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
        if decision.entry_price is not None:
            self._trade.entry_price = decision.entry_price
        self._trade.entry_order_id = order_id
        self.telemetry.log(
            f"✅ 已提交 {side} 市价单，数量 {self._quantity_as_float(quantity):.6f}",
            str(self._instrument_id),
        )
        self._update_triggers(decision)

    def _close_position(self, decision: AIDecision, payload: DecisionPayload) -> None:
        if not self._state.has_position:
            self.telemetry.log("⚠️ 当前无持仓，忽略平仓信号。", str(self._instrument_id))
            return

        try:
            self.close_position(self._instrument_id)  # type: ignore[attr-defined]
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
        if not self._is_same_instrument(event.instrument_id):
            return

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
        if not self._is_same_instrument(event.instrument_id):
            return

        if self._state.is_closing:
            self.telemetry.log("⚠️ 平仓订单被取消，保持持仓等待。", str(self._instrument_id))
            self._state.is_closing = False
            return

        if self._trade.entry_time is None:
            self.telemetry.log("⚠️ 开仓订单被取消。", str(self._instrument_id))
            self._reset_pending_order()

    def on_order_rejected(self, event: OrderRejected) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        if not self._is_same_instrument(event.instrument_id):
            return

        reason = getattr(event, "reason", "未知原因")
        if self._state.is_closing:
            self.telemetry.log(f"❌ 平仓订单被拒绝：{reason}", str(self._instrument_id))
            self._state.is_closing = False
            return

        self.telemetry.log(f"❌ 开仓订单被拒绝：{reason}", str(self._instrument_id))
        self._reset_pending_order()

    def on_position_closed(self, event: PositionClosed) -> None:  # pragma: no cover - 依赖 Nautilus 回调
        if not self._is_same_instrument(event.instrument_id):
            return

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

    def _is_same_instrument(self, instrument_id: Any) -> bool:
        return str(instrument_id) == str(self._instrument_id)

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
