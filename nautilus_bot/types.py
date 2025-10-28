from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional


class DecisionAction(str, Enum):
    """AI 决策动作类型。"""

    OPEN = "OPEN_POSITION"
    CLOSE = "CLOSE_POSITION"
    MODIFY = "MODIFY_POSITION"
    WAIT = "WAIT"


@dataclass(slots=True)
class TriggerSpec:
    """触发条件定义。"""

    label: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AIDecision:
    """AI 返回的结构化决策。"""

    action: DecisionAction
    reasoning: str
    confidence: Optional[str] = None
    side: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: Optional[int] = None
    risk_percent: Optional[float] = None
    trailing_distance_pct: Optional[float] = None
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    trigger_timeout: Optional[int] = None
    triggers: List[TriggerSpec] = field(default_factory=list)
    raw_decision: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MarketSnapshot:
    """策略分析时使用的行情快照。"""

    instrument_id: str
    timeframe: str
    current_price: float
    ohlcv: "pd.DataFrame"  # type: ignore[name-defined]
    metadata: Dict[str, Any] = field(default_factory=dict)


if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


@dataclass(slots=True)
class StrategyState:
    """策略运行期状态，用于风险与持仓协同。"""

    has_position: bool = False
    position_side: Optional[str] = None
    position_size: float = 0.0
    entry_price: Optional[float] = None
    last_decision: Optional[AIDecision] = None
    open_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    is_closing: bool = False
    pending_quantity: float = 0.0
    pending_side: Optional[str] = None
    last_trigger_reason: str = ""


@dataclass(slots=True)
class TradeLifecycle:
    """跟踪单笔交易的生命周期，用于总结与遥测。"""

    symbol: str
    side: Optional[str] = None
    quantity: float = 0.0
    entry_filled: float = 0.0
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    entry_order_id: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_order_id: Optional[str] = None
    realized_pnl: Optional[float] = None
    realized_return: Optional[float] = None
    reasoning: str = ""
    exit_reason: str = ""
    trigger_reason: str = ""
    context_notes: List[str] = field(default_factory=list)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Optional[float] = None
    market_regime: Optional[str] = None
    news_digest: str = ""
    fills: List[Dict[str, Any]] = field(default_factory=list)
    leverage: float = 1.0
    risk_fraction: float = 0.0
    notional: float = 0.0
