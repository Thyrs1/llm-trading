from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from nautilus_bot.config import RiskSettings


@dataclass(slots=True)
class RiskStatus:
    """当前风险状态快照。"""

    daily_start_equity: float = 0.0
    daily_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    trading_halted_until: float = 0.0
    last_check_day: int = -1
    is_trading_halted: bool = False


class RiskController:
    """实现日内回撤与连亏控制。"""

    def __init__(self, settings: RiskSettings):
        self.settings = settings
        self.status = RiskStatus()

    # ------------------------------------------------------------------ #
    # 状态更新与校验
    # ------------------------------------------------------------------ #

    def refresh_daily_state(self, equity: float, now: datetime) -> None:
        """每日开始或首次调用时重置基准。"""

        current_day = now.timetuple().tm_yday
        if self.status.last_check_day != current_day:
            self.status.daily_start_equity = max(equity, 0.0)
            self.status.daily_drawdown_pct = 0.0
            self.status.consecutive_losses = 0
            self.status.last_check_day = current_day
            self.status.is_trading_halted = False
            self.status.trading_halted_until = 0.0

    def update_drawdown(self, equity: float) -> None:
        """更新回撤百分比。"""

        if self.status.daily_start_equity <= 0:
            return
        drawdown = (self.status.daily_start_equity - equity) / self.status.daily_start_equity * 100.0
        self.status.daily_drawdown_pct = max(self.status.daily_drawdown_pct, drawdown)
        if drawdown >= self.settings.max_daily_drawdown_pct:
            self.status.is_trading_halted = True

    def check_halt(self, now_ts: float) -> bool:
        """检查是否处于风控暂停期。"""

        if not self.status.is_trading_halted:
            return False
        return now_ts < self.status.trading_halted_until

    def schedule_halt(self, now_ts: float) -> None:
        """触发暂停交易。"""

        self.status.is_trading_halted = True
        self.status.trading_halted_until = now_ts + self.settings.trading_pause_seconds

    def record_trade_result(self, pnl: float, now_ts: float) -> None:
        """记录单笔交易盈亏，更新连亏状态。"""

        if pnl >= 0:
            self.status.consecutive_losses = 0
            return
        self.status.consecutive_losses += 1
        if self.status.consecutive_losses >= self.settings.max_consecutive_losses:
            self.schedule_halt(now_ts)

    def can_open_new_trade(self, equity: float, now: datetime, now_ts: float) -> bool:
        """综合判断当前是否允许开仓。"""

        self.refresh_daily_state(equity, now)
        self.update_drawdown(equity)
        if self.check_halt(now_ts):
            return False
        if self.status.daily_drawdown_pct >= self.settings.max_daily_drawdown_pct:
            self.schedule_halt(now_ts)
            return False
        return True
