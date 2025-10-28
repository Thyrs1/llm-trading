from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta

from nautilus_bot.types import TriggerSpec


class TriggerManager:
    """管理 AI 返回的动态触发条件。"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.triggers: List[TriggerSpec] = []
        self.timeout_at: Optional[float] = None
        self._last_check = 0.0

    def update(self, triggers: List[TriggerSpec], timeout_seconds: Optional[int]) -> None:
        self.triggers = triggers
        self.timeout_at = time.time() + timeout_seconds if timeout_seconds else None
        self._last_check = time.time()

    def clear(self) -> None:
        self.triggers = []
        self.timeout_at = None

    def should_analyze(self, df_5m: pd.DataFrame, fallback_interval: int) -> Tuple[bool, str]:
        """根据触发条件判断是否需要重新分析。"""

        if not self.triggers:
            now = time.time()
            if now - self._last_check >= fallback_interval:
                self._last_check = now
                return True, "定期触发"
            return False, ""

        if self.timeout_at and time.time() > self.timeout_at:
            self.clear()
            return True, "触发器超时"

        for trigger in self.triggers:
            if self._is_condition_met(trigger, df_5m):
                label = trigger.label or trigger.type
                return True, f"触发器命中: {label}"

        return False, ""

    def _is_condition_met(self, trigger: TriggerSpec, df_5m: pd.DataFrame) -> bool:
        if df_5m.empty:
            return False
        latest = df_5m.iloc[-1]
        params = trigger.params
        try:
            if trigger.type == "PRICE_CROSS":
                level = float(params.get("level", 0))
                direction = params.get("direction")
                return (direction == "ABOVE" and latest["high"] >= level) or (
                    direction == "BELOW" and latest["low"] <= level
                )
            if trigger.type == "RSI_CROSS":
                level = float(params.get("level", 0))
                direction = params.get("direction")
                current_rsi = ta.rsi(df_5m["close"], length=int(params.get("length", 14))).iloc[-1]
                return (direction == "ABOVE" and current_rsi >= level) or (
                    direction == "BELOW" and current_rsi <= level
                )
            if trigger.type == "EMA_CROSS":
                fast = ta.ema(df_5m["close"], length=int(params.get("fast", 20)))
                slow = ta.ema(df_5m["close"], length=int(params.get("slow", 50)))
                direction = params.get("direction")
                if len(fast) < 2 or len(slow) < 2:
                    return False
                if direction == "GOLDEN":
                    return fast.iloc[-2] < slow.iloc[-2] and fast.iloc[-1] >= slow.iloc[-1]
                if direction == "DEATH":
                    return fast.iloc[-2] > slow.iloc[-2] and fast.iloc[-1] <= slow.iloc[-1]
            if trigger.type == "PRICE_EMA_DISTANCE":
                period = int(params.get("period", 20))
                target_pct = float(params.get("percent", 0))
                condition = params.get("condition")
                ema_val = ta.ema(df_5m["close"], length=period).iloc[-1]
                distance_pct = (latest["close"] - ema_val) / ema_val * 100
                if condition == "BELOW":
                    return distance_pct <= target_pct
                if condition == "ABOVE":
                    return distance_pct >= target_pct
            if trigger.type == "BBAND_WIDTH":
                period = int(params.get("period", 20))
                target_pct = float(params.get("percent", 0))
                condition = params.get("condition")
                bbands = ta.bbands(df_5m["close"], length=period)
                width_pct = (bbands[f"BBU_{period}_2.0"] - bbands[f"BBL_{period}_2.0"]) / bbands[
                    f"BBM_{period}_2.0"
                ] * 100
                current_width = width_pct.iloc[-1]
                if condition == "BELOW":
                    return current_width <= target_pct
                if condition == "ABOVE":
                    return current_width >= target_pct
            if trigger.type == "MACD_HIST_SIGN":
                condition = params.get("condition")
                macd = ta.macd(df_5m["close"])
                hist = macd["MACDh_12_26_9"]
                if len(hist) < 2:
                    return False
                if condition == "POSITIVE":
                    return hist.iloc[-2] <= 0 < hist.iloc[-1]
                if condition == "NEGATIVE":
                    return hist.iloc[-2] >= 0 > hist.iloc[-1]
        except Exception:
            return False
        return False
