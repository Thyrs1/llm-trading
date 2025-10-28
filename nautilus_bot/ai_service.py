from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import json
import re

import pandas as pd
import pandas_ta as ta
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from nautilus_bot.config import BotSettings
from nautilus_bot.types import AIDecision, DecisionAction, MarketSnapshot, TriggerSpec


@dataclass(slots=True)
class DecisionPayload:
    """AI 决策调用的返回载体。"""

    decision: Optional[AIDecision]
    context_update: Dict[str, Any]
    raw_response: str
    chain_of_thought: str
    sentiment: float
    news_digest: str
    market_regime: str


class AIService:
    """
    负责 LLM 决策、情绪分析与新闻聚合，完全独立于旧版模块。
    """

    def __init__(self, settings: BotSettings):
        self.settings = settings
        self._client: Optional[OpenAI] = None
        self._sentiment_analyzer = None
        self._memory_file = Path("trade_memory.txt")
        self._initialized = False

    # ------------------------------------------------------------------ #
    # 生命周期管理
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """初始化 LLM 客户端与情绪分析器。"""

        if self._initialized:
            return

        self._client = OpenAI(
            api_key=self.settings.ai.api_key,
            base_url=self.settings.ai.base_url,
        )
        # 触发一次模型列表请求以验证凭证
        self._client.models.list()

        if self.settings.ai.enable_sentiment:
            self._initialize_finbert()

        self._initialized = True

    # ------------------------------------------------------------------ #
    # 决策流程
    # ------------------------------------------------------------------ #

    def request_decision(
        self,
        snapshot: MarketSnapshot,
        position_text: str,
        context_summary: str,
        live_equity: float,
        active_triggers: Optional[List[Dict[str, Any]]] = None,
        trigger_reason: str = "",
    ) -> DecisionPayload:
        """
        基于最新行情快照请求 AI 做出交易决策。
        """

        if not self._initialized or self._client is None:
            raise RuntimeError("AI 服务尚未初始化，请先调用 initialize()。")

        analysis_text = self._analyze_market(snapshot.ohlcv, snapshot.current_price)
        news_digest = self._collect_news(snapshot.instrument_id)
        sentiment = self._compute_sentiment(news_digest)
        market_regime = self._determine_market_regime(snapshot.ohlcv)
        active_triggers = active_triggers or snapshot.metadata.get("active_triggers", [])
        trigger_reason = trigger_reason or snapshot.metadata.get("pending_trigger_reason", "")

        lessons = self._load_lessons()
        prompt_body = self._compose_prompt(
            position_text=position_text,
            context_summary=context_summary,
            analysis_text=analysis_text,
            news_digest=news_digest,
            sentiment=sentiment,
            market_regime=market_regime,
            live_equity=live_equity,
            lessons=lessons,
            active_triggers=active_triggers,
            trigger_reason=trigger_reason,
        )

        response = self._client.chat.completions.create(
            model=self.settings.ai.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt_body},
            ],
            temperature=0.0,
            timeout=45,
        )
        raw_response = response.choices[0].message.content or ""
        decision = _parse_decision(_extract_decision_block(raw_response))
        context_update = _extract_context_block(raw_response)
        chain_of_thought = _extract_chain_of_thought(raw_response)

        return DecisionPayload(
            decision=decision,
            context_update=context_update,
            raw_response=raw_response,
            chain_of_thought=chain_of_thought,
            sentiment=sentiment,
            news_digest=news_digest,
            market_regime=market_regime,
        )

    def summarize_trade(self, trade_summary: str, symbol: str) -> None:
        """将成交结果摘要写入 AI 记忆。"""

        if not self._initialized or self._client is None:
            return

        prompt = (
            f"你是资深交易复盘专家。请将以下交易总结为一句以“Lesson:”开头的中文经验：{trade_summary}"
        )

        response = self._client.chat.completions.create(
            model=self.settings.ai.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        lesson = (response.choices[0].message.content or "").strip()
        if lesson:
            self._memory_file.parent.mkdir(parents=True, exist_ok=True)
            with self._memory_file.open("a", encoding="utf-8") as fp:
                fp.write(f"- [{symbol}] {lesson}\n")

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    @property
    def _system_prompt(self) -> str:
        from config import AI_SYSTEM_PROMPT

        return AI_SYSTEM_PROMPT

    def _initialize_finbert(self) -> None:
        """加载 FinBERT 模型用于情绪打分。"""

        if self._sentiment_analyzer is not None:
            return

        device = 0 if self._cuda_available() else -1
        local_path = Path("./local_finbert")

        if local_path.is_dir():
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        self._sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _collect_news(self, instrument_id: str) -> str:
        """抓取相关新闻标题。"""

        feeds = self.settings.ai.rss_feeds
        base_part = instrument_id.split(".")[0]
        base_symbol = base_part.replace("-", "/").split("/")[0].upper()
        alias_map = {
            "SOL": ["solana"],
            "BTC": ["bitcoin"],
            "ETH": ["ethereum"],
            "BNB": ["binance coin"],
            "XRP": ["ripple"],
            "ADA": ["cardano"],
        }
        search_terms = [base_symbol.lower()] + alias_map.get(base_symbol, [])
        pattern = re.compile(r"\b(" + "|".join(search_terms) + r")\b", re.IGNORECASE)

        headlines: List[str] = []
        for url in feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    title = getattr(entry, "title", "")
                    summary = getattr(entry, "summary", "")
                    if pattern.search(title) or pattern.search(summary):
                        headlines.append(title)
            except Exception:
                continue

        if not headlines:
            return f"暂无针对 {base_symbol} 的相关新闻。"
        deduped = list(dict.fromkeys(headlines))
        return "；".join(deduped[:10])

    def _compute_sentiment(self, text: str) -> float:
        """对新闻摘要执行情绪分析。"""

        if not text or self._sentiment_analyzer is None:
            return 0.0
        try:
            results = self._sentiment_analyzer(text, max_length=512, truncation=True)
        except Exception:
            return 0.0
        if not results:
            return 0.0
        score = 0.0
        for res in results:
            label = res["label"].lower()
            value = res["score"]
            if label == "positive":
                score += value
            elif label == "negative":
                score -= value
        return round(score / len(results), 3)

    def _analyze_market(self, df_5m: pd.DataFrame, current_price: float) -> str:
        """复刻旧版多周期技术分析，生成给 LLM 的文本。"""

        if df_5m.empty or len(df_5m) < 60:
            return "数据不足，无法生成有效分析。"

        df_clean = df_5m[~df_5m.index.duplicated(keep="last")]
        if len(df_clean) < 60:
            return "数据不足，无法生成有效分析。"

        report = [f"### 当前价格\n- **最新成交价：** {current_price:.4f} USDT\n"]
        risk_flags: List[str] = []
        tf_settings = {"4h": "4h", "1h": "1h", "15m": "15min", "5m": "5min"}

        for label, rule in tf_settings.items():
            resampled = df_clean.resample(rule).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            if len(resampled) < 50:
                report.append(f"### {label} 级别\n- 数据不足，无法计算指标。\n")
                continue

            resampled.ta.ema(length=20, append=True)
            resampled.ta.ema(length=50, append=True)
            resampled.ta.rsi(length=14, append=True)
            resampled.ta.adx(length=14, append=True)
            resampled.ta.macd(append=True)
            resampled.ta.bbands(length=20, append=True)

            latest = resampled.iloc[-1]
            lines: List[str] = [f"### {label} 级别"]
            metrics_ready = False

            ema_fast = self._safe_indicator(latest, "EMA_20")
            ema_slow = self._safe_indicator(latest, "EMA_50")
            if ema_fast is not None and ema_slow is not None:
                lines.append(f"- EMA20/50：{ema_fast:.4f} / {ema_slow:.4f}")
                metrics_ready = True
            else:
                lines.append("- EMA20/50：数据不足，暂无法评估均线结构。")

            rsi_val = self._safe_indicator(latest, "RSI_14")
            if rsi_val is not None:
                lines.append(f"- RSI14：{rsi_val:.2f}")
                metrics_ready = True
                if rsi_val >= 80:
                    risk_flags.append(f"{label} RSI 已达 {rsi_val:.1f}（严重超买），禁止追多。")
                elif rsi_val <= 20:
                    risk_flags.append(f"{label} RSI 已降至 {rsi_val:.1f}（严重超卖），谨慎追空。")
            else:
                lines.append("- RSI14：数据不足。")

            adx_val = self._safe_indicator(latest, "ADX_14")
            if adx_val is not None:
                lines.append(f"- ADX14：{adx_val:.2f}")
                metrics_ready = True
            else:
                lines.append("- ADX14：数据不足。")

            macd_hist = self._safe_indicator(latest, "MACDh_12_26_9")
            if macd_hist is not None:
                lines.append(f"- MACD(直方)：{macd_hist:.4f}")
                metrics_ready = True
            else:
                lines.append("- MACD：数据不足。")

            bb_upper = self._safe_indicator(latest, "BBU_20_2.0")
            bb_lower = self._safe_indicator(latest, "BBL_20_2.0")
            bb_mid = self._safe_indicator(latest, "BBM_20_2.0")
            if bb_upper is not None and bb_lower is not None and bb_mid not in (None, 0.0):
                bband_width = (bb_upper - bb_lower) / bb_mid * 100
                lines.append(f"- 布林带宽度：{bband_width:.2f}%")
                metrics_ready = True
                if bband_width <= 5.0:
                    risk_flags.append(f"{label} 布林带宽度仅 {bband_width:.2f}%（波动收缩），警惕假突破。")
            else:
                lines.append("- 布林带：数据不足。")

            if not metrics_ready:
                lines.append("- 当前周期指标尚未就绪，请等待更多历史数据。")

            report.append("\n".join(lines) + "\n")

        if risk_flags:
            warning_block = "### 风险警示\n" + "\n".join(f"- {flag}" for flag in dict.fromkeys(risk_flags)) + "\n"
            report.insert(1, warning_block)

        return "\n".join(report)

    def _determine_market_regime(self, df_5m: pd.DataFrame) -> str:
        """根据 EMA 与布林带判断市场状态。"""

        if df_5m.empty or len(df_5m) < 200:
            return "UNKNOWN"

        df_clean = df_5m[~df_5m.index.duplicated(keep="last")].dropna(subset=["close"])
        if len(df_clean) < 200:
            return "UNKNOWN"

        ema_series = ta.ema(df_clean["close"], length=200)
        if ema_series.empty or pd.isna(ema_series.iloc[-1]):
            return "UNKNOWN"
        ema_200 = float(ema_series.iloc[-1])

        latest_close = float(df_clean["close"].iloc[-1])

        bbands = ta.bbands(df_clean["close"], length=20)
        if bbands is None or bbands.empty:
            return "UNKNOWN"
        latest_bb = bbands.iloc[-1]
        bb_upper = self._safe_indicator(latest_bb, "BBU_20_2.0")
        bb_lower = self._safe_indicator(latest_bb, "BBL_20_2.0")
        bb_mid = self._safe_indicator(latest_bb, "BBM_20_2.0")

        bb_width_pct: Optional[float] = None
        if bb_upper is not None and bb_lower is not None and bb_mid not in (None, 0.0):
            bb_width_pct = (bb_upper - bb_lower) / bb_mid * 100

        regime = "BULLISH" if latest_close > ema_200 else "BEARISH"
        if bb_width_pct is not None and bb_width_pct < 15:
            adx = ta.adx(df_clean["high"], df_clean["low"], df_clean["close"], length=14)
            if not adx.empty:
                adx_latest = adx["ADX_14"].dropna()
                if not adx_latest.empty and adx_latest.iloc[-1] < 25:
                    regime = "RANGE"
        return regime

    @staticmethod
    def _safe_indicator(series: pd.Series, key: str) -> Optional[float]:
        """安全读取技术指标，若不存在或为 NaN 则返回 None。"""

        if key not in series.index:
            return None
        value = series.get(key)
        if pd.isna(value):
            return None
        return float(value)

    def _compose_prompt(
        self,
        position_text: str,
        context_summary: str,
        analysis_text: str,
        news_digest: str,
        sentiment: float,
        market_regime: str,
        live_equity: float,
        lessons: str,
        active_triggers: List[Dict[str, Any]],
        trigger_reason: str,
    ) -> str:
        """拼装交给 LLM 的主体提示词。"""

        trigger_plan = self._format_triggers(active_triggers, trigger_reason)

        return (
            f"**市场环境**：{market_regime}\n"
            f"**账户权益**：${live_equity:.2f}\n"
            f"**情绪分数**：{sentiment:+.2f}\n"
            "**风险护栏**：\n"
            "- 如“风险警示”段落出现超买/超卖提示，必须以 WAIT 为首选，并说明等待条件；除非当前已持有与提示方向一致的仓位。\n"
            "- 当 RSI ≥ 80 时禁止给出 LONG / OPEN_POSITION 指令；当 RSI ≤ 20 时禁止给出 SHORT 指令。\n"
            "- 未满足已安装触发器或明确的风险回报优势时，不得贸然 OPEN_POSITION。\n"
            f"**记忆库**：\n{lessons}\n\n"
            f"**历史上下文**：\n{context_summary}\n\n"
            f"**当前仓位**：{position_text}\n\n"
            f"**触发器计划**：\n{trigger_plan}\n\n"
            f"**技术分析**：\n{analysis_text}\n\n"
            f"**新闻摘要**：{news_digest}\n"
        )

    @staticmethod
    def _format_triggers(triggers: List[Dict[str, Any]], reason: str) -> str:
        if not triggers:
            base = "当前无活跃触发器，请在需要时新建。"
        else:
            lines = []
            for idx, trig in enumerate(triggers, start=1):
                label = trig.get("label") or f"Trigger-{idx}"
                t_type = trig.get("type", "UNKNOWN")
                detail_parts = []
                for key, value in trig.items():
                    if key in {"label", "type"}:
                        continue
                    detail_parts.append(f"{key}={value}")
                details = ", ".join(detail_parts) if detail_parts else "无附加参数"
                lines.append(f"- {label} | 类型={t_type} | {details}")
            base = "\n".join(lines)
        if reason:
            return base + f"\n- 最近触发原因：{reason}"
        return base

    def _load_lessons(self) -> str:
        if not self._memory_file.exists():
            return "尚无历史经验。"
        lines = self._memory_file.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-15:])


def _extract_chain_of_thought(raw: str) -> str:
    try:
        return raw.split("[CHAIN_OF_THOUGHT_BLOCK]")[1].split("[END_CHAIN_OF_THOUGHT_BLOCK]")[0].strip()
    except (IndexError, AttributeError):
        return ""


def _extract_context_block(raw: str) -> Dict[str, Any]:
    try:
        block = raw.split("[MARKET_CONTEXT_BLOCK]")[1].split("[END_CONTEXT_BLOCK]")[0].strip()
    except (IndexError, AttributeError):
        return {}
    result: Dict[str, Any] = {}
    list_keys = {"KEY_SUPPORT_LEVELS", "KEY_RESISTANCE_LEVELS"}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key in list_keys:
            try:
                result[key.lower()] = [float(v.strip()) for v in value.split(",") if v.strip()]
            except ValueError:
                result[key.lower()] = []
        else:
            result[key.lower()] = value
    if result:
        result["last_full_analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


def _extract_decision_block(raw: str) -> Dict[str, Any]:
    try:
        block = raw.split("[DECISION_BLOCK]")[1].split("[END_BLOCK]")[0].strip()
    except (IndexError, AttributeError):
        return {}

    lines = block.splitlines()
    decision: Dict[str, Any] = {}
    json_buffer = ""
    parsing_triggers = False

    type_map = {
        "ENTRY_PRICE": float,
        "STOP_LOSS": float,
        "TAKE_PROFIT": float,
        "LEVERAGE": int,
        "RISK_PERCENT": float,
        "TRAILING_DISTANCE_PCT": float,
        "NEW_STOP_LOSS": float,
        "NEW_TAKE_PROFIT": float,
        "TRIGGER_TIMEOUT": int,
    }

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("TRIGGERS"):
            parsing_triggers = True
            json_buffer += stripped.split(":", 1)[1].strip()
            continue
        if parsing_triggers:
            json_buffer += stripped
            if stripped.endswith("]"):
                try:
                    decision["triggers"] = json.loads(json_buffer)
                except json.JSONDecodeError:
                    decision["triggers"] = []
                parsing_triggers = False
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key in type_map:
            converter = type_map[key]
            try:
                decision[key.lower()] = converter(value)
            except Exception:
                decision[key.lower()] = None
        else:
            decision[key.lower()] = value
    return decision


def _parse_decision(decision_dict: Dict[str, Any]) -> Optional[AIDecision]:
    if not decision_dict:
        return None

    action_raw = str(decision_dict.get("action", "WAIT")).upper()
    action = DecisionAction._value2member_map_.get(action_raw, DecisionAction.WAIT)

    triggers = [
        TriggerSpec(
            label=trigger.get("label", "Unnamed"),
            type=trigger.get("type", "CUSTOM"),
            params={k: v for k, v in trigger.items() if k not in {"label", "type"}},
        )
        for trigger in decision_dict.get("triggers", [])
        if isinstance(trigger, dict)
    ]

    return AIDecision(
        action=action,
        reasoning=decision_dict.get("reasoning", ""),
        confidence=decision_dict.get("confidence"),
        side=decision_dict.get("decision"),
        entry_price=_safe_float(decision_dict.get("entry_price")),
        stop_loss=_safe_float(decision_dict.get("stop_loss")),
        take_profit=_safe_float(decision_dict.get("take_profit")),
        leverage=_safe_int(decision_dict.get("leverage")),
        risk_percent=_safe_float(decision_dict.get("risk_percent")),
        trailing_distance_pct=_safe_float(decision_dict.get("trailing_distance_pct")),
        new_stop_loss=_safe_float(decision_dict.get("new_stop_loss")),
        new_take_profit=_safe_float(decision_dict.get("new_take_profit")),
        trigger_timeout=_safe_int(decision_dict.get("trigger_timeout")),
        triggers=triggers,
        raw_decision=decision_dict,
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
