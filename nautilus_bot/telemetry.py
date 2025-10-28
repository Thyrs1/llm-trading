from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import json
import sqlite3

from nautilus_bot.config import BotSettings

if TYPE_CHECKING:  # pragma: no cover
    from nautilus_bot.risk import RiskStatus


@dataclass(slots=True)
class PositionRecord:
    """持仓信息，用于数据库记录。"""

    side: Optional[str] = None
    entry_price: Optional[float] = None
    quantity: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class TelemetryStore:
    """
    SQLite 遥测写入组件，供策略与仪表盘复用。
    """

    def __init__(self, settings: BotSettings):
        self.settings = settings
        self._db_path = settings.telemetry.database_path
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    # 数据库基础
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                message TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                pnl_pct REAL,
                reasoning TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_state (
                symbol TEXT PRIMARY KEY,
                is_in_position INTEGER NOT NULL,
                side TEXT,
                entry_price REAL,
                quantity REAL,
                unrealized_pnl REAL,
                last_analysis_time TEXT,
                market_context TEXT,
                active_triggers TEXT,
                last_ai_response TEXT,
                last_sentiment_score REAL,
                last_known_price REAL,
                chain_of_thought TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS account_vitals (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_equity REAL,
                available_margin REAL,
                timestamp TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    # 写入接口
    # ------------------------------------------------------------------ #

    def log(self, message: str, symbol: str = "SYSTEM") -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        conn.execute(
            "INSERT INTO logs (timestamp, symbol, message) VALUES (?, ?, ?)",
            (timestamp, symbol, message),
        )
        conn.commit()
        conn.close()
        print(f"[{timestamp}] [{symbol}] {message}")

    def record_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        reasoning: str,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO trades (timestamp, symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct, reasoning),
        )
        conn.commit()
        conn.close()

    def update_bot_state(
        self,
        symbol: str,
        is_in_position: bool,
        position: PositionRecord,
        context: Dict[str, Any],
    ) -> None:
        market_context = json.dumps(context.get("market_context", {}), ensure_ascii=False)
        triggers = json.dumps(context.get("active_triggers", []), ensure_ascii=False)
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO bot_state (
                symbol,
                is_in_position,
                side,
                entry_price,
                quantity,
                unrealized_pnl,
                last_analysis_time,
                market_context,
                active_triggers,
                last_ai_response,
                last_sentiment_score,
                last_known_price,
                chain_of_thought
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                1 if is_in_position else 0,
                position.side,
                position.entry_price,
                position.quantity,
                position.unrealized_pnl,
                datetime.now(timezone.utc).isoformat(),
                market_context,
                triggers,
                context.get("last_ai_response", ""),
                context.get("last_sentiment_score", 0.0),
                context.get("last_known_price", 0.0),
                context.get("chain_of_thought", ""),
            ),
        )
        conn.commit()
        conn.close()

    def update_account_vitals(self, total_equity: float, available_margin: float) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO account_vitals (id, total_equity, available_margin, timestamp)
            VALUES (1, ?, ?, ?)
            """,
            (
                total_equity,
                available_margin,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    # 仪表盘数据
    # ------------------------------------------------------------------ #

    def get_bot_states(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM bot_state").fetchall()
        finally:
            conn.close()
        states: List[Dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            for key in ("market_context", "active_triggers"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    try:
                        payload[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
            states.append(payload)
        return states

    def write_dashboard_snapshot(self, risk_status: "RiskStatus") -> Path:
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "risk": asdict(risk_status),
            "states": self.get_bot_states(),
        }
        output_path = Path(self._db_path).resolve().parent / "dashboard_snapshot.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(snapshot, fp, ensure_ascii=False, indent=2)
        return output_path
