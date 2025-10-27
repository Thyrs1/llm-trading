# database_manager.py

import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List
import json
from decimal import Decimal

DATABASE_FILE = 'trading_data.db'

def json_default_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, (datetime, Decimal)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def get_db_connection():
    """Creates and returns a connection to the SQLite database."""
    # check_same_thread=False is required because Flask and the bot will access the DB from different threads.
    conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def setup_database():
    """Initializes and updates the database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Log Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            message TEXT NOT NULL
        )
    """)

    # 2. Trade Table
    cursor.execute("""
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
    """)

    # 3. State Table (EXPANDED SCHEMA)
    cursor.execute("""
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
    """)

    # 4. NEW: Account Vitals Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS account_vitals (
            id INTEGER PRIMARY KEY CHECK (id = 1), -- Enforces a single row for the single account
            total_equity REAL,
            available_margin REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialized/verified at {DATABASE_FILE}")

def log_system_message(message: str, symbol: str = "SYSTEM"):
    """Logs a message to the database and prints it."""
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    conn.execute("INSERT INTO logs (timestamp, symbol, message) VALUES (?, ?, ?)",
                 (timestamp, symbol, message))
    conn.commit()
    conn.close()
    # Also print to console for immediate feedback
    print(f"[{timestamp}] [{symbol}] {message}")

def log_trade(symbol: str, side: str, entry_price: float, exit_price: float, quantity: float, pnl: float, pnl_pct: float, reasoning: str):
    """Logs a completed trade to the database."""
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO trades (timestamp, symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct, reasoning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct, reasoning))
    conn.commit()
    conn.close()

def update_bot_state(symbol: str, is_in_position: bool, pos_data: Dict, state_data: Dict):
    """Updates the detailed state of a symbol in the database."""
    conn = get_db_connection()

    try:
        context_json = json.dumps(state_data.get('market_context', {}), default=json_default_serializer)
        triggers_json = json.dumps(state_data.get('active_triggers', []), default=json_default_serializer)
    except Exception as e:
        log_system_message(f"❌ Failed to serialize data for {symbol}: {e}", "DB_ERROR")
        context_json = '{"error": "serialization failed"}'
        triggers_json = '[]'

    conn.execute("""
        INSERT OR REPLACE INTO bot_state
        (symbol, is_in_position, side, entry_price, quantity, unrealized_pnl, last_analysis_time, market_context, active_triggers, last_ai_response, last_sentiment_score, last_known_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        symbol,
        1 if is_in_position else 0,
        pos_data.get('side'),
        pos_data.get('entry_price'),
        pos_data.get('quantity'),
        pos_data.get('unrealized_pnl'),
        datetime.now(timezone.utc).isoformat(),
        context_json,
        triggers_json,
        state_data.get('last_ai_response', ''),
        state_data.get('last_sentiment_score', 0.0),
        state_data.get('last_known_price', 0.0)
    ))
    conn.commit()
    conn.close()

def update_account_vitals(vitals: Dict):
    """Updates the global account vitals in the database."""
    conn = get_db_connection()
    conn.execute("""
        INSERT OR REPLACE INTO account_vitals (id, total_equity, available_margin, timestamp)
        VALUES (1, ?, ?, ?)
    """, (vitals.get('total_equity', 0.0), vitals.get('available_margin', 0.0), datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

def get_dashboard_data() -> Dict[str, Any]:
    """Fetches all data required for the dashboard."""
    conn = get_db_connection()

    state_data = conn.execute("SELECT * FROM bot_state ORDER BY symbol").fetchall()
    trades_data = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 50").fetchall()
    logs_data = conn.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 50").fetchall()
    vitals_data = conn.execute("SELECT * FROM account_vitals WHERE id = 1").fetchone()

    conn.close()

    return {
        'state': [dict(row) for row in state_data],
        'trades': [dict(row) for row in trades_data],
        'logs': [dict(row) for row in logs_data],
        'vitals': dict(vitals_data) if vitals_data else {}
    }