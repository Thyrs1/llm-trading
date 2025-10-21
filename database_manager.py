# database_manager.py

import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List

DATABASE_FILE = 'trading_data.db'

def get_db_connection():
    """Creates and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def setup_database():
    """Initializes the database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Log Table (for all system messages)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            message TEXT NOT NULL
        )
    """)
    
    # 2. Trade Table (for executed trades)
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
    
    # 3. State Table (for current positions and market context)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bot_state (
            symbol TEXT PRIMARY KEY,
            is_in_position INTEGER NOT NULL,
            side TEXT,
            entry_price REAL,
            quantity REAL,
            unrealized_pnl REAL,
            last_analysis_time TEXT,
            market_context TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DATABASE_FILE}")

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

def update_bot_state(symbol: str, is_in_position: bool, pos_data: Dict, context: Dict):
    """Updates the current state of a symbol in the database."""
    conn = get_db_connection()
    
    # Serialize context to JSON string
    context_json = json.dumps(context)
    
    conn.execute("""
        INSERT OR REPLACE INTO bot_state 
        (symbol, is_in_position, side, entry_price, quantity, unrealized_pnl, last_analysis_time, market_context) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        symbol,
        1 if is_in_position else 0,
        pos_data.get('side'),
        pos_data.get('entry_price'),
        pos_data.get('quantity'),
        pos_data.get('unrealized_pnl'),
        datetime.now(timezone.utc).isoformat(),
        context_json
    ))
    conn.commit()
    conn.close()

def get_dashboard_data() -> Dict[str, Any]:
    """Fetches all data required for the dashboard."""
    conn = get_db_connection()
    
    # Fetch current state
    state_data = conn.execute("SELECT * FROM bot_state").fetchall()
    
    # Fetch recent trades (e.g., last 50)
    trades_data = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 50").fetchall()
    
    # Fetch recent logs (e.g., last 20)
    logs_data = conn.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 20").fetchall()
    
    conn.close()
    
    # Convert Row objects to dictionaries
    return {
        'state': [dict(row) for row in state_data],
        'trades': [dict(row) for row in trades_data],
        'logs': [dict(row) for row in logs_data]
    }

if __name__ == '__main__':
    setup_database()