# config.py

# --- API Keys ---
BINANCE_API_KEY = "YOUR_BINANCE_TESTNET_API_KEY"
BINANCE_API_SECRET = "YOUR_BINANCE_TESTNET_API_SECRET"

# --- NEW: List of Gemini API Keys for Rotation ---
# Add as many keys as you have to avoid rate limits.
GEMINI_API_KEYS = [
    "YOUR_FIRST_GEMINI_API_KEY",
    "YOUR_SECOND_GEMINI_API_KEY",
    "YOUR_THIRD_GEMINI_API_KEY", # Add more if needed
]

TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_PERSONAL_OR_GROUP_CHAT_ID"

# --- Binance Settings ---
BINANCE_TESTNET = True

# --- Core Capital and Risk Management ---
TOTAL_CAPITAL_USDT = 10000.0
MAX_RISK_PER_TRADE = 0.03

# --- Trading Parameters ---
SYMBOL = "SOLUSDT"
ANALYSIS_TIMEFRAMES = ['5m', '15m', '1h']

# --- Bot Logic Parameters (Triggers) ---
NORMAL_VOLUME_SPIKE = 3.0
NORMAL_VOLATILITY_SPIKE = 2.5
ALERT_VOLUME_SPIKE = 2.0
ALERT_VOLATILITY_SPIKE = 1.8