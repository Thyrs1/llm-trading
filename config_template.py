# config.py

# API KEYS
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

# --- NEW: List of Gemini API Keys for Rotation ---
# Add as many keys as you have to avoid rate limits.
GEMINI_API_KEYS = [
    "",
    "",
    "", # Add more if needed
    "",
]

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
CRYPTOPANIC_API_KEY = ""

# --- Binance Settings ---
BINANCE_TESTNET = False


# CRITICAL SAFETY CAP: The bot will NEVER risk more than this percentage,
# regardless of what the AI suggests. 10% is already very high.
MAX_RISK_PER_TRADE = 0.90 # 10%

# --- Trading Parameters ---
SYMBOL = "SOLUSDT"
ANALYSIS_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h'] # Focus on shorter timeframes

# --- Bot Logic Parameters ---
# Default interval (in seconds) for Gemini to be called if no specific trigger is set.
DEFAULT_MONITORING_INTERVAL = 60

# NEW: Sanity Check Threshold
# If AI suggests a price more than 10% away from the current market price, reject it.
PRICE_SANITY_CHECK_PERCENT = 0.10 # 10%