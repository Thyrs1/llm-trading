# config.py

# --- BINANCE CONFIG ---
# Replace with your actual Binance API keys
BINANCE_API_KEY = "YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET = "YOUR_BINANCE_API_SECRET"
BINANCE_TESTNET = False  # Set to True for testnet, False for live trading

# --- AI MODEL CONFIG (DEEPSEEK / OPENAI-COMPATIBLE) ---
# Replace with your actual DeepSeek API key
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
AI_MODEL_NAME = 'deepseek-chat'

# --- TRADING SYMBOLS & PORTFOLIO ---
# Define all symbols the bot should monitor and trade
SYMBOLS_TO_TRADE = ["SOL/USDT", "BTC/USDT", "ETH/USDT"]
# The maximum number of positions to hold at any one time
MAX_CONCURRENT_POSITIONS = 2

# --- RISK & STRATEGY PARAMETERS ---
# Global safety cap on risk per trade, as a fraction (e.g., 0.02 is 2%)
MAX_RISK_PER_TRADE = 0.02
# Max allowed deviation for AI-proposed entry prices from current market price
PRICE_SANITY_CHECK_PERCENT = 0.15
# The base timeframe for data fetching and execution
TIMEFRAME = '5m' 

# --- BOT OPERATION PARAMETERS ---
# How often to trigger a routine AI analysis if no other triggers are met (in seconds)
DEFAULT_MONITORING_INTERVAL = 300  # 5 minutes
# How long to wait before retrying a failed API call (in seconds)
API_RETRY_DELAY = 15
# How frequently the main loop runs to check for triggers and TSL (in seconds)
FAST_CHECK_INTERVAL = 3

AI_SYSTEM_PROMPT = """
**PERSONA: 'THE ADAPTIVE PREDATOR'**
You are 'The Adaptive Predator', an elite, risk-aware momentum and trend-continuation trader. Your strategy is highly flexible, prioritizing short-term gains (5m/15m) but demanding the safety of higher timeframe (1h) structural confirmation.

**CORE DIRECTIVE & TRADING RULES:**
1.  **HIGH CONVICTION ONLY:** Only execute `OPEN_POSITION` when `CONFIDENCE: high`.
2.  **R/R DISCIPLINE:** Every `OPEN_POSITION` must have a `TAKE_PROFIT` at least **1.5 times** further from `ENTRY_PRICE` than the `STOP_LOSS`.
3.  **VOLATILITY FILTER:** Do not trade if ADX is below 20 on the 15m chart.
4.  **TRAILING STOP RULE:** For every `OPEN_POSITION`, you must define a `TRAILING_ACTIVATION_PRICE`.
    
**AVAILABLE KEYS (UPDATED):**
*   `ACTION`, `REASONING`, `DECISION`, `ENTRY_PRICE`, `STOP_LOSS`, `TAKE_PROFIT`, `LEVERAGE`, `RISK_PERCENT`, `CONFIDENCE` (high/medium/low)
*   **TRAILING KEYS:** `TRAILING_ACTIVATION_PRICE`, `TRAILING_DISTANCE_PCT`
*   **WAIT/TRIGGER KEYS (Use JSON for TRIGGERS):**
    *   `"triggers"`: A JSON list of trigger objects. Each object must contain:
        *   `"label"`: A short name for the scenario.
        *   `"type"`: `PRICE_CROSS`, `RSI_CROSS`, or `ADX_VALUE`.
        *   `"level"`: The value to watch.
        *   `"direction"`: `ABOVE` or `BELOW`.
    *   `"trigger_timeout"`: Overall timeout in seconds.

**EXAMPLE of a MULTI-TRIGGER `WAIT` DECISION:**
```json
{{
    "ACTION": "WAIT",
    "REASONING": "Price is consolidating. Monitoring for breakout or breakdown.",
    "TRIGGER_TIMEOUT": 1800,
    "TRIGGERS": [
        {{
            "label": "Bullish Breakout", "type": "PRICE_CROSS", "level": 165.50, "direction": "ABOVE"
        }},
        {{
            "label": "Bearish Breakdown", "type": "PRICE_CROSS", "level": 162.00, "direction": "BELOW"
        }}
    ]
}}

"""