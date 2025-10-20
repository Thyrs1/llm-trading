# trading_bot.py (v13 - Fully Integrated Memory and Learning)

# --- Standard Library Imports ---
import os
import time
import json
import math
from enum import Enum
from collections import deque
from datetime import datetime, timezone
import traceback
from typing import Optional

# --- Third-Party Library Imports ---
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Correct, modern imports for Google GenAI SDK
import google.generativeai as genai
from google.generativeai import types
from google.api_core import exceptions

# --- Local Imports ---
import config

# NEW LOGIC FOR FINBERT IMPLEMENTATION

import requests, feedparser, re
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --- NEW OPENAI/DEEPSEEK IMPORTS ---
from openai import OpenAI
from openai import APIStatusError, APITimeoutError # New error types

# --- 1. State Management ---
class BotState(Enum):
    SEARCHING = 1
    ANALYZING = 2
    ORDER_PENDING = 3
    IN_POSITION = 4
    COOL_DOWN = 5

bot_status = { "bot_state": "INITIALIZING", 
              "symbol": config.SYMBOL, 
              "position": {"side": None, 
                           "quantity": 0, 
                           "entry_price": 0}, 
              "pnl": {"usd": None, "percentage": 0}, 
              "last_ai_decision": None, "log": deque(maxlen=30), "last_update": None, }
current_bot_state = BotState.SEARCHING
last_ai_decision = {}
current_key_index = 0
price_precision = 2
quantity_precision = 2

# --- 2. Utility Functions ---
def save_status():
    bot_status['last_update'] = datetime.now(timezone.utc).isoformat()
    status_to_save = bot_status.copy()
    status_to_save['log'] = list(bot_status['log'])
    try:
        with open('status.json', 'w') as f:
            json.dump(status_to_save, f, indent=4)
    except Exception as e:
        print(f"!!! CRITICAL ERROR: Could not write status file: {e}")

def add_log(message):
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    bot_status['log'].appendleft(log_entry)
    print(log_entry)
    save_status()

# Initialize the model once to avoid reloading it every time
try:
    add_log("🤖 Loading FinBERT sentiment model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    add_log("✅ FinBERT model loaded successfully.")
except Exception as e:
    add_log(f"❌ CRITICAL: Could not load FinBERT model: {e}. Sentiment analysis will be disabled.")
    sentiment_analyzer = None

def get_sentiment_score(text: str) -> float:
    """Analyzes text using FinBERT and returns a single numerical score."""
    if not sentiment_analyzer or not text or "failed" in text or "Could not" in text:
        return 0.0 # Return neutral if the model or text is unavailable

    try:
        results = sentiment_analyzer(text)
        # Convert 'positive', 'negative', 'neutral' to a numerical score.
        # Example logic: positive is +score, negative is -score.
        score = 0.0
        for res in results:
            if res['label'] == 'positive':
                score += res['score']
            elif res['label'] == 'negative':
                score -= res['score']
        # Normalize the score to be between -1 and 1
        return round(max(-1.0, min(1.0, score / len(results))), 2) if results else 0.0
    except Exception as e:
        add_log(f"❌ Error during sentiment analysis: {e}")
        return 0.0 # Return neutral on error

    
def get_news_from_rss(symbol: str, limit=15):
    """
    Fetches headlines from multiple RSS feeds and filters them for a specific symbol.
    """
    add_log("📰 Fetching and filtering news from RSS feeds...")
    
    rss_feeds = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        # Add more general crypto news feeds here
    ]
    
    relevant_headlines = []
    
    # Create a regex pattern to find the symbol (case-insensitive)
    # This will match "SOL", "Solana", "sol", etc.
    # We can expand this with a dictionary for other coins.
    coin_names = {'SOL': ['solana', 'sol']}
    search_terms = [symbol.lower()] + coin_names.get(symbol.upper(), [])
    pattern = re.compile(r'\b(' + '|'.join(search_terms) + r')\b', re.IGNORECASE)

    for url in rss_feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # Check if the title or summary contains our symbol
                if pattern.search(entry.title) or (hasattr(entry, 'summary') and pattern.search(entry.summary)):
                    relevant_headlines.append(entry.title)
        except Exception as e:
            add_log(f"⚠️ Could not parse RSS feed {url}: {e}")
            
    if not relevant_headlines:
        add_log(f"No specific news found for {symbol} in RSS feeds.")
        return f"No specific news found for {symbol}."

    # Remove duplicate headlines, preserving order
    unique_headlines = list(dict.fromkeys(relevant_headlines))
    
    return ". ".join(unique_headlines[:limit])

# --- 3. API Client Initialization ---
# (This section is correct and remains unchanged)
FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"
try:
    add_log("Initializing Binance client...")
    binance_client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, testnet=config.BINANCE_TESTNET)
    if config.BINANCE_TESTNET:
        binance_client.API_URL = FUTURES_TESTNET_URL
        add_log(f"✅ Client API URL manually set to Futures Testnet: {binance_client.API_URL}")
    server_time = binance_client.get_server_time()['serverTime']
    binance_client.timestamp_offset = server_time - int(time.time() * 1000)
    add_log("✅ Time synchronization successful.")
    add_log(f"✅ Binance client initialization successful (Testnet: {config.BINANCE_TESTNET}).")
    try:
        binance_client.futures_change_margin_type(symbol=config.SYMBOL, marginType='ISOLATED')
        add_log("✅ Margin type set to ISOLATED.")
    except BinanceAPIException as e:
        if e.code == -4046: add_log("✅ Margin type was already ISOLATED.")
        else: raise e
    add_log("Fetching exchange information for precision rules...")
    exchange_info = binance_client.futures_exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == config.SYMBOL:
            price_precision = int(s['pricePrecision'])
            quantity_precision = int(s['quantityPrecision'])
            add_log(f"✅ Precision rules for {config.SYMBOL}: Price={price_precision}, Quantity={quantity_precision}")
            break
except Exception as e:
    add_log(f"❌ Binance initialization failed: {e}")
    exit()

# try:
#     gemini_model = genai.GenerativeModel('gemini-flash-latest') # Using the latest flash model
#     add_log(f"✅ Gemini AI model loaded. Using {len(config.GEMINI_API_KEYS)} API keys for rotation.")
# except Exception as e:
#     add_log(f"❌ Gemini AI initialization failed: {e}")
#     exit()

try:
    add_log("🤖 Initializing OpenAI/DeepSeek client...")
    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    # client.models.lists()  # Test call to ensure client works
    add_log("✅ OpenAI/DeepSeek client initialized successfully.")
    
    AI_MODEL_NAME = "deepseek-chat"
    
except Exception as e:
    add_log(f"❌ OpenAI/DeepSeek initialization failed: {e}")
    exit()

# --- 4. AI Model Master Prompt ---
AI_SYSTEM_PROMPT_TEXT_BASED = """
**PERSONA: 'THE MOMENTUM SCALPER'**

You are 'The Momentum Scalper', an elite, high-velocity trading specialist. Your strategy is focused purely on **short-term momentum and quick, efficient profit-taking**. Your typical trade duration is minutes, aiming to capture small, confirmed movements. Discipline is paramount: take profit quickly and cut losses instantly.

**CORE DIRECTIVE & TRADING RULES:**

1.  **PRIMARY FOCUS (HTF ID):** Your core bias and trend direction MUST be determined by the **15-minute chart**. The 5-minute chart is used for execution. Ignore 1h/4h noise unless a major resistance/support level is hit.

2.  **ENTRY STRATEGY (IMPULSE):** You execute a trade only on signs of strong continuation or breakout from a micro-consolidation, always aiming for the current trend's direction (15m). Avoid pullbacks that last longer than 3 candles.

3.  **EXECUTE WITH PRECISION:** Your confidence must be **`high`** to execute a trade. If confidence is `medium` or `low`, you must **`WAIT`**.

4.  **DEFINED RISK/REWARD PROFILE (FAST EXITS):**
    *   **Risk/Reward Ratio:** Every `OPEN_POSITION` decision **must** have a `TAKE_PROFIT` that is at least **1.2 times** further from your `ENTRY_PRICE` than your `STOP_LOSS`. (e.g., If Stop Loss is $1, Take Profit is $1.20). The priority is a high win rate with frequent exits.
    *   **Risk Percentage (`RISK_PERCENT`):** Use a slightly lower risk range of **2% to 6%** per trade. This protects capital against high-frequency drawdowns.
    *   **Leverage:** Use a moderate leverage range between **20x and 30x**.

5.  **NEWS AS A VETO:** Use the `News Sentiment Score` primarily as a *filter*. If sentiment strongly conflicts with the technical setup (e.g., highly bullish technicals but highly negative news), you MUST stand down and **`WAIT`** to avoid event risk.

**YOUR TASK:**
Analyze the provided data. Is there a high-momentum setup that confirms the 15-minute trend and meets the 1.2 R/R target? If so, provide the `[DECISION_BLOCK]`. In all other scenarios, you will patiently **`WAIT`**.

**FORMATTING RULES FOR [DECISION_BLOCK]**
1.  Starts with `[DECISION_BLOCK]` and ends with `[END_BLOCK]`.
2.  Inside, each line must be a `KEY: VALUE` pair.
3.  **CRITICAL:** Only include KEYs relevant to your chosen `ACTION`.

**AVAILABLE KEYS and WHEN TO USE THEM:**
*   `ACTION`: (REQUIRED) Must be one of: `OPEN_POSITION`, `CLOSE_POSITION`, `MODIFY_POSITION`, `WAIT`.
*   `REASONING`: (REQUIRED) A brief, one-sentence explanation for your action.
*   **--- If ACTION is `OPEN_POSITION`:**
    *   `DECISION`: `LONG` or `SHORT`.
    *   `CONFIDENCE`: Your confidence level. Must be one of: `high`, `medium`, `low`.
    *   `ENTRY_PRICE`: The target entry price.
    *   `STOP_LOSS`: The mandatory stop loss price.
    *   `TAKE_PROFIT`: The initial take profit price.
    *   `LEVERAGE`: Integer leverage to use.
    *   `RISK_PERCENT`: Percentage of capital to risk (Min 10%, max 90%, dont place orders for those that you dont have confidence).
*   **--- If ACTION is `WAIT`:**
    *   `TRIGGER_PRICE`: The price that triggers the next analysis.
    *   `TRIGGER_DIRECTION`: `ABOVE` or `BELOW`.
    *   `TRIGGER_TIMEOUT`: Timeout in seconds.
*   **--- If ACTION is `MODIFY_POSITION`:**
    *   `NEW_STOP_LOSS`: New stop loss price. (Use 0 if not changing)
    *   `NEW_TAKE_PROFIT`: New take profit price. (Use 0 if not changing)

**ADDITIONAL REQUIREMENT: Market Context Summary**
After `[END_BLOCK]`, you MUST provide a `[MARKET_CONTEXT_BLOCK]`. This is your persistent view of the market.
**FORMATTING RULES FOR [MARKET_CONTEXT_BLOCK]**
...
*   `MARKET_THESIS`: A short summary of your overall market view.
*   `KEY_SUPPORT_LEVELS`: A comma-separated list of the 2-3 most important support price levels you are watching (e.g., "188.50, 185.00, 182.25").
*   `KEY_RESISTANCE_LEVELS`: A comma-separated list of the 2-3 most important resistance price levels (e.g., "192.80, 195.00, 200.00").
*   `DOMINANT_TREND_TIMEFRAME`: The timeframe (e.g., 5m, 15m, 1h, 4h) that you believe is currently driving the price action.
...

**TO BE NOTED: Response Length Limit**
Your Response must as clear and concise as possible while still in the met of FORMATTING RULES. Use abbreviations where possible without losing clarity.

**EXAMPLE of the full output:**
[DECISION_BLOCK]
ACTION: WAIT
REASONING: The market is approaching a major resistance; waiting for a confirmed breakout or rejection.
TRIGGER_PRICE: 193.20
TRIGGER_DIRECTION: ABOVE
TRIGGER_TIMEOUT: 900
[END_BLOCK]
[MARKET_CONTEXT_BLOCK]
MARKET_THESIS: Bullish but overbought, approaching major 4h resistance.
KEY_SUPPORT: 189.75
KEY_RESISTANCE: 193.20
DOMINANT_TREND: 15m
[END_CONTEXT_BLOCK]

Now, analyze the following data and provide your full response.
"""
def is_decision_sane(decision):
    """
    Performs a sanity check on the prices provided by AI Model to prevent hallucinations.
    """
    try:
        # Get the current mark price for comparison
        ticker = binance_client.futures_mark_price(symbol=config.SYMBOL)
        current_price = float(ticker['markPrice'])
        
        # Define the acceptable price range (e.g., +/- 10% from current price)
        max_price = current_price * (1 + config.PRICE_SANITY_CHECK_PERCENT)
        min_price = current_price * (1 - config.PRICE_SANITY_CHECK_PERCENT)
        
        # Check all relevant prices from the decision
        prices_to_check = [
            decision.get('entry_price'),
            decision.get('stop_loss'),
            decision.get('take_profit'),
            decision.get('new_stop_loss'),
            decision.get('new_take_profit'),
            decision.get('trigger_price')
        ]
        
        for price in prices_to_check:
            # Skip if the price is not provided or is 0
            if price is None or price == 0:
                continue
            
            # If any price is outside the sane range, the decision is invalid
            if not (min_price <= price <= max_price):
                add_log(f"🚨 SANITY CHECK FAILED: AI proposed price {price} is outside the acceptable range ({min_price:.2f} - {max_price:.2f}). Current price is {current_price:.2f}.")
                return False
                
        add_log("✅ Sanity check passed. AI decision is within reasonable price limits.")
        return True
        
    except BinanceAPIException as e:
        add_log(f"❌ Error during sanity check while fetching mark price: {e}")
        return False # Fail safe
    except Exception as e:
        add_log(f"❌ An unexpected error occurred during sanity check: {e}")
        return False # Fail safe

def run_diagnostic_query(analysis_data, position_data):
    """
    Asks AI Model to "think out loud" in plain English without JSON constraints.
    This is used to diagnose why it might be generating faulty data.
    """
    global current_key_index
    add_log("--- 🧠 Requesting AI Model RAW THOUGHT PROCESS for diagnostics ---")
    
    # A different, more open-ended prompt
    diagnostic_prompt = f"""
    **DIAGNOSTIC MODE**

    You are an expert trading analyst. Your previous attempt to generate a JSON response for the following data resulted in a nonsensical price.
    
    **Your Task:**
    Ignore all JSON formatting. In plain English, I want you to "think out loud". Analyze the data step-by-step and formulate a trading plan.
    
    1.  Start by stating the current price you see in the data.
    2.  State your overall market thesis (bullish, bearish, neutral).
    3.  Explain how the data points (Vitals and K-lines) support your thesis.
    4.  Conclude with a specific, actionable trade idea, including your recommended entry price, stop loss, and take profit.

    **Here is the data you must analyze:**
    
    **Position Status:**
    {position_data}

    **Market Analysis:**
    {analysis_data}
    """
    
    # # This loop is for a single, non-structured request
    # for i in range(len(config.GEMINI_API_KEYS)):
    #     try:
    #         key = config.GEMINI_API_KEYS[current_key_index]
    #         # Create a fresh, temporary model instance
    #         genai.configure(api_key=key) # type: ignore
    #         model = genai.GenerativeModel('gemini-flash-latest') # type: ignore
    #         current_key_index = (current_key_index + 1) % len(config.GEMINI_API_KEYS)
            
    #         # Make a standard text-only API call
    #         response = model.generate_content(diagnostic_prompt)
            
    #         return response.text # Return the plain text response
            
    #     except exceptions.ResourceExhausted:
    #         add_log(f"⚠️ Gemini API key at index {current_key_index-1} is rate-limited. Switching...")
    #         continue
    #     except Exception as e:
    #         add_log(f"❌ Error during diagnostic query: {e}")
    #         return f"Failed to get diagnostic response: {e}"

    # return "All Gemini API keys failed during diagnostic query."

# --- 5. Data Acquisition and Analysis Functions ---
def get_klines_robust(symbol, interval, limit=200, retries=3, delay=5):
    for i in range(retries):
        try:
            klines = binance_client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            df = pd.DataFrame(klines, columns=columns)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.set_index('open_time')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            return df.drop(columns=['ignore', 'close_time'])
        except BinanceAPIException as e:
            add_log(f"Binance API Error (Attempt {i+1}/{retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
    add_log(f"Failed to fetch {interval} K-lines after {retries} attempts.")
    return None

def get_market_vitals(symbol):
    vitals = {}
    try:
        depth = binance_client.futures_order_book(symbol=symbol, limit=20)
        total_bids_qty = sum([float(qty) for price, qty in depth['bids']])
        total_asks_qty = sum([float(qty) for price, qty in depth['asks']])
        total_qty = total_bids_qty + total_asks_qty
        vitals['order_book_imbalance'] = total_bids_qty / total_qty if total_qty > 0 else 0.5
        mark_price_data = binance_client.futures_mark_price(symbol=symbol)
        vitals['funding_rate'] = float(mark_price_data['lastFundingRate'])
        open_interest_data = binance_client.futures_open_interest(symbol=symbol)
        vitals['open_interest'] = float(open_interest_data['openInterest'])
        if config.BINANCE_TESTNET:
            add_log("ℹ️ Skipping Top Trader L/S Ratio on Testnet.")
            vitals['top_trader_long_short_ratio'] = 1.0
        else:
            long_short_ratio = binance_client.futures_top_longshort_account_ratio(symbol=symbol, period='5m', limit=1)
            if long_short_ratio:
                vitals['top_trader_long_short_ratio'] = float(long_short_ratio[0]['longShortRatio'])
            else:
                vitals['top_trader_long_short_ratio'] = 1.0
                add_log("⚠️ Warning: Top Trader L/S Ratio returned empty list. Defaulting to 1.0.")
        return vitals
    except BinanceAPIException as e:
        add_log(f"❌ Error fetching market vitals: {e}")
        return None

# trading_bot.py (run_heavy_analysis function - Corrected)

def run_heavy_analysis():
    """Generates the full holographic analysis report for Gemini."""
    add_log("🚀 Starting holographic analysis...")
    
    # --- NEW: Get current price to anchor the AI ---
    try:
        ticker = binance_client.futures_mark_price(symbol=config.SYMBOL)
        current_price = float(ticker['markPrice'])
    except Exception as e:
        add_log(f"Could not fetch current price for analysis bundle: {e}")
        return None

    market_vitals = get_market_vitals(config.SYMBOL)
    if not market_vitals: return None
        
    # --- NEW: Add the anchor price to the top of the report ---
    
    market_vitals = get_market_vitals(config.SYMBOL)
    if not market_vitals: return None
    
    # --- NEW: Integrate Sentiment Analysis ---
    news_headlines = get_news_from_rss(symbol=config.SYMBOL.replace("USDT", ""))
    sentiment_score = get_sentiment_score(news_headlines)
    # --- END of new code ---

    
    all_data_content = f"### 0. Current Market Price (Anchor)\n- **Current Price:** {current_price:.2f} USDT\n\n"
    all_data_content = "### 1. Live Market Vitals\n"
    all_data_content += f"- Order Book Imbalance (Buy Pressure): {market_vitals['order_book_imbalance']:.2%}\n"
    all_data_content += f"- Funding Rate: {market_vitals['funding_rate']:.4%}\n"
    all_data_content += f"- Open Interest (USDT): {market_vitals['open_interest']:,.2f}\n"
    all_data_content += f"- Top Trader L/S Ratio: {market_vitals['top_trader_long_short_ratio']:.2f}\n\n"
    all_data_content += "### 2. News Sentiment Analysis (FinBERT via API)\n"
    all_data_content += f"- Overall Sentiment Score: {sentiment_score:.2f} (-1 Negative, +1 Positive)\n"
    all_data_content += "### 3. Multi-Timeframe K-line Depth Analysis\n"

    for tf in config.ANALYSIS_TIMEFRAMES:
        # --- CRITICAL FIX: Fetch more data to ensure indicators can warm up ---
        # We need at least 200 candles for the EMA_200, so fetching 250 gives a buffer.
        df = get_klines_robust(config.SYMBOL, tf, limit=250)
        if df is None: continue
        
        # Explicitly name the indicator columns
        df['EMA_20'] = df.ta.ema(length=20)
        df['EMA_50'] = df.ta.ema(length=50)
        
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df = df.join(macd)
        
        df['RSI_14'] = df.ta.rsi(length=14)
        df['ATRr_14'] = df.ta.atr(length=14)
        
        # ADX returns a DataFrame, so we select the column
        adx_df = df.ta.adx(length=14)
        if adx_df is not None and 'ADX_14' in adx_df.columns:
            df['ADX_14'] = adx_df['ADX_14']
        
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()

        latest = df.iloc[-1]
        
        atr_percentage = (latest.get('ATRr_14', 0) / latest['close']) * 100 if latest['close'] > 0 else 0
        volume_strength = latest.get('volume', 0) / latest.get('volume_ma_20', 1) if latest.get('volume_ma_20', 0) > 0 else 0
        
        report = f"--- Analysis Report ({tf} Timeframe) ---\n"
        report += f"Close Price: {latest['close']:.4f}\n"
        report += f"Trend Strength (ADX_14): {latest.get('ADX_14', 0):.2f}\n"
        report += f"EMA 20/50: {latest.get('EMA_20', 0):.4f} / {latest.get('EMA_50', 0):.4f}\n"
        report += f"RSI_14: {latest.get('RSI_14', 0):.2f}\n"
        report += f"ATR_Volatility_Percent: {atr_percentage:.2f}%\n"
        report += f"Volume_Strength_Ratio: {volume_strength:.2f}x\n"
        all_data_content += report
    
    return all_data_content
def save_market_context(context_data: dict):
    if not context_data: return
    try:
        with open('market_context.json', 'w') as f:
            json.dump(context_data, f, indent=4)
        add_log("💾 Market context saved.")
    except Exception as e:
        add_log(f"❌ Error saving market context: {e}")

def load_market_context() -> dict:
    try:
        with open('market_context.json', 'r') as f:
            context = json.load(f)
            add_log("🧠 Market context loaded from last session.")
            return context
    except FileNotFoundError:
        add_log("No previous market context found. Starting fresh.")
        return {}
    except Exception as e:
        add_log(f"❌ Error loading market context: {e}")
        return {}

def parse_decision_block(raw_text: str) -> dict:
    decision = {}
    in_block = False
    type_map = {"ENTRY_PRICE": float, "STOP_LOSS": float, "TAKE_PROFIT": float, "LEVERAGE": int, "RISK_PERCENT": float, "TRIGGER_PRICE": float, "TRIGGER_TIMEOUT": int, "NEW_STOP_LOSS": float, "NEW_TAKE_PROFIT": float}
    for line in raw_text.splitlines():
        line = line.strip()
        if line == '[DECISION_BLOCK]': in_block = True; continue
        if line == '[END_BLOCK]': break
        if in_block and ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            if key in type_map:
                try: decision[key.lower()] = type_map[key](value)
                except (ValueError, TypeError): decision[key.lower()] = None
            else: decision[key.lower()] = value
    return decision

def parse_context_block(raw_text: str) -> dict:
    """Parses the market context block, including comma-separated levels, into a dictionary."""
    context = {}
    in_block = False
    
    # Define keys that should be parsed as lists of floats
    list_keys = ["KEY_SUPPORT_LEVELS", "KEY_RESISTANCE_LEVELS"]

    for line in raw_text.splitlines():
        line = line.strip()
        if line == '[MARKET_CONTEXT_BLOCK]':
            in_block = True
            continue
        if line == '[END_CONTEXT_BLOCK]':
            break
        if in_block and ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper() # Use upper for consistent matching
            value = value.strip()
            
            if key in list_keys:
                try:
                    # Split by comma, strip whitespace from each part, convert to float
                    levels = [float(level.strip()) for level in value.split(',') if level.strip()]
                    context[key.lower()] = levels
                except (ValueError, TypeError):
                    context[key.lower()] = [] # Default to empty list on error
            else:
                # Handle single value keys as before
                context[key.lower()] = value
    
    if context:
        context['last_full_analysis_timestamp'] = datetime.now(timezone.utc).isoformat()

    return context

# --- 6. AI Interaction and Learning ---

def summarize_and_learn(trade_history_entry: str):
    """Asks AI to analyze a closed trade, extract a lesson, and update the memory file."""
    global current_key_index
    add_log("🧠 Performing post-trade analysis to update memory...")
    system_prompt = "You are a master trading analyst. Your goal is to learn from every trade."
    memory_prompt = f""" Below is the data for a recently closed trade.
    **Trade Data:**\n{trade_history_entry}
    **Your Task:** Analyze this trade. Was it a good entry? Was the outcome due to a good strategy or just luck? Condense your analysis into a single, powerful, one-sentence "lesson learned" starting with "Lesson:".
    **Example Lessons:**
    - "Lesson: Shorting into a strong 15m uptrend, even with a high L/S ratio, is risky and often results in being stopped out."
    - "Lesson: Entering a long position near the 1h EMA50 after a period of consolidation has a high probability of success."
    Provide only the single "Lesson:" line."""

    try:
        key = config.DEEPSEEK_API_KEY[current_key_index]
        client = OpenAI(
            api_key=key,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        add_log(f"🧠 Requesting trade summarization from DeepSeek with key index")
        response = client.chat.completions.create(
            model=AI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": memory_prompt}
            ],
            temperature=0.3
        )
        lesson = response.choices[0].message.content.strip()
        if "Lesson:" in lesson:
            add_log(f"💡 New lesson learned: {lesson}")
            with open("trade_memory.txt", "a") as f:
                f.write(f"- {lesson}\n")
            return
        else:
                    add_log("⚠️ Could not extract a valid lesson from the trade analysis.")
    except Exception as e:
               add_log(f"❌ Error during trade summarization on key index {current_key_index-1}: {e}")
    add_log("🚨 All API keys failed during trade summarization.")


def get_ai_decision(analysis_data, position_data, last_context_summary, live_equity):
    """Gets a trading decision from AI Models, including long-term memory."""
    global current_key_index
    try:
        with open("trade_memory.txt", "r") as f:
            lessons = "".join(f.readlines()[-10:])
    except FileNotFoundError:
        lessons = "No past trade lessons available yet."

    # live_equity = get_total_equity()

    prompt = f"""
    
**--- CRITICAL ACCOUNT CONSTRAINTS ---**
My current account equity is only ${live_equity:.2f} USDT. You MUST provide parameters that are realistic for this small account size. A large position with a tight stop-loss may be impossible to open due to margin requirements. Be pragmatic.
**----------------------------------------------------**

**--- STRATEGIC MEMORY: LESSONS FROM PAST TRADES ---**
{lessons}
**----------------------------------------------------**
**--- MARKET CONTEXT (YOUR LAST ANALYSIS) ---**
{last_context_summary}
**----------------------------------------------------**
**--- CURRENT DATA FOR ANALYSIS ---**
**1. Current Position Status:**
{position_data}
**2. Holographic Market Analysis:**
{analysis_data}
Based on your memory, your last analysis, and the new data, provide your full response."""
    try:
        key = config.DEEPSEEK_API_KEY
        add_log(f"🧠 Requesting text-based decision from OpenAI/Deepseek")
        genai.configure(api_key=key)
        client = OpenAI(
            api_key=key,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        response = client.chat.completions.create(
            model=AI_MODEL_NAME,
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT_TEXT_BASED},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        raw_response_text = response.choices[0].message.content
        add_log("--- 🧠 AI MODEL RAW TEXT RESPONSE ---"); add_log(raw_response_text); add_log("--- END RAW TEXT RESPONSE ---")
            
        decision_dict = parse_decision_block(raw_response_text)
        context_dict = parse_context_block(raw_response_text)
            
        if not decision_dict or 'action' not in decision_dict:
            add_log("❌ Parsing failed or ACTION key is missing.")
        if context_dict:
            save_market_context(context_dict)
        add_log(f"✅ AI Model decision parsed. Action: {decision_dict.get('action')}")
        return decision_dict, context_dict
    except exceptions.ResourceExhausted:
        add_log(f"⚠️ OpenAI/Deepseek API key is rate-limited.")
    except Exception as e:
        add_log(f"❌ An unexpected error during OpenAI API call: {traceback.format_exc()}")

# --- 7. Trading Execution and Management Functions ---
def get_current_position():
    try:
        positions = binance_client.futures_position_information()
        for p in positions:
            if p['symbol'] == config.SYMBOL:
                pos_amt = float(p['positionAmt'])
                if pos_amt != 0:
                    return {"side": "LONG" if pos_amt > 0 else "SHORT", "quantity": abs(pos_amt), "entry_price": float(p['entryPrice'])}
        return {"side": None, "quantity": 0, "entry_price": 0}
    except BinanceAPIException as e:
        add_log(f"Error checking position: {e}")
        return {"side": None, "quantity": 0, "entry_price": 0}

def get_available_margin():
    """Fetches the available USDT balance from the futures account."""
    try:
        balances = binance_client.futures_account_balance()
        for balance in balances:
            if balance['asset'] == 'USDT':
                return float(balance['availableBalance'])
        return 0.0
    except BinanceAPIException as e:
        add_log(f"❌ Could not fetch available margin: {e}")
        return 0.0

    
# Add this new helper function first, to get your total equity
def get_total_equity():
    """Fetches the total wallet balance (equity) from the futures account."""
    try:
        balances = binance_client.futures_account_balance()
        for balance in balances:
            if balance['asset'] == 'USDT':
                # 'balance' represents the total equity (wallet balance)
                return float(balance['balance'])
        return 0.0
    except BinanceAPIException as e:
        add_log(f"❌ Could not fetch total equity: {e}")
        return 0.0

# Now, replace your entire old calculate_position_size function with this one
def calculate_position_size(entry_price, stop_loss_price, risk_percent):
    """
    Calculates position size based on a percentage of the *live* account equity.
    """
    try:
        # --- DYNAMIC CAPITAL LOGIC ---
        live_equity = get_total_equity()
        if live_equity <= 0:
            add_log("⚠️ Cannot calculate position size, account equity is zero or unavailable.")
            return 0
        
        add_log(f"Calculating risk based on LIVE equity of {live_equity:.2f} USDT.")
        
        # Ensure risk percent is a fraction
        actual_risk_fraction = min(risk_percent / 100, config.MAX_RISK_PER_TRADE)
        amount_to_risk_usdt = live_equity * actual_risk_fraction
        # --- END DYNAMIC CAPITAL LOGIC ---

        price_delta_per_unit = abs(entry_price - stop_loss_price)
        if price_delta_per_unit == 0:
            add_log("⚠️ Price delta is zero, cannot calculate position size.")
            return 0
            
        position_size_units = amount_to_risk_usdt / price_delta_per_unit
        
        # We return the unrounded size and let the open_position function handle final precision
        return position_size_units

    except Exception as e:
        add_log(f"❌ Error calculating position size: {e}")
        return 0

def open_position(decision):
    """Executes the logic to open a new position with a margin pre-check."""
    global bot_status
    
    side = decision.get('decision')
    entry_price = decision.get('entry_price')
    stop_loss_price = decision.get('stop_loss')
    leverage = decision.get('leverage')
    risk_percent = decision.get('risk_percent')
    
    if not all([side, entry_price, stop_loss_price, leverage, risk_percent is not None]):
        add_log(f"❌ AI OPEN decision missing required fields. Decision: {decision}")
        return

    # Calculate the desired position size based on risk
    position_size = calculate_position_size(entry_price, stop_loss_price, risk_percent)
    if position_size <= 0:
        add_log("Calculated position size is zero or invalid, trade cancelled.")
        return
        
    add_log(f"💎 Decision: {side} | Desired Size: {position_size} | Risk: {risk_percent}% | Leverage: {leverage}x")

    # --- NEW: MARGIN PRE-FLIGHT CHECK ---
    try:
        position_value_usdt = position_size * entry_price
        required_margin = position_value_usdt / leverage
        available_margin = get_available_margin()

        add_log(f"🔬 Pre-flight Check: Required Margin ≈ {required_margin:.2f} USDT, Available Margin = {available_margin:.2f} USDT")

        if required_margin * 1.05 > available_margin: # Added 5% buffer for safety
            add_log(f"🚨 MARGIN PRE-CHECK FAILED: Required margin ({required_margin:.2f} USDT) exceeds available balance ({available_margin:.2f} USDT). Order cancelled.")
            return
        add_log("✅ Margin pre-check passed.")

    except Exception as e:
        add_log(f"❌ Error during margin pre-check: {e}")
        return
    # --- END OF PRE-FLIGHT CHECK ---
    
    bot_status['bot_state'] = "ORDER_PENDING"
    try:
        binance_client.futures_change_leverage(symbol=config.SYMBOL, leverage=leverage)
        add_log(f"⚙️ Leverage set to {leverage}x.")
    except BinanceAPIException as e:
        add_log(f"❌ Failed to set leverage: {e}")
        return

    try:
        formatted_price = f"{entry_price:.{price_precision}f}"
        formatted_quantity = f"{position_size:.{quantity_precision}f}"
        add_log(f"Formatted Order: Price={formatted_price}, Qty={formatted_quantity}")

        order = binance_client.futures_create_order(
            symbol=config.SYMBOL,
            side='BUY' if side == 'LONG' else 'SELL',
            type='LIMIT',
            timeInForce='GTC',
            price=formatted_price,
            quantity=formatted_quantity,
        )
        add_log(f"✅ Limit Order placed @ {formatted_price} (ID: {order['orderId']})")
        # You might need logic here to set SL/TP after the order fills
        
    except BinanceAPIException as e:
        # The pre-check should prevent -2019, but we keep this for other errors
        add_log(f"❌ Binance order failed: {e}")

def close_position(position):
    add_log(f"Executing AI's instruction to CLOSE position...")
    try:
        binance_client.futures_cancel_all_open_orders(symbol=config.SYMBOL)
        close_side = 'BUY' if position['side'] == 'SHORT' else 'SELL'
        binance_client.futures_create_order(symbol=config.SYMBOL, side=close_side, type='MARKET', quantity=position['quantity'], reduceOnly=True)
        add_log(f"✅ Market close order sent successfully.")
    except BinanceAPIException as e:
        add_log(f"❌ Failed to close position: {e}")

def modify_position(decision, position):
    """Modifies the Stop Loss and/or Take Profit with correct precision."""
    add_log(f"Executing AI's instruction to MODIFY position...")
    try:
        binance_client.futures_cancel_all_open_orders(symbol=config.SYMBOL)
        sl_price = decision.get('new_stop_loss')
        tp_price = decision.get('new_take_profit')
        side = position['side']
        
        if sl_price and sl_price > 0:
            # --- RE-INTRODUCED PYTHON FORMATTING ---
            formatted_sl = f"{sl_price:.{price_precision}f}"
            sl_side = 'SELL' if side == 'LONG' else 'BUY'
            binance_client.futures_create_order(
                symbol=config.SYMBOL, side=sl_side, type='STOP_MARKET',
                stopPrice=formatted_sl, closePosition=True
            )
            add_log(f"✅ New Stop Loss set @ {formatted_sl}")
            
        if tp_price and tp_price > 0:
            # --- RE-INTRODUCED PYTHON FORMATTING ---
            formatted_tp = f"{tp_price:.{price_precision}f}"
            tp_side = 'SELL' if side == 'LONG' else 'BUY'
            binance_client.futures_create_order(
                symbol=config.SYMBOL, side=tp_side, type='TAKE_PROFIT_MARKET',
                stopPrice=formatted_tp, closePosition=True
            )
            add_log(f"✅ New Take Profit set @ {formatted_tp}")
            
    except BinanceAPIException as e:
        add_log(f"❌ Failed to modify position: {e}")

def wait_for_trigger(decision):
    """Enters a fast loop to check for a specific condition set by AI."""
    # --- MODIFICATION START ---
    # Get parameters directly, matching the new robust logic.
    price = decision.get('trigger_price')
    direction = decision.get('trigger_direction')
    timeout_seconds = decision.get('trigger_timeout', 300) # Use default if not provided
    
    # Simplified, robust check for valid parameters.
    if not all([price, direction]) or price <= 0:
        add_log("⚠️ AI WAIT decision missing valid trigger parameters. Re-analyzing immediately.")
        return
    # --- MODIFICATION END ---

    add_log(f"Entering fast-check mode. WAITING for price to cross {direction} {price}...")
    
    timeout = time.time() + timeout_seconds
    
    while time.time() < timeout:
        try:
            ticker = binance_client.futures_mark_price(symbol=config.SYMBOL)
            current_price = float(ticker['markPrice'])
            
            if direction.upper() == 'ABOVE' and current_price > price:
                add_log(f"🎯 Trigger condition MET: Price {current_price:.4f} crossed ABOVE {price:.4f}.")
                return
            if direction.upper() == 'BELOW' and current_price < price:
                add_log(f"🎯 Trigger condition MET: Price {current_price:.4f} crossed BELOW {price:.4f}.")
                return
                
            # Check every 5 seconds
            time.sleep(5)
        except Exception as e:
            add_log(f"Error in wait_for_trigger loop: {e}")
            time.sleep(15) # Wait longer on error
            
    add_log(f"⏳ Wait condition timed out after {timeout_seconds} seconds. Re-analyzing.")

def close_position(position):
    add_log(f"Executing AI's instruction to CLOSE position...")
    try:
        binance_client.futures_cancel_all_open_orders(symbol=config.SYMBOL)
        close_side = 'BUY' if position['side'] == 'SHORT' else 'SELL'
        binance_client.futures_create_order(symbol=config.SYMBOL, side=close_side, type='MARKET', quantity=position['quantity'], reduceOnly=True)
        add_log(f"✅ Market close order sent successfully.")
    except BinanceAPIException as e:
        add_log(f"❌ Failed to close position: {e}")

# --- 8. Main Loop ---
def main_loop():
    global bot_status, last_ai_decision
    add_log("🤖 AI-Driven Trading Bot Main Loop Started.")
    
    was_in_position = False
    last_position_details = {}
    market_context = load_market_context()

    while True:
        try:
            pos = get_current_position()
            is_in_position = pos['side'] is not None

            if was_in_position and not is_in_position:
                add_log("📉 Position closed. Triggering post-trade analysis and learning...")
                try:
                    trades = binance_client.futures_account_trade_list(symbol=config.SYMBOL, limit=2)
                    if len(trades) > 0:
                        closing_trade = trades[-1]
                        pnl = float(closing_trade['realizedPnl'])
                        outcome = "WIN" if pnl > 0 else "LOSS"
                        trade_summary = (f"--- TRADE OUTCOME: {outcome} ---\n"
                                         f"Direction: {last_position_details.get('side', 'N/A')}\n"
                                         f"Entry Price: {last_position_details.get('entry_price', 'N/A')}\n"
                                         f"Closing Price: {float(closing_trade['price']):.4f}\n"
                                         f"Realized PNL (USDT): {pnl:.4f}\n"
                                         f"Reason for Entry: {last_ai_decision.get('reasoning', 'N/A')}\n"
                                         f"Closing Reason: Automatically closed by SL/TP or external action.\n")
                        summarize_and_learn(trade_summary)
                    else:
                        add_log("⚠️ Could not find recent trades to analyze for learning.")
                except Exception as e:
                    add_log(f"❌ An error occurred during post-trade analysis: {e}")
            
            was_in_position = is_in_position
            if is_in_position:
                last_position_details = pos.copy()
            
            bot_status['market_context'] = market_context
            bot_status['bot_state'] = "SEARCHING"
            position_status_report = f"Position: {pos['side'] or 'FLAT'}, Size: {pos['quantity']}, Entry Price: {pos['entry_price']}"
            
            analysis_bundle = run_heavy_analysis()
            if not analysis_bundle:
                time.sleep(60); continue
            
            context_summary = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in market_context.items()])
            if not context_summary:
                context_summary = "No market context from previous sessions is available."
            
            current_live_equity = get_total_equity()
            decision, new_context = get_ai_decision(analysis_bundle, position_status_report, context_summary, current_live_equity)
            
            if new_context:
                market_context = new_context
            
            if not decision:
                time.sleep(60); continue
            
            bot_status['last_ai_decision'] = decision
            last_ai_decision = decision
            
            if not is_decision_sane(decision):
                add_log("🚨 SANITY CHECK FAILED. Aborting action.")
                time.sleep(60); continue
            
            add_log(f"💡 AI Action Plan: {decision.get('action')}. Reason: {decision.get('reasoning')}")
            action = decision.get('action')
            
            if action == 'OPEN_POSITION':
                if pos['side']:
                    add_log("⚠️ AI wants to open but already in position. Holding.")
                else:
                    confidence = decision.get('confidence')
                    if confidence == 'low':
                        add_log(f"📉 SKIPPING TRADE: AI's confidence is LOW.")
                    elif confidence in ['medium', 'high']:
                        add_log(f"🔥 Executing trade with {confidence.upper()} confidence.")
                        open_position(decision)
                    else:
                        add_log(f"⚠️ Unknown confidence level '{confidence}'. Skipping trade for safety.")
            
            elif action == 'CLOSE_POSITION':
                if pos['side']: close_position(pos)
                else: add_log("⚠️ AI wants to close but position is already flat.")
            
            elif action == 'MODIFY_POSITION':
                if pos['side']: modify_position(decision, pos)
                else: add_log("⚠️ AI wants to modify but position is flat.")

            elif action == 'WAIT':
                trigger_price = decision.get('trigger_price')
                trigger_direction = decision.get('trigger_direction')
                if trigger_price and trigger_direction and trigger_price > 0:
                    add_log("Trigger conditions found, entering fast-check wait mode.")
                    wait_for_trigger(decision)
                else:
                    add_log(f"AI instructed to wait without a specific trigger. Monitoring continuously...")
            
            else:
                add_log(f"Unknown action from AI: {action}. Waiting.")
            
            # Default sleep interval after an action
            time.sleep(config.DEFAULT_MONITORING_INTERVAL)

        except KeyboardInterrupt:
            add_log("🛑 User interrupted. Shutting down...")
            break
        except Exception as e:
            add_log(f"💥 CRITICAL ERROR in main loop: {traceback.format_exc()}")
            time.sleep(60)

if __name__ == "__main__":
    # Ensure you have pasted all required function definitions into this single file
    # before running, especially the data analysis and execution functions.
    main_loop()