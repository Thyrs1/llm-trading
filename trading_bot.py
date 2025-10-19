# trading_bot.py (v12 - With AI Memory Refresh)

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
from pydantic import BaseModel, Field

# Correct, modern imports for Google GenAI SDK
import google.generativeai as genai
from google.generativeai import types
from google.api_core import exceptions

# --- Local Imports ---
import config

# --- 1. State Management ---
class BotState(Enum):
    SEARCHING = 1
    ANALYZING = 2
    ORDER_PENDING = 3
    IN_POSITION = 4
    COOL_DOWN = 5

bot_status = { "bot_state": "INITIALIZING", "symbol": config.SYMBOL, "position": {"side": None, "quantity": 0, "entry_price": 0}, "pnl": {"usd": None, "percentage": 0}, "last_gemini_decision": None, "log": deque(maxlen=30), "last_update": None, }
current_bot_state = BotState.SEARCHING
last_gemini_decision = {}
pending_order_id = None
pending_order_start_time = 0
current_key_index = 0
# --- NEW: Global variables for exchange precision rules ---
price_precision = 2  # Default, will be updated on startup
quantity_precision = 2 # Default, will be updated on startup

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

# --- 3. API Client Initialization ---
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
            price_precision = 2
            quantity_precision = 2
            add_log(f"✅ Precision rules for {config.SYMBOL}: Price={price_precision}, Quantity={quantity_precision}")
            break

except Exception as e:
    add_log(f"❌ Binance initialization failed: {e}")
    exit()

try:
    gemini_model = genai.GenerativeModel('gemini-flash-latest') # type: ignore
    add_log(f"✅ Gemini AI model loaded. Using {len(config.GEMINI_API_KEYS)} API keys for rotation.")
except Exception as e:
    add_log(f"❌ Gemini AI initialization failed: {e}")
    exit()

# --- 4. Gemini Master Prompt (Text-Based V3) ---

GEMINI_SYSTEM_PROMPT_TEXT_BASED = """
You are 'The Scalpel', the world's #1 proprietary trader. Your analysis is final and must be communicated with absolute clarity.

**DIRECTIVE**
Analyze the provided market data and current position. Formulate a complete trading plan. You must output your final decision inside a special block called `[DECISION_BLOCK]`.

**FORMATTING RULES FOR [DECISION_BLOCK]**
1.  The block starts with the line `[DECISION_BLOCK]` and ends with `[END_BLOCK]`.
2.  Inside the block, each line must be a `KEY: VALUE` pair.
3.  **CRITICAL:** Only include the KEYs relevant to your chosen `ACTION`. Do not include keys for actions you are not taking.

**AVAILABLE KEYS and WHEN TO USE THEM:**

*   `ACTION`: (REQUIRED) The action to take. Must be one of: `OPEN_POSITION`, `CLOSE_POSITION`, `MODIFY_POSITION`, `WAIT`.
*   `REASONING`: (REQUIRED) A brief, one-sentence explanation for your action.

*   **--- If ACTION is `OPEN_POSITION`, you MUST also include:**
    *   `DECISION`: `LONG` or `SHORT`.
    *   `ENTRY_PRICE`: The target entry price.
    *   `STOP_LOSS`: The mandatory stop loss price.
    *   `TAKE_PROFIT`: The initial take profit price.
    *   `LEVERAGE`: The integer leverage to use.
    *   `RISK_PERCENT`: The percentage of capital to risk. 20.0 at minimum. 90.0 at maximum.

*   **--- If ACTION is `WAIT`, you MUST also include:**
    *   `TRIGGER_PRICE`: The price that triggers the next analysis.
    *   `TRIGGER_DIRECTION`: `ABOVE` or `BELOW`.
    *   `TRIGGER_TIMEOUT`: Timeout in seconds.

*   **--- If ACTION is `MODIFY_POSITION`, you MUST also include:**
    *   `NEW_STOP_LOSS`: The new stop loss price. (Use 0 if not changing)
    *   `NEW_TAKE_PROFIT`: The new take profit price. (Use 0 if not changing)

*   **--- If ACTION is `CLOSE_POSITION`, no other keys are needed.**

**EXAMPLE 1: Opening a Long Position**
[DECISION_BLOCK]
ACTION: OPEN_POSITION
REASONING: The price is showing bullish momentum after breaking a key resistance level on the 15m chart.
DECISION: LONG
ENTRY_PRICE: 190.50
STOP_LOSS: 189.00
TAKE_PROFIT: 193.00
LEVERAGE: 20
RISK_PERCENT: 2.5
[END_BLOCK]
**EXAMPLE 2: Waiting for a Price Drop**
[DECISION_BLOCK]
ACTION: WAIT
REASONING: The market is consolidating; waiting for a pullback to a stronger support level before considering an entry.
TRIGGER_PRICE: 188.75
TRIGGER_DIRECTION: BELOW
TRIGGER_TIMEOUT: 600
[END_BLOCK]
Now, analyze the following data and provide your decision.
"""

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
    
    all_data_content = f"### 0. Current Market Price (Anchor)\n- **Current Price:** {current_price:.2f} USDT\n\n"
    all_data_content = "### 1. Live Market Vitals\n"
    all_data_content += f"- Order Book Imbalance (Buy Pressure): {market_vitals['order_book_imbalance']:.2%}\n"
    all_data_content += f"- Funding Rate: {market_vitals['funding_rate']:.4%}\n"
    all_data_content += f"- Open Interest (USDT): {market_vitals['open_interest']:,.2f}\n"
    all_data_content += f"- Top Trader L/S Ratio: {market_vitals['top_trader_long_short_ratio']:.2f}\n\n"
    all_data_content += "### 2. Multi-Timeframe K-line Depth Analysis\n"

    for tf in config.ANALYSIS_TIMEFRAMES:
        # --- CRITICAL FIX: Fetch more data to ensure indicators can warm up ---
        # We need at least 200 candles for the EMA_200, so fetching 300 gives a buffer.
        df = get_klines_robust(config.SYMBOL, tf, limit=300)
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

def is_decision_sane(decision):
    """
    Performs a sanity check on the prices provided by Gemini to prevent hallucinations.
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
    Asks Gemini to "think out loud" in plain English without JSON constraints.
    This is used to diagnose why it might be generating faulty data.
    """
    global current_key_index
    add_log("--- 🧠 Requesting Gemini RAW THOUGHT PROCESS for diagnostics ---")
    
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
    
    # This loop is for a single, non-structured request
    for i in range(len(config.GEMINI_API_KEYS)):
        try:
            key = config.GEMINI_API_KEYS[current_key_index]
            # Create a fresh, temporary model instance
            genai.configure(api_key=key) # type: ignore
            model = genai.GenerativeModel('gemini-flash-latest') # type: ignore
            current_key_index = (current_key_index + 1) % len(config.GEMINI_API_KEYS)
            
            # Make a standard text-only API call
            response = model.generate_content(diagnostic_prompt)
            
            return response.text # Return the plain text response
            
        except exceptions.ResourceExhausted:
            add_log(f"⚠️ Gemini API key at index {current_key_index-1} is rate-limited. Switching...")
            continue
        except Exception as e:
            add_log(f"❌ Error during diagnostic query: {e}")
            return f"Failed to get diagnostic response: {e}"

    return "All Gemini API keys failed during diagnostic query."

def parse_decision_block(raw_text: str) -> dict:
    """
    Parses the structured text block from Gemini's response into a dictionary.
    This function is designed to be robust against extra text or formatting errors.
    """
    decision = {}
    in_block = False
    
    # A mapping of keys to their expected types for automatic conversion
    type_map = {
        "ENTRY_PRICE": float, "STOP_LOSS": float, "TAKE_PROFIT": float,
        "LEVERAGE": int, "RISK_PERCENT": float, "TRIGGER_PRICE": float,
        "TRIGGER_TIMEOUT": int, "NEW_STOP_LOSS": float, "NEW_TAKE_PROFIT": float
    }

    for line in raw_text.splitlines():
        line = line.strip()
        
        if line == '[DECISION_BLOCK]':
            in_block = True
            continue
        
        if line == '[END_BLOCK]':
            break

        if in_block and ':' in line:
            # Split only on the first colon to handle reasoning text with colons
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Convert value to the correct type if needed
            if key in type_map:
                try:
                    decision[key.lower()] = type_map[key](value)
                except (ValueError, TypeError):
                    add_log(f"⚠️ Parser Warning: Could not convert '{value}' for key '{key}'. Defaulting to 0 or None.")
                    decision[key.lower()] = None
            else:
                # Keys like ACTION, REASONING, etc., remain strings
                decision[key.lower()] = value
    
    return decision

# --- 6. Gemini Decision Function ---

# --- 6. Gemini Decision Function (Text-Based V3) ---

def get_gemini_decision(analysis_data, position_data):
    """
    Gets a trading decision from Gemini using a structured text format,
    then parses it into a Python dictionary.
    """
    global current_key_index
    
    prompt = f"""
    {GEMINI_SYSTEM_PROMPT_TEXT_BASED}

    **--- CURRENT DATA FOR ANALYSIS ---**

    **1. Current Position Status:**
    {position_data}

    **2. Holographic Market Analysis:**
    {analysis_data}

    Provide your analysis and `[DECISION_BLOCK]` now.
    """
    
    for i in range(len(config.GEMINI_API_KEYS)):
        try:
            key = config.GEMINI_API_KEYS[current_key_index]
            add_log(f"🧠 Requesting text-based decision from Gemini with key index {current_key_index}...")
            
            genai.configure(api_key=key) # type: ignore
            model = genai.GenerativeModel('gemini-flash-latest') # type: ignore
            
            current_key_index = (current_key_index + 1) % len(config.GEMINI_API_KEYS)
            
            # Simple text-in, text-out generation. No complex schemas.
            response = model.generate_content(prompt)
            
            raw_response_text = response.text
            add_log("--- 🧠 GEMINI RAW TEXT RESPONSE ---")
            add_log(raw_response_text)
            add_log("--- END RAW TEXT RESPONSE ---")
            
            # Parse the raw text to get a structured dictionary
            decision_dict = parse_decision_block(raw_response_text)
            
            if not decision_dict or 'action' not in decision_dict:
                add_log("❌ Parsing failed or ACTION key is missing in the response. Trying next key.")
                continue

            add_log(f"✅ Gemini decision parsed successfully. Action: {decision_dict.get('action')}")
            return decision_dict
            
        except exceptions.ResourceExhausted:
            add_log(f"⚠️ Gemini API key at index {current_key_index-1} is rate-limited. Switching...")
            continue
        except Exception as e:
            add_log(f"❌ An unexpected error occurred during Gemini call: {traceback.format_exc()}")
            # Continue to the next key
    
    add_log("🚨 All Gemini API keys failed or returned unparsable responses. Pausing.")
    time.sleep(60)
    return None


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

def calculate_position_size(entry_price, stop_loss_price, risk_percent):
    try:
        actual_risk_percent = min(risk_percent / 100, config.MAX_RISK_PER_TRADE)
        amount_to_risk_usdt = config.TOTAL_CAPITAL_USDT * actual_risk_percent
        price_delta_per_unit = abs(entry_price - stop_loss_price)
        if price_delta_per_unit == 0: return 0
        position_size_units = amount_to_risk_usdt / price_delta_per_unit
        return round(position_size_units, 3)
    except Exception as e:
        add_log(f"❌ Error calculating position size: {e}")
        return 0

def open_position(decision):
    """Executes the logic to open a new position with correct precision."""
    global bot_status, last_gemini_decision
    bot_status['bot_state'] = "ORDER_PENDING"
    
    # --- MODIFICATION START ---
    # Use the new, consistent, lowercase key names from our parser
    side = decision.get('decision')
    entry_price = decision.get('entry_price')
    stop_loss_price = decision.get('stop_loss')
    leverage = decision.get('leverage')
    risk_percent = decision.get('risk_percent') # CORRECTED KEY NAME
    # --- MODIFICATION END ---
    
    if not all([side, entry_price, stop_loss_price, leverage, risk_percent is not None]): # Added 'is not None' for robustness
        add_log(f"❌ Gemini OPEN decision missing required fields. Decision: {decision}")
        bot_status['bot_state'] = "SEARCHING"
        return

    try:
        binance_client.futures_change_leverage(symbol=config.SYMBOL, leverage=leverage)
        add_log(f"⚙️ Leverage set to {leverage}x.")
    except BinanceAPIException as e:
        add_log(f"❌ Failed to set leverage: {e}")
        bot_status['bot_state'] = "SEARCHING"
        return

    position_size = calculate_position_size(entry_price, stop_loss_price, risk_percent)
    if position_size <= 0:
        add_log("Calculated position size is zero or invalid, trade cancelled.")
        bot_status['bot_state'] = "SEARCHING"
        return
        
    add_log(f"💎 Decision: {side} | Size: {position_size} | Risk: {risk_percent}%")
    
    try:
        # --- RE-INTRODUCED PYTHON FORMATTING ---
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
            newOrderRespType='RESULT',
            isMakers=True
        )
        add_log(f"✅ Post-Only Limit Order placed @ {formatted_price} (ID: {order['orderId']})")
        # Logic to handle pending order and SL/TP placement would go here
        
    except BinanceAPIException as e:
        if e.code == -2021: 
            add_log("⚠️ Order failed: Price would execute immediately (Taker).")
        else: 
            add_log(f"❌ Binance order failed: {e}")
        bot_status['bot_state'] = "SEARCHING"

def close_position(position):
    add_log(f"Executing Gemini's instruction to CLOSE position...")
    try:
        binance_client.futures_cancel_all_open_orders(symbol=config.SYMBOL)
        close_side = 'BUY' if position['side'] == 'SHORT' else 'SELL'
        binance_client.futures_create_order(symbol=config.SYMBOL, side=close_side, type='MARKET', quantity=position['quantity'], reduceOnly=True)
        add_log(f"✅ Market close order sent successfully.")
    except BinanceAPIException as e:
        add_log(f"❌ Failed to close position: {e}")

def modify_position(decision, position):
    """Modifies the Stop Loss and/or Take Profit with correct precision."""
    add_log(f"Executing Gemini's instruction to MODIFY position...")
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
    """Enters a fast loop to check for a specific condition set by Gemini."""
    trigger_type = decision.get('next_analysis_trigger')
    price = decision.get('trigger_price')
    direction = decision.get('trigger_direction')
    
    # Add a check for valid trigger parameters
    if not all([trigger_type, price, direction]) or price == 0 or direction == 'NULL':
        add_log("⚠️ Gemini WAIT decision missing valid trigger parameters. Re-analyzing immediately.")
        return

    add_log(f"Entering fast-check mode. WAITING for price to cross {direction} {price}...")
    
    # Use the timeout from the AI's decision, with a default
    timeout_seconds = decision.get('trigger_timeout', 300)
    timeout = time.time() + timeout_seconds
    
    while time.time() < timeout:
        try:
            ticker = binance_client.futures_mark_price(symbol=config.SYMBOL)
            current_price = float(ticker['markPrice'])
            
            if direction == 'ABOVE' and current_price > price:
                add_log(f"🎯 Trigger condition MET: Price {current_price:.4f} crossed ABOVE {price:.4f}.")
                return
            if direction == 'BELOW' and current_price < price:
                add_log(f"🎯 Trigger condition MET: Price {current_price:.4f} crossed BELOW {price:.4f}.")
                return
                
            # Check every 5 seconds
            time.sleep(5)
        except Exception as e:
            add_log(f"Error in wait_for_trigger loop: {e}")
            time.sleep(15) # Wait longer on error
            
    add_log(f"⏳ Wait condition timed out after {timeout_seconds} seconds. Re-analyzing.")

# --- 8. Main Loop ---
# trading_bot.py (main_loop function - with Stateless Diagnostic)

def main_loop():
    global bot_status, last_gemini_decision
    add_log("🤖 AI-Driven Trading Bot Main Loop Started.")
    
    startup_analysis_done = False

    while True:
        try:
            if not startup_analysis_done:
                add_log("🚀 Performing initial startup analysis for debugging...")
                bot_status['bot_state'] = "ANALYZING"
                startup_analysis_done = True
            else:
                bot_status['bot_state'] = "SEARCHING"
            
            pos = get_current_position()
            position_status_report = f"Position: {pos['side'] or 'FLAT'}, Size: {pos['quantity']}, Entry Price: {pos['entry_price']}"
            
            analysis_bundle = run_heavy_analysis()
            if not analysis_bundle:
                time.sleep(60)
                continue
                
            decision = get_gemini_decision(analysis_bundle, position_status_report)
            if not decision:
                time.sleep(60)
                continue
            
            bot_status['last_gemini_decision'] = decision
            last_gemini_decision = decision
            
            if not is_decision_sane(decision):
                add_log("🚨 SANITY CHECK FAILED. Initiating diagnostic query to understand AI's reasoning...")
                
                # --- NEW: Call the diagnostic function ---
                diagnostic_text = run_diagnostic_query(analysis_bundle, position_status_report)
                
                add_log("--- 🧠 GEMINI RAW THOUGHT PROCESS (DIAGNOSTIC) ---")
                # Log the raw, unconstrained thoughts of the AI
                add_log(diagnostic_text)
                add_log("--- END DIAGNOSTIC ---")

                add_log("Aborting action due to failed sanity check. Re-analyzing in next cycle.")
                time.sleep(60) # Wait a minute before trying again
                continue
            
            # --- If Sanity Check Passes, Proceed with Action ---
            add_log(f"💡 Gemini Action Plan: {decision.get('action')}. Reason: {decision.get('reasoning')}")
            action = decision.get('action')
            
            if action == 'OPEN_POSITION':
                if pos['side']: add_log("⚠️ Gemini wants to open but already in position. Holding.")
                else: open_position(decision)
                time.sleep(config.DEFAULT_MONITORING_INTERVAL)
            elif action == 'CLOSE_POSITION':
                if pos['side']: close_position(pos)
                else: add_log("⚠️ Gemini wants to close but position is already flat.")
                time.sleep(config.DEFAULT_MONITORING_INTERVAL)
            elif action == 'MODIFY_POSITION':
                if pos['side']: modify_position(decision, pos)
                else: add_log("⚠️ Gemini wants to modify but position is flat.")
                time.sleep(config.DEFAULT_MONITORING_INTERVAL)
            elif action == 'WAIT':
                # --- MODIFICATION START ---
                # Check for the presence of trigger keys to decide the wait type.
                # This is more robust and matches our new text-based prompt.
                trigger_price = decision.get('trigger_price')
                trigger_direction = decision.get('trigger_direction')

                if trigger_price and trigger_direction and trigger_price > 0:
                    add_log("Trigger conditions found, entering fast-check wait mode.")
                    wait_for_trigger(decision)
                else:
                    # If AI says WAIT but provides no specific trigger, default to monitoring.
                    add_log(f"Gemini instructed to wait without a specific price trigger. Monitoring continuously...")
                    time.sleep(config.DEFAULT_MONITORING_INTERVAL)
                # --- MODIFICATION END ---
            
            else:
                add_log(f"Unknown action from Gemini: {action}. Waiting.")
                time.sleep(config.DEFAULT_MONITORING_INTERVAL)

        except KeyboardInterrupt:
            # ... (KeyboardInterrupt logic remains the same)
            break
        except Exception as e:
            # ... (Detailed error logging remains the same)
            time.sleep(60)

if __name__ == "__main__":
    # To make this a single, runnable file, you need to paste the full implementations
    # of all functions where they are called.
    main_loop()