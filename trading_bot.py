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
quantity_precision = 3 # Default, will be updated on startup

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
            price_precision = s['pricePrecision']
            quantity_precision = s['quantityPrecision']
            add_log(f"✅ Precision rules for {config.SYMBOL}: Price={price_precision}, Quantity={quantity_precision}")
            break

except Exception as e:
    add_log(f"❌ Binance initialization failed: {e}")
    exit()

try:
    gemini_model = genai.GenerativeModel('gemini-2.5-pro') # type: ignore
    add_log(f"✅ Gemini AI model loaded. Using {len(config.GEMINI_API_KEYS)} API keys for rotation.")
except Exception as e:
    add_log(f"❌ Gemini AI initialization failed: {e}")
    exit()

# --- 4. Gemini Pydantic Schema and Master Prompt ---
# trading_bot.py (TradeDecision class - Corrected)


class InitialAction(BaseModel):
    """Step 1: AI decides the type of action to take."""
    action: str = Field(description="The high-level action to take: 'OPEN_POSITION', 'CLOSE_POSITION', 'MODIFY_POSITION', or 'WAIT'.")
    reasoning: str = Field(description="Brief reasoning for this high-level action.")

class OpenPositionParams(BaseModel):
    """Step 2: AI provides specific parameters for opening a position."""
    decision: str = Field(description="The direction: 'LONG' or 'SHORT'. Set to 'NULL' to abort.")
    confidence: str = Field(description="Confidence: 'high', 'medium', or 'low'. Set to 'NULL' to abort.")
    
    # --- PRECISION INSTRUCTION ADDED ---
    entry_price: float = Field(description="Target entry price (format: '%.2f'). Set to 0 to abort.")
    stop_loss: float = Field(description="Mandatory stop loss (format: '%.2f'). Set to 0 to abort.")
    take_profit: float = Field(description="Initial take profit (format: '%.2f'). Set to 0 to abort.")
    
    leverage: int = Field(description="Leverage (10-30). Set to 0 to abort.")
    risk_per_trade_percent: float = Field(description="Capital risk percentage (15-30). Set to 0 to abort.")

class ModifyPositionParams(BaseModel):
    """Step 2: AI provides specific parameters for modifying a position."""
    # --- PRECISION INSTRUCTION ADDED ---
    new_stop_loss: float = Field(description="The new stop loss price (format: '%.2f'). Set to 0 to abort or if unchanged.")
    new_take_profit: float = Field(description="The new take profit price (format: '%.2f'). Set to 0 to abort or if unchanged.")

class WaitParams(BaseModel):
    """Step 2: AI provides specific parameters for waiting."""
    next_analysis_trigger: str = Field(description="Condition for next analysis: 'IMMEDIATE' or 'PRICE_CROSS'.")
    
    # --- PRECISION INSTRUCTION ADDED ---
    trigger_price: float = Field(description="Required if trigger is 'PRICE_CROSS' (format: '%.2f'). Set to 0 otherwise.")
    trigger_timeout: int = Field(description="Required if trigger is 'PRICE_CROSS' Time in seconds to wait for the trigger condition before re-analyzing.")
    trigger_direction: str = Field(description="Required if trigger is 'PRICE_CROSS': 'ABOVE' or 'BELOW'. Set to 'NULL' otherwise.")

# --- MASTER PROMPT FOR MEMORY REFRESH ---
GEMINI_SYSTEM_PROMPT = """
You are 'The Scalpel', the world's #1 proprietary trader, known for your precision, strict discipline, and mastery of high-risk, high-reward scalping. Your analysis is final. Your output MUST be a single, complete JSON object conforming to the provided schema. There is no room for error.

**DIRECTIVE**

**Persona:** You are 'The Scalpel'. Your strategy is high-frequency scalping and momentum trading on 1m, 5m, and 15m timeframes.
**Risk Profile:** High risk (10-30% of capital per trade), high profit. Discipline is paramount. Every trade MUST have a stop loss.
**Task:** Analyze the following market data and my current position status. Return a complete JSON action plan.

**Your Decision Logic:**
- **If FLAT:** Is there a high-probability scalping opportunity RIGHT NOW? If yes, set `action: 'OPEN_POSITION'` and fill all trade parameters. If not, set `action: 'WAIT'` and define the precise condition (`next_analysis_trigger`) that would create an opportunity (e.g., a price breakout or breakdown).
- **If IN_POSITION:** Is the reason for holding the trade still valid? Should the stop loss be moved to lock in profit? Is it time to take profit? Set `action` to `'MODIFY_POSITION'`, `'CLOSE_POSITION'`, or `'WAIT'` accordingly.
- **Discipline:** If no clear edge exists, the correct action is always `'WAIT'`. Do not force trades. For a 'WAIT' decision, you must still define the `next_analysis_trigger`.
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

def run_heavy_analysis():
    add_log("🚀 Starting holographic analysis...")
    market_vitals = get_market_vitals(config.SYMBOL)
    if not market_vitals: return None
    all_data_content = "### 1. Live Market Vitals\n"
    all_data_content += f"- Order Book Imbalance (Buy Pressure): {market_vitals['order_book_imbalance']:.2%}\n"
    all_data_content += f"- Funding Rate: {market_vitals['funding_rate']:.4%}\n"
    all_data_content += f"- Open Interest ({config.SYMBOL}): {market_vitals['open_interest']:,.2f}\n"
    all_data_content += f"- Top Trader L/S Ratio: {market_vitals['top_trader_long_short_ratio']:.2f}\n\n"
    all_data_content += "### 2. Multi-Timeframe K-line Depth Analysis\n"
    for tf in config.ANALYSIS_TIMEFRAMES:
        df = get_klines_robust(config.SYMBOL, tf, limit=200)
        if df is None: continue
        df.ta.ema(lengths=[20, 50], append=True)
        df.ta.macd(append=True)
        df.ta.rsi(append=True)
        df.ta.atr(append=True)
        df.ta.adx(append=True)
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

# --- 6. Gemini Decision Function ---
def get_gemini_decision(analysis_data, position_data):
    """
    Performs a two-step confirmation to get a robust, structured trading decision from Gemini.
    """
    global current_key_index
    add_log("🧠 Step 1: Requesting initial action from Gemini...")
    
    # --- STEP 1: Get the high-level action ---
    
    system_instruction_step1 = (
        "You are 'The Scalpel', the world's top proprietary trader. Your task is to first decide the high-level action to take: "
        "'OPEN_POSITION', 'CLOSE_POSITION', 'MODIFY_POSITION', or 'WAIT'. Your output MUST be a single JSON object conforming to the InitialAction schema."
    )
    prompt_step1 = f"""
    **DIRECTIVE (Step 1/2): Determine High-Level Action**
    Analyze the following data and decide on the general course of action.

    **1. Current Position Status:**
    {position_data}

    **2. Holographic Market Analysis:**
    {analysis_data}

    Return the InitialAction JSON object now.
    """
    
    initial_decision = None
    for i in range(len(config.GEMINI_API_KEYS)):
        try:
            key = config.GEMINI_API_KEYS[current_key_index]
            genai.configure(api_key=key) # type: ignore
            current_key_index = (current_key_index + 1) % len(config.GEMINI_API_KEYS)
            
            response = gemini_model.generate_content(
                prompt_step1,
                generation_config=types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=InitialAction,
                    temperature=0,
                )
            )
            initial_decision = json.loads(response.text)
            add_log(f"✅ Gemini Step 1 successful. Action: {initial_decision.get('action')}")
            break # Success, exit the loop
        except exceptions.ResourceExhausted:
            add_log(f"⚠️ Gemini API key at index {current_key_index-1} is rate-limited. Switching...")
            continue
        except Exception as e:
            add_log(f"❌ Error during Gemini Step 1: {e}")
            return None
            
    if not initial_decision:
        add_log("🚨 All Gemini API keys failed for Step 1. Pausing.")
        time.sleep(60)
        return None

    # --- STEP 2: Get the specific parameters for the chosen action ---
    
    action = initial_decision.get('action')
    if not action:
        add_log("❌ Gemini Step 1 returned no action. Aborting.")
        return None

    add_log(f"🧠 Step 2: Requesting parameters for action '{action}'...")
    
    schema_map = {
        'OPEN_POSITION': OpenPositionParams,
        'MODIFY_POSITION': ModifyPositionParams,
        'WAIT': WaitParams,
    }
    
    # For CLOSE_POSITION, we don't need more parameters.
    if action == 'CLOSE_POSITION':
        return initial_decision

    target_schema = schema_map.get(action)
    if not target_schema:
        add_log(f"Unknown action '{action}' from Step 1. Aborting.")
        return initial_decision # Return the partial decision

    prompt_step2 = f"""
    **DIRECTIVE (Step 2/2): Provide Specific Parameters**
    Your initial decision was '{action}' with the reasoning: "{initial_decision.get('reasoning')}"

    Now, provide the exact parameters for this action.
    - If the opportunity is still valid, fill all fields in the schema.
    - **If the opportunity has passed or is no longer valid, you can abort by setting numerical values to 0 and string values to 'NULL'.**

    Return the {target_schema.__name__} JSON object now.
    """
    
    for i in range(len(config.GEMINI_API_KEYS)):
        try:
            key = config.GEMINI_API_KEYS[current_key_index]
            genai.configure(api_key=key) # type: ignore
            current_key_index = (current_key_index + 1) % len(config.GEMINI_API_KEYS)
            
            response = gemini_model.generate_content(
                prompt_step2,
                generation_config=types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=target_schema,
                    temperature=0,
                )
            )
            
            params = json.loads(response.text)
            add_log(f"✅ Gemini Step 2 successful.")
            
            # Combine the results from both steps into one final decision object
            final_decision = {**initial_decision, **params}
            return final_decision
            
        except exceptions.ResourceExhausted:
            add_log(f"⚠️ Gemini API key at index {current_key_index-1} is rate-limited. Switching...")
            continue
        except Exception as e:
            add_log(f"❌ Error during Gemini Step 2: {e}")
            return None

    add_log("🚨 All Gemini API keys failed for Step 2. Pausing.")
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
    global bot_status
    bot_status['bot_state'] = "ORDER_PENDING"
    side = decision.get('decision')
    entry_price = decision.get('entry_price')
    stop_loss_price = decision.get('stop_loss')
    leverage = decision.get('leverage')
    risk_percent = decision.get('risk_per_trade_percent')
    if not all([side, entry_price, stop_loss_price, leverage, risk_percent]):
        add_log(f"❌ Gemini OPEN decision missing required fields. Decision: {decision}")
        return
    try:
        binance_client.futures_change_leverage(symbol=config.SYMBOL, leverage=leverage)
        add_log(f"⚙️ Leverage set to {leverage}x.")
    except BinanceAPIException as e:
        add_log(f"❌ Failed to set leverage: {e}")
        return
    position_size = calculate_position_size(entry_price, stop_loss_price, risk_percent)
    if position_size <= 0:
        add_log("Calculated position size is zero, trade cancelled.")
        return
    add_log(f"💎 Decision: {side} | Size: {position_size} | Risk: {risk_percent}%")
    try:
        order = binance_client.futures_create_order(symbol=config.SYMBOL, side='BUY' if side == 'LONG' else 'SELL', type='LIMIT', timeInForce='GTC', price=f"{entry_price:.4f}", quantity=position_size, newOrderRespType='RESULT', isMakers=True)
        add_log(f"✅ Post-Only Limit Order placed @ {entry_price:.4f} (ID: {order['orderId']})")
        # Logic to handle pending order and SL/TP placement would go here
    except BinanceAPIException as e:
        if e.code == -2021: add_log("⚠️ Order failed: Price would execute immediately (Taker).")
        else: add_log(f"❌ Binance order failed: {e}")

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
    add_log(f"Executing Gemini's instruction to MODIFY position...")
    try:
        binance_client.futures_cancel_all_open_orders(symbol=config.SYMBOL)
        sl_price = decision.get('new_stop_loss')
        tp_price = decision.get('new_take_profit')
        side = position['side']
        if sl_price:
            sl_side = 'SELL' if side == 'LONG' else 'BUY'
            binance_client.futures_create_order(symbol=config.SYMBOL, side=sl_side, type='STOP_MARKET', stopPrice=f"{sl_price:.4f}", closePosition=True)
            add_log(f"✅ New Stop Loss set @ {sl_price:.4f}")
        if tp_price:
            tp_side = 'SELL' if side == 'LONG' else 'BUY'
            binance_client.futures_create_order(symbol=config.SYMBOL, side=tp_side, type='TAKE_PROFIT_MARKET', stopPrice=f"{tp_price:.4f}", closePosition=True)
            add_log(f"✅ New Take Profit set @ {tp_price:.4f}")
    except BinanceAPIException as e:
        add_log(f"❌ Failed to modify position: {e}")

def wait_for_trigger(decision):
    trigger_type = decision.get('next_analysis_trigger')
    price = decision.get('trigger_price')
    direction = decision.get('trigger_direction')
    trigger_timeout = decision.get('trigger_timeout')
    add_log(f"Entering fast-check mode. WAITING for price to cross {direction} {price} in {trigger_timeout} seconds...")
    timeout = time.time() + trigger_timeout
    while time.time() < timeout:
        try:
            ticker = binance_client.futures_mark_price(symbol=config.SYMBOL)
            current_price = float(ticker['markPrice'])
            if direction == 'ABOVE' and current_price > price:
                add_log(f"🎯 Trigger condition MET: Price {current_price} crossed ABOVE {price}.")
                return
            if direction == 'BELOW' and current_price < price:
                add_log(f"🎯 Trigger condition MET: Price {current_price} crossed BELOW {price}.")
                return
            time.sleep(5)
        except Exception as e:
            add_log(f"Error in wait_for_trigger loop: {e}")
            time.sleep(15)
    add_log("⏳ Wait condition timed out after {trigger_timeout} seconds}. Re-analyzing.")

# --- 8. Main Loop ---
def main_loop():
    global bot_status, last_gemini_decision
    add_log("🤖 AI-Driven Trading Bot Main Loop Started.")
    while True:
        try:
            pos = get_current_position()
            # Update dashboard status
            bot_status['position'] = pos
            if pos['side']:
                # PNL calculation logic...
                pass
            else:
                bot_status['pnl'] = {"usd": None, "percentage": 0}
            save_status()

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
                trigger_type = decision.get('next_analysis_trigger')
                if trigger_type == 'PRICE_CROSS':
                    wait_for_trigger(decision)
                else:
                    add_log(f"Gemini instructed to wait. Monitoring continuously...")
                    time.sleep(config.DEFAULT_MONITORING_INTERVAL)
            else:
                add_log(f"Unknown action from Gemini: {action}. Waiting.")
                time.sleep(config.DEFAULT_MONITORING_INTERVAL)

        except KeyboardInterrupt:
            add_log("\nProgram manually stopped. Cancelling all open orders...")
            try:
                binance_client.futures_cancel_all_open_orders(symbol=config.SYMBOL)
                add_log("All orders cancelled.")
            except BinanceAPIException as e:
                add_log(f"Failed to cancel orders: {e}")
            break
        except Exception as e:
            error_details = traceback.format_exc()
            add_log(f"‼️ SEVERE ERROR in main loop: {repr(e)}")
            add_log(f"--- Full Traceback ---\n{error_details}")
            time.sleep(60)

if __name__ == "__main__":
    # To make this a single, runnable file, you need to paste the full implementations
    # of all functions where they are called.
    main_loop()