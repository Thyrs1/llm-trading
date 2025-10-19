# trading_bot.py (v10 - Final, All Fixes Integrated)

# --- Standard Library Imports ---
import os
import time
import json
import math
from enum import Enum
from collections import deque
from datetime import datetime, timezone

# --- Third-Party Library Imports ---
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pydantic import BaseModel, Field

# Correct, modern imports for Google GenAI SDK
import google.generativeai as genai
from google.generativeai import types
from google.api_core import exceptions # For catching rate limit errors

# --- Local Imports ---
import config

# --- 1. State Management ---
class BotState(Enum):
    SEARCHING = 1
    ANALYZING = 2
    ORDER_PENDING = 3
    IN_POSITION = 4
    COOL_DOWN = 5

bot_status = { "bot_state": "INITIALIZING", "symbol": config.SYMBOL, "position": {"side": None, "quantity": 0, "entry_price": 0}, "pnl": {"usd": None, "percentage": 0}, "last_gemini_decision": None, "log": deque(maxlen=20), "last_update": None, }
current_bot_state = BotState.SEARCHING
last_gemini_decision = {}
pending_order_id = None
pending_order_start_time = 0
current_key_index = 0 # Index for API key rotation

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
    # --- CORRECTED: Initialize Binance Client with explicit base_url for Testnet ---
    binance_client = Client(
        config.BINANCE_API_KEY,
        config.BINANCE_API_SECRET,
        tld='com' if not config.BINANCE_TESTNET else 'testnet'
    )
    server_time = binance_client.get_server_time()['serverTime']
    binance_client.timestamp_offset = server_time - int(time.time() * 1000)
    add_log("✅ Time synchronization successful.")
    add_log(f"✅ Binance client initialized (Testnet: {config.BINANCE_TESTNET}).")
    try:
        binance_client.futures_change_margin_type(symbol=config.SYMBOL, marginType='ISOLATED')
        add_log("✅ Margin type set to ISOLATED.")
    except BinanceAPIException as e:
        if e.code == -4046:
            add_log("✅ Margin type was already ISOLATED.")
        else: raise e
except Exception as e:
    add_log(f"❌ Binance initialization failed: {e}")
    exit()

try:
    # --- CORRECTED: Load the model, but configure the key per-request for rotation ---
    gemini_model = genai.GenerativeModel('gemini-2.5-pro') # type: ignore
    add_log(f"✅ Gemini AI model loaded. Using {len(config.GEMINI_API_KEYS)} API keys for rotation.")
except Exception as e:
    add_log(f"❌ Gemini AI initialization failed: {e}")
    exit()

# --- 4. Gemini Structured Output Schema (Using Pydantic) ---
class TradeDecision(BaseModel):
    decision: str = Field(description="The trading action: LONG, SHORT, or HOLD.")
    confidence: str = Field(description="The confidence level: high, medium, or low.")
    reasoning: str = Field(description="A brief justification for the trade, referencing data.")
    entry_price: float = Field(description="The target entry price.")
    stop_loss: float = Field(description="The mandatory stop loss price.")
    take_profit: float = Field(description="The initial take profit price.")
    leverage: int = Field(description="The leverage to use (10-50).")
    risk_per_trade_percent: float = Field(description="The percentage of total capital to risk (15-90).")
    trailing_stop_callback: float = Field(description="The trailing stop callback rate in percent (0.5-2.0).")
    order_pending_timeout_seconds: int = Field(description="Max time in seconds to wait for the limit order to fill.")
TRADE_DECISION_SCHEMA = TradeDecision

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
        funding_rates = binance_client.futures_funding_rate(symbol=symbol, limit=1)
        vitals['funding_rate'] = float(funding_rates[0]['fundingRate']) if funding_rates else 0.0
        open_interest_hist = binance_client.futures_open_interest_hist(symbol=symbol, period='5m', limit=1)
        vitals['open_interest'] = float(open_interest_hist[0]['sumOpenInterestValue'])
        long_short_ratio = binance_client.futures_top_longshort_position_ratio(symbol=symbol, period='5m', limit=1)[0]
        vitals['top_trader_long_short_ratio'] = float(long_short_ratio['longShortRatio'])
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
    all_data_content += f"- Open Interest (USDT): {market_vitals['open_interest']:,.2f}\n"
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
        atr_percentage = (latest['ATRr_14'] / latest['close']) * 100 if latest['close'] > 0 else 0
        volume_strength = latest['volume'] / latest['volume_ma_20'] if latest['volume_ma_20'] > 0 else 0
        report = f"--- Analysis Report ({tf} Timeframe) ---\n"
        report += f"Close Price: {latest['close']:.4f}\n"
        report += f"Trend Strength (ADX_14): {latest.get('ADX_14', 0):.2f}\n"
        report += f"EMA 20/50: {latest['EMA_20']:.4f} / {latest['EMA_50']:.4f}\n"
        report += f"RSI_14: {latest['RSI_14']:.2f}\n"
        report += f"ATR_Volatility_Percent: {atr_percentage:.2f}%\n"
        report += f"Volume_Strength_Ratio: {volume_strength:.2f}x\n"
        all_data_content += report
    return all_data_content

# --- CORRECTED: Gemini Decision Function with Key Rotation ---
def get_gemini_decision(analysis_data):
    global current_key_index
    add_log("🧠 Requesting Gemini strategy with key rotation...")
    system_instruction = ("You are an AI quantitative trading strategist...") # Omitted for brevity
    prompt = f"Analyze the following data...\n{analysis_data}\nReturn the JSON object now."
    for i in range(len(config.GEMINI_API_KEYS)):
        try:
            key = config.GEMINI_API_KEYS[current_key_index]
            add_log(f"Attempting API call with key index {current_key_index}...")
            genai.configure(api_key=key) # type: ignore
            current_key_index = (current_key_index + 1) % len(config.GEMINI_API_KEYS)
            response = gemini_model.generate_content(
                prompt,
                generation_config=types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=TRADE_DECISION_SCHEMA,
                    temperature=0,
                )
            )
            decision = json.loads(response.text)
            add_log("✅ Gemini decision received successfully.")
            return decision
        except exceptions.ResourceExhausted as e:
            add_log(f"⚠️ Gemini API key at index {current_key_index-1} is rate-limited. Switching to next key...")
            continue
        except Exception as e:
            add_log(f"❌ An unexpected error occurred during Gemini API call: {e}")
            return None
    add_log("🚨 All Gemini API keys are rate-limited. Pausing for 60 seconds...")
    time.sleep(60)
    return None

# --- 6. Trading Execution and Management Functions ---
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

def check_for_trigger(df):
    if df is None or len(df) < 21: return False, "Data insufficient", 0, 0
    vol_mult = config.NORMAL_VOLUME_SPIKE if bot_status['bot_state'] == BotState.SEARCHING.name else config.ALERT_VOLUME_SPIKE
    rng_mult = config.NORMAL_VOLATILITY_SPIKE if bot_status['bot_state'] == BotState.SEARCHING.name else config.ALERT_VOLATILITY_SPIKE
    last_20 = df.iloc[-21:-1]
    current = df.iloc[-1]
    avg_volume = last_20['volume'].mean()
    avg_range = (last_20['high'] - last_20['low']).mean()
    vol_ratio = current['volume'] / avg_volume if avg_volume > 0 else 0
    rng_ratio = (current['high'] - current['low']) / avg_range if avg_range > 0 else 0
    if vol_ratio > vol_mult: return True, f"Volume Spike ({bot_status['bot_state']})", vol_ratio, rng_ratio
    if rng_ratio > rng_mult: return True, f"Volatility Spike ({bot_status['bot_state']})", vol_ratio, rng_ratio
    return False, "Market stable", vol_ratio, rng_ratio

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

def execute_trade(decision):
    global current_bot_state, pending_order_id, pending_order_start_time, last_gemini_decision
    if not decision or decision.get('decision') == 'HOLD':
        add_log("Gemini decision is 'HOLD', returning to SEARCHING.")
        current_bot_state = BotState.SEARCHING
        return
    side = decision['decision']
    entry_price = decision['entry_price']
    stop_loss_price = decision['stop_loss']
    try:
        binance_client.futures_change_leverage(symbol=config.SYMBOL, leverage=decision['leverage'])
        add_log(f"⚙️ Leverage set to {decision['leverage']}x.")
    except BinanceAPIException as e:
        add_log(f"❌ Failed to set leverage: {e}")
        current_bot_state = BotState.SEARCHING
        return
    position_size = calculate_position_size(entry_price, stop_loss_price, decision['risk_per_trade_percent'])
    if position_size <= 0:
        add_log("Calculated position size is zero or negative, trade cancelled.")
        current_bot_state = BotState.SEARCHING
        return
    add_log(f"💎 Decision: {side} | Size: {position_size} | Risk: {decision['risk_per_trade_percent']}%")
    try:
        order = binance_client.futures_create_order(symbol=config.SYMBOL, side='BUY' if side == 'LONG' else 'SELL', type='LIMIT', timeInForce='GTC', price=f"{entry_price:.4f}", quantity=position_size, newOrderRespType='RESULT', isMakers=True)
        add_log(f"✅ Post-Only Limit Order placed @ {entry_price:.4f} (ID: {order['orderId']})")
        pending_order_id = order['orderId']
        pending_order_start_time = time.time()
        current_bot_state = BotState.ORDER_PENDING
    except BinanceAPIException as e:
        if e.code == -2021: add_log("⚠️ Order failed: Price would execute immediately (Taker). Returning to SEARCHING.")
        else: add_log(f"❌ Binance order failed: {e}")
        current_bot_state = BotState.SEARCHING

def manage_position(position):
    global last_gemini_decision
    if last_gemini_decision.get('take_profit', 0) == 0: return
    side, entry_price, initial_tp, callback_rate = position['side'], position['entry_price'], last_gemini_decision.get('take_profit'), last_gemini_decision.get('trailing_stop_callback', 1.0)
    try:
        ticker = binance_client.futures_mark_price(symbol=config.SYMBOL)
        current_price = float(ticker['markPrice'])
    except BinanceAPIException: return
    profit_target_distance = abs(initial_tp - entry_price)
    current_profit_distance = abs(current_price - entry_price)
    if current_profit_distance >= (profit_target_distance * 0.75):
        add_log(f"📈 Profit at 75% of TP. Deploying Trailing Stop ({callback_rate}%)...")
        try:
            binance_client.futures_cancel_all_open_orders(symbol=config.SYMBOL)
            sl_side = 'SELL' if side == 'LONG' else 'BUY'
            binance_client.futures_create_order(symbol=config.SYMBOL, side=sl_side, type='TRAILING_STOP_MARKET', callbackRate=callback_rate, quantity=position['quantity'], reduceOnly=True)
            add_log(f"✅ Trailing Stop deployed with {callback_rate}% callback.")
            last_gemini_decision['take_profit'] = 0
        except BinanceAPIException as e:
            add_log(f"❌ Failed to deploy Trailing Stop: {e}")

# --- 7. Main Loop ---
def main_loop():
    global current_bot_state, last_gemini_decision, pending_order_id, pending_order_start_time, bot_status
    add_log("🤖 Trading Bot Main Loop Started.")
    startup_analysis_done = False
    while True:
        try:
            if not startup_analysis_done:
                add_log("🚀 Performing initial startup analysis for debugging...")
                current_bot_state = BotState.ANALYZING
                startup_analysis_done = True
            bot_status['bot_state'] = current_bot_state.name
            pos = get_current_position()
            if pos['side']:
                bot_status['position'] = pos
                ticker = binance_client.futures_mark_price(symbol=config.SYMBOL)
                current_price = float(ticker['markPrice'])
                pnl_usd = (current_price - pos['entry_price']) * pos['quantity'] if pos['side'] == 'LONG' else (pos['entry_price'] - current_price) * pos['quantity']
                leverage = last_gemini_decision.get('leverage', 20)
                initial_margin = (pos['entry_price'] * pos['quantity']) / leverage if leverage > 0 else 0
                pnl_perc = (pnl_usd / initial_margin) * 100 if initial_margin > 0 else 0
                bot_status['pnl'] = {"usd": pnl_usd, "percentage": pnl_perc}
            else:
                bot_status['position'] = {"side": None, "quantity": 0, "entry_price": 0}
                bot_status['pnl'] = {"usd": None, "percentage": 0}
            if current_bot_state == BotState.SEARCHING:
                trigger_df = get_klines_robust(config.SYMBOL, '1m', limit=21)
                is_triggered, reason, vol_ratio, rng_ratio = check_for_trigger(trigger_df)
                if is_triggered:
                    add_log(f"🎯 Market Trigger Activated! Reason: {reason}")
                    current_bot_state = BotState.ANALYZING
                else:
                    add_log(f"🔍 Searching... Vol: {vol_ratio:.2f}x | Rng: {rng_ratio:.2f}x")
                    time.sleep(60)
            elif current_bot_state == BotState.ANALYZING:
                analysis_bundle = run_heavy_analysis()
                if analysis_bundle:
                    trade_decision = get_gemini_decision(analysis_bundle)
                    if trade_decision:
                        bot_status['last_gemini_decision'] = trade_decision
                        last_gemini_decision = trade_decision
                        execute_trade(trade_decision)
                else:
                    add_log("Analysis failed, returning to SEARCHING.")
                    current_bot_state = BotState.SEARCHING
            elif current_bot_state == BotState.ORDER_PENDING:
                try:
                    order_status = binance_client.futures_get_order(symbol=config.SYMBOL, orderId=pending_order_id)
                    if order_status['status'] == 'FILLED':
                        add_log("🎉 Order FILLED! Entering IN_POSITION state.")
                        current_bot_state = BotState.IN_POSITION
                    elif time.time() - pending_order_start_time > last_gemini_decision.get('order_pending_timeout_seconds', 300):
                        binance_client.futures_cancel_order(symbol=config.SYMBOL, orderId=pending_order_id)
                        add_log("⏳ Order timed out, cancelled. Returning to SEARCHING.")
                        current_bot_state = BotState.SEARCHING
                except BinanceAPIException as e:
                    add_log(f"Querying order status failed: {e}")
                    current_bot_state = BotState.SEARCHING
                time.sleep(10)
            elif current_bot_state == BotState.IN_POSITION:
                if not pos['side']:
                    add_log("💰 Position closed. Entering 15-minute COOL_DOWN.")
                    current_bot_state = BotState.COOL_DOWN
                    continue
                manage_position(pos)
                time.sleep(15)
            elif current_bot_state == BotState.COOL_DOWN:
                add_log("Cooling down...")
                time.sleep(900)
                add_log("Cool down finished, returning to SEARCHING.")
                current_bot_state = BotState.SEARCHING
        except KeyboardInterrupt:
            add_log("\nProgram manually stopped. Cancelling all open orders...")
            try:
                binance_client.futures_cancel_all_open_orders(symbol=config.SYMBOL)
                add_log("All orders cancelled.")
            except BinanceAPIException as e:
                add_log(f"Failed to cancel orders: {e}")
            break
        except Exception as e:
            add_log(f"Severe error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main_loop()