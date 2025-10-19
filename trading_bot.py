# trading_bot.py

import os
import time
import json
import math
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import google.generativeai as genai
from google.generativeai import types
from enum import Enum
from collections import deque
from datetime import datetime
import config_template

# --- 1. State Management ---
class BotState(Enum):
    SEARCHING = 1
    ANALYZING = 2
    ORDER_PENDING = 3
    IN_POSITION = 4
    COOL_DOWN = 5

# Global state dictionary for dashboard
bot_status = {
    "bot_state": "INITIALIZING",
    "symbol": config_template.SYMBOL,
    "position": {"side": None, "quantity": 0, "entry_price": 0},
    "pnl": {"usd": None, "percentage": 0},
    "last_gemini_decision": None,
    "log": deque(maxlen=20),
    "last_update": None,
}
current_bot_state = BotState.SEARCHING
last_gemini_decision = {}
pending_order_id = None
pending_order_start_time = 0

# --- 2. Utility Functions ---

def save_status():
    """Writes the global status dictionary to a JSON file for the dashboard."""
    bot_status['last_update'] = datetime.utcnow().isoformat() + 'Z'
    status_to_save = bot_status.copy()
    status_to_save['log'] = list(bot_status['log'])
    try:
        with open('status.json', 'w') as f:
            json.dump(status_to_save, f, indent=4)
    except Exception as e:
        print(f"!!! CRITICAL ERROR: Could not write status file: {e}")

def add_log(message):
    """Adds a timestamped entry to the log and saves status."""
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    bot_status['log'].appendleft(log_entry)
    print(log_entry)
    save_status()

# --- 3. API Client Initialization ---

try:
    binance_client = Client(config_template.BINANCE_API_KEY, config_template.BINANCE_API_SECRET, testnet=config_template.BINANCE_TESTNET)
    binance_client.futures_change_margin_type(symbol=config_template.SYMBOL, marginType='ISOLATED')
    add_log(f"✅ Binance client initialized (Testnet: {config_template.BINANCE_TESTNET}).")
except Exception as e:
    add_log(f"❌ Binance initialization failed: {e}")
    exit()

try:
    # Using gemini-2.5-flash-latest as requested
    genai.configure(api_key=config_template.GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-latest')
    add_log("✅ Gemini AI client initialized (Flash).")
except Exception as e:
    add_log(f"❌ Gemini AI initialization failed: {e}")
    exit()

# --- 4. Gemini Structured Output Schema ---

TRADE_DECISION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "decision": types.Schema(type=types.Type.STRING, enum=["LONG", "SHORT", "HOLD"]),
        "entry_price": types.Schema(type=types.Type.NUMBER),
        "stop_loss": types.Schema(type=types.Type.NUMBER),
        "take_profit": types.Schema(type=types.Type.NUMBER),
        "confidence": types.Schema(type=types.Type.STRING, enum=["high", "medium", "low"]),
        "leverage": types.Schema(type=types.Type.INTEGER),
        "risk_per_trade_percent": types.Schema(type=types.Type.NUMBER),
        "trailing_stop_callback": types.Schema(type=types.Type.NUMBER),
        "order_pending_timeout_seconds": types.Schema(type=types.Type.INTEGER),
        "reasoning": types.Schema(type=types.Type.STRING),
    },
    required=["decision", "entry_price", "stop_loss", "take_profit", "confidence", "leverage", "risk_per_trade_percent", "trailing_stop_callback", "order_pending_timeout_seconds", "reasoning"]
)

# --- 5. Data Acquisition and Analysis Functions ---

def get_klines_robust(symbol, interval, limit=200, retries=3, delay=5):
    """Robust K-line fetching with retry logic."""
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
    """Fetches real-time market sentiment and structure data."""
    vitals = {}
    try:
        depth = binance_client.futures_order_book(symbol=symbol, limit=20)
        total_bids_qty = sum([float(qty) for price, qty in depth['bids']])
        total_asks_qty = sum([float(qty) for price, qty in depth['asks']])
        total_qty = total_bids_qty + total_asks_qty
        vitals['order_book_imbalance'] = total_bids_qty / total_qty if total_qty > 0 else 0.5
        premium_index = binance_client.futures_premium_index(symbol=symbol)
        vitals['funding_rate'] = float(premium_index['lastFundingRate'])
        vitals['open_interest'] = float(premium_index['openInterest'])
        long_short_ratio = binance_client.futures_top_long_short_account_ratio(symbol=symbol, period='5m', limit=1)[0]
        vitals['top_trader_long_short_ratio'] = float(long_short_ratio['longShortRatio'])
        return vitals
    except BinanceAPIException as e:
        add_log(f"❌ Error fetching market vitals: {e}")
        return None

def run_heavy_analysis():
    """Generates the full holographic analysis report for Gemini."""
    add_log("🚀 Starting holographic analysis...")
    market_vitals = get_market_vitals(config_template.SYMBOL)
    if not market_vitals: return None
        
    all_data_content = "### 1. Live Market Vitals\n"
    all_data_content += f"- Order Book Imbalance (Buy Pressure): {market_vitals['order_book_imbalance']:.2%}\n"
    all_data_content += f"- Funding Rate: {market_vitals['funding_rate']:.4%}\n"
    all_data_content += f"- Open Interest: {market_vitals['open_interest']:,.0f} {config_template.SYMBOL}\n"
    all_data_content += f"- Top Trader L/S Ratio: {market_vitals['top_trader_long_short_ratio']:.2f}\n\n"
    all_data_content += "### 2. Multi-Timeframe K-line Depth Analysis\n"

    for tf in config_template.ANALYSIS_TIMEFRAMES:
        df = get_klines_robust(config_template.SYMBOL, tf, limit=200)
        if df is None: continue
        
        # Calculate indicators
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

def get_gemini_decision(analysis_data):
    """Requests Gemini for a structured trading decision with dynamic risk parameters."""
    add_log("🧠 Requesting Gemini strategy...")
    
    system_instruction = (
        "You are an AI quantitative trading strategist. Before making a decision, "
        "you must internally synthesize the Market Vitals and K-line data to form a clear thesis. "
        "Your final output MUST be a single JSON object conforming to the provided schema. "
        "Dynamically set risk parameters (leverage, trailing_stop_callback, risk_per_trade_percent) "
        "based on the confidence derived from the data synthesis."
    )

    prompt = f"""
    Analyze the following comprehensive market data for {config_template.SYMBOL}.
    Synthesize all layers of data (Vitals and K-lines) to determine the highest probability trade.

    Here is the full data bundle:
    {analysis_data}

    Return the JSON object now.
    """
    
    try:
        response = gemini_model.generate_content(
            prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=TRADE_DECISION_SCHEMA,
                temperature=0,
            )
        )
        
        # The response text is guaranteed to be valid JSON due to structured output
        decision = json.loads(response.text)
        return decision
    except Exception as e:
        add_log(f"❌ Failed to parse Gemini response: {e}")
        return None

# --- 6. Trading Execution and Management Functions ---

def get_current_position():
    """Checks current position via Binance API."""
    try:
        positions = binance_client.futures_position_information()
        for p in positions:
            if p['symbol'] == config_template.SYMBOL:
                pos_amt = float(p['positionAmt'])
                if pos_amt != 0:
                    return {
                        "side": "LONG" if pos_amt > 0 else "SHORT",
                        "quantity": abs(pos_amt),
                        "entry_price": float(p['entryPrice'])
                    }
        return {"side": None, "quantity": 0, "entry_price": 0}
    except BinanceAPIException as e:
        add_log(f"Error checking position: {e}")
        return {"side": None, "quantity": 0, "entry_price": 0}

def check_for_trigger(df):
    """Dynamic check for market anomalies based on alert level."""
    if df is None or len(df) < 21: return False, "Data insufficient"

    vol_mult = config_template.NORMAL_VOLUME_SPIKE if bot_status['bot_state'] == BotState.SEARCHING.name else config_template.ALERT_VOLUME_SPIKE
    rng_mult = config_template.NORMAL_VOLATILITY_SPIKE if bot_status['bot_state'] == BotState.SEARCHING.name else config_template.ALERT_VOLATILITY_SPIKE

    last_20 = df.iloc[-21:-1]
    current = df.iloc[-1]
    
    avg_volume = last_20['volume'].mean()
    avg_range = (last_20['high'] - last_20['low']).mean()
    
    if current['volume'] > avg_volume * vol_mult:
        return True, f"Volume Spike ({bot_status['bot_state']})"
    if (current['high'] - current['low']) > avg_range * rng_mult:
        return True, f"Volatility Spike ({bot_status['bot_state']})"
    
    return False, "Market stable"

def calculate_position_size(entry_price, stop_loss_price, risk_percent):
    """Calculates position size based on capital, risk, and stop distance."""
    try:
        actual_risk_percent = min(risk_percent / 100, config_template.MAX_RISK_PER_TRADE)
        amount_to_risk_usdt = config_template.TOTAL_CAPITAL_USDT * actual_risk_percent
        
        price_delta_per_unit = abs(entry_price - stop_loss_price)
        if price_delta_per_unit == 0: return 0

        position_size_units = amount_to_risk_usdt / price_delta_per_unit
        
        sol_precision = 3 
        return round(position_size_units, sol_precision)
    except Exception as e:
        add_log(f"❌ Error calculating position size: {e}")
        return 0

def execute_trade(decision):
    """Executes the trade using dynamic parameters and Post-Only limit order."""
    global current_bot_state, pending_order_id, pending_order_start_time, last_gemini_decision

    if not decision or decision.get('decision') == 'HOLD':
        add_log("Gemini decision is 'HOLD', returning to SEARCHING.")
        current_bot_state = BotState.SEARCHING
        return

    side = decision['decision']
    entry_price = decision['entry_price']
    stop_loss_price = decision['stop_loss']
    
    # 1. Set dynamic leverage
    try:
        binance_client.futures_change_leverage(symbol=config_template.SYMBOL, leverage=decision['leverage'])
        add_log(f"⚙️ Leverage set to {decision['leverage']}x.")
    except BinanceAPIException as e:
        add_log(f"❌ Failed to set leverage: {e}")
        current_bot_state = BotState.SEARCHING
        return

    # 2. Calculate Position Size
    position_size = calculate_position_size(entry_price, stop_loss_price, decision['risk_per_trade_percent'])
    if position_size <= 0:
        add_log("Calculated position size is zero or negative, trade cancelled.")
        current_bot_state = BotState.SEARCHING
        return

    add_log(f"💎 Decision: {side} | Size: {position_size} | Risk: {decision['risk_per_trade_percent']}%")

    try:
        # 3. Place Post-Only Limit Order (Maker)
        order = binance_client.futures_create_order(
            symbol=config_template.SYMBOL,
            side='BUY' if side == 'LONG' else 'SELL',
            type='LIMIT',
            timeInForce='GTC',
            price=f"{entry_price:.4f}",
            quantity=position_size,
            newOrderRespType='RESULT',
            isMakers=True 
        )
        add_log(f"✅ Post-Only Limit Order placed @ {entry_price:.4f} (ID: {order['orderId']})")
        pending_order_id = order['orderId']
        pending_order_start_time = time.time()
        current_bot_state = BotState.ORDER_PENDING

    except BinanceAPIException as e:
        if e.code == -2021: # Post-Only order would be taker
            add_log("⚠️ Order failed: Price would execute immediately (Taker). Returning to SEARCHING.")
        else:
            add_log(f"❌ Binance order failed: {e}")
        current_bot_state = BotState.SEARCHING

def manage_position(position):
    """Manages position, deploying Trailing Stop Loss near TP."""
    global last_gemini_decision
    
    # Check if initial TP is set and if TSL has already been deployed (TP=0 means TSL is active)
    if last_gemini_decision.get('take_profit', 0) == 0:
        return # TSL already active

    side = position['side']
    entry_price = position['entry_price']
    initial_tp = last_gemini_decision.get('take_profit')
    callback_rate = last_gemini_decision.get('trailing_stop_callback', 1.0) # Default 1.0%
    
    # Get current price
    try:
        ticker = binance_client.futures_mark_price(symbol=config_template.SYMBOL)
        current_price = float(ticker['markPrice'])
    except BinanceAPIException:
        return

    # Activation logic: Activate TSL when price reaches 75% of the distance to TP
    profit_target_distance = abs(initial_tp - entry_price)
    current_profit_distance = abs(current_price - entry_price)

    if current_profit_distance >= (profit_target_distance * 0.75):
        add_log(f"📈 Profit reached 75% of TP distance. Deploying Trailing Stop ({callback_rate}%)...")
        try:
            # 1. Cancel initial fixed TP/SL orders
            binance_client.futures_cancel_all_open_orders(symbol=config_template.SYMBOL)

            # 2. Set Trailing Stop Loss
            sl_side = 'SELL' if side == 'LONG' else 'BUY'
            binance_client.futures_create_order(
                symbol=config_template.SYMBOL,
                side=sl_side,
                type='TRAILING_STOP_MARKET',
                callbackRate=callback_rate,
                quantity=position['quantity'],
                reduceOnly=True
            )
            add_log(f"✅ Trailing Stop deployed with {callback_rate}% callback. Running profit.")
            
            # Mark TP as deployed to prevent re-deployment
            last_gemini_decision['take_profit'] = 0 
            
        except BinanceAPIException as e:
            add_log(f"❌ Failed to deploy Trailing Stop: {e}")

# --- 7. Main Loop ---

def main_loop():
    global current_bot_state, last_gemini_decision, pending_order_id, pending_order_start_time, bot_status

    add_log("🤖 Trading Bot Main Loop Started.")

    while True:
        try:
            # Update status for dashboard
            bot_status['bot_state'] = current_bot_state.name
            
            # --- PNL and Position Update ---
            pos = get_current_position()
            if pos['side']:
                bot_status['position'] = pos
                ticker = binance_client.futures_mark_price(symbol=config_template.SYMBOL)
                current_price = float(ticker['markPrice'])
                pnl_usd = (current_price - pos['entry_price']) * pos['quantity'] if pos['side'] == 'LONG' else (pos['entry_price'] - current_price) * pos['quantity']
                
                # Use the leverage from the last decision, default to 20 if missing
                leverage = last_gemini_decision.get('leverage', 20)
                initial_margin = (pos['entry_price'] * pos['quantity']) / leverage
                pnl_perc = (pnl_usd / initial_margin) * 100 if initial_margin > 0 else 0
                bot_status['pnl'] = {"usd": pnl_usd, "percentage": pnl_perc}
            else:
                bot_status['position'] = {"side": None, "quantity": 0, "entry_price": 0}
                bot_status['pnl'] = {"usd": None, "percentage": 0}
            
            # --- State Machine Logic ---

            if current_bot_state == BotState.SEARCHING:
                trigger_df = get_klines_robust(config_template.SYMBOL, '1m', limit=21)
                is_triggered, reason = check_for_trigger(trigger_df)
                if is_triggered:
                    add_log(f"🎯 Market Trigger Activated! Reason: {reason}")
                    current_bot_state = BotState.ANALYZING
                time.sleep(60)

            elif current_bot_state == BotState.ANALYZING:
                analysis_bundle = run_heavy_analysis()
                if analysis_bundle:
                    trade_decision = get_gemini_decision(analysis_bundle)
                    if trade_decision:
                        bot_status['last_gemini_decision'] = trade_decision
                        execute_trade(trade_decision) # Changes state to PENDING or SEARCHING
                else:
                    add_log("Analysis failed, returning to SEARCHING.")
                    current_bot_state = BotState.SEARCHING
            
            elif current_bot_state == BotState.ORDER_PENDING:
                # Check order status and timeout
                # ... (Implementation details omitted for brevity) ...
                time.sleep(10)

            elif current_bot_state == BotState.IN_POSITION:
                pos = get_current_position()
                if not pos['side']:
                    add_log("💰 Position closed. Entering 15-minute COOL_DOWN.")
                    current_bot_state = BotState.COOL_DOWN
                    continue
                
                manage_position(pos)
                time.sleep(15)

            elif current_bot_state == BotState.COOL_DOWN:
                add_log("Cooling down to avoid re-entry after volatility...")
                time.sleep(900)
                add_log("Cool down finished, returning to SEARCHING.")
                current_bot_state = BotState.SEARCHING

        except KeyboardInterrupt:
            add_log("\nProgram manually stopped. Cancelling all open orders...")
            try:
                binance_client.futures_cancel_all_open_orders(symbol=config_template.SYMBOL)
                add_log("All orders cancelled.")
            except BinanceAPIException as e:
                add_log(f"Failed to cancel orders: {e}")
            break
        except Exception as e:
            add_log(f"Severe error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main_loop()