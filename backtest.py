# backtester.py (v1.0 - AI Strategy Validation Engine)

# --- Standard Library Imports ---
import os
import time
import json
import traceback

# --- Third-Party Library Imports ---
import pandas as pd
import pandas_ta as ta
from binance.client import Client
import quantstats as qs

# --- AI and Local Imports ---
import google.generativeai as genai
from google.api_core import exceptions
import config 
# We will import the logic from the trading bot
from trading_bot import (
    GEMINI_SYSTEM_PROMPT_TEXT_BASED,
    parse_decision_block,
    parse_context_block,
)

# --- Backtester Configuration ---
INITIAL_CAPITAL = 1000.00  # Starting balance in USDT
COMMISSION_PCT = 0.04       # Commission fee per trade (0.04% is standard for Binance Futures)
API_RETRY_DELAY = 10        # Seconds to wait after a Gemini API error

# --- Gemini API Initialization ---
current_key_index = 0
try:
    genai.configure(api_key=config.GEMINI_API_KEYS[0])
    print(f"✅ Gemini AI model configured. Using {len(config.GEMINI_API_KEYS)} API keys for rotation.")
except Exception as e:
    print(f"❌ Gemini AI initialization failed: {e}")
    exit()

# --- 1. Historical Data Retrieval ---

def get_historical_data(symbol, start_str, end_str, interval=Client.KLINE_INTERVAL_5MINUTE):
    """Downloads historical K-line data from Binance and saves it to a CSV."""
    filepath = f"hist_data_{symbol}_{start_str.replace(' ', '')}_{end_str.replace(' ', '')}.csv"
    if os.path.exists(filepath):
        print(f"💾 Loading historical data from local file: {filepath}")
        return pd.read_csv(filepath, index_col='open_time', parse_dates=True)

    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    print(f"Downloading historical data for {symbol} from {start_str} to {end_str}...")
    
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
        
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time')
    
    df_to_save = df[['open', 'high', 'low', 'close', 'volume']]
    df_to_save.to_csv(filepath)
    print(f"✅ Downloaded and saved {len(df_to_save)} candles to {filepath}")
    return df_to_save

# --- 2. The Simulated Exchange Environment ---

class SimulatedExchange:
    def __init__(self, initial_balance, commission_pct):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_pct = commission_pct / 100
        self.position = {"side": None, "quantity": 0, "entry_price": 0, "stop_loss": 0, "take_profit": 0}
        self.trades = []
        self.balance_history = []
        self.timestamps = []
        print(f"🏦 Simulated Exchange initialized with {self.balance:.2f} USDT.")

    def record_equity(self, timestamp, current_price):
        current_equity = self.get_total_equity(current_price)
        self.balance_history.append(current_equity)
        self.timestamps.append(timestamp)

    def get_total_equity(self, current_price=0):
        if self.position['side']:
            pnl = 0
            if self.position['side'] == 'LONG': pnl = (current_price - self.position['entry_price']) * self.position['quantity']
            elif self.position['side'] == 'SHORT': pnl = (self.position['entry_price'] - current_price) * self.position['quantity']
            return self.balance + pnl
        return self.balance
    
    def get_current_position(self):
        return self.position.copy()

    def calculate_position_size(self, current_equity, risk_percent, entry_price, stop_loss_price):
        amount_to_risk = current_equity * (risk_percent / 100)
        price_delta = abs(entry_price - stop_loss_price)
        if price_delta == 0: return 0
        return amount_to_risk / price_delta

    def open_position(self, side, price, decision):
        if self.position['side']: return False
        
        leverage = decision.get('leverage', 20)
        risk_percent = decision.get('risk_percent', 5)
        stop_loss = decision.get('stop_loss')
        take_profit = decision.get('take_profit')

        if not all([stop_loss, take_profit]):
            print("⚠️ Decision missing SL or TP, cannot open position.")
            return False

        quantity = self.calculate_position_size(self.get_total_equity(price), risk_percent, price, stop_loss)
        required_margin = (quantity * price) / leverage
        fee = (quantity * price) * self.commission_pct

        if self.balance < required_margin + fee:
            print(f"🚨 MARGIN INSUFFICIENT (Simulated): Required {required_margin:.2f} but only have {self.balance:.2f}")
            return False

        self.position = {"side": side, "quantity": quantity, "entry_price": price, "stop_loss": stop_loss, "take_profit": take_profit}
        self.trades.append(f"OPEN {side} | Qty: {quantity:.3f} @ {price:.4f}")
        print(f"✅ OPEN {side} | Size: {quantity:.3f} @ {price:.4f} | SL: {stop_loss} TP: {take_profit}")
        return True

    def close_position(self, price, reason="AI Decision"):
        if not self.position['side']: return

        pnl = 0
        if self.position['side'] == 'LONG': pnl = (price - self.position['entry_price']) * self.position['quantity']
        elif self.position['side'] == 'SHORT': pnl = (self.position['entry_price'] - price) * self.position['quantity']
            
        opening_fee = (self.position['quantity'] * self.position['entry_price']) * self.commission_pct
        closing_fee = (self.position['quantity'] * price) * self.commission_pct
        net_pnl = pnl - opening_fee - closing_fee
        self.balance += net_pnl
        
        trade_log = f"CLOSE {self.position['side']} @ {price:.4f} | PNL: {net_pnl:.2f} | Reason: {reason} | Balance: {self.balance:.2f}"
        self.trades.append(trade_log)
        print(f"❌ {trade_log}")
        
        self.position = {"side": None, "quantity": 0, "entry_price": 0, "stop_loss": 0, "take_profit": 0}

    def generate_report(self):
        if not self.balance_history:
            print("⚠️ No trading activity to report.")
            return

        returns = pd.Series(self.balance_history, index=pd.to_datetime(self.timestamps))
        returns = returns.pct_change().fillna(0)
        
        print("\n--- 📊 Generating QuantStats Report... ---")
        try:
            qs.reports.html(returns, output='report.html', title='AI Bot Backtest Performance')
            print("✅ Report saved as 'report.html'. Open this file in your browser.")
        except Exception as e:
            print(f"❌ An error occurred while generating the report: {e}")

# --- 3. Adapted AI and Analysis Logic for Backtesting ---

def create_backtest_analysis_bundle(df_slice):
    """
    Creates a simplified analysis bundle using only historical data.
    No live vitals, news, or order book data is used.
    """
    current_price = df_slice.iloc[-1]['close']
    report = f"### 0. Current Market Price (Anchor)\n- **Current Price:** {current_price:.4f} USDT\n\n"
    report += "### 1. Multi-Timeframe K-line Depth Analysis\n"
    
    # We use the same dataframe for all "timeframes" in this simplified version
    for tf_length in [20, 50, 200]:
        df_slice[f'EMA_{tf_length}'] = ta.ema(df_slice['close'], length=tf_length)
    df_slice['RSI_14'] = ta.rsi(df_slice['close'], length=14)
    
    latest = df_slice.iloc[-1]
    
    report += f"--- Analysis Report (Input Timeframe) ---\n"
    report += f"Close Price: {latest['close']:.4f}\n"
    report += f"EMA 20/50: {latest.get('EMA_20', 0):.4f} / {latest.get('EMA_50', 0):.4f}\n"
    report += f"RSI_14: {latest.get('RSI_14', 0):.2f}\n"
    return report

def get_backtest_gemini_decision(analysis_data, position_data):
    """Calls the Gemini API. This is the bottleneck of the backtest."""
    global current_key_index
    prompt = f"""{GEMINI_SYSTEM_PROMPT_TEXT_BASED}
**--- IMPORTANT: THIS IS A BACKTESTING SIMULATION ---**
You are operating on historical data. News sentiment and live market vitals are NOT available. Base your decision solely on the provided price action and technical indicators.
**----------------------------------------------------**
**--- CURRENT DATA FOR ANALYSIS ---**
**1. Current Position Status:** {position_data}
**2. Holographic Market Analysis:** {analysis_data}
Provide your full response."""
    for i in range(len(config.GEMINI_API_KEYS)):
        try:
            key = config.GEMINI_API_KEYS[current_key_index]
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            current_key_index = (current_key_index + 1) % len(config.GEMINI_API_KEYS)
            response = model.generate_content(prompt, generation_config={"temperature": 0.2})
            decision = parse_decision_block(response.text)
            if decision and 'action' in decision:
                return decision
        except exceptions.ResourceExhausted:
            print(f"Key {current_key_index-1} rate-limited. Switching...")
            time.sleep(API_RETRY_DELAY)
        except Exception as e:
            print(f"❌ Gemini API Error: {e}. Retrying in {API_RETRY_DELAY}s")
            time.sleep(API_RETRY_DELAY)
    return None

# --- 4. The Main Backtesting Loop ---

def run_backtest(historical_data, sim_exchange):
    print("\n--- ▶️ Starting Backtest Simulation Loop... ---")
    for i in range(200, len(historical_data)): # Start from 200 to ensure indicators are well-established
        
        current_data_slice = historical_data.iloc[0:i]
        current_candle = current_data_slice.iloc[-1]
        current_price = current_candle['close']
        
        # --- A. Check for SL/TP Triggers ---
        pos = sim_exchange.get_current_position()
        if pos['side'] == 'LONG':
            if current_candle['low'] <= pos['stop_loss']:
                sim_exchange.close_position(pos['stop_loss'], "Stop Loss Triggered")
                sim_exchange.record_equity(current_candle.name, pos['stop_loss'])
                continue
            elif current_candle['high'] >= pos['take_profit']:
                sim_exchange.close_position(pos['take_profit'], "Take Profit Triggered")
                sim_exchange.record_equity(current_candle.name, pos['take_profit'])
                continue
        elif pos['side'] == 'SHORT':
            if current_candle['high'] >= pos['stop_loss']:
                sim_exchange.close_position(pos['stop_loss'], "Stop Loss Triggered")
                sim_exchange.record_equity(current_candle.name, pos['stop_loss'])
                continue
            elif current_candle['low'] <= pos['take_profit']:
                sim_exchange.close_position(pos['take_profit'], "Take Profit Triggered")
                sim_exchange.record_equity(current_candle.name, pos['take_profit'])
                continue
        
        # --- B. Call AI for a Decision (e.g., once every 3 candles to save API calls) ---
        if i % 3 == 0: 
            print(f"\n--- Candle {i} | Time: {current_candle.name} | Price: {current_price:.4f} ---")
            analysis_bundle = create_backtest_analysis_bundle(current_data_slice.copy()) # Use copy to avoid SettingWithCopyWarning
            position_status_report = f"Position: {pos['side'] or 'FLAT'}, Size: {pos['quantity']:.3f}, Entry: {pos['entry_price']:.4f}"
            
            decision = get_backtest_gemini_decision(analysis_bundle, position_status_report)
            
            if decision:
                action = decision.get('action')
                if action == 'OPEN_POSITION' and not pos['side']:
                    sim_exchange.open_position(decision.get('decision'), current_price, decision)
                elif action == 'CLOSE_POSITION' and pos['side']:
                    sim_exchange.close_position(current_price)

        # --- C. Record Equity at the end of every candle ---
        sim_exchange.record_equity(current_candle.name, current_price)


# --- 5. Main Execution Block ---

if __name__ == '__main__':
    # Define backtest parameters here
    backtest_symbol = "SOLUSDT"
    start_date = "1 May, 2024"
    end_date = "7 May, 2024" # A shorter period is better for initial tests due to API call speed
    
    # 1. Download data
    hist_data = get_historical_data(
        symbol=backtest_symbol,
        start_str=start_date,
        end_str=end_date,
        interval=Client.KLINE_INTERVAL_5MINUTE
    )
    
    # 2. Initialize simulation
    sim_exchange = SimulatedExchange(
        initial_balance=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT
    )
    
    # 3. Run the backtest
    run_backtest(hist_data, sim_exchange)
    
    # 4. Print summary and generate report
    print("\n\n--- 📈 BACKTEST COMPLETE 📈 ---")
    final_equity = sim_exchange.get_total_equity()
    print(f"Initial Balance: {sim_exchange.initial_balance:.2f} USDT")
    print(f"Final Balance:   {final_equity:.2f} USDT")
    pnl_pct = ((final_equity - sim_exchange.initial_balance) / sim_exchange.initial_balance) * 100
    print(f"Net PNL:         {final_equity - sim_exchange.initial_balance:.2f} USDT ({pnl_pct:.2f}%)")
    if sim_exchange.trades:
        print(f"Total Trades:    {int(len(sim_exchange.trades) / 2)}")
    
    sim_exchange.generate_report()