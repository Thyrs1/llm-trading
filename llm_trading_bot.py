
# LLM_Trading_Bot.py (V19.0 - Multi-Process Multi-Symbol Engine)

# --- Standard Library Imports ---
import asyncio
import json
import time
import traceback
from collections import deque
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import signal
import sys

# --- Project Modules ---
import config
# Note: Renaming module imports to avoid conflicts if needed, but not strictly necessary here
from execution_manager import ExchangeManager 
from ai_processor import (
    init_ai_client, # Now imported directly
    init_finbert_process, get_sentiment_score_async, get_news_from_rss,
    get_ai_decision, analyze_freqtrade_data, process_klines,
    summarize_and_learn
)
import pandas as pd
import pandas_ta as ta

# --- Global State & Utilities (Local to each Process) ---
# These are local to the current process and manage the state for the *single* symbol it handles.
LOG_DEQUE = deque(maxlen=200)
HISTORICAL_DATA: Dict[str, pd.DataFrame] = {} # Keyed by symbol, but only one symbol per process
SYMBOL_BOT_STATE: Dict[str, Any] = {} # State for the single symbol this process manages

# --- Utility Functions (Adapted for Per-Symbol Context) ---
def add_log(message: str, symbol: str = "SYSTEM"):
    """Adds a message to the log deque and prints it."""
    # Prefix log with PID to easily distinguish processes
    pid = os.getpid()
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PID:{pid}] [{symbol}] {message}"
    LOG_DEQUE.appendleft(log_entry)
    print(log_entry)

def load_market_context(symbol: str) -> Dict:
    """Loads the last market context for the specific symbol from a file."""
    context_file = f'market_context_{symbol.replace("/", "_")}.json'
    try:
        with open(context_file, 'r') as f:
            context = json.load(f)
            add_log(f"🧠 Market context loaded from last session for {symbol}.", symbol)
            return context
    except Exception:
        add_log(f"No previous market context found for {symbol}. Starting fresh.", symbol)
        return {}

def save_market_context(symbol: str, context: Dict):
    """Saves the current market context for the symbol to a file."""
    context_file = f'market_context_{symbol.replace("/", "_")}.json'
    try:
        with open(context_file, 'w') as f:
            json.dump(context, f, indent=4)
    except Exception as e:
        add_log(f"❌ Error saving market context for {symbol}: {e}", symbol)

def calculate_position_size(equity, available_margin, risk_percent, entry_price, stop_loss_price, leverage, symbol, exchange_manager: ExchangeManager):
    """Calculates position size based on risk and caps it by available margin."""
    if any(p is None or p <= 0 for p in [equity, available_margin, risk_percent, entry_price, stop_loss_price, leverage]):
        return 0.0
    try:
        amount_to_risk = equity * min(risk_percent / 100, config.MAX_RISK_PER_TRADE)
        price_delta = abs(entry_price - stop_loss_price)
        risk_based_size = amount_to_risk / price_delta if price_delta > 0 else 0
        
        # Max position value based on margin and leverage
        # Using a safety margin of 95% of available margin * leverage
        max_position_value = available_margin * leverage * 0.95
        margin_based_size = max_position_value / entry_price
        
        final_size = min(risk_based_size, margin_based_size)
        
        if final_size < risk_based_size and risk_based_size > 0:
            add_log(f"⚠️ Risk-based size ({risk_based_size:.4f}) unaffordable. Capped by margin to {final_size:.4f}.", symbol)
        
        # Format size to exchange's precision
        return float(exchange_manager.client.amount_to_precision(symbol, final_size))
    except Exception as e:
        add_log(f"❌ Error in calculate_position_size for {symbol}: {e}", symbol)
        return 0.0

# --- Advanced Trigger & State Management (Keep this class for logic isolation) ---
class DynamicTriggerManager:
    """Manages complex, multi-scenario triggers set by the AI."""
    def __init__(self, symbol: str):
        self.triggers: List[Dict] = []
        self.timeout = None
        self._last_check = 0
        self.symbol = symbol

    def set_triggers(self, decision: Dict):
        """Sets new triggers from an AI decision."""
        if decision.get('action') == 'WAIT' and 'triggers' in decision and isinstance(decision['triggers'], list):
            self.triggers = decision['triggers']
            self.timeout = time.time() + decision.get('trigger_timeout', 1800)
            add_log(f"💤 Setting {len(self.triggers)} new triggers. Timeout in {decision.get('trigger_timeout', 1800)/60:.1f} mins.", self.symbol)
        else:
            self.triggers = []
            self.timeout = None
            # Reset last check time to ensure next interval check is on time
            self._last_check = time.time()

    def check_triggers(self, df_5m: pd.DataFrame) -> Tuple[bool, str]:
        """Checks if any active trigger is met. Returns (is_triggered, reason)."""
        if not self.triggers:
            # Check for default monitoring interval if no explicit WAIT triggers
            if time.time() > self._last_check + config.DEFAULT_MONITORING_INTERVAL:
                self._last_check = time.time()
                return True, "Scheduled Analysis"
            return False, ""

        # Check for trigger timeout
        if self.timeout and time.time() > self.timeout:
            add_log("⏳ Trigger timeout reached.", self.symbol)
            return True, "Timeout"

        # Check explicit triggers
        for trigger in self.triggers:
            try:
                if self._is_condition_met(trigger, df_5m):
                    reason = f"Trigger Met: {trigger.get('label', 'Unnamed')}"
                    add_log(f"🎯 {reason}", self.symbol)
                    return True, reason
            except Exception as e:
                add_log(f"⚠️ Error checking trigger '{trigger.get('label')}': {e}", self.symbol)
        return False, ""
    
    def _is_condition_met(self, trigger: Dict, df_5m: pd.DataFrame) -> bool:
        """Evaluates a single trigger object."""
        trigger_type = trigger.get('type')
        level = trigger.get('level')
        direction = trigger.get('direction')
        
        # Ensure we have enough data and valid parameters
        if df_5m.empty or level is None or direction is None:
            return False

        if trigger_type == 'PRICE_CROSS':
            latest_candle = df_5m.iloc[-1]
            if direction == 'ABOVE' and latest_candle['high'] >= level: return True
            if direction == 'BELOW' and latest_candle['low'] <= level: return True
        
        elif trigger_type == 'RSI_CROSS':
            # Calculate RSI on the fly for the latest candle
            current_rsi = ta.rsi(df_5m['close'], length=14).iloc[-1]
            if direction == 'ABOVE' and current_rsi >= level: return True
            if direction == 'BELOW' and current_rsi <= level: return True
        
        elif trigger_type == 'ADX_VALUE':
            # Calculate ADX on the fly for the latest candle
            adx_data = ta.adx(df_5m['high'], df_5m['low'], df_5m['close'], length=14)
            current_adx = adx_data['ADX_14'].iloc[-1]
            if direction == 'ABOVE' and current_adx >= level: return True
            if direction == 'BELOW' and current_adx <= level: return True

        return False

# --- Core Asynchronous Logic (One Symbol per Instance) ---
async def symbol_main_loop(symbol: str, exchange: ExchangeManager, executor: ProcessPoolExecutor):
    """The main trading loop for a single symbol, executed in its own process."""
    
    # Initialize AI Client for this process's event loop
    ai_client = await init_ai_client()
    if not ai_client:
        add_log(f"❌ Aborting {symbol} bot due to AI client failure.", symbol)
        return
    
    # Global state for this process
    initial_context = load_market_context(symbol)
    SYMBOL_BOT_STATE[symbol] = {
        "trigger_manager": DynamicTriggerManager(symbol), 
        "market_context": initial_context, 
        "last_decision": {},
        "trade_state": {"is_trailing_active": False, "current_stop_loss": None}, 
        "was_in_position": False
    }
    symbol_state = SYMBOL_BOT_STATE[symbol]

    # Initial data hydration (now fetch for a single symbol)
    add_log(f"💧 Hydrating historical data for {symbol}...", symbol)
    klines_map = await exchange.fetch_historical_klines(symbol, config.TIMEFRAME, 500)
    klines = klines_map.get(symbol, [])
    if klines:
        HISTORICAL_DATA[symbol] = process_klines(klines)
        add_log(f"✅ Data hydration complete for {symbol}.", symbol)
    else:
        add_log(f"⚠️ Initial data fetch failed for {symbol}. Cannot start trading.", symbol)
        return

    while True:
        try:
            # 1. Fetch Account Vitals and Positions
            vitals = await exchange.get_account_vitals()
            all_positions = await exchange.fetch_positions([symbol]) # Fetch only for this symbol
            pos = all_positions[0] if all_positions else {"side": None}
            is_in_position = pos.get('side') is not None
            current_price = await exchange.get_current_mark_price(symbol)
            
            # 2. Position Close & Learning Cycle
            if symbol_state['was_in_position'] and not is_in_position:
                add_log(f"📉 Position closed for {symbol}. Triggering learning cycle.", symbol)
                # Fetch recent trades to calculate PNL for learning
                trades = await exchange.fetch_account_trade_list(symbol, limit=5)
                # Assuming the most recent trade is the closing trade (complex on Binance, simple approximation)
                if trades:
                    # In a real system, realizedPNL would be better tracked, but we use the first trade's PNL here as a heuristic
                    pnl = float(trades[0].get('info', {}).get('realizedPnl', 0)) if trades[0].get('info', {}).get('realizedPnl') else 0.0
                    entry_reason = symbol_state['last_decision'].get('reasoning', 'N/A')
                    trade_summary = f"Outcome: {'WIN' if pnl > 0 else 'LOSS'}, PNL: {pnl:.2f} USDT. Entry Reason: {entry_reason}"
                    await summarize_and_learn(trade_summary, symbol)
                
                # Reset trade state
                symbol_state["trade_state"] = {"is_trailing_active": False, "current_stop_loss": None}
            symbol_state['was_in_position'] = is_in_position
            
            # 3. Trailing Stop / Breakeven Update (High-Frequency Check)
            if is_in_position and symbol_state['last_decision'].get('action') == 'OPEN_POSITION':
                activation_price = symbol_state['last_decision'].get('trailing_activation_price')
                
                # Check for Breakeven activation
                if activation_price and not symbol_state['trade_state']['is_trailing_active']:
                    if (pos['side'] == 'LONG' and current_price >= activation_price) or \
                       (pos['side'] == 'SHORT' and current_price <= activation_price):
                        add_log(f"🔒 Breakeven Trigger for {symbol}. Moving SL to entry: {pos['entry_price']}.", symbol)
                        res = await exchange.modify_protective_orders(symbol, pos['side'], pos['quantity'], new_sl=pos['entry_price'])
                        if res['status'] == 'success':
                            symbol_state['trade_state']['is_trailing_active'] = True
                            symbol_state['trade_state']['current_stop_loss'] = pos['entry_price']
                
                # TODO: Implement full Trailing Stop Logic here if 'is_trailing_active' is True and current_price allows a tighter SL

            # 4. Data Refresh
            # Fetch the latest 1-min/5-min candle to keep data up-to-date
            df_update_map = await exchange.fetch_historical_klines(symbol, config.TIMEFRAME, limit=1)
            df_5m_update = process_klines(df_update_map.get(symbol, []))
            if not df_5m_update.empty:
                # Append new data and keep the frame size capped
                HISTORICAL_DATA[symbol] = pd.concat([HISTORICAL_DATA[symbol], df_5m_update]).drop_duplicates()
                if len(HISTORICAL_DATA[symbol]) > 500: 
                    HISTORICAL_DATA[symbol] = HISTORICAL_DATA[symbol].iloc[-500:]

            # 5. Check Triggers for AI Analysis
            if symbol not in HISTORICAL_DATA or HISTORICAL_DATA[symbol].empty: 
                await asyncio.sleep(config.API_RETRY_DELAY) # Wait and re-try in case of data failure
                continue
            
            is_triggered, reason = symbol_state['trigger_manager'].check_triggers(HISTORICAL_DATA[symbol])

            # 6. AI Decision Making
            if is_triggered:
                add_log(f"Analysis for {symbol} triggered by: {reason}", symbol)
                
                analysis_bundle = analyze_freqtrade_data(HISTORICAL_DATA[symbol], current_price)
                
                base_symbol = symbol.split('/')[0]
                news_text = get_news_from_rss(base_symbol)
                # Run sentiment analysis in the ProcessPoolExecutor
                sentiment_score = await get_sentiment_score_async(executor, news_text)
                
                pos_report = f"Side: {pos.get('side', 'FLAT')}, Entry: {pos.get('entry_price', 0):.4f}"
                context_summary = json.dumps(symbol_state['market_context'])
                
                decision, new_context = await get_ai_decision(
                    analysis_bundle, pos_report, context_summary, vitals['total_equity'], sentiment_score
                )
                
                # 7. Execute AI Action
                if new_context: symbol_state['market_context'] = new_context
                
                if decision and decision.get('action'):
                    symbol_state['last_decision'] = decision
                    action = decision['action']
                    
                    if action == 'OPEN_POSITION' and not is_in_position:
                        # With multi-process, we simplify this: only one position per symbol process.
                        # Global max positions check is removed.
                        qty = calculate_position_size(vitals['total_equity'], vitals['available_margin'], decision.get('risk_percent', 0), decision.get('entry_price', 0), decision.get('stop_loss', 0), decision.get('leverage', 0), symbol, exchange)
                        
                        if qty > 0:
                            res = await exchange.place_limit_order(symbol, decision['decision'], qty, decision['entry_price'])
                            if res['status'] == 'success':
                                add_log(f"✅ Entry order placed for {symbol}. Setting SL/TP.", symbol)
                                await asyncio.sleep(2) # Give a small pause
                                # The quantity here should be the *filled* quantity, but since we are placing a limit order, we assume the intended size for the protective order.
                                await exchange.modify_protective_orders(symbol, decision['decision'], qty, decision.get('stop_loss'), decision.get('take_profit'))
                                symbol_state['trade_state']['current_stop_loss'] = decision.get('stop_loss')
                            else:
                                add_log(f"❌ Failed to place entry order: {res['message']}", symbol)
                        else:
                            add_log(f"⚠️ Calculated position size is zero. Skipping trade.", symbol)
                            
                    elif action == 'CLOSE_POSITION' and is_in_position:
                        await exchange.close_position_market(symbol, pos)
                        add_log(f"✅ Position market closed for {symbol}.", symbol)
                        
                    elif action == 'MODIFY_POSITION' and is_in_position:
                        res = await exchange.modify_protective_orders(
                            symbol, pos['side'], pos['quantity'], 
                            decision.get('new_stop_loss'), decision.get('new_take_profit')
                        )
                        if res['status'] == 'success':
                            add_log(f"✅ Protective orders modified for {symbol}.", symbol)
                            if decision.get('new_stop_loss'):
                                symbol_state['trade_state']['current_stop_loss'] = decision.get('new_stop_loss')
                        
                    # Always set new triggers/wait state after an analysis is performed
                    symbol_state['trigger_manager'].set_triggers(decision)
            
            # 8. Save Context and Wait
            save_market_context(symbol, symbol_state['market_context'])
            await asyncio.sleep(config.FAST_CHECK_INTERVAL) 

        except asyncio.CancelledError:
            add_log(f"🛑 {symbol} loop cancelled.", symbol)
            break
        except Exception as e:
            add_log(f"💥 CRITICAL ERROR in {symbol} engine loop: {traceback.format_exc()}", symbol)
            await asyncio.sleep(config.API_RETRY_DELAY) # Wait before retrying loop

# --- Multi-Process Management ---
def run_symbol_bot(symbol: str):
    """Entry point for each trading process."""
    add_log(f"Starting dedicated bot process for {symbol}...", symbol)
    exchange = ExchangeManager()
    
    # We must start the ProcessPoolExecutor for FinBERT inside the process
    # This keeps the sentiment analysis isolated and non-blocking.
    with ProcessPoolExecutor(max_workers=2, initializer=init_finbert_process) as executor:
        try:
            # Run the main asynchronous loop inside this process
            asyncio.run(exchange.load_markets())
            asyncio.run(symbol_main_loop(symbol, exchange, executor))
        except KeyboardInterrupt:
            add_log(f"Keyboard Interrupt in {symbol} process.", symbol)
        except Exception as e:
            add_log(f"Unhandled error in {symbol} process: {e}", symbol)
        finally:
            add_log(f"Shutting down {symbol} process.", symbol)
            asyncio.run(exchange.close_client())

def main_process_manager():
    """Manages the creation and cleanup of all symbol processes."""
    processes = []
    
    # Simple signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Received shutdown signal. Terminating all symbol processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
        sys.exit(0)

    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("--- Multi-Process Bot Manager Started ---")
    
    # Create and start a new process for each symbol
    for symbol in config.SYMBOLS_TO_TRADE:
        p = mp.Process(target=run_symbol_bot, args=(symbol,))
        processes.append(p)
        p.start()
        print(f"✅ Launched process {p.pid} for {symbol}.")
        time.sleep(1) # Stagger start-up to avoid API key rate limits on initialization

    # Main process waits for all child processes to finish (or for a signal)
    try:
        while True:
            # Check status of child processes
            all_dead = True
            for p in processes:
                if p.is_alive():
                    all_dead = False
                elif p.exitcode is not None and p.exitcode != 0:
                    print(f"⚠️ Process for a symbol died unexpectedly with exit code {p.exitcode}. Restarting is an option here, but for now, we just log it.")
            
            if all_dead and processes:
                print("All child processes have terminated. Exiting manager.")
                break

            time.sleep(5)
    except Exception:
        # Pass to allow signal handler to catch the KeyboardInterrupt/Termination
        pass
    finally:
        # Final cleanup for any lingering processes
        signal_handler(None, None)

if __name__ == '__main__':
    # Set the start method for multiprocessing (important on some OS/platforms)
    try:
        mp.set_start_method('spawn', force=True)
    except ValueError:
        pass # Ignore if already set or not necessary

    main_process_manager()