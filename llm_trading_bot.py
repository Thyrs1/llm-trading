# LLM_Trading_Bot.py (V22.0 - Final Synchronous Polling with DB and WebUI)

# --- Standard Library Imports ---
import time
import json
import traceback
from collections import deque
from typing import Dict, Any, List, Tuple
import threading # For running the web dashboard
import sys

# --- Project Modules ---
import config
from execution_manager import ExchangeManager 
from ai_processor import (
    init_ai_client,
    init_finbert_analyzer, get_sentiment_score_sync, get_news_from_rss,
    get_ai_decision_sync, analyze_freqtrade_data, process_klines,
    summarize_and_learn_sync
)
from database_manager import setup_database, log_system_message, log_trade, update_bot_state # NEW DB IMPORTS
from web_dashboard import start_dashboard # NEW WEBUI IMPORT
import pandas as pd
import pandas_ta as ta

# --- Global State & Utilities (Single Process) ---
# LOG_DEQUE is now redundant but kept for local logging before DB is ready
LOG_DEQUE = deque(maxlen=200) 
HISTORICAL_DATA: Dict[str, pd.DataFrame] = {} 
BOT_STATE: Dict[str, Any] = {} 
AI_CLIENT = None 

# --- Utility Functions (Modified to use DB logging) ---
def add_log(message: str, symbol: str = "SYSTEM"):
    """Logs a message to the database and prints it."""
    # Use the DB function for logging
    log_system_message(message, symbol)

def load_market_context(symbol: str) -> Dict:
    """Loads the last market context for the specific symbol from the DB/file."""
    # In a real system, this would load from the DB state table. 
    # For simplicity, we keep the file-based loading as a fallback/initialization.
    context_file = f'market_context_{symbol.replace("/", "_")}.json'
    try:
        with open(context_file, 'r') as f:
            context = json.load(f)
            log_system_message(f"🧠 Market context loaded from last session for {symbol}.", symbol)
            return context
    except Exception:
        log_system_message(f"No previous market context found for {symbol}. Starting fresh.", symbol)
        return {}

def save_market_context():
    """Saves the current market context for all symbols to the database."""
    for symbol, state in BOT_STATE.items():
        # Prepare position data for DB update
        pos_data = {
            'side': state['current_position'].get('side'),
            'entry_price': state['current_position'].get('entry_price'),
            'quantity': state['current_position'].get('quantity'),
            'unrealized_pnl': state['current_position'].get('unrealized_pnl', 0.0)
        }
        is_in_position = pos_data['side'] is not None
        update_bot_state(symbol, is_in_position, pos_data, state['market_context'])

def calculate_position_size(equity, available_margin, risk_percent, entry_price, stop_loss_price, leverage, symbol, exchange_manager: ExchangeManager):
    # ... (Unchanged, uses exchange_manager.client)
    if any(p is None or p <= 0 for p in [equity, available_margin, risk_percent, entry_price, stop_loss_price, leverage]):
        return 0.0
    try:
        amount_to_risk = equity * min(risk_percent / 100, config.MAX_RISK_PER_TRADE)
        price_delta = abs(entry_price - stop_loss_price)
        risk_based_size = amount_to_risk / price_delta if price_delta > 0 else 0
        
        max_position_value = available_margin * leverage * 0.95
        margin_based_size = max_position_value / entry_price
        
        final_size = min(risk_based_size, margin_based_size)
        
        if final_size < risk_based_size and risk_based_size > 0:
            add_log(f"⚠️ Risk-based size ({risk_based_size:.4f}) unaffordable. Capped by margin to {final_size:.4f}.", symbol)
        
        return float(exchange_manager.client.amount_to_precision(symbol, final_size))
    except Exception as e:
        add_log(f"❌ Error in calculate_position_size for {symbol}: {e}", symbol)
        return 0.0

# --- DynamicTriggerManager (Unchanged) ---
class DynamicTriggerManager:
    # ... (Unchanged)
    def __init__(self, symbol: str):
        self.triggers: List[Dict] = []
        self.timeout = None
        self._last_check = 0
        self.symbol = symbol

    def set_triggers(self, decision: Dict):
        if decision.get('action') == 'WAIT' and 'triggers' in decision and isinstance(decision['triggers'], list):
            self.triggers = decision['triggers']
            self.timeout = time.time() + decision.get('trigger_timeout', 1800)
            add_log(f"💤 Setting {len(self.triggers)} new triggers. Timeout in {decision.get('trigger_timeout', 1800)/60:.1f} mins.", self.symbol)
        else:
            self.triggers = []
            self.timeout = None
            self._last_check = time.time()

    def check_triggers(self, df_5m: pd.DataFrame) -> Tuple[bool, str]:
        if not self.triggers:
            if time.time() > self._last_check + config.DEFAULT_MONITORING_INTERVAL:
                self._last_check = time.time()
                return True, "Scheduled Analysis"
            return False, ""

        if self.timeout and time.time() > self.timeout:
            add_log("⏳ Trigger timeout reached.", self.symbol)
            return True, "Timeout"

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
        trigger_type = trigger.get('type')
        level = trigger.get('level')
        direction = trigger.get('direction')
        
        if df_5m.empty or level is None or direction is None:
            return False

        if trigger_type == 'PRICE_CROSS':
            latest_candle = df_5m.iloc[-1]
            if direction == 'ABOVE' and latest_candle['high'] >= level: return True
            if direction == 'BELOW' and latest_candle['low'] <= level: return True
        
        elif trigger_type == 'RSI_CROSS':
            current_rsi = ta.rsi(df_5m['close'], length=14).iloc[-1]
            if direction == 'ABOVE' and current_rsi >= level: return True
            if direction == 'BELOW' and current_rsi <= level: return True
        
        elif trigger_type == 'ADX_VALUE':
            adx_data = ta.adx(df_5m['high'], df_5m['low'], df_5m['close'], length=14)
            current_adx = adx_data['ADX_14'].iloc[-1]
            if direction == 'ABOVE' and current_adx >= level: return True
            if direction == 'BELOW' and current_adx <= level: return True

        return False

# --- Main Bot Execution Loop (Synchronous) ---
def main():
    global AI_CLIENT
    
    # 1. Database Setup
    setup_database()
    
    # 2. Start Web Dashboard in a separate thread
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    
    # 3. Initialize Exchange and AI
    exchange = ExchangeManager()
    exchange.load_markets()
    
    AI_CLIENT = init_ai_client()
    if not AI_CLIENT: return

    init_finbert_analyzer()

    # 4. Initialize state for all symbols
    for symbol in config.SYMBOLS_TO_TRADE:
        BOT_STATE[symbol] = {
            "trigger_manager": DynamicTriggerManager(symbol), 
            "market_context": load_market_context(symbol), 
            "last_decision": {},
            "trade_state": {"is_trailing_active": False, "current_stop_loss": None}, 
            "was_in_position": False,
            "current_position": {"side": None} # Placeholder for current position data
        }

    add_log("💧 Hydrating historical data for all symbols sequentially...")
    initial_klines_map = exchange.fetch_historical_klines(
        config.SYMBOLS_TO_TRADE, 
        config.TIMEFRAME, 
        500
    )
    for symbol, klines in initial_klines_map.items():
        if klines: 
            HISTORICAL_DATA[symbol] = process_klines(klines)
        else:
            add_log(f"⚠️ Could not hydrate data for {symbol}. It will be skipped until data is available.", "SYSTEM")
    add_log("✅ Data hydration complete.")

    add_log(f"🚀 Bot engine is live for symbols: {', '.join(config.SYMBOLS_TO_TRADE)}. Polling every {config.FAST_CHECK_INTERVAL}s.")
    
    while True:
        start_time = time.time()
        try:
            vitals = exchange.get_account_vitals()
            all_positions = exchange.fetch_positions(config.SYMBOLS_TO_TRADE)
            open_positions_map = {pos['symbol']: pos for pos in all_positions}
            
            latest_klines_map = exchange.fetch_historical_klines(config.SYMBOLS_TO_TRADE, config.TIMEFRAME, limit=2)
            
            for symbol in config.SYMBOLS_TO_TRADE:
                if symbol not in BOT_STATE: continue
                
                symbol_state = BOT_STATE[symbol]
                pos = open_positions_map.get(symbol, {"side": None})
                is_in_position = pos.get('side') is not None
                
                # Update current position data in state for DB logging
                symbol_state['current_position'] = pos

                # 1. Update Historical Data
                df_5m_update = process_klines(latest_klines_map.get(symbol, []))
                if not df_5m_update.empty:
                    HISTORICAL_DATA[symbol] = df_5m_update.combine_first(HISTORICAL_DATA.get(symbol, pd.DataFrame()))
                    if len(HISTORICAL_DATA[symbol]) > 500: 
                        HISTORICAL_DATA[symbol] = HISTORICAL_DATA[symbol].iloc[-500:].copy()

                if symbol not in HISTORICAL_DATA or HISTORICAL_DATA[symbol].empty: continue
                
                current_price = exchange.get_current_mark_price(symbol)

                # 2. Position Close & Learning Cycle
                if symbol_state['was_in_position'] and not is_in_position:
                    add_log(f"📉 Position closed for {symbol}. Triggering learning cycle.", symbol)
                    trades = exchange.fetch_account_trade_list(symbol, limit=5)
                    
                    # Log trade to DB
                    if trades:
                        pnl = float(trades[0].get('info', {}).get('realizedPnl', 0)) if trades[0].get('info', {}).get('realizedPnl') else 0.0
                        entry_reason = symbol_state['last_decision'].get('reasoning', 'N/A')
                        
                        # Approximate exit price and PNL% for logging (requires more complex logic for true values)
                        entry_price = symbol_state['current_position'].get('entry_price', 0)
                        quantity = symbol_state['current_position'].get('quantity', 0)
                        exit_price = current_price # Approximation
                        pnl_pct = (pnl / (entry_price * quantity)) if entry_price * quantity else 0.0
                        
                        log_trade(symbol, symbol_state['current_position'].get('side', 'UNKNOWN'), entry_price, exit_price, quantity, pnl, pnl_pct, entry_reason)
                        
                        trade_summary = f"Outcome: {'WIN' if pnl > 0 else 'LOSS'}, PNL: {pnl:.2f} USDT. Entry Reason: {entry_reason}"
                        summarize_and_learn_sync(trade_summary, symbol)
                        
                    symbol_state["trade_state"] = {"is_trailing_active": False, "current_stop_loss": None}
                symbol_state['was_in_position'] = is_in_position
                
                # 3. Trailing Stop / Breakeven Update
                if is_in_position and symbol_state['last_decision'].get('action') == 'OPEN_POSITION':
                    activation_price = symbol_state['last_decision'].get('trailing_activation_price')
                    if activation_price and not symbol_state['trade_state']['is_trailing_active']:
                        if (pos['side'] == 'LONG' and current_price >= activation_price) or \
                           (pos['side'] == 'SHORT' and current_price <= activation_price):
                            add_log(f"🔒 Breakeven Trigger for {symbol}. Moving SL to entry: {pos['entry_price']}.", symbol)
                            res = exchange.modify_protective_orders(symbol, pos['side'], pos['quantity'], new_sl=pos['entry_price'])
                            if res['status'] == 'success':
                                symbol_state['trade_state']['is_trailing_active'] = True
                                symbol_state['trade_state']['current_stop_loss'] = pos['entry_price']

                # 4. Check Triggers for AI Analysis
                is_triggered, reason = symbol_state['trigger_manager'].check_triggers(HISTORICAL_DATA[symbol])

                # 5. AI Decision Making
                if is_triggered:
                    add_log(f"Analysis for {symbol} triggered by: {reason}", symbol)
                    
                    analysis_bundle = analyze_freqtrade_data(HISTORICAL_DATA[symbol], current_price)
                    
                    base_symbol = symbol.split('/')[0]
                    news_text = get_news_from_rss(base_symbol)
                    sentiment_score = get_sentiment_score_sync(news_text)
                    
                    pos_report = f"Side: {pos.get('side', 'FLAT')}, Entry: {pos.get('entry_price', 0):.4f}"
                    context_summary = json.dumps(symbol_state['market_context'])
                    
                    decision, new_context = get_ai_decision_sync(
                        analysis_bundle, pos_report, context_summary, vitals['total_equity'], sentiment_score
                    )
                    
                    # 6. Execute AI Action
                    if new_context: symbol_state['market_context'] = new_context
                    
                    if decision and decision.get('action'):
                        symbol_state['last_decision'] = decision
                        action = decision['action']
                        
                        if action == 'OPEN_POSITION' and not is_in_position:
                            if len(open_positions_map) >= config.MAX_CONCURRENT_POSITIONS:
                                add_log(f"🚨 Max positions reached. Skipping open for {symbol}.", symbol)
                            else:
                                qty = calculate_position_size(vitals['total_equity'], vitals['available_margin'], decision.get('risk_percent', 0), decision.get('entry_price', 0), decision.get('stop_loss', 0), decision.get('leverage', 0), symbol, exchange)
                                
                                if qty > 0:
                                    res = exchange.place_limit_order(symbol, decision['decision'], qty, decision['entry_price'])
                                    if res['status'] == 'success':
                                        add_log(f"✅ Entry order placed for {symbol}. Setting SL/TP.", symbol)
                                        time.sleep(2) 
                                        exchange.modify_protective_orders(symbol, decision['decision'], qty, decision.get('stop_loss'), decision.get('take_profit'))
                                        symbol_state['trade_state']['current_stop_loss'] = decision.get('stop_loss')
                                    else:
                                        add_log(f"❌ Failed to place entry order: {res['message']}", symbol)
                                else:
                                    add_log(f"⚠️ Calculated position size is zero. Skipping trade.", symbol)
                                    
                        elif action == 'CLOSE_POSITION' and is_in_position:
                            exchange.close_position_market(symbol, pos)
                            
                        elif action == 'MODIFY_POSITION' and is_in_position:
                            res = exchange.modify_protective_orders(
                                symbol, pos['side'], pos['quantity'], 
                                decision.get('new_stop_loss'), decision.get('new_take_profit')
                            )
                            if res['status'] == 'success' and decision.get('new_stop_loss'):
                                symbol_state['trade_state']['current_stop_loss'] = decision.get('new_stop_loss')
                            
                        symbol_state['trigger_manager'].set_triggers(decision)
            
            # 7. Save all state to DB
            save_market_context()
            
        except Exception:
            add_log(f"💥 CRITICAL ERROR in main loop: {traceback.format_exc()}", "SYSTEM")
            time.sleep(config.API_RETRY_DELAY) 
        
        # Calculate time spent and sleep for the remainder of the polling interval
        time_spent = time.time() - start_time
        sleep_time = max(0, config.FAST_CHECK_INTERVAL - time_spent)
        time.sleep(sleep_time)

    exchange.close_client()
    add_log("🛑 Bot shutting down.")

if __name__ == '__main__':
    print("--- Single-Process Synchronous Bot Manager Started ---")
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 User interrupted. Shutting down.")
    except Exception as e:
        print(f"\n🛑 CRITICAL UNHANDLED EXCEPTION: {e}")