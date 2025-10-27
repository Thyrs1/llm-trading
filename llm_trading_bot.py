# LLM_Trading_Bot.py (V22.3 - Final Synchronous Polling with Detailed Dashboard Data)

# --- Standard Library Imports ---
import time
import json
import traceback
from collections import deque
from typing import Dict, Any, List, Tuple
import threading 
import sys
import os

# --- Project Modules ---
import config
from execution_manager import ExchangeManager 
from ai_processor import (
    init_ai_client,
    init_finbert_analyzer, get_sentiment_score_sync, get_news_from_rss,
    get_ai_decision_sync, analyze_freqtrade_data, process_klines,
    summarize_and_learn_sync
)
from database_manager import setup_database, log_system_message, log_trade, update_bot_state, update_account_vitals
import pandas as pd
import pandas_ta as ta

# --- Global State & Utilities (Single Process) ---
HISTORICAL_DATA: Dict[str, pd.DataFrame] = {} 
BOT_STATE: Dict[str, Any] = {} 
AI_CLIENT = None 

# --- Utility Functions ---
def add_log(message: str, symbol: str = "SYSTEM"):
    log_system_message(message, symbol)

def load_market_context(symbol: str) -> Dict:
    context_file = f'market_context_{symbol.replace("/", "_")}.json'
    try:
        with open(context_file, 'r') as f:
            context = json.load(f)
            log_system_message(f"🧠 Market context loaded from last session file for {symbol}.", symbol)
            return context
    except Exception:
        log_system_message(f"No previous market context file found for {symbol}. Starting fresh.", symbol)
        return {}

# ########################################################################### #
# ################## START OF MODIFIED SECTION ############################## #
# ########################################################################### #
# CORE CHANGE: Reverted to the risk-based position sizing function.
def calculate_position_size(equity, available_margin, risk_percent, entry_price, stop_loss_price, leverage, symbol, exchange_manager: ExchangeManager):
    if any(p is None or p <= 0 for p in [equity, available_margin, risk_percent, entry_price, stop_loss_price, leverage]): return 0.0
    try:
        # The amount to risk is the LESSER of the AI's proposal and the global hard cap.
        risk_fraction = min(risk_percent / 100.0, config.MAX_RISK_PER_TRADE)
        amount_to_risk = equity * risk_fraction
        
        price_delta = abs(entry_price - stop_loss_price)
        risk_based_size = amount_to_risk / price_delta if price_delta > 0 else 0
        
        # Check if this size is affordable with the available margin and leverage
        max_position_value = available_margin * leverage * 0.98 # 2% safety buffer
        margin_based_size = max_position_value / entry_price
        
        final_size = min(risk_based_size, margin_based_size)
        
        if final_size < risk_based_size and risk_based_size > 0:
            add_log(f"⚠️ Risk-based size ({risk_based_size:.4f}) unaffordable. Capped by margin to {final_size:.4f}.", symbol)
        
        add_log(f"RISK SIZING: Equity=${equity:.2f}, AI Risk={risk_percent}%, Final Risk={risk_fraction*100:.2f}%, Size={final_size:.4f}", symbol)
        return float(exchange_manager.client.amount_to_precision(symbol, final_size))
    except Exception as e:
        add_log(f"❌ Error in calculate_position_size for {symbol}: {e}", symbol)
        return 0.0
# ########################################################################### #
# ################### END OF MODIFIED SECTION ############################### #
# ########################################################################### #

# --- DynamicTriggerManager ---
class DynamicTriggerManager:
    # ... (This class remains unchanged) ...
    def __init__(self, symbol: str):
        self.triggers: List[Dict] = []
        self.timeout = None
        self._last_check = 0
        self.symbol = symbol
    def set_triggers(self, decision: Dict):
        if decision.get('action') == 'WAIT' and 'triggers' in decision and isinstance(decision['triggers'], list):
            self.triggers = decision.get('triggers', [])
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
        """Checks if a specific trigger condition is met based on the latest data."""
        trigger_type = trigger.get('type')
        if df_5m.empty: return False

        # --- Pre-calculation for indicators ---
        # This ensures we calculate indicators only once if multiple triggers need them.
        latest_candle = df_5m.iloc[-1]
        
        try:
            # --- TYPE 1: PRICE_CROSS ---
            if trigger_type == 'PRICE_CROSS':
                level = float(trigger.get('level', 0))
                direction = trigger.get('direction')
                if direction == 'ABOVE' and latest_candle['high'] >= level: return True
                if direction == 'BELOW' and latest_candle['low'] <= level: return True

            # --- TYPE 2: RSI_CROSS ---
            elif trigger_type == 'RSI_CROSS':
                level = float(trigger.get('level', 0))
                direction = trigger.get('direction')
                current_rsi = ta.rsi(df_5m['close'], length=14).iloc[-1]
                if direction == 'ABOVE' and current_rsi >= level: return True
                if direction == 'BELOW' and current_rsi <= level: return True

            # --- NEW TYPE 3: EMA_CROSS ---
            elif trigger_type == 'EMA_CROSS':
                fast_period = int(trigger.get('fast', 20))
                slow_period = int(trigger.get('slow', 50))
                direction = trigger.get('direction')
                
                ema_fast = ta.ema(df_5m['close'], length=fast_period)
                ema_slow = ta.ema(df_5m['close'], length=slow_period)
                
                # Golden Cross: Fast EMA crosses ABOVE Slow EMA
                if direction == 'GOLDEN' and ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] >= ema_slow.iloc[-1]:
                    return True
                # Death Cross: Fast EMA crosses BELOW Slow EMA
                if direction == 'DEATH' and ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] <= ema_slow.iloc[-1]:
                    return True

            # --- NEW TYPE 4: PRICE_EMA_DISTANCE ---
            elif trigger_type == 'PRICE_EMA_DISTANCE':
                period = int(trigger.get('period', 20))
                target_pct = float(trigger.get('percent', 0))
                condition = trigger.get('condition') # 'ABOVE' or 'BELOW'
                
                ema = ta.ema(df_5m['close'], length=period).iloc[-1]
                distance_pct = ((latest_candle['close'] - ema) / ema) * 100
                
                if condition == 'BELOW' and distance_pct <= target_pct: return True
                if condition == 'ABOVE' and distance_pct >= target_pct: return True

            # --- NEW TYPE 5: BBAND_WIDTH ---
            elif trigger_type == 'BBAND_WIDTH':
                period = int(trigger.get('period', 20))
                target_pct = float(trigger.get('percent', 0))
                condition = trigger.get('condition') # 'BELOW' (squeeze) or 'ABOVE' (expansion)
                
                bbands = ta.bbands(df_5m['close'], length=period)
                # Calculate Bollinger Band Width Percentage
                bb_width_pct = ((bbands[f'BBU_{period}_2.0'] - bbands[f'BBL_{period}_2.0']) / bbands[f'BBM_{period}_2.0'] * 100).iloc[-1]

                if condition == 'BELOW' and bb_width_pct <= target_pct: return True
                if condition == 'ABOVE' and bb_width_pct >= target_pct: return True

            # --- NEW TYPE 6: MACD_HIST_SIGN ---
            elif trigger_type == 'MACD_HIST_SIGN':
                condition = trigger.get('condition') # 'POSITIVE' or 'NEGATIVE'
                macd = ta.macd(df_5m['close'])
                hist = macd['MACDh_12_26_9']
                
                # Crosses to positive
                if condition == 'POSITIVE' and hist.iloc[-2] <= 0 and hist.iloc[-1] > 0: return True
                # Crosses to negative
                if condition == 'NEGATIVE' and hist.iloc[-2] >= 0 and hist.iloc[-1] < 0: return True

        except Exception as e:
            add_log(f"⚠️ Error evaluating trigger '{trigger.get('label')}': {e}", self.symbol)
            return False
            
        return False

# --- Main Bot Execution Loop ---
def main():
    global AI_CLIENT
    
    setup_database()
    
    try:
        from web_dashboard import start_dashboard
        dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
        dashboard_thread.start()
    except ImportError:
        add_log("❌ CRITICAL: Could not import web_dashboard. Dashboard will not run.", "SYSTEM")
    
    exchange = ExchangeManager()
    exchange.load_markets()
    
    AI_CLIENT = init_ai_client()
    if not AI_CLIENT: return

    init_finbert_analyzer()

    for symbol in config.SYMBOLS_TO_TRADE:
        BOT_STATE[symbol] = {
            "trigger_manager": DynamicTriggerManager(symbol), 
            # ################## START OF MODIFIED SECTION ##############################
            # 2. USE A DEQUE TO STORE A HISTORY OF CONTEXTS (e.g., last 12 analyses = 1 hour)
            "market_context_history": deque(maxlen=12), 
            # ################### END OF MODIFIED SECTION ###############################
            "last_decision": {}, 
            "trade_state": {"current_stop_loss": None, "trailing_distance_pct": None, "pending_order_id": None}, 
            "was_in_position": False, "current_position": {"side": None},
            "last_ai_response": "No analysis yet.", "chain_of_thought": "No thought process recorded yet.",
            "last_sentiment_score": 0.0, "last_known_price": 0.0
        }

    add_log("💧 Hydrating full historical data for all symbols sequentially...")
    for symbol in config.SYMBOLS_TO_TRADE:
        klines = exchange.fetch_full_historical_data(symbol, config.TIMEFRAME, days_of_data=60)
        if klines:
            HISTORICAL_DATA[symbol] = process_klines(klines)
        else:
            add_log(f"⚠️ Could not hydrate data for {symbol}. It will be skipped.", "SYSTEM")
    add_log("✅ Data hydration complete.")
    add_log(f"🚀 Bot engine is live. Polling every {config.FAST_CHECK_INTERVAL}s.")
    
    while True:
        start_time = time.time()
        try:
            vitals = exchange.get_account_vitals()
            update_account_vitals(vitals)
            
            all_positions = exchange.fetch_positions(config.SYMBOLS_TO_TRADE)
            open_positions_map = {pos['symbol']: pos for pos in all_positions}
            
            latest_klines_map = exchange.fetch_historical_klines(config.SYMBOLS_TO_TRADE, config.TIMEFRAME, limit=2)
            
            for symbol in config.SYMBOLS_TO_TRADE:
                if symbol not in BOT_STATE or symbol not in HISTORICAL_DATA: continue
                
                symbol_state = BOT_STATE[symbol]
                pos = open_positions_map.get(symbol, {"side": None})
                is_in_position = pos.get('side') is not None
                symbol_state['current_position'] = pos

                df_5m_update = process_klines(latest_klines_map.get(symbol, []))
                if not df_5m_update.empty:
                    combined_df = pd.concat([HISTORICAL_DATA.get(symbol, pd.DataFrame()), df_5m_update])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.sort_index(inplace=True)
                    HISTORICAL_DATA[symbol] = combined_df.tail(18000)

                if HISTORICAL_DATA[symbol].empty: continue
                
                current_price = exchange.get_current_mark_price(symbol)
                symbol_state['last_known_price'] = current_price

                if symbol_state['was_in_position'] and not is_in_position:
                    add_log(f"📉 Position closed for {symbol}. Triggering learning cycle.", symbol)
                    try:
                        exchange.client.cancel_all_orders(symbol)
                        add_log(f"✅ Canceled all open orders for {symbol} post-closure.", symbol)
                    except Exception as e:
                        add_log(f"⚠️ Could not cancel open orders for {symbol}: {e}", symbol)
                    trades = exchange.fetch_account_trade_list(symbol, limit=5)
                    
                    if trades:
                        last_trade = trades
                        pnl = float(last_trade.get('info', {}).get('realizedPnl', 0))
                        entry_reason = symbol_state['last_decision'].get('reasoning', 'N/A')
                        entry_price = symbol_state['current_position'].get('entry_price', 0)
                        quantity = symbol_state['current_position'].get('quantity', 0)
                        exit_price = current_price 
                        pnl_pct = (pnl / (entry_price * quantity)) if entry_price and quantity else 0.0
                        log_trade(symbol, symbol_state['current_position'].get('side', 'UNKNOWN'), entry_price, exit_price, quantity, pnl, pnl_pct, entry_reason)
                        trade_summary = f"Outcome: {'WIN' if pnl > 0 else 'LOSS'}, PNL: {pnl:.2f} USDT. Entry Reason: {entry_reason}"
                        summarize_and_learn_sync(trade_summary, symbol)
                        
                    # CORE CHANGE: Reset the full trade state on close
                    symbol_state["trade_state"] = {"current_stop_loss": None, "trailing_distance_pct": None}
                symbol_state['was_in_position'] = is_in_position
                
                if is_in_position and symbol_state['trade_state'].get('pending_order_id'):
                    add_log(f"✅ Limit order {symbol_state['trade_state']['pending_order_id']} filled. Position is now active.", symbol)
                    
                    # Safely set SL/TP because we are now in a confirmed position
                    decision = symbol_state['last_decision']
                    qty = pos.get('quantity', decision.get('quantity', 0)) # Use actual position quantity
                    
                    add_log(f"Setting SL/TP for active position. SL: {decision.get('stop_loss')}, TP: {decision.get('take_profit')}", symbol)
                    exchange.modify_protective_orders(symbol, decision['decision'], qty, decision.get('stop_loss'), decision.get('take_profit'))
                    
                    symbol_state['trade_state']['current_stop_loss'] = decision.get('stop_loss')
                    symbol_state['trade_state']['trailing_distance_pct'] = decision.get('trailing_distance_pct')
                    symbol_state['trade_state']['pending_order_id'] = None # Clear the pending state

                # Update position info after potential SL/TP placement
                symbol_state['current_position'] = pos

                is_triggered, reason = symbol_state['trigger_manager'].check_triggers(HISTORICAL_DATA[symbol])

                if is_triggered:
                    add_log(f"Analysis for {symbol} triggered by: {reason}", symbol)
                    
                    analysis_bundle = analyze_freqtrade_data(HISTORICAL_DATA[symbol], current_price)
                    
                    news_text = get_news_from_rss(symbol.split('/')[0])
                    sentiment_score = get_sentiment_score_sync(news_text)
                    symbol_state['last_sentiment_score'] = sentiment_score
                    
                    pos_report = f"Side: {pos.get('side', 'FLAT')}, Entry: {pos.get('entry_price', 0):.4f}"
                    # ################## START OF MODIFIED SECTION ##############################
                    # 3. FORMAT THE HISTORY OF CONTEXTS INTO A STRING FOR THE AI
                    context_history_list = list(symbol_state['market_context_history'])
                    context_summary_string = "No historical analysis available yet."
                    if context_history_list:
                        formatted_contexts = []
                        # Iterate in reverse to show the most recent analysis first
                        for i, context in enumerate(reversed(context_history_list)):
                            # The timestamp is already in the context dict, added by parse_context_block
                            ts = context.get('last_full_analysis_timestamp', 'N/A').split('T')[1].split('.')[0]
                            formatted_contexts.append(f"--- Analysis @ {ts} UTC ({i*5} mins ago) ---\n" + json.dumps(context, indent=2))
                        context_summary_string = "\n\n".join(formatted_contexts)
                    # ################### END OF MODIFIED SECTION ###############################
                    
                    decision, new_context, raw_response, chain_of_thought = get_ai_decision_sync(
                        analysis_bundle, pos_report, context_summary_string, vitals['total_equity'], sentiment_score
                    )
                    symbol_state['last_ai_response'] = raw_response
                    symbol_state['chain_of_thought'] = chain_of_thought # Store the thought process
                    
                    # Log the thought process for immediate debugging
                    add_log(f"AI Chain of Thought:\n--- START ---\n{chain_of_thought}\n--- END ---", symbol)
                    
                    if new_context: 
                        symbol_state['market_context_history'].append(new_context)
                    
                    if decision and decision.get('action'):
                        symbol_state['last_decision'] = decision
                        action = decision['action']
                        
                        if action == 'OPEN_POSITION' and not is_in_position:
                            # ########################################################################### #
                            # ################## START OF MODIFIED SECTION ############################## #
                            # ########################################################################### #
                            # CORE CHANGE: Re-instated the check for max concurrent positions.
                            if len(open_positions_map) >= config.MAX_CONCURRENT_POSITIONS:
                                add_log(f"🚨 Max positions reached ({config.MAX_CONCURRENT_POSITIONS}). Skipping open for {symbol}.", symbol)
                            else:
                                # CORE CHANGE: Using the risk-based sizing function again.
                                qty = calculate_position_size(
                                    vitals['total_equity'], 
                                    vitals['available_margin'], 
                                    decision.get('risk_percent', 0), 
                                    decision.get('entry_price', 0), 
                                    decision.get('stop_loss', 0), 
                                    decision.get('leverage', 0), 
                                    symbol, 
                                    exchange
                                )
                                decision['quantity'] = qty
                                
                    if decision and decision.get('action'):
                        symbol_state['last_decision'] = decision
                        action = decision['action']
                        
                        # STATE 2: AI wants to open a position, and we are flat with no pending orders.
                        if action == 'OPEN_POSITION' and not is_in_position and not symbol_state['trade_state'].get('pending_order_id'):
                            
                            # --- Pre-flight Check 1: Max Concurrent Positions ---
                            if len(open_positions_map) >= config.MAX_CONCURRENT_POSITIONS:
                                add_log(f"🚨 Max positions reached ({config.MAX_CONCURRENT_POSITIONS}). Skipping open for {symbol}.", symbol)
                                continue # Skip to the next symbol in the main loop

                            # --- Pre-flight Check 2: Calculate Position Size ---
                            qty = calculate_position_size(
                                vitals['total_equity'], 
                                vitals['available_margin'], 
                                decision.get('risk_percent', 0), 
                                decision.get('entry_price', 0), 
                                decision.get('stop_loss', 0), 
                                decision.get('leverage', 0), 
                                symbol, 
                                exchange
                            )
                            decision['quantity'] = qty
                            
                            if qty > 0:
                                # --- Pre-flight Check 3: Minimum Notional Value ---
                                ai_entry_price = decision.get('entry_price', 0)
                                calculated_notional = qty * ai_entry_price
                                
                                min_notional = 0
                                try:
                                    min_notional = exchange.client.markets[symbol]['limits']['cost']['min']
                                except (KeyError, TypeError):
                                    add_log(f"⚠️ Could not find minimum notional value for {symbol} in market data. Using fallback.", symbol)
                                    min_notional = 5 # Binance's typical minimum is 5 USDT for most pairs

                                if calculated_notional < min_notional:
                                    add_log(f"⚠️ Order size is too small. Calculated Notional: ${calculated_notional:.2f}, Minimum Required: ${min_notional:.2f}. Skipping trade.", symbol)
                                
                                else:
                                    # --- Execution Step 1: Set Leverage ---
                                    exchange.set_leverage_for_symbol(symbol, decision.get('leverage', 20))
                                    time.sleep(1) # Small delay after setting leverage is good practice

                                    # --- Execution Step 2: Place the Limit Order ---
                                    if ai_entry_price > 0:
                                        add_log(f"Placing standard Limit order at AI's specified price: {ai_entry_price}", symbol)
                                        res = exchange.place_limit_order(symbol, decision['decision'], qty, ai_entry_price)

                                        if res['status'] == 'success' and res.get('order'):
                                            order_id = res['order']['id']
                                            add_log(f"✅ Limit order placed successfully. Order ID: {order_id}. Waiting for fill.", symbol)
                                            # --- Execution Step 3: Set the pending state. DO NOT set SL/TP yet. ---
                                            symbol_state['trade_state']['pending_order_id'] = order_id
                                        else:
                                            add_log(f"❌ Failed to place limit order: {res.get('message', 'Unknown error')}", symbol)
                                    else:
                                        add_log(f"⚠️ AI provided an invalid entry price of zero. Skipping trade.", symbol)
                            else:
                                add_log(f"⚠️ Calculated position size is zero. Skipping trade.", symbol)
                        
                        elif action == 'CLOSE_POSITION':
                            # If we are in a position, close it.
                            if is_in_position:
                                add_log(f"AI decision to CLOSE. Closing active position for {symbol}.", symbol)
                                exchange.close_position_market(symbol, pos)
                            # If we have a pending order, cancel it.
                            elif symbol_state['trade_state'].get('pending_order_id'):
                                add_log(f"AI decision to CLOSE. Cancelling pending order {symbol_state['trade_state']['pending_order_id']} for {symbol}.", symbol)
                                exchange.cancel_order(symbol_state['trade_state']['pending_order_id'], symbol)
                                symbol_state['trade_state']['pending_order_id'] = None

                        elif action == 'MODIFY_POSITION' and is_in_position:
                            add_log(f"AI decision to MODIFY. Updating SL/TP for {symbol}.", symbol)
                            res = exchange.modify_protective_orders(symbol, pos['side'], pos['quantity'], decision.get('new_stop_loss'), decision.get('new_take_profit'))
                            if res['status'] == 'success' and decision.get('new_stop_loss'):
                                symbol_state['trade_state']['current_stop_loss'] = decision.get('new_stop_loss')
                                
                        elif action == 'WAIT':
                            # This also handles the "HOLD" case when in a position
                            add_log(f"AI decision: WAIT/HOLD. Reason: {decision.get('reasoning', 'N/A')}", symbol)
                            symbol_state['trigger_manager'].set_triggers(decision)
                
                update_bot_state(symbol, is_in_position, pos, {
                    # Pass the list representation of the deque for JSON serialization
                    'market_context': list(symbol_state['market_context_history']),
                    'active_triggers': symbol_state['trigger_manager'].triggers,
                    'last_ai_response': symbol_state['last_ai_response'],
                    'chain_of_thought': symbol_state['chain_of_thought'], # Pass to DB
                    'last_sentiment_score': symbol_state['last_sentiment_score'],
                    'last_known_price': symbol_state['last_known_price']
                })
            
        except Exception:
            add_log(f"💥 CRITICAL ERROR in main loop: {traceback.format_exc()}", "SYSTEM")
            time.sleep(config.API_RETRY_DELAY) 
        
        time_spent = time.time() - start_time
        sleep_time = max(0, config.FAST_CHECK_INTERVAL - time_spent)
        time.sleep(sleep_time)

    exchange.close_client()
    add_log("🛑 Bot shutting down.")

if __name__ == '__main__':
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
        
    print("--- Single-Process Synchronous Bot Manager Started ---")
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 User interrupted. Shutting down.")
    except Exception as e:
        print(f"\n🛑 CRITICAL UNHANDLED EXCEPTION: {e}")