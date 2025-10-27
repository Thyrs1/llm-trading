# llm_trading_bot.py

# --- Standard Library Imports ---
import time
import json
import traceback
from collections import deque
from typing import Dict, Any, List, Tuple
import threading 
import sys
import os
from datetime import datetime, time as dt_time

# --- Project Modules ---
import config
from execution_manager import ExchangeManager 
from ai_processor import (
    init_ai_client,
    init_finbert_analyzer, get_sentiment_score_sync, get_news_from_rss,
    get_ai_decision_sync, analyze_freqtrade_data, process_klines,
    summarize_and_learn_sync, get_market_regime
)
from database_manager import setup_database, log_system_message, log_trade, update_bot_state, update_account_vitals
import pandas as pd
import pandas_ta as ta

# --- Global State & Utilities (Single Process) ---
HISTORICAL_DATA: Dict[str, pd.DataFrame] = {} 
BOT_STATE: Dict[str, Any] = {} 
AI_CLIENT = None 

# --- Global Risk Management State ---
RISK_STATE = {
    "daily_start_equity": 0.0,
    "consecutive_losses": 0,
    "is_trading_halted": False,
    "halt_until_timestamp": 0,
    "last_check_day": -1
}
MAX_DAILY_DRAWDOWN_PCT = 5.0
MAX_CONSECUTIVE_LOSSES = 3
TRADING_PAUSE_DURATION_S = 3600

# --- Utility Functions ---
def add_log(message: str, symbol: str = "SYSTEM"):
    log_system_message(message, symbol)

def calculate_position_size(equity, available_margin, risk_percent, entry_price, stop_loss_price, leverage, symbol, exchange_manager: ExchangeManager):
    if any(p is None or p <= 0 for p in [equity, available_margin, risk_percent, entry_price, stop_loss_price, leverage]): return 0.0
    try:
        risk_fraction = min(risk_percent / 100.0, config.MAX_RISK_PER_TRADE)
        amount_to_risk = equity * risk_fraction
        price_delta = abs(entry_price - stop_loss_price)
        risk_based_size = amount_to_risk / price_delta if price_delta > 0 else 0
        max_position_value = available_margin * leverage * 0.98
        margin_based_size = max_position_value / entry_price
        final_size = min(risk_based_size, margin_based_size)
        if final_size < risk_based_size and risk_based_size > 0:
            add_log(f"⚠️ Risk-based size ({risk_based_size:.4f}) unaffordable. Capped by margin to {final_size:.4f}.", symbol)
        add_log(f"RISK SIZING: Equity=${equity:.2f}, AI Risk={risk_percent}%, Final Risk={risk_fraction*100:.2f}%, Size={final_size:.4f}", symbol)
        return float(exchange_manager.client.amount_to_precision(symbol, final_size))
    except Exception as e:
        add_log(f"❌ Error in calculate_position_size for {symbol}: {e}", symbol)
        return 0.0

# --- DynamicTriggerManager ---
class DynamicTriggerManager:
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
        trigger_type = trigger.get('type')
        if df_5m.empty: return False
        latest_candle = df_5m.iloc[-1]
        try:
            if trigger_type == 'PRICE_CROSS':
                level = float(trigger.get('level', 0))
                direction = trigger.get('direction')
                if direction == 'ABOVE' and latest_candle['high'] >= level: return True
                if direction == 'BELOW' and latest_candle['low'] <= level: return True
            elif trigger_type == 'RSI_CROSS':
                level = float(trigger.get('level', 0))
                direction = trigger.get('direction')
                current_rsi = ta.rsi(df_5m['close'], length=14).iloc[-1]
                if direction == 'ABOVE' and current_rsi >= level: return True
                if direction == 'BELOW' and current_rsi <= level: return True
            elif trigger_type == 'EMA_CROSS':
                fast_period = int(trigger.get('fast', 20))
                slow_period = int(trigger.get('slow', 50))
                direction = trigger.get('direction')
                ema_fast = ta.ema(df_5m['close'], length=fast_period)
                ema_slow = ta.ema(df_5m['close'], length=slow_period)
                if direction == 'GOLDEN' and ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] >= ema_slow.iloc[-1]: return True
                if direction == 'DEATH' and ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] <= ema_slow.iloc[-1]: return True
            elif trigger_type == 'PRICE_EMA_DISTANCE':
                period = int(trigger.get('period', 20))
                target_pct = float(trigger.get('percent', 0))
                condition = trigger.get('condition')
                ema = ta.ema(df_5m['close'], length=period).iloc[-1]
                distance_pct = ((latest_candle['close'] - ema) / ema) * 100
                if condition == 'BELOW' and distance_pct <= target_pct: return True
                if condition == 'ABOVE' and distance_pct >= target_pct: return True
            elif trigger_type == 'BBAND_WIDTH':
                period = int(trigger.get('period', 20))
                target_pct = float(trigger.get('percent', 0))
                condition = trigger.get('condition')
                bbands = ta.bbands(df_5m['close'], length=period)
                bb_width_pct = ((bbands[f'BBU_{period}_2.0'] - bbands[f'BBL_{period}_2.0']) / bbands[f'BBM_{period}_2.0'] * 100).iloc[-1]
                if condition == 'BELOW' and bb_width_pct <= target_pct: return True
                if condition == 'ABOVE' and bb_width_pct >= target_pct: return True
            elif trigger_type == 'MACD_HIST_SIGN':
                condition = trigger.get('condition')
                macd = ta.macd(df_5m['close'])
                hist = macd['MACDh_12_26_9']
                if condition == 'POSITIVE' and hist.iloc[-2] <= 0 and hist.iloc[-1] > 0: return True
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
    if not exchange.connect_and_load_markets():
        add_log("🛑 Bot cannot start: Failed to connect to the exchange and load markets.", "SYSTEM")
        return
        
    AI_CLIENT = init_ai_client()
    if not AI_CLIENT: return
    init_finbert_analyzer()

    for symbol in config.SYMBOLS_TO_TRADE:
        BOT_STATE[symbol] = {
            "trigger_manager": DynamicTriggerManager(symbol), 
            "market_context_history": deque(maxlen=12), 
            "last_decision": {}, 
            "trade_state": {
                "pending_order_id": None, "stop_loss_order_id": None, "take_profit_order_id": None,
                "current_stop_loss": None, "trailing_distance_pct": None
            }, 
            "was_in_position": False, "current_position": {"side": None},
            "last_ai_response": "No analysis yet.", "chain_of_thought": "No thought process recorded yet.",
            "last_sentiment_score": 0.0, "last_known_price": 0.0
        }

    add_log("💧 Hydrating full historical data for all symbols sequentially...")
    for symbol in config.SYMBOLS_TO_TRADE:
        klines = exchange.fetch_full_historical_data(symbol, config.TIMEFRAME, days_of_data=210)
        if klines:
            HISTORICAL_DATA[symbol] = process_klines(klines)
        else:
            add_log(f"⚠️ Could not hydrate data for {symbol}. It will be skipped.", "SYSTEM")
    add_log("✅ Data hydration complete.")
    add_log(f"🚀 Bot engine is live. Polling every {config.FAST_CHECK_INTERVAL}s.")
    
    while True:
        start_time = time.time()
        try:
            # ... (Risk management logic remains the same) ...
            now = datetime.now()
            if now.day != RISK_STATE["last_check_day"]:
                vitals = exchange.get_account_vitals()
                RISK_STATE["daily_start_equity"] = vitals.get('total_equity', 0.0)
                RISK_STATE["is_trading_halted"] = False
                RISK_STATE["last_check_day"] = now.day
                add_log(f"☀️ New trading day. Daily start equity set to: ${RISK_STATE['daily_start_equity']:.2f}", "RISK_MANAGER")
            
            if RISK_STATE["is_trading_halted"] and time.time() < RISK_STATE["halt_until_timestamp"]:
                if now.day != RISK_STATE["last_check_day"]: pass
                else:
                    add_log(f"Trading is paused. Resumes at {datetime.fromtimestamp(RISK_STATE['halt_until_timestamp']).strftime('%H:%M:%S')}.", "RISK_MANAGER",)
                    time.sleep(config.FAST_CHECK_INTERVAL)
                    continue
            elif RISK_STATE["is_trading_halted"] and time.time() >= RISK_STATE["halt_until_timestamp"]:
                RISK_STATE["is_trading_halted"] = False
                RISK_STATE["consecutive_losses"] = 0
                add_log("✅ Trading pause has ended. Resuming operations.", "RISK_MANAGER")

            vitals = exchange.get_account_vitals()
            current_equity = vitals.get('total_equity', 0.0)
            if RISK_STATE["daily_start_equity"] > 0:
                daily_drawdown_limit = RISK_STATE["daily_start_equity"] * (1 - MAX_DAILY_DRAWDOWN_PCT / 100.0)
                if current_equity < daily_drawdown_limit:
                    add_log(f"🚨 DAILY DRAWDOWN LIMIT HIT! Current Equity ${current_equity:.2f} is below limit ${daily_drawdown_limit:.2f}.", "RISK_MANAGER")
                    add_log("--- CLOSING ALL POSITIONS AND HALTING TRADING FOR THE DAY ---", "RISK_MANAGER")
                    all_positions = exchange.fetch_positions(config.SYMBOLS_TO_TRADE)
                    for pos in all_positions:
                        exchange.close_position_market(pos['symbol'], pos)
                    RISK_STATE["is_trading_halted"] = True
                    RISK_STATE["halt_until_timestamp"] = time.time() + (24 * 3600) 
                    continue

            all_positions = exchange.fetch_positions(config.SYMBOLS_TO_TRADE)
            open_positions_map = {pos['symbol']: pos for pos in all_positions}
            latest_klines_map = exchange.fetch_historical_klines(config.SYMBOLS_TO_TRADE, config.TIMEFRAME, limit=2)
            
            for symbol in config.SYMBOLS_TO_TRADE:
                if symbol not in BOT_STATE or symbol not in HISTORICAL_DATA: continue
                
                symbol_state = BOT_STATE[symbol]
                pos = open_positions_map.get(symbol, {"side": None})
                is_in_position = pos.get('side') is not None
                
                if is_in_position and symbol_state['trade_state'].get('pending_order_id'):
                    add_log(f"✅ Limit order {symbol_state['trade_state']['pending_order_id']} filled. Position is now active.", symbol)
                    decision = symbol_state['last_decision']
                    qty = pos.get('quantity', decision.get('quantity', 0))
                    add_log(f"Setting initial SL/TP for active position. SL: {decision.get('stop_loss')}, TP: {decision.get('take_profit')}", symbol)
                    res = exchange.modify_protective_orders(symbol, decision['decision'], qty, decision.get('stop_loss'), decision.get('take_profit'))
                    if res['status'] == 'success':
                        if 'sl' in res['orders']: symbol_state['trade_state']['stop_loss_order_id'] = res['orders']['sl']['id']
                        if 'tp' in res['orders']: symbol_state['trade_state']['take_profit_order_id'] = res['orders']['tp']['id']
                    symbol_state['trade_state']['current_stop_loss'] = decision.get('stop_loss')
                    symbol_state['trade_state']['trailing_distance_pct'] = decision.get('trailing_distance_pct')
                    symbol_state['trade_state']['pending_order_id'] = None

                symbol_state['current_position'] = pos
                df_5m_update = process_klines(latest_klines_map.get(symbol, []))
                if not df_5m_update.empty:
                    combined_df = pd.concat([HISTORICAL_DATA.get(symbol, pd.DataFrame()), df_5m_update])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.sort_index(inplace=True)
                    # ########################################################################### #
                    # ################## START OF MODIFIED SECTION ############################## #
                    # ########################################################################### #
                    # CRITICAL FIX: Ensure we keep enough data for 210 days.
                    # 210 days * 288 (5m candles per day) = 60480. We'll keep a buffer.
                    HISTORICAL_DATA[symbol] = combined_df.tail(61000)
                    # ########################################################################### #
                    # ################### END OF MODIFIED SECTION ############################### #
                    # ########################################################################### #
                
                if HISTORICAL_DATA[symbol].empty: continue
                current_price = exchange.get_current_mark_price(symbol)
                symbol_state['last_known_price'] = current_price

                if symbol_state['was_in_position'] and not is_in_position:
                    add_log(f"📉 Position closed for {symbol}. Clearing all trade state.", symbol)
                    symbol_state['trade_state'] = {"pending_order_id": None, "stop_loss_order_id": None, "take_profit_order_id": None, "current_stop_loss": None, "trailing_distance_pct": None}
                    trades = exchange.fetch_account_trade_list(symbol, limit=5)
                    if trades:
                        last_trade = trades[0]
                        pnl = float(last_trade.get('info', {}).get('realizedPnl', 0))
                        if pnl < 0:
                            RISK_STATE["consecutive_losses"] += 1
                            add_log(f"Consecutive loss count: {RISK_STATE['consecutive_losses']}", "RISK_MANAGER")
                        else:
                            RISK_STATE["consecutive_losses"] = 0
                        if RISK_STATE["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                            RISK_STATE["is_trading_halted"] = True
                            RISK_STATE["halt_until_timestamp"] = time.time() + TRADING_PAUSE_DURATION_S
                            add_log(f"🚨 {MAX_CONSECUTIVE_LOSSES} CONSECUTIVE LOSSES HIT. Pausing all trading for {TRADING_PAUSE_DURATION_S / 60} minutes.", "RISK_MANAGER")
                        entry_reason = symbol_state['last_decision'].get('reasoning', 'N/A')
                        entry_price = symbol_state['current_position'].get('entry_price', 0)
                        quantity = symbol_state['current_position'].get('quantity', 0)
                        exit_price = current_price 
                        pnl_pct = (pnl / (entry_price * quantity)) if entry_price and quantity else 0.0
                        log_trade(symbol, symbol_state['current_position'].get('side', 'UNKNOWN'), entry_price, exit_price, quantity, pnl, pnl_pct, entry_reason)
                        trade_summary = f"Outcome: {'WIN' if pnl > 0 else 'LOSS'}, PNL: {pnl:.2f} USDT. Entry Reason: {entry_reason}"
                        summarize_and_learn_sync(trade_summary, symbol)
                symbol_state['was_in_position'] = is_in_position

                # ... (The rest of the file, including trailing stop, AI analysis, and order placement, remains the same) ...
                if is_in_position:
                    current_sl = symbol_state['trade_state'].get('current_stop_loss')
                    trail_pct = symbol_state['trade_state'].get('trailing_distance_pct')
                    if current_sl and trail_pct:
                        potential_new_sl = 0.0
                        if pos['side'] == 'LONG' and current_price * (1 - (trail_pct / 100.0)) > current_sl:
                            potential_new_sl = current_price * (1 - (trail_pct / 100.0))
                        elif pos['side'] == 'SHORT' and current_price * (1 + (trail_pct / 100.0)) < current_sl:
                            potential_new_sl = current_price * (1 + (trail_pct / 100.0))
                        if potential_new_sl > 0:
                            add_log(f"📈 Trailing SL ({pos['side']}) for {symbol}. New SL: {potential_new_sl:.4f}", symbol)
                            old_sl_id = symbol_state['trade_state'].get('stop_loss_order_id')
                            if old_sl_id: exchange.cancel_order(old_sl_id, symbol)
                            res = exchange.modify_protective_orders(symbol, pos['side'], pos['quantity'], new_sl=potential_new_sl)
                            if res['status'] == 'success' and 'sl' in res['orders']:
                                symbol_state['trade_state']['stop_loss_order_id'] = res['orders']['sl']['id']
                                symbol_state['trade_state']['current_stop_loss'] = potential_new_sl

                is_triggered, reason = symbol_state['trigger_manager'].check_triggers(HISTORICAL_DATA[symbol])
                if is_triggered:
                    add_log(f"Analysis for {symbol} triggered by: {reason}", symbol)
                    analysis_bundle = analyze_freqtrade_data(HISTORICAL_DATA[symbol], current_price)
                    news_text = get_news_from_rss(symbol.split('/')[0])
                    sentiment_score = get_sentiment_score_sync(news_text)
                    pos_report = f"Side: {pos.get('side', 'FLAT')}, Entry: {pos.get('entry_price', 0):.4f}"
                    context_history_list = list(symbol_state['market_context_history'])
                    context_summary_string = "No historical analysis available yet."
                    if context_history_list:
                        formatted_contexts = []
                        for i, context in enumerate(reversed(context_history_list)):
                            ts = context.get('last_full_analysis_timestamp', 'N/A').split('T')[1].split('.')[0]
                            formatted_contexts.append(f"--- Analysis @ {ts} UTC ({i*5} mins ago) ---\n" + json.dumps(context, indent=2))
                        context_summary_string = "\n\n".join(formatted_contexts)
                    df_1d = HISTORICAL_DATA[symbol].resample('1d').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
                    market_regime = get_market_regime(df_1d)
                    add_log(f"Determined Market Regime for {symbol}: {market_regime}", "SYSTEM")
                    decision, new_context, raw_response, chain_of_thought = get_ai_decision_sync(analysis_bundle, pos_report, context_summary_string, vitals['total_equity'], sentiment_score, market_regime)
                    symbol_state['last_ai_response'] = raw_response
                    symbol_state['chain_of_thought'] = chain_of_thought
                    add_log(f"AI Chain of Thought:\n--- START ---\n{chain_of_thought}\n--- END ---", symbol)
                    if new_context: symbol_state['market_context_history'].append(new_context)
                    
                    if decision and decision.get('action'):
                        symbol_state['last_decision'] = decision
                        action = decision['action']
                        
                        if action == 'OPEN_POSITION' and not is_in_position and not symbol_state['trade_state'].get('pending_order_id'):
                            if len(open_positions_map) >= config.MAX_CONCURRENT_POSITIONS:
                                add_log(f"🚨 Max positions reached ({config.MAX_CONCURRENT_POSITIONS}). Skipping open for {symbol}.", symbol)
                                continue
                            qty = calculate_position_size(vitals['total_equity'], vitals['available_margin'], decision.get('risk_percent', 0), decision.get('entry_price', 0), decision.get('stop_loss', 0), decision.get('leverage', 0), symbol, exchange)
                            decision['quantity'] = qty
                            if qty > 0:
                                ai_entry_price = decision.get('entry_price', 0)
                                calculated_notional = qty * ai_entry_price
                                min_notional = 5
                                try: min_notional = exchange.client.markets[symbol]['limits']['cost']['min']
                                except (KeyError, TypeError): add_log(f"⚠️ Could not find minimum notional for {symbol}. Using fallback ${min_notional}.", symbol)
                                if calculated_notional < min_notional:
                                    add_log(f"⚠️ Order size is too small. Calculated Notional: ${calculated_notional:.2f}, Minimum: ${min_notional:.2f}. Skipping.", symbol)
                                else:
                                    exchange.set_leverage_for_symbol(symbol, decision.get('leverage', 20))
                                    time.sleep(1)
                                    if ai_entry_price > 0:
                                        add_log(f"Placing standard Limit order at AI's specified price: {ai_entry_price}", symbol)
                                        res = exchange.place_limit_order(symbol, decision['decision'], qty, ai_entry_price)
                                        if res['status'] == 'success' and res.get('order'):
                                            order_id = res['order']['id']
                                            add_log(f"✅ Limit order placed successfully. Order ID: {order_id}. Waiting for fill.", symbol)
                                            symbol_state['trade_state']['pending_order_id'] = order_id
                                        else:
                                            add_log(f"❌ Failed to place limit order: {res.get('message', 'Unknown error')}", symbol)
                                    else:
                                        add_log(f"⚠️ AI provided an invalid entry price of zero. Skipping trade.", symbol)
                            else:
                                add_log(f"⚠️ Calculated position size is zero. Skipping trade.", symbol)
                        
                        elif action == 'CLOSE_POSITION':
                            if is_in_position:
                                add_log(f"AI decision to CLOSE. Closing active position for {symbol}.", symbol)
                                exchange.close_position_market(symbol, pos)
                            elif symbol_state['trade_state'].get('pending_order_id'):
                                add_log(f"AI decision to CLOSE. Cancelling pending order {symbol_state['trade_state']['pending_order_id']} for {symbol}.", symbol)
                                exchange.cancel_order(symbol_state['trade_state']['pending_order_id'], symbol)
                                symbol_state['trade_state']['pending_order_id'] = None

                        elif action == 'MODIFY_POSITION' and is_in_position:
                            add_log(f"AI decision to MODIFY. Updating SL/TP for {symbol}.", symbol)
                            old_sl_id = symbol_state['trade_state'].get('stop_loss_order_id')
                            old_tp_id = symbol_state['trade_state'].get('take_profit_order_id')
                            if old_sl_id: exchange.cancel_order(old_sl_id, symbol)
                            if old_tp_id: exchange.cancel_order(old_tp_id, symbol)
                            time.sleep(1)
                            res = exchange.modify_protective_orders(symbol, pos['side'], pos['quantity'], decision.get('new_stop_loss'), decision.get('new_take_profit'))
                            if res['status'] == 'success':
                                if 'sl' in res['orders']: symbol_state['trade_state']['stop_loss_order_id'] = res['orders']['sl']['id']
                                if 'tp' in res['orders']: symbol_state['trade_state']['take_profit_order_id'] = res['orders']['tp']['id']
                                if decision.get('new_stop_loss'): symbol_state['trade_state']['current_stop_loss'] = decision.get('new_stop_loss')
                                
                        elif action == 'WAIT':
                            add_log(f"AI decision: WAIT/HOLD. Reason: {decision.get('reasoning', 'N/A')}", symbol)
                            symbol_state['trigger_manager'].set_triggers(decision)
                
                update_bot_state(symbol, is_in_position, pos, {
                    'market_context': list(symbol_state['market_context_history']),
                    'active_triggers': symbol_state['trigger_manager'].triggers,
                    'last_ai_response': symbol_state['last_ai_response'],
                    'chain_of_thought': symbol_state['chain_of_thought'],
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