# execution_manager.py

import ccxt
import time
from typing import List, Dict, Any
import pandas as pd

# --- Local Imports ---
import config

class ExchangeManager:
    def __init__(self):
        """Initializes the CCXT client configuration."""
        client_config = {
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 10000,
            },
        }
        self.client = getattr(ccxt, 'binance')(client_config)
        if config.BINANCE_TESTNET:
            self.client.set_sandbox_mode(True)
            print("✅ CCXT client configured for SANDBOX (Testnet) mode.")
        
        print("✅ Exchange Manager configured.")

    def connect_and_load_markets(self, retries=5, delay=5):
        """
        Connects to the exchange, loads markets, and verifies the connection.
        This is a critical, robust initialization step.
        """
        for i in range(retries):
            try:
                print(f"Attempting to connect and load markets (Attempt {i+1}/{retries})...")
                self.client.load_markets()
                
                if 'BTC/USDT:USDT' in self.client.markets:
                    print("✅ Exchange markets loaded successfully.")
                    print(f"   - Found {len(self.client.markets)} markets.")
                    return True
                else:
                    raise ccxt.ExchangeError("Market loading verification failed: 'BTC/USDT:USDT' not found.")

            except Exception as e:
                print(f"❌ Connection/Market loading failed: {e}")
                if i < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("❌ All attempts to connect and load markets have failed. Aborting.")
                    return False

    def get_account_vitals(self) -> Dict[str, float]:
        """Synchronously retrieves global account equity and available margin."""
        try:
            balance = self.client.fetch_balance({'type': 'future'})
            usdt_info = balance.get('USDT', {})
            return {
                'total_equity': usdt_info.get('total', 0.0),
                'available_margin': usdt_info.get('free', 0.0),
            }
        except Exception as e:
            print(f"❌ Error fetching account vitals: {e}")
            return {'total_equity': 0.0, 'available_margin': 0.0}

    def get_current_mark_price(self, symbol: str) -> float:
        """Fetches the current mark price for a specific symbol."""
        try:
            ticker = self.client.fetch_ticker(symbol)
            return float(ticker.get('mark', ticker.get('last', 0.0)))
        except Exception as e:
            print(f"❌ Error fetching mark price for {symbol}: {e}")
            return 0.0

    def set_leverage_for_symbol(self, symbol: str, leverage: int):
        """Sets the leverage for a specific symbol."""
        try:
            market_id = self.client.market(symbol)['id']
            self.client.set_leverage(leverage, market_id)
            print(f"✅ Leverage for {symbol} set to {leverage}x.")
            return True
        except Exception as e:
            print(f"❌ Failed to set leverage for {symbol}: {e}")
            return False

    def get_orderbook_ticker(self, symbol: str) -> Dict:
        """
        Fetches the best bid and ask prices with a built-in retry mechanism
        to handle transient network errors.
        """
        retries = 3
        for i in range(retries):
            try:
                ticker = self.client.fetch_ticker(symbol)
                if ticker and ticker.get('bid') is not None and ticker.get('ask') is not None:
                    return {'bid': ticker['bid'], 'ask': ticker['ask']}
            except Exception as e:
                print(f"⚠️ [Attempt {i+1}/{retries}] Could not fetch orderbook ticker for {symbol}: {e}")
            if i < retries - 1:
                time.sleep(0.5)
        return {}

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float):
        """Synchronously executes a standard Limit order."""
        try:
            formatted_price = self.client.price_to_precision(symbol, price)
            formatted_quantity = self.client.amount_to_precision(symbol, quantity)
            ccxt_side = 'buy' if side.upper() == 'LONG' else 'sell'
            
            order = self.client.create_order(
                symbol=symbol, 
                type='LIMIT', 
                side=ccxt_side,
                amount=formatted_quantity, 
                price=formatted_price
            )
            return {'status': 'success', 'order': order}
        except ccxt.InsufficientFunds as e:
            return {'status': 'error', 'message': f"Insufficient Funds: {e}"}
        except Exception as e:
            return {'status': 'error', 'message': f"Error on order placement for {symbol}: {e}"}

    def place_market_order(self, symbol: str, side: str, quantity: float):
        """Synchronously executes a Market order."""
        try:
            formatted_quantity = self.client.amount_to_precision(symbol, quantity)
            ccxt_side = 'buy' if side.upper() == 'LONG' else 'sell'
            order = self.client.create_order(
                symbol=symbol, type='MARKET', side=ccxt_side,
                amount=formatted_quantity
            )
            return {'status': 'success', 'order': order}
        except Exception as e:
            return {'status': 'error', 'message': f"Error on market order placement for {symbol}: {e}"}

    def fetch_order_status(self, order_id: str, symbol: str):
        """Fetches the status of a specific order."""
        try:
            return self.client.fetch_order(order_id, symbol)
        except Exception as e:
            print(f"⚠️ Could not fetch order status for {order_id}: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str):
        """Cancels a specific order."""
        try:
            self.client.cancel_order(order_id, symbol)
            print(f"✅ Order {order_id} for {symbol} cancelled.")
            return True
        except Exception as e:
            print(f"⚠️ Could not cancel order {order_id}: {e}")
            return False

    def modify_protective_orders(self, symbol: str, side: str, quantity: float, new_sl: float = None, new_tp: float = None):
        """
        Places new protective orders (SL/TP). It no longer cancels all orders beforehand.
        The calling logic is now responsible for managing existing orders.
        """
        close_side = 'sell' if side.upper() == 'LONG' else 'buy'
        formatted_quantity = self.client.amount_to_precision(symbol, quantity)
        
        placed_orders = {}
        try:
            if new_tp and new_tp > 0:
                tp_order = self.client.create_order(
                    symbol=symbol, 
                    type='TAKE_PROFIT_MARKET', 
                    side=close_side,
                    amount=formatted_quantity, 
                    stopPrice=self.client.price_to_precision(symbol, new_tp),
                    params={'closePosition': True}
                )
                placed_orders['tp'] = tp_order
                print(f"✅ Placed Take Profit for {symbol} at {new_tp}")

            if new_sl and new_sl > 0:
                sl_order = self.client.create_order(
                    symbol=symbol, 
                    type='STOP_MARKET', 
                    side=close_side,
                    amount=formatted_quantity, 
                    stopPrice=self.client.price_to_precision(symbol, new_sl),
                    params={'closePosition': True}
                )
                placed_orders['sl'] = sl_order
                print(f"✅ Placed Stop Loss for {symbol} at {new_sl}")
            
            return {'status': 'success', 'orders': placed_orders}

        except Exception as e:
            for order_type, order in placed_orders.items():
                try:
                    self.cancel_order(order['id'], symbol)
                except Exception as cancel_e:
                    print(f"⚠️ Failed to clean up partially placed protective order: {cancel_e}")
            return {'status': 'error', 'message': f"Failed to modify protective orders for {symbol}: {e}"}

    def close_position_market(self, symbol: str, position: Dict):
        """Executes a market order to close a specific position."""
        if not position.get('side'): return {'status': 'error', 'message': 'No position to close.'}
        close_side = 'buy' if position['side'].upper() == 'SHORT' else 'sell'
        formatted_quantity = self.client.amount_to_precision(symbol, position['quantity'])
        try:
            order = self.client.create_order(
                symbol=symbol, type='MARKET', side=close_side,
                amount=formatted_quantity, params={'reduceOnly': True} 
            )
            return {'status': 'success', 'order': order}
        except Exception as e:
            return {'status': 'error', 'message': f"Failed to market close {symbol}: {e}"}

    def fetch_account_trade_list(self, symbol: str, limit: int) -> List[Dict]:
        """Fetches recent trades for PNL calculation."""
        try:
            return self.client.fetch_my_trades(symbol, limit=limit)
        except Exception as e:
            print(f"❌ Error fetching trade list for {symbol}: {e}")
            return []
            
    # ########################################################################### #
    # ################## START OF MODIFIED SECTION ############################## #
    # ########################################################################### #
    def fetch_positions(self, symbols: List[str]) -> List[Dict]:
        """
        Fetches current position details for a list of symbols, ensuring correct
        and reliable retrieval by passing the symbols list to the underlying CCXT call.
        """
        if not symbols:
            return []
        try:
            # CRITICAL FIX: Pass the 'symbols' list directly to the ccxt call.
            # This is the most robust way to ensure we get immediate and accurate
            # position data for the symbols we care about.
            positions = self.client.fetch_positions(symbols=symbols)
            
            active_positions = []
            for pos in positions:
                if pos.get('contracts') is not None and float(pos['contracts']) != 0:
                    active_positions.append({
                        "symbol": pos['symbol'], 
                        "side": pos['side'].upper(),
                        "quantity": float(pos['contracts']), 
                        "entry_price": float(pos['entryPrice']),
                        "unrealized_pnl": float(pos.get('unrealizedPnl', 0.0))
                    })
            return active_positions
        except Exception as e:
            print(f"❌ Error fetching positions: {e}")
            return []
    # ########################################################################### #
    # ################### END OF MODIFIED SECTION ############################### #
    # ########################################################################### #

    def fetch_historical_klines(self, symbols: List[str], timeframe: str, limit: int) -> Dict[str, List[List]]:
        """Fetches klines for multiple symbols sequentially."""
        klines_map = {}
        for symbol in symbols:
            try:
                result = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
                klines_map[symbol] = result
            except Exception as e:
                print(f"❌ Error fetching k-lines for {symbol}: {e}")
                klines_map[symbol] = []
        return klines_map

    def fetch_full_historical_data(self, symbol: str, timeframe: str, days_of_data: int = 60) -> List[List]:
        """Fetches a large, continuous historical dataset for a symbol using multiple paginated requests."""
        print(f"💧 Fetching full historical data for {symbol} ({days_of_data} days)...")
        try:
            ms_per_candle = self.client.parse_timeframe(timeframe) * 1000
            candles_per_day = (24 * 60 * 60 * 1000) / ms_per_candle
            total_candles_needed = int(candles_per_day * days_of_data)
            limit_per_call = 1000
            all_klines = []
            since = self.client.milliseconds() - (total_candles_needed * ms_per_candle)

            while len(all_klines) < total_candles_needed:
                klines = self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_call)
                if not klines: break
                all_klines.extend(klines)
                since = klines[-1][0] + ms_per_candle
                print(f"   - Fetched {len(klines)} candles for {symbol}, total: {len(all_klines)}/{total_candles_needed}")
                time.sleep(self.client.rateLimit / 1000)

            df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            df.sort_values(by='timestamp', inplace=True)
            print(f"✅ Successfully fetched {len(df)} unique historical candles for {symbol}.")
            return df.values.tolist()
        except Exception as e:
            print(f"❌ CRITICAL ERROR fetching full history for {symbol}: {e}")
            return []