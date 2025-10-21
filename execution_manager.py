# Execution_Manager.py (Final, Complete Asynchronous CCXT Pro Module)

import ccxt.pro as ccxt # Use ccxt.pro for async and websocket support
import asyncio
from typing import List, Dict, Union, Any

# --- Local Imports ---
import config

class ExchangeManager:
    def __init__(self):
        """Initializes the single, reusable async CCXT client."""
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
            print("✅ CCXT client is in SANDBOX (Testnet) mode.")
        
        print("✅ Exchange Manager initialized.")

    async def load_markets(self):
        """Loads markets to enable precision functions. Call once at startup."""
        await self.client.load_markets()
        print("✅ Exchange markets loaded.")
        
    async def close_client(self):
        """Properly closes the client connection on shutdown."""
        if self.client:
            await self.client.close()
            print("✅ CCXT client connection closed.")

    async def get_account_vitals(self) -> Dict[str, float]:
        """Asynchronously retrieves global account equity and available margin."""
        try:
            balance = await self.client.fetch_balance({'type': 'future'})
            usdt_info = balance.get('USDT', {})
            return {
                'total_equity': usdt_info.get('total', 0.0),
                'available_margin': usdt_info.get('free', 0.0),
            }
        except Exception as e:
            print(f"❌ Error fetching account vitals: {e}")
            return {'total_equity': 0.0, 'available_margin': 0.0}

    async def get_current_mark_price(self, symbol: str) -> float:
        """Fetches the current mark price for a specific symbol."""
        try:
            ticker = await self.client.fetch_ticker(symbol)
            return float(ticker.get('mark', ticker.get('last', 0.0)))
        except Exception as e:
            print(f"❌ Error fetching mark price for {symbol}: {e}")
            return 0.0

    async def fetch_historical_klines(self, symbols: List[str], timeframe: str, limit: int) -> Dict[str, List[List]]:
        """Asynchronously fetches klines for multiple symbols concurrently and returns a dictionary."""
        print(f"💧 Hydrating historical k-lines for {len(symbols)} symbols...")
        
        # --- CORRECTED LOGIC ---
        # Create a list of awaitable tasks, one for each symbol
        tasks = [self.client.fetch_ohlcv(symbol, timeframe, limit=limit) for symbol in symbols]
        
        # Await all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Create a dictionary mapping each symbol to its result
        klines_map = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                print(f"❌ Error fetching initial k-lines for {symbol}: {result}")
                klines_map[symbol] = []
            else:
                klines_map[symbol] = result
        
        return klines_map
        # --- END CORRECTION ---

    async def fetch_positions(self, symbols: List[str]) -> List[Dict]:
        """Fetches current position details for a list of symbols."""
        try:
            positions = await self.client.fetch_positions(symbols)
            active_positions = []
            for pos in positions:
                if pos.get('contracts') is not None and float(pos['contracts']) != 0:
                    active_positions.append({
                        "symbol": pos['symbol'], "side": pos['side'].upper(),
                        "quantity": float(pos['contracts']), "entry_price": float(pos['entryPrice']),
                        "unrealized_pnl": float(pos.get('unrealizedPnl', 0.0))
                    })
            return active_positions
        except Exception as e:
            print(f"❌ Error fetching positions: {e}")
            return []

    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float):
        """Asynchronously executes a Limit Post-Only order."""
        try:
            formatted_price = self.client.price_to_precision(symbol, price)
            formatted_quantity = self.client.amount_to_precision(symbol, quantity)
            ccxt_side = side.lower()
            
            order = await self.client.create_order(
                symbol=symbol, type='LIMIT', side=ccxt_side,
                amount=formatted_quantity, price=formatted_price, params={'postOnly': True}
            )
            return {'status': 'success', 'order': order}
        except ccxt.InsufficientFunds as e:
            return {'status': 'error', 'message': f"Insufficient Funds: {e}"}
        except Exception as e:
            return {'status': 'error', 'message': f"Error on order placement for {symbol}: {e}"}

    async def modify_protective_orders(self, symbol: str, side: str, quantity: float, new_sl: float = None, new_tp: float = None):
        """Asynchronously cancels all existing conditional orders and places new ones."""
        try:
            await self.client.cancel_all_orders(symbol)
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Could not cancel orders for {symbol}, proceeding: {e}")

        close_side = 'sell' if side.upper() == 'LONG' else 'buy'
        formatted_quantity = self.client.amount_to_precision(symbol, quantity)
        
        try:
            tasks = []
            if new_tp and new_tp > 0:
                tp_params = {'stopPrice': self.client.price_to_precision(symbol, new_tp), 'closePosition': True}
                tasks.append(self.client.create_order(
                    symbol=symbol, type='TAKE_PROFIT_MARKET', side=close_side,
                    amount=formatted_quantity, params=tp_params
                ))
            if new_sl and new_sl > 0:
                sl_params = {'stopPrice': self.client.price_to_precision(symbol, new_sl), 'closePosition': True}
                tasks.append(self.client.create_order(
                    symbol=symbol, type='STOP_MARKET', side=close_side,
                    amount=formatted_quantity, params=sl_params
                ))
            
            if not tasks: return {'status': 'noop', 'message': 'No new SL or TP provided.'}
            await asyncio.gather(*tasks, return_exceptions=False)
            return {'status': 'success'}

        except Exception as e:
            try: await self.client.cancel_all_orders(symbol)
            except: pass
            return {'status': 'error', 'message': f"Failed to modify protective orders for {symbol}: {e}"}

    async def close_position_market(self, symbol: str, position: Dict):
        """Asynchronously executes a market order to close a specific position."""
        if not position.get('side'): return {'status': 'error', 'message': 'No position to close.'}
        close_side = 'buy' if position['side'].upper() == 'SHORT' else 'sell'
        formatted_quantity = self.client.amount_to_precision(symbol, position['quantity'])
        try:
            await self.client.cancel_all_orders(symbol)
            await asyncio.sleep(0.5)
            order = await self.client.create_order(
                symbol=symbol, type='MARKET', side=close_side,
                amount=formatted_quantity, params={'reduceOnly': True}
            )
            return {'status': 'success', 'order': order}
        except Exception as e:
            return {'status': 'error', 'message': f"Failed to market close {symbol}: {e}"}

    async def fetch_account_trade_list(self, symbol: str, limit: int) -> List[Dict]:
        """Asynchronously fetches recent trades for PNL calculation."""
        try:
            return await self.client.fetch_my_trades(symbol, limit=limit)
        except Exception as e:
            print(f"❌ Error fetching trade list for {symbol}: {e}")
            return []