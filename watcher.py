# watcher.py

import asyncio
from binance import AsyncClient, BinanceSocketManager
import requests
import time
import config_template

# --- Telegram Notification Function ---
def send_telegram_notification(message):
    """Sends a formatted message to your Telegram."""
    bot_token = config_template.TELEGRAM_BOT_TOKEN
    chat_id = config_template.TELEGRAM_CHAT_ID
    
    # Using a code block in Telegram for clean formatting
    formatted_message = f"```\n{message}\n```"
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": formatted_message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Sent notification: {message.splitlines()[0]}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Telegram notification: {e}")

# --- WebSocket Message Processor ---
async def process_message(msg):
    """Processes messages from the Binance User Data Stream."""
    if msg.get('e') == 'ORDER_TRADE_UPDATE':
        order_data = msg.get('o', {})
        
        symbol = order_data.get('s')
        side = order_data.get('S')
        order_type = order_data.get('o')
        quantity = order_data.get('q')
        
        execution_type = order_data.get('x')
        order_status = order_data.get('X')
        
        if execution_type == 'NEW':
            message = (
                f"🔔 New Order Placed 🔔\n\n"
                f"Symbol: {symbol}\n"
                f"Side:   {side}\n"
                f"Type:   {order_type}\n"
                f"Qty:    {quantity}\n"
            )
            send_telegram_notification(message)

        elif execution_type == 'TRADE': # A fill occurred
            trade_price = order_data.get('L')
            trade_qty = order_data.get('l')
            total_cost = float(order_data.get('n', 0))
            
            action = "✅ Position Opened/Increased" if order_status in ['PARTIALLY_FILLED', 'FILLED'] else "💰 Position Closed/Reduced"
            
            message = (
                f"{action} ✅\n\n"
                f"Symbol:    {symbol}\n"
                f"Side:      {side}\n"
                f"Fill Qty:  {trade_qty}\n"
                f"Fill Price:{trade_price}\n"
                f"Total Cost:{total_cost:.2f} USDT\n"
                f"Status:    {order_status}\n"
            )
            send_telegram_notification(message)

# --- Main Execution ---
async def main():
    """Initializes the client and starts the WebSocket listener."""
    print("="*40)
    print("🛡️ Binance Account Watcher 🛡️")
    print("="*40)
    print("Listening for all account activity in real-time...")
    
    client = await AsyncClient.create(config_template.BINANCE_API_KEY, config_template.BINANCE_API_SECRET, testnet=config_template.BINANCE_TESTNET)
    bm = BinanceSocketManager(client)
    
    user_socket = bm.user_socket()
    
    async with user_socket as stream:
        while True:
            try:
                res = await stream.recv()
                await process_message(res)
            except Exception as e:
                print(f"An error occurred in the WebSocket stream: {e}")
                time.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())