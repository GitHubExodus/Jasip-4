from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import time

API_KEY = "PK6DV6Q3VUIQRY5RL3JCCPIHLY"
SECRET_KEY = "J3zuRUyHaNpBqPGN21NeRdUnhnnHJomLuYbB7VZbsbYu"

client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)


positions = client.get_all_positions()

for p in positions:
    symbol = p.symbol

    print(f"Cancelling orders for {symbol}...")

    try:
        client.cancel_orders()
    except Exception as e:
        print("Cancel error:", e)

    time.sleep(0.5)

print("\nRemaining orders:")

orders = client.get_orders()
for o in orders:
    print(o.symbol, o.side, o.qty, o.status, o.id)

print("\nPositions:")

for p in client.get_all_positions():
    print(p.symbol, p.qty)