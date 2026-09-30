import os
import math
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request, jsonify
import logging

logging.getLogger("werkzeug").setLevel(logging.ERROR)

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.trading.requests import MarketOrderRequest


API_KEY = "PK6DV6Q3VUIQRY5RL3JCCPIHLY"
SECRET_KEY = "J3zuRUyHaNpBqPGN21NeRdUnhnnHJomLuYbB7VZbsbYu"

BUYING_POWER = 10_000
TRADE_PERCENT = 0.002
STOP_LOSS_PERCENT = 0.005

NY = ZoneInfo("America/New_York")

app = Flask(__name__)
alpaca = TradingClient(API_KEY, SECRET_KEY, paper=True)

stocks = {}
lock = threading.Lock()


@dataclass
class Entry:
    id: str
    shares: int
    entry_price: float
    stop_price: float


@dataclass
class StockState:
    symbol: str
    price: float = 0.0
    candles: dict = field(default_factory=dict)
    entries: list = field(default_factory=list)
    active: bool = False
    last_buy_bucket: str | None = None
    last_sell_bucket: str | None = None


def minute_bucket():
    return datetime.now(NY).strftime("%Y-%m-%d %H:%M")


def candle_valid(timeframe, now):
    minute = now.minute
    hour = now.hour

    if timeframe == "5m":
        return minute % 5 != 0

    if timeframe == "15m":
        return minute % 15 != 0

    if timeframe == "30m":
        return minute % 30 != 0

    if timeframe == "1h":
        return minute != 0

    if timeframe == "4h":
        return not (minute == 0 and hour % 4 == 0)

    if timeframe == "1d":
        return not (hour == 0 and minute == 0)

    return True


def market_is_open():
    return alpaca.get_clock().is_open


def buy_condition(state):
    now = datetime.now(NY)

    for timeframe in ("5m", "15m", "30m", "1h", "4h", "1d"):
        if not candle_valid(timeframe, now):
            continue

        value = state.candles.get(timeframe)

        if value is None or value <= 0:
            return False

    return True


def target_shares(price):
    if price <= 0:
        return 0

    amount = BUYING_POWER * TRADE_PERCENT
    return math.floor(amount / price)


def submit_buy(state, bucket):
    if state.last_buy_bucket == bucket:
        return

    shares = target_shares(state.price)

    if shares <= 0:
        return

    order = alpaca.submit_order(
        MarketOrderRequest(
            symbol=state.symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
    )

    while True:
        filled = alpaca.get_order_by_id(order.id)

        if filled.status == OrderStatus.FILLED:
            break

        if filled.status in {
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }:
            return

        time.sleep(0.05)

    fill_price = float(filled.filled_avg_price)
    filled_shares = int(float(filled.filled_qty))

    entry = Entry(
        id=str(filled.id),
        shares=filled_shares,
        entry_price=fill_price,
        stop_price=fill_price * (1 - STOP_LOSS_PERCENT),
    )

    state.entries.append(entry)
    state.last_buy_bucket = bucket

    print(
        f"ENTRY | {state.symbol} | "
        f"{filled_shares} | ${fill_price:.4f} | "
        f"STOP ${entry.stop_price:.4f}"
    )


def submit_stop_loss(state, bucket):
    if state.last_sell_bucket == bucket:
        return

    stopped_entry = None

    for entry in state.entries:
        if state.price <= entry.stop_price:
            stopped_entry = entry
            break

    if stopped_entry is None:
        return

    order = alpaca.submit_order(
        MarketOrderRequest(
            symbol=state.symbol,
            qty=stopped_entry.shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
    )

    while True:
        filled = alpaca.get_order_by_id(order.id)

        if filled.status == OrderStatus.FILLED:
            break

        if filled.status in {
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }:
            return

        time.sleep(0.05)

    state.entries.remove(stopped_entry)
    state.last_sell_bucket = bucket

    fill_price = float(filled.filled_avg_price)

    print(
        f"EXIT | {state.symbol} | "
        f"{stopped_entry.shares} | ${fill_price:.4f} | STOP"
    )


def process_stock(state):
    if not state.active:
        return

    if not market_is_open():
        return

    bucket = minute_bucket()

    submit_stop_loss(state, bucket)

    if state.last_buy_bucket != bucket and buy_condition(state):
        submit_buy(state, bucket)


def update_screener(data):
    with lock:
        for item in data:
            symbol = str(item.get("symbol") or item.get("TickerUniversal") or "").strip()

            if not symbol:
                continue

            if symbol not in stocks:
                stocks[symbol] = StockState(symbol=symbol)

            state = stocks[symbol]

            price = item.get("Price")

            if isinstance(price, (int, float)) and price > 0:
                state.price = float(price)

            state.candles = {
                "1m": item.get("ChangeFromOpen|TimeResolution1"),
                "5m": item.get("ChangeFromOpen|TimeResolution5"),
                "15m": item.get("ChangeFromOpen|TimeResolution15"),
                "30m": item.get("ChangeFromOpen|TimeResolution30"),
                "1h": item.get("ChangeFromOpen|TimeResolution60"),
                "4h": item.get("ChangeFromOpen|TimeResolution240"),
                "1d": item.get("ChangeFromOpen|TimeResolution1D"),
            }

            state.active = True


@app.post("/screener")
def screener():
    data = request.get_json(silent=True)

    if not isinstance(data, list):
        return jsonify({"error": "invalid payload"}), 400

    update_screener(data)

    return jsonify({"ok": True})


def trading_loop():
    last_bucket = None

    while True:
        now_bucket = minute_bucket()

        if now_bucket != last_bucket:
            last_bucket = now_bucket

            with lock:
                current_stocks = list(stocks.values())

            for state in current_stocks:
                try:
                    process_stock(state)
                except Exception:
                    pass

        time.sleep(0.1)


if __name__ == "__main__":
    threading.Thread(
        target=trading_loop,
        daemon=True,
    ).start()

    app.run(
        host="127.0.0.1",
        port=8765,
        threaded=True,
    )