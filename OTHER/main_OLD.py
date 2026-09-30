import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.stream import TradingStream


# ============================================================
# CONFIG
# ============================================================

ALPACA_API_KEY = "PKA4A6THLEKI6QD2MQPOAO25J3"
ALPACA_SECRET_KEY = "4nj9w53vMrNKJGZsHqN7Siqy34z2Gis9TffWi2beszNU"

PAPER_TRADING = True

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765

STARTING_BUYING_POWER = 10_000.00

ENTRY_CHECK_SECONDS = 60

INTERVAL_MINUTES = 30

FIRST_MINUTE_BUFFER = 0.01

TAKE_PROFIT_PERCENT = 0.15

TRAILING_STOP_PERCENT = 0.10


# ============================================================
# TIME
# ============================================================

NY = ZoneInfo("America/New_York")


def now_ny():
    return datetime.now(NY)


def get_interval_start(dt):
    dt = dt.astimezone(NY)

    minute = (dt.hour * 60) + dt.minute
    interval_minute = (minute // 30) * 30

    return dt.replace(
        hour=interval_minute // 60,
        minute=interval_minute % 60,
        second=0,
        microsecond=0,
    )


# ============================================================
# DATA
# ============================================================

@dataclass
class Interval:
    start: datetime
    end: datetime
    max_price: float


@dataclass
class Stock:
    symbol: str

    price: float = 0.0
    percent_change: float = 0.0

    current_interval_start: Optional[datetime] = None
    current_interval_max: float = 0.0

    intervals: dict = field(default_factory=dict)

    entry_order_id: Optional[str] = None
    entry_order_qty: int = 0

    desired_shares: int = 0


@dataclass
class Trade:
    trade_id: str
    symbol: str

    entry_order_id: str

    qty: int
    entry_price: float
    entry_time: datetime

    take_profit: float
    high_water_mark: float
    trailing_stop: float

    exit_order_id: Optional[str] = None

    closed: bool = False
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None


# ============================================================
# STATE
# ============================================================

lock = threading.RLock()

stocks = {}
trades = {}

buying_power = STARTING_BUYING_POWER
trading_day = now_ny().date()


# ============================================================
# ALPACA
# ============================================================

alpaca = TradingClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    paper=PAPER_TRADING,
)


# ============================================================
# LOG
# ============================================================

def log(message):
    print(
        f"[{now_ny().strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        flush=True,
    )


# ============================================================
# DAILY RESET
# ============================================================

def check_day():
    global trading_day
    global buying_power

    today = now_ny().date()

    if today != trading_day:
        trading_day = today
        buying_power = STARTING_BUYING_POWER

        log(
            f"NEW DAY | BUYING POWER=${buying_power:.2f}"
        )


# ============================================================
# INTERVALS
# ============================================================

def update_interval(stock, price, current_time):
    current_start = get_interval_start(current_time)

    if stock.current_interval_start is None:
        stock.current_interval_start = current_start
        stock.current_interval_max = price
        return

    if current_start != stock.current_interval_start:

        old_start = stock.current_interval_start
        old_end = old_start + timedelta(minutes=30)

        if stock.current_interval_max > 0:
            stock.intervals[old_start] = Interval(
                start=old_start,
                end=old_end,
                max_price=stock.current_interval_max,
            )

        stock.current_interval_start = current_start
        stock.current_interval_max = price

    else:

        stock.current_interval_max = max(
            stock.current_interval_max,
            price,
        )


def get_previous_interval(stock):
    if stock.current_interval_start is None:
        return None

    previous_start = (
        stock.current_interval_start
        - timedelta(minutes=30)
    )

    previous = stock.intervals.get(
        previous_start
    )

    if previous is None:
        return None

    if previous.end != stock.current_interval_start:
        return None

    return previous


def get_entry_price(stock, current_time):
    previous = get_previous_interval(stock)

    if previous is None:
        return None

    price = previous.max_price

    interval_start = stock.current_interval_start

    if current_time < interval_start + timedelta(minutes=1):
        price *= 1.0 + FIRST_MINUTE_BUFFER

    return price


# ============================================================
# ALLOCATION
# ============================================================

def calculate_allocations(rows):
    positive = [
        row
        for row in rows
        if row["percent_change"] > 0
        and row["price"] > 0
    ]

    total = sum(
        row["percent_change"]
        for row in positive
    )

    if total <= 0:
        return {}

    with lock:
        power = buying_power

    result = {}

    for row in positive:

        allocation = (
            row["percent_change"] / total
        )

        dollars = power * allocation

        shares = math.floor(
            dollars / row["price"]
        )

        result[row["symbol"]] = shares

    return result


# ============================================================
# SCREENER UPDATE
# ============================================================

def process_screener(rows):
    check_day()

    current_time = now_ny()

    allocations = calculate_allocations(rows)

    symbols_seen = set()

    with lock:

        for row in rows:

            symbol = row["symbol"]
            price = row["price"]
            pct = row["percent_change"]

            symbols_seen.add(symbol)

            if symbol not in stocks:
                stocks[symbol] = Stock(
                    symbol=symbol
                )

                log(
                    f"NEW STOCK | {symbol}"
                )

            stock = stocks[symbol]

            stock.price = price
            stock.percent_change = pct

            update_interval(
                stock,
                price,
                current_time,
            )

            stock.desired_shares = allocations.get(
                symbol,
                0,
            )

    manage_entry_orders()


# ============================================================
# ENTRY ORDERS
# ============================================================

def manage_entry_orders():
    current_time = now_ny()

    with lock:
        stock_list = list(stocks.values())

    for stock in stock_list:

        entry_price = get_entry_price(
            stock,
            current_time,
        )

        desired = stock.desired_shares

        existing_id = stock.entry_order_id

        # No valid previous interval.
        if entry_price is None:
            continue

        # Nothing allocated.
        if desired <= 0:
            continue

        # Already have a buy order waiting/filling.
        if existing_id:
            continue

        # Synthetic buy-stop:
        # Do not buy until the current price reaches
        # the fixed entry level.
        if stock.price < entry_price:
            continue

        submit_entry(
            stock,
            desired,
            entry_price,
        )

def submit_entry(stock, qty, price):

    if qty <= 0:
        return

    request = LimitOrderRequest(
        symbol=stock.symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=round(price, 2),
    )

    try:

        order = alpaca.submit_order(
            order_data=request
        )

        with lock:
            stock.entry_order_id = str(order.id)
            stock.entry_order_qty = qty

        log(
            f"BUY ORDER | {stock.symbol} | "
            f"QTY={qty} | "
            f"PRICE=${price:.2f}"
        )

    except Exception as exc:

        log(
            f"BUY ERROR | {stock.symbol} | {exc}"
        )


def cancel_entry(stock):

    order_id = stock.entry_order_id

    if not order_id:
        return

    with lock:
        stock.entry_order_id = None
        stock.entry_order_qty = 0

    try:

        alpaca.cancel_order_by_id(
            order_id
        )

        log(
            f"CANCEL BUY | {stock.symbol}"
        )

    except Exception as exc:

        log(
            f"CANCEL BUY ERROR | "
            f"{stock.symbol} | {exc}"
        )


# ============================================================
# FILLED BUY
# ============================================================

def handle_buy_fill(order):

    global buying_power

    qty = int(float(order.filled_qty))

    if qty <= 0:
        return

    price = float(
        order.filled_avg_price
    )

    symbol = order.symbol

    trade_id = str(uuid.uuid4())

    trade = Trade(
        trade_id=trade_id,
        symbol=symbol,
        entry_order_id=str(order.id),
        qty=qty,
        entry_price=price,
        entry_time=now_ny(),
        take_profit=price * (
            1.0 + TAKE_PROFIT_PERCENT
        ),
        high_water_mark=price,
        trailing_stop=price * (
            1.0 - TRAILING_STOP_PERCENT
        ),
    )

    spent = qty * price

    with lock:

        trades[trade_id] = trade

        buying_power -= spent

        stock = stocks.get(symbol)

        if stock:
            stock.entry_order_id = None
            stock.entry_order_qty = 0

    log(
        f"BUY FILLED | {symbol} | "
        f"QTY={qty} | "
        f"PRICE=${price:.4f} | "
        f"SPENT=${spent:.2f} | "
        f"BP=${buying_power:.2f}"
    )


# ============================================================
# TRADE PRICE UPDATE
# ============================================================

def update_trade_price(symbol, price):

    with lock:

        matching = [
            trade
            for trade in trades.values()
            if (
                trade.symbol == symbol
                and not trade.closed
                and trade.exit_order_id is None
            )
        ]

        for trade in matching:

            if price > trade.high_water_mark:
                trade.high_water_mark = price

                trade.trailing_stop = (
                    price *
                    (1.0 - TRAILING_STOP_PERCENT)
                )

            if price >= trade.take_profit:

                submit_exit(
                    trade,
                    "TAKE_PROFIT",
                )

            elif price <= trade.trailing_stop:

                submit_exit(
                    trade,
                    "TRAILING_STOP",
                )


# ============================================================
# EXIT ORDER
# ============================================================

def submit_exit(trade, reason):

    if trade.closed:
        return

    if trade.exit_order_id is not None:
        return

    qty = trade.qty

    if qty <= 0:
        return

    # Reserve this trade's quantity immediately.
    # This prevents the same trade from creating
    # multiple sell orders.

    request = MarketOrderRequest(
        symbol=trade.symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )

    try:

        order = alpaca.submit_order(
            order_data=request
        )

        trade.exit_order_id = str(order.id)

        log(
            f"SELL ORDER | {trade.symbol} | "
            f"QTY={qty} | "
            f"REASON={reason} | "
            f"TRADE={trade.trade_id}"
        )

    except Exception as exc:

        log(
            f"SELL ERROR | "
            f"{trade.symbol} | "
            f"TRADE={trade.trade_id} | "
            f"{exc}"
        )


# ============================================================
# FILLED SELL
# ============================================================

def handle_sell_fill(order):

    global buying_power

    order_id = str(order.id)

    qty = int(float(order.filled_qty))

    if qty <= 0:
        return

    price = float(
        order.filled_avg_price
    )

    with lock:

        trade = None

        for candidate in trades.values():

            if candidate.exit_order_id == order_id:
                trade = candidate
                break

        if trade is None:

            log(
                f"UNKNOWN SELL FILL | "
                f"ORDER={order_id}"
            )

            return

        trade.closed = True
        trade.exit_price = price
        trade.exit_time = now_ny()

        proceeds = qty * price

        buying_power += proceeds

    log(
        f"SELL FILLED | {trade.symbol} | "
        f"QTY={qty} | "
        f"PRICE=${price:.4f} | "
        f"PROCEEDS=${proceeds:.2f} | "
        f"BP=${buying_power:.2f} | "
        f"TRADE={trade.trade_id}"
    )


# ============================================================
# ALPACA TRADE STREAM
# ============================================================

async def on_trade_update(data):

    event = str(data.event)

    order = data.order

    if event != "fill":
        return

    if order.side == OrderSide.BUY:

        handle_buy_fill(order)

    elif order.side == OrderSide.SELL:

        handle_sell_fill(order)


def alpaca_stream_loop():

    stream = TradingStream(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        paper=PAPER_TRADING,
    )

    stream.subscribe_trade_updates(
        on_trade_update
    )

    log("ALPACA STREAM STARTED")

    stream.run()


# ============================================================
# ENTRY CHECK LOOP
# ============================================================

def entry_loop():

    while True:

        try:
            check_day()
            manage_entry_orders()

        except Exception as exc:

            log(
                f"ENTRY LOOP ERROR | {exc}"
            )

        time.sleep(
            ENTRY_CHECK_SECONDS
        )


# ============================================================
# HTTP SERVER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    def cors(self):

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_POST(self):

        if self.path != "/screener":

            self.send_response(404)

            self.cors()

            self.end_headers()

            return

        try:

            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            body = self.rfile.read(length)

            data = json.loads(
                body.decode("utf-8")
            )

            rows = []

            for item in data:

                symbol = str(
                    item["symbol"]
                ).upper().strip()

                price = float(
                    item["price"]
                )

                pct = float(
                    item["percent_change"]
                )

                if (
                    symbol
                    and price > 0
                ):

                    rows.append({
                        "symbol": symbol,
                        "price": price,
                        "percent_change": pct,
                    })

            process_screener(rows)

            response = json.dumps({
                "ok": True,
                "rows": len(rows),
            }).encode()

            self.send_response(200)

            self.cors()

            self.send_header(
                "Content-Type",
                "application/json",
            )

            self.send_header(
                "Content-Length",
                str(len(response)),
            )

            self.end_headers()

            self.wfile.write(response)

        except Exception as exc:

            log(
                f"HTTP ERROR | {exc}"
            )

            response = json.dumps({
                "ok": False,
                "error": str(exc),
            }).encode()

            self.send_response(500)

            self.cors()

            self.send_header(
                "Content-Type",
                "application/json",
            )

            self.send_header(
                "Content-Length",
                str(len(response)),
            )

            self.end_headers()

            self.wfile.write(response)

    def log_message(self, format, *args):
        return


# ============================================================
# MAIN SERVER
# ============================================================

def main():

    log("SYSTEM STARTING")

    log(
        f"NEW YORK TIME | "
        f"{now_ny().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    threading.Thread(
        target=alpaca_stream_loop,
        daemon=True,
    ).start()

    threading.Thread(
        target=entry_loop,
        daemon=True,
    ).start()

    server = ThreadingHTTPServer(
        (
            SERVER_HOST,
            SERVER_PORT,
        ),
        Handler,
    )

    log(
        f"SCREENER SERVER | "
        f"http://{SERVER_HOST}:{SERVER_PORT}/screener"
    )

    server.serve_forever()


if __name__ == "__main__":
    main()