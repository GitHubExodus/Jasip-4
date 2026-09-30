from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from allocation_handler import AllocationHandler
from interval_tracking import IntervalTracking
from trading import Trading
from alpaca_wrapper import Alpaca
from trade_interval_tracking import TradeIntervalTracking
from live_trade_tracking import LiveTradeTracking


NEW_YORK = ZoneInfo("America/New_York")


class TradingSystem:
    def __init__(
        self,
        alpaca_api_key: str,
        alpaca_secret_key: str,
        paper: bool = True,
    ):
        self.alpaca = Alpaca(
            api_key=alpaca_api_key,
            secret_key=alpaca_secret_key,
            paper=paper,
        )

        self.allocation_handler = AllocationHandler(
            initial_buying_power=10_000.0
        )

        self.interval_tracking = IntervalTracking(
            interval_minutes=5
        )

        self.trade_interval_tracking = (
            TradeIntervalTracking(
                interval_minutes=5
            )
        )

        self.trading = Trading(
            alpaca=self.alpaca,
            allocation_handler=self.allocation_handler,
            interval_tracking=self.interval_tracking,
            order_update_minutes=5,
            stop_limit_offset_percent=30,
        )

        self.live_trade_tracking = LiveTradeTracking(
            alpaca=self.alpaca,
            allocation_handler=self.allocation_handler,
            interval_tracking=self.interval_tracking,
            trade_interval_tracking=self.trade_interval_tracking,
        )

        self.alpaca.add_trade_update_callback(
            self.trading.handle_trade_update
        )

        self.alpaca.add_trade_update_callback(
            self.live_trade_tracking.handle_trade_update
        )

        self.running = False

    def start(self) -> None:
        if self.running:
            return

        self.running = True

        self.alpaca.start_trade_update_stream()


    def process_screener_update(
        self,
        screener_data: list[dict],
    ) -> None:
        now = datetime.now(NEW_YORK)

        active_rows: dict[str, dict] = {}

        for row in screener_data:
            symbol = row.get("symbol")

            if symbol is None:
                continue

            symbol = str(symbol).upper()

            price = row.get("price")
            percent_change = row.get(
                "percent_change"
            )

            if price is None or percent_change is None:
                continue

            price = float(price)
            percent_change = float(
                percent_change
            )

            active_rows[symbol] = {
                "price": price,
                "percent_change": percent_change,
            }

        active_symbols = set(active_rows)

        for symbol, row in active_rows.items():
            self.allocation_handler.update_stock(
                symbol=symbol,
                price=row["price"],
                percent_change=row["percent_change"],
            )

            self.interval_tracking.update(
                symbol=symbol,
                price=row["price"],
                timestamp=now,
            )

        self.allocation_handler.mark_inactive_stocks(
            active_symbols
        )

        self.trading.update_active_stocks(
            active_symbols
        )

        self.allocation_handler.recalculate()

        for symbol, row in active_rows.items():
            self.trading.update_stock(
                symbol=symbol,
                price=row["price"],
                now=now,
            )

        self.live_trade_tracking.update(
            now=now
        )


    def get_buying_power(self) -> float:
        return (
            self.allocation_handler
            .get_buying_power()
        )

    def get_allocation_states(self):
        return (
            self.allocation_handler
            .get_all_states()
        )

    def get_interval_states(self):
        return (
            self.interval_tracking
            .get_all_states()
        )

    def get_trading_states(self):
        return (
            self.trading
            .get_all_states()
        )

    def get_live_trade_states(self):
        return (
            self.live_trade_tracking
            .get_all_trades()
        )


if __name__ == "__main__":
    API_KEY = "YOUR_ALPACA_API_KEY"
    SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"

    trading_system = TradingSystem(
        alpaca_api_key=API_KEY,
        alpaca_secret_key=SECRET_KEY,
        paper=True,
    )

    trading_system.start()

    print("Trading system started.")

    while True:
        time.sleep(1)