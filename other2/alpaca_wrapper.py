from __future__ import annotations

import threading
import time
from typing import Any, Callable


class Alpaca:
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        self.client = None
        self.stream = None

        self.stream_thread: threading.Thread | None = None
        self.stream_running = False

        self.trade_update_callbacks: list[
            Callable[[Any], None]
        ] = []

        self._initialize_client()
        self._initialize_stream()

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def _initialize_client(self) -> None:
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

    def _initialize_stream(self) -> None:
        from alpaca.trading.stream import TradingStream

        self.stream = TradingStream(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

    # =========================================================
    # TRADE UPDATE CALLBACKS
    # =========================================================

    def add_trade_update_callback(
        self,
        callback: Callable[[Any], None],
    ) -> None:
        if callback not in self.trade_update_callbacks:
            self.trade_update_callbacks.append(callback)

    async def _handle_trade_update(
        self,
        update: Any,
    ) -> None:
        for callback in self.trade_update_callbacks:
            callback(update)

    # =========================================================
    # START STREAM
    # =========================================================

    def start_trade_update_stream(self) -> None:
        if self.stream is None:
            raise RuntimeError(
                "Alpaca stream is not initialized."
            )

        if self.stream_running:
            return

        self.stream.subscribe_trade_updates(
            self._handle_trade_update
        )

        self.stream_running = True

        self.stream_thread = threading.Thread(
            target=self._run_stream,
            daemon=True,
        )

        self.stream_thread.start()

    def _run_stream(self) -> None:
        while self.stream_running:
            try:
                self.stream.run()

            except Exception as error:
                print(
                    f"[ALPACA STREAM ERROR] {error}"
                )

            if not self.stream_running:
                break

            time.sleep(2)

            print(
                "[ALPACA STREAM] Reconnecting..."
            )

    def stop_trade_update_stream(self) -> None:
        self.stream_running = False

        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass

    # =========================================================
    # ACCOUNT
    # =========================================================

    def get_account(self) -> Any:
        return self.client.get_account()

    def get_account_buying_power(self) -> float:
        return 10_000.0

    # =========================================================
    # ENTRY ORDERS
    # =========================================================

    def place_stop_limit_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float,
    ) -> Any:

        if qty <= 0:
            raise ValueError(
                "Order quantity must be greater than zero."
            )

        if stop_price <= 0:
            raise ValueError(
                "Stop price must be greater than zero."
            )

        if limit_price <= 0:
            raise ValueError(
                "Limit price must be greater than zero."
            )

        from alpaca.trading.enums import (
            OrderSide,
            TimeInForce,
        )

        from alpaca.trading.requests import (
            StopLimitOrderRequest,
        )

        request = StopLimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
            limit_price=round(limit_price, 2),
        )

        return self.client.submit_order(
            order_data=request
        )

    # =========================================================
    # ORDER REPLACEMENT
    # =========================================================

    def replace_order_qty(
        self,
        order_id: str,
        new_qty: int,
    ) -> Any:

        if new_qty <= 0:
            raise ValueError(
                "Replacement quantity must be greater than zero."
            )

        from alpaca.trading.requests import (
            ReplaceOrderRequest,
        )

        request = ReplaceOrderRequest(
            qty=new_qty
        )

        return self.client.replace_order_by_id(
            order_id,
            order_data=request,
        )

    # =========================================================
    # ORDER CANCELLATION
    # =========================================================

    def cancel_order(
        self,
        order_id: str,
    ) -> None:

        self.client.cancel_order_by_id(
            order_id
        )

    def cancel_order_if_exists(
        self,
        order_id: str | None,
    ) -> None:

        if order_id is None:
            return

        try:
            self.cancel_order(
                order_id
            )
        except Exception:
            pass

    # =========================================================
    # ORDER QUERIES
    # =========================================================

    def get_open_orders(
        self,
        symbol: str | None = None,
    ) -> list[Any]:

        orders = self.client.get_orders()

        if symbol is None:
            return list(orders)

        return [
            order
            for order in orders
            if str(
                getattr(order, "symbol", "")
            ).upper()
            == symbol.upper()
        ]

    def get_order(
        self,
        order_id: str,
    ) -> Any:

        return self.client.get_order_by_id(
            order_id
        )

    # =========================================================
    # TAKE PROFIT
    # =========================================================

    def submit_take_profit_order(
        self,
        symbol: str,
        qty: int,
        fill_price: float,
    ) -> Any:

        if qty <= 0:
            raise ValueError(
                "Take-profit quantity must be greater than zero."
            )

        if fill_price <= 0:
            raise ValueError(
                "Fill price must be greater than zero."
            )

        take_profit_price = (
            fill_price * 1.20
        )

        from alpaca.trading.enums import (
            OrderSide,
            TimeInForce,
        )

        from alpaca.trading.requests import (
            LimitOrderRequest,
        )

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            limit_price=round(take_profit_price, 2),
        )

        return self.client.submit_order(
            order_data=request
        )

    # =========================================================
    # MARKET SELL
    # =========================================================

    def submit_market_sell_order(
        self,
        symbol: str,
        qty: int,
    ) -> Any:

        if qty <= 0:
            raise ValueError(
                "Sell quantity must be greater than zero."
            )

        from alpaca.trading.enums import (
            OrderSide,
            TimeInForce,
        )

        from alpaca.trading.requests import (
            MarketOrderRequest,
        )

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        return self.client.submit_order(
            order_data=request
        )

    # =========================================================
    # POSITION QUERIES
    # =========================================================

    def get_position(
        self,
        symbol: str,
    ) -> Any | None:

        try:
            return self.client.get_open_position(
                symbol
            )

        except Exception:
            return None

    def get_position_qty(
        self,
        symbol: str,
    ) -> int:

        position = self.get_position(symbol)

        if position is None:
            return 0

        return int(
            float(position.qty)
        )

    def get_position_fill_price(
        self,
        symbol: str,
    ) -> float | None:

        position = self.get_position(symbol)

        if position is None:
            return None

        return float(
            position.avg_entry_price
        )

    # =========================================================
    # SELL ORDER CLEANUP
    # =========================================================

    def cancel_all_sell_orders(
        self,
        symbol: str,
    ) -> None:

        from alpaca.trading.enums import OrderSide

        orders = self.get_open_orders(
            symbol
        )

        for order in orders:

            if getattr(
                order,
                "side",
                None,
            ) != OrderSide.SELL:
                continue

            order_id = getattr(
                order,
                "id",
                None,
            )

            if order_id is not None:
                self.cancel_order(
                    str(order_id)
                )