from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")


@dataclass
class TradingState:
    symbol: str

    is_active: bool = False

    active_order_id: str | None = None
    active_order_qty: int = 0

    fill_price: float | None = None
    position_qty: int = 0

    last_order_update: datetime | None = None


class Trading:
    def __init__(
        self,
        alpaca,
        allocation_handler,
        interval_tracking,
        live_trade_tracking=None,
        order_update_minutes: int = 5,
        stop_limit_offset_percent: float = 30,
    ):
        self.alpaca = alpaca
        self.allocation_handler = allocation_handler
        self.interval_tracking = interval_tracking
        self.live_trade_tracking = live_trade_tracking

        self.order_update_minutes = order_update_minutes
        self.stop_limit_offset_percent = (
            stop_limit_offset_percent
        )

        self.stocks: dict[str, TradingState] = {}

    # =========================================================
    # STOCK MANAGEMENT
    # =========================================================

    def add_stock(
        self,
        symbol: str,
    ) -> None:

        if symbol not in self.stocks:
            self.stocks[symbol] = TradingState(
                symbol=symbol
            )

    def get_state(
        self,
        symbol: str,
    ) -> TradingState | None:

        return self.stocks.get(symbol)

    # =========================================================
    # ACTIVE STOCK SYNCHRONIZATION
    # =========================================================

    def update_active_stocks(
        self,
        active_symbols: set[str],
    ) -> None:

        for symbol in active_symbols:
            self.add_stock(symbol)

        for symbol, state in self.stocks.items():

            was_active = state.is_active

            state.is_active = (
                symbol in active_symbols
            )

            if (
                was_active
                and not state.is_active
            ):
                self._handle_stock_inactive(
                    symbol
                )

    # =========================================================
    # INACTIVE STOCK
    # =========================================================

    def _handle_stock_inactive(
        self,
        symbol: str,
    ) -> None:

        state = self.stocks.get(symbol)

        if state is None:
            return

        # A stock disappearing from the screener
        # does not sell an existing position.
        #
        # Only cancel an unfilled BUY order.

        if state.active_order_id is not None:

            self.alpaca.cancel_order(
                state.active_order_id
            )

            state.active_order_id = None
            state.active_order_qty = 0
            state.last_order_update = None

    # =========================================================
    # MAIN TRADING UPDATE
    # =========================================================

    def update_stock(
        self,
        symbol: str,
        price: float,
        now: datetime | None = None,
    ) -> None:

        self.add_stock(symbol)

        state = self.stocks[symbol]

        if not state.is_active:
            return

        if now is None:
            now = datetime.now(
                NEW_YORK
            )
        else:
            now = now.astimezone(
                NEW_YORK
            )

        interval_state = (
            self.interval_tracking.get_state(
                symbol
            )
        )

        if interval_state is None:
            return

        # -----------------------------------------------------
        # Already in a trade.
        #
        # No new BUY orders or BUY order updates.
        # -----------------------------------------------------

        if interval_state.in_trade:
            return

        # -----------------------------------------------------
        # A previous interval must exist.
        # -----------------------------------------------------

        if interval_state.prev_max_high is None:
            return

        # -----------------------------------------------------
        # Only update orders every 5 minutes.
        #
        # The breakout itself is evaluated every JS tick.
        # -----------------------------------------------------

        if (
            state.last_order_update is not None
            and (
                now
                - state.last_order_update
            )
            < timedelta(
                minutes=self.order_update_minutes
            )
        ):
            return

        target_shares = (
            self.allocation_handler
            .get_target_shares(symbol)
        )

        # -----------------------------------------------------
        # Price has not broken previous interval high.
        # -----------------------------------------------------

        if (
            price
            <= interval_state.prev_max_high
        ):
            return

        # -----------------------------------------------------
        # No shares allocated.
        # -----------------------------------------------------

        if target_shares <= 0:

            if state.active_order_id is not None:

                self.alpaca.cancel_order(
                    state.active_order_id
                )

                state.active_order_id = None
                state.active_order_qty = 0

            state.last_order_update = now

            return

        # -----------------------------------------------------
        # Existing order with same quantity.
        # -----------------------------------------------------

        if (
            state.active_order_id is not None
            and (
                state.active_order_qty
                == target_shares
            )
        ):
            state.last_order_update = now
            return

        # -----------------------------------------------------
        # Existing order with different quantity.
        # -----------------------------------------------------

        if state.active_order_id is not None:

            self.alpaca.cancel_order(
                state.active_order_id
            )

            state.active_order_id = None
            state.active_order_qty = 0

        # -----------------------------------------------------
        # Create STOP-LIMIT BUY.
        # -----------------------------------------------------

        stop_price = (
            interval_state.prev_max_high
        )

        limit_price = (
            stop_price
            * (
                1.0
                + (
                    self.stop_limit_offset_percent
                    / 100.0
                )
            )
        )

        order = (
            self.alpaca.place_stop_limit_order(
                symbol=symbol,
                qty=target_shares,
                stop_price=stop_price,
                limit_price=limit_price,
            )
        )

        order_id = getattr(
            order,
            "id",
            None,
        )

        if order_id is None:
            raise RuntimeError(
                "Alpaca returned an order without an ID."
            )

        state.active_order_id = str(
            order_id
        )

        state.active_order_qty = (
            target_shares
        )

        state.last_order_update = now

    # =========================================================
    # ALPACA TRADE UPDATE
    # =========================================================

    def handle_trade_update(
        self,
        update: Any,
    ) -> None:

        event = getattr(
            update,
            "event",
            None,
        )

        order = getattr(
            update,
            "order",
            None,
        )

        if order is None:
            return

        symbol = getattr(
            order,
            "symbol",
            None,
        )

        if symbol is None:
            return

        symbol = str(symbol).upper()

        self.add_stock(symbol)

        state = self.stocks[symbol]

        # =====================================================
        # ALPACA ENUM VALUES
        # =====================================================

        side_value = getattr(
            getattr(
                order,
                "side",
                None,
            ),
            "value",
            None,
        )

        status_value = getattr(
            getattr(
                order,
                "status",
                None,
            ),
            "value",
            None,
        )

        side = (
            str(side_value).lower()
            if side_value is not None
            else ""
        )

        status = (
            str(status_value).lower()
            if status_value is not None
            else ""
        )

        # =====================================================
        # BUY FILLED
        # =====================================================

        if (
            event == "fill"
            and side == "buy"
        ):
            self._handle_buy_fill(
                symbol,
                order,
            )

            return

        # =====================================================
        # BUY CANCELED / REJECTED / EXPIRED
        # =====================================================

        if (
            side == "buy"
            and status in {
                "canceled",
                "cancelled",
                "rejected",
                "expired",
            }
        ):

            order_id = getattr(
                order,
                "id",
                None,
            )

            if (
                state.active_order_id is not None
                and order_id is not None
                and str(order_id)
                == str(
                    state.active_order_id
                )
            ):
                state.active_order_id = None
                state.active_order_qty = 0
                state.last_order_update = None

    # =========================================================
    # BUY FILL
    # =========================================================

    def _handle_buy_fill(
        self,
        symbol: str,
        order: Any,
    ) -> None:

        state = self.stocks[symbol]

        fill_price = getattr(
            order,
            "filled_avg_price",
            None,
        )

        filled_qty = getattr(
            order,
            "filled_qty",
            None,
        )

        if fill_price is None:
            return

        if filled_qty is None:
            return

        fill_price = float(
            fill_price
        )

        filled_qty = int(
            float(filled_qty)
        )

        if filled_qty <= 0:
            return

        # -----------------------------------------------------
        # Clear entry order.
        # -----------------------------------------------------
        self.alpaca.cancel_order_if_exists(
            state.active_order_id
        )

        state.active_order_id = None
        state.active_order_qty = 0
        state.last_order_update = None

        state.fill_price = fill_price
        state.position_qty = filled_qty

        # -----------------------------------------------------
        # Lock this stock for the current 30-minute interval.
        # -----------------------------------------------------

        self.interval_tracking.set_in_trade(
            symbol,
            True,
        )

        # -----------------------------------------------------
        # Deduct actual money spent.
        # -----------------------------------------------------

        self.allocation_handler.record_buy_fill(
            symbol=symbol,
            quantity=filled_qty,
            fill_price=fill_price,
        )

        # -----------------------------------------------------
        # Submit 20% take-profit.
        # -----------------------------------------------------
       

    # =========================================================
    # FORCE RESET ENTRY ORDER
    # =========================================================

    def cancel_entry_order(
        self,
        symbol: str,
    ) -> None:

        state = self.stocks.get(symbol)

        if state is None:
            return

        if state.active_order_id is None:
            return

        self.alpaca.cancel_order(
            state.active_order_id
        )

        state.active_order_id = None
        state.active_order_qty = 0
        state.last_order_update = None

    # =========================================================
    # STATE ACCESS
    # =========================================================

    def get_all_states(
        self,
    ) -> dict[str, TradingState]:

        return self.stocks