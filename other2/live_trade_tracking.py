from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")


@dataclass
class LiveTradeState:
    symbol: str

    total_entry_shares: int
    remaining_shares: int

    entry_timestamp: datetime
    entry_price: float

    sell_count: int = 0

    alpaca_tp_order_id: str | None = None

    pending_sell_order_id: str | None = None
    pending_sell_qty: int = 0

    active: bool = True


class LiveTradeTracking:
    def __init__(
        self,
        alpaca,
        allocation_handler,
        interval_tracking,
        trade_interval_tracking,
    ):
        self.alpaca = alpaca
        self.allocation_handler = allocation_handler
        self.interval_tracking = interval_tracking
        self.trade_interval_tracking = trade_interval_tracking

        self.trades: dict[str, LiveTradeState] = {}

    def get_trade(
        self,
        symbol: str,
    ) -> LiveTradeState | None:
        return self.trades.get(symbol)

    def get_all_trades(
        self,
    ) -> dict[str, LiveTradeState]:
        return self.trades

    def handle_trade_update(
        self,
        update: Any,
    ) -> None:
        order = getattr(update, "order", None)

        if order is None:
            return

        symbol = getattr(order, "symbol", None)

        if symbol is None:
            return

        symbol = str(symbol)

        side_value = getattr(
            getattr(order, "side", None),
            "value",
            None,
        )

        status_value = getattr(
            getattr(order, "status", None),
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

        order_id = getattr(order, "id", None)

        if order_id is not None:
            order_id = str(order_id)

        if side == "buy" and status == "filled":
            self._handle_entry_fill(
                symbol,
                order,
            )
            return

        state = self.trades.get(symbol)

        if state is None:
            return

        if (
            state.alpaca_tp_order_id is not None
            and order_id == state.alpaca_tp_order_id
        ):
            if side == "sell":
                self._handle_tp_update(
                    symbol,
                    order,
                    status,
                )
                return

        if (
            state.pending_sell_order_id is not None
            and order_id == state.pending_sell_order_id
        ):
            if side == "sell":
                self._handle_force_sell_update(
                    symbol,
                    order,
                    status,
                )

    def update(
        self,
        now: datetime | None = None,
    ) -> None:
        if now is None:
            now = datetime.now(NEW_YORK)
        else:
            now = self._normalize_timestamp(now)

        for symbol in list(self.trades):
            state = self.trades.get(symbol)

            if state is None or not state.active:
                continue

            if state.remaining_shares <= 0:
                continue

            if state.pending_sell_order_id is not None:
                continue

            if not self.trade_interval_tracking.is_new_interval(
                symbol,
                now,
            ):
                continue

            interval_index = (
                self.trade_interval_tracking
                .get_interval_index(
                    symbol,
                    now,
                )
            )

            if interval_index < 1:
                continue

            self._submit_scheduled_sell(
                symbol,
                now,
            )

    def _handle_entry_fill(
        self,
        symbol: str,
        order: Any,
    ) -> None:
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

        if fill_price is None or filled_qty is None:
            return

        fill_price = float(fill_price)
        filled_qty = int(float(filled_qty))

        if filled_qty <= 0:
            return

        existing = self.trades.get(symbol)

        if existing is not None and existing.active:
            return

        entry_timestamp = (
            getattr(order, "filled_at", None)
            or getattr(order, "updated_at", None)
            or datetime.now(NEW_YORK)
        )

        entry_timestamp = self._normalize_timestamp(
            entry_timestamp
        )

        state = LiveTradeState(
            symbol=symbol,
            total_entry_shares=filled_qty,
            remaining_shares=filled_qty,
            entry_timestamp=entry_timestamp,
            entry_price=fill_price,
        )

        self.trades[symbol] = state

        self.interval_tracking.set_in_trade(
            symbol,
            True,
        )

        self.trade_interval_tracking.add_trade(
            symbol,
            entry_timestamp,
        )

        tp_order = self.alpaca.submit_take_profit_order(
            symbol=symbol,
            qty=filled_qty,
            fill_price=fill_price,
        )

        tp_order_id = getattr(
            tp_order,
            "id",
            None,
        )

        if tp_order_id is not None:
            state.alpaca_tp_order_id = str(
                tp_order_id
            )


    def register_tp_order(
        self,
        symbol: str,
        order_id: str,
    ) -> None:
        state = self.trades.get(symbol)

        if state is None:
            return

        state.alpaca_tp_order_id = str(
            order_id
        )

    def _submit_scheduled_sell(
        self,
        symbol: str,
        now: datetime,
    ) -> None:
        state = self.trades.get(symbol)

        if state is None or not state.active:
            return

        if state.remaining_shares <= 0:
            return

        if state.pending_sell_order_id is not None:
            return

        state.sell_count += 1

        if state.sell_count >= 5:
            sell_qty = state.remaining_shares
        else:
            sell_qty = int(
                state.total_entry_shares * 0.25
            )

            sell_qty = min(
                sell_qty,
                state.remaining_shares,
            )

        if sell_qty <= 0:
            self._finish_trade(
                symbol
            )
            return

        if state.sell_count >= 5:
            self.alpaca.cancel_order_if_exists(
                state.alpaca_tp_order_id
            )

        else:
            self._reduce_take_profit_order(
                state
            )

        order = self.alpaca.submit_market_sell_order(
            symbol=symbol,
            qty=sell_qty,
        )

        order_id = getattr(
            order,
            "id",
            None,
        )

        if order_id is None:
            state.sell_count -= 1
            return

        state.pending_sell_order_id = str(
            order_id
        )

        state.pending_sell_qty = sell_qty

        self.trade_interval_tracking.mark_interval_processed(
            symbol,
            now,
        )

    def _reduce_take_profit_order(
        self,
        state: LiveTradeState,
    ) -> None:
        if state.alpaca_tp_order_id is None:
            return

        remaining_after_sell = (
            state.remaining_shares
            - int(
                state.total_entry_shares * 0.25
            )
        )

        remaining_after_sell = max(
            0,
            remaining_after_sell,
        )

        if remaining_after_sell <= 0:
            self.alpaca.cancel_order(
                state.alpaca_tp_order_id
            )

            state.alpaca_tp_order_id = None
            return

        try:
            self.alpaca.replace_order_qty(
                state.alpaca_tp_order_id,
                remaining_after_sell,
            )
        except Exception:
            self.alpaca.cancel_order(
                state.alpaca_tp_order_id
            )

            state.alpaca_tp_order_id = None

            tp_order = (
                self.alpaca.submit_take_profit_order(
                    symbol=state.symbol,
                    qty=remaining_after_sell,
                    fill_price=state.entry_price,
                )
            )

            tp_order_id = getattr(
                tp_order,
                "id",
                None,
            )

            if tp_order_id is not None:
                state.alpaca_tp_order_id = str(
                    tp_order_id
                )

    def _handle_tp_update(
        self,
        symbol: str,
        order: Any,
        status: str,
    ) -> None:
        state = self.trades.get(symbol)

        if state is None:
            return

        if status != "filled":
            if status in {
                "canceled",
                "cancelled",
                "rejected",
                "expired",
            }:
                state.alpaca_tp_order_id = None

            return

        filled_price = getattr(
            order,
            "filled_avg_price",
            None,
        )

        filled_qty = getattr(
            order,
            "filled_qty",
            None,
        )

        if filled_price is None or filled_qty is None:
            return

        filled_price = float(filled_price)
        filled_qty = int(float(filled_qty))

        if filled_qty <= 0:
            return

        state.remaining_shares = max(
            0,
            state.remaining_shares - filled_qty,
        )

        state.alpaca_tp_order_id = None

        self.allocation_handler.record_sell_fill(
            symbol=symbol,
            quantity=filled_qty,
            fill_price=filled_price,
        )

        if state.remaining_shares <= 0:
            self._finish_trade(symbol)
        else:
            state.alpaca_tp_order_id = None
    def _handle_force_sell_update(
        self,
        symbol: str,
        order: Any,
        status: str,
    ) -> None:
        state = self.trades.get(symbol)

        if state is None:
            return

        if status in {
            "canceled",
            "cancelled",
            "rejected",
            "expired",
        }:
            state.pending_sell_order_id = None
            state.pending_sell_qty = 0
            return

        if status != "filled":
            return

        filled_price = getattr(
            order,
            "filled_avg_price",
            None,
        )

        filled_qty = getattr(
            order,
            "filled_qty",
            None,
        )

        if filled_price is None or filled_qty is None:
            return

        filled_price = float(filled_price)
        filled_qty = int(float(filled_qty))

        if filled_qty <= 0:
            return

        state.remaining_shares = max(
            0,
            state.remaining_shares - filled_qty,
        )

        state.pending_sell_order_id = None
        state.pending_sell_qty = 0

        self.allocation_handler.record_sell_fill(
            symbol=symbol,
            quantity=filled_qty,
            fill_price=filled_price,
        )

        if state.remaining_shares <= 0:
            self._finish_trade(symbol)
            return

        if state.sell_count >= 5:
            self._finish_trade(symbol)

    def _finish_trade(
        self,
        symbol: str,
    ) -> None:
        state = self.trades.get(symbol)

        if state is None:
            return

        if state.alpaca_tp_order_id is not None:
            try:
                self.alpaca.cancel_order(
                    state.alpaca_tp_order_id
                )
            except Exception:
                pass

            state.alpaca_tp_order_id = None

        state.pending_sell_order_id = None
        state.pending_sell_qty = 0
        state.remaining_shares = 0
        state.active = False

        self.trade_interval_tracking.remove_trade(
            symbol
        )

        self.interval_tracking.set_in_trade(
            symbol,
            False,
        )

    def _normalize_timestamp(
        self,
        timestamp: datetime,
    ) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(
                tzinfo=NEW_YORK
            )

        return timestamp.astimezone(
            NEW_YORK
        )
