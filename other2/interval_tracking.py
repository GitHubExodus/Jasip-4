from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")


@dataclass
class IntervalState:
    prev_max_high: float | None = None
    current_max_high: float = 0.0
    in_trade: bool = False
    is_new_interval: bool = False
    interval_key: int | None = None


class IntervalTracking:
    def __init__(self, interval_minutes: int = 30):
        self.interval_minutes = interval_minutes
        self.stocks: dict[str, IntervalState] = {}

    def add_stock(self, symbol: str) -> None:
        if symbol not in self.stocks:
            self.stocks[symbol] = IntervalState()

    def remove_stock(self, symbol: str) -> None:
        self.stocks.pop(symbol, None)

    def get_state(self, symbol: str) -> IntervalState | None:
        return self.stocks.get(symbol)

    def get_prev_max_high(self, symbol: str) -> float | None:
        state = self.stocks.get(symbol)

        if state is None:
            return None

        return state.prev_max_high

    def get_current_max_high(self, symbol: str) -> float | None:
        state = self.stocks.get(symbol)

        if state is None:
            return None

        return state.current_max_high

    def is_in_trade(self, symbol: str) -> bool:
        state = self.stocks.get(symbol)

        if state is None:
            return False

        return state.in_trade

    def set_in_trade(self, symbol: str, value: bool) -> None:
        self.add_stock(symbol)
        self.stocks[symbol].in_trade = value

    def is_ready_to_trade(self, symbol: str) -> bool:
        state = self.stocks.get(symbol)

        if state is None:
            return False

        return (
            state.prev_max_high is not None
            and not state.in_trade
        )

    def get_interval_key(self, timestamp: datetime) -> int:
        timestamp = timestamp.astimezone(NEW_YORK)

        total_minutes = (
            timestamp.hour * 60
            + timestamp.minute
        )

        return (
            total_minutes // self.interval_minutes
        ) * self.interval_minutes

    def update(
        self,
        symbol: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> IntervalState:
        self.add_stock(symbol)

        if timestamp is None:
            timestamp = datetime.now(NEW_YORK)
        else:
            timestamp = timestamp.astimezone(NEW_YORK)

        state = self.stocks[symbol]

        current_interval_key = self.get_interval_key(timestamp)

        # First time this stock is seen.
        if state.interval_key is None:
            state.interval_key = current_interval_key
            state.current_max_high = price
            state.is_new_interval = False

            return state

        # New 30-minute interval.
        if current_interval_key != state.interval_key:
            state.prev_max_high = state.current_max_high
            state.current_max_high = price
            state.in_trade = False
            state.is_new_interval = True
            state.interval_key = current_interval_key

            return state

        # Normal tick inside the current interval.
        state.is_new_interval = False
        state.current_max_high = max(
            state.current_max_high,
            price,
        )

        return state

    def get_all_states(self) -> dict[str, IntervalState]:
        return self.stocks
